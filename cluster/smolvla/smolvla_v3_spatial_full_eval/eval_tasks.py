"""Evaluate a model on LIBERO-Plus tasks in plan.tsv order, one trial per task.

The episode protocol (VLA-Adapter / OpenVLA-style LIBERO rollout) lives here and is the same for every
model; the model only enters through the adapter (see model_adapter.py). The model is loaded once and
each task's result is appended to results.jsonl as soon as it finishes. Tasks already in results.jsonl
are skipped, as are tasks with two recorded hangs/crashes. Before each task, current.json records which
task is in progress so the watchdog in run_plan.sh can attribute a hang.

Exit codes: 0 plan finished, 2 restart requested (leaked file handles), 3 stopped (deadline or stop.sh).

    python eval_tasks.py [--deadline EPOCH_SECONDS]
"""

import argparse
import csv
import collections
import importlib
import json
import logging
import os
import random
import resource
import sys
import time
from pathlib import Path

import imageio
import numpy as np

import kitconfig

KIT = kitconfig.KIT
# Parallel workers (set by eval_job.sbatch): worker k of N takes plan rows k, k+N, ... and writes its own files.
WORKER, NWORKERS = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
SUFFIX = f".{WORKER}" if NWORKERS > 1 else ""
RESULTS, CURRENT = KIT / f"results{SUFFIX}.jsonl", KIT / f"current{SUFFIX}.json"
TASK_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
NUM_STEPS_WAIT = 10
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
# Worst-case seconds per task before durations are measured (step limit at ~10 steps/s).
DEFAULT_TASK_SECONDS = {"libero_spatial": 60, "libero_object": 60, "libero_goal": 70, "libero_10": 120}


def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_state():
    done, hangs, durations = set(), collections.Counter(), collections.defaultdict(list)
    for results in sorted(KIT.glob("results*.jsonl")):  # all workers' results
        for line in open(results):
            record = json.loads(line)
            key = (record["suite"], record["task_id"])
            if record.get("hung"):
                hangs[key] += 1
            else:
                done.add(key)
                durations[record["suite"]].append(record["seconds"])
    return done, hangs, durations


def expected_seconds(durations, suite):
    recent = sorted(durations[suite][-30:])
    if len(recent) < 5:
        return DEFAULT_TASK_SECONDS[suite]
    return recent[int(0.9 * (len(recent) - 1))] + 15  # 90th percentile plus env-creation margin


def append_result(record):
    record["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_episode(env, adapter, instruction, initial_state, max_steps):
    """One episode; returns (success, frames). Frames are agentview|wrist, rotated upright."""
    env.reset()
    obs = env.set_init_state(initial_state)
    adapter.reset()
    queue, frames = [], []
    for t in range(max_steps + NUM_STEPS_WAIT):
        if t < NUM_STEPS_WAIT:
            obs, _, _, _ = env.step(DUMMY_ACTION)
            continue
        frames.append(np.concatenate([obs["agentview_image"][::-1, ::-1], obs["robot0_eye_in_hand_image"][::-1, ::-1]], axis=1))
        if not queue:
            queue = [np.asarray(a, dtype=np.float32) for a in adapter.predict(obs, instruction)]
        action = queue.pop(0) if queue else np.asarray(DUMMY_ACTION, dtype=np.float32)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True, frames
    return False, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", type=float, default=None, help="stop before a task that would end after this epoch time")
    args = parser.parse_args()

    kitconfig.load()
    seed = int(os.environ.get("SEED", 7))
    save_video = os.environ.get("SAVE_VIDEO", "1") == "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    plan = [line.rstrip("\n").split("\t") for line in open(KIT / "plan.tsv") if not line.startswith("#")]
    plan = [(row[1], int(row[2])) for row in plan]
    with (KIT / "instruction_map.csv").open(newline="", encoding="utf-8") as handle:
        instruction_map = {(row["suite"], int(row["task_id"])): row for row in csv.DictReader(handle)}
    if len(instruction_map) != 10030 or any(key not in instruction_map for key in plan):
        raise ValueError("Clean instruction map does not cover the frozen benchmark plan")
    done, hangs, durations = load_state()
    todo = [key for i, key in enumerate(plan) if i % NWORKERS == WORKER and key not in done and hangs[key] < 2]
    logging.info("worker %d/%d | plan: %d tasks, %d remaining for this worker, %d skipped after repeated hangs",
                 WORKER, NWORKERS, len(plan), len(todo), sum(1 for key in plan if key not in done and hangs[key] >= 2))
    if not todo:
        return 0

    module_name, class_name = os.environ["ADAPTER"].split(":")
    sys.path.insert(0, str(KIT))
    adapter = getattr(importlib.import_module(module_name), class_name)()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suites = {}
    for suite, task_id in todo:
        if (KIT / "STOP").exists():  # created by stop.sh; exit like a deadline stop
            logging.info("stop requested (stop.sh); stopping before %s %d", suite, task_id)
            CURRENT.unlink(missing_ok=True)
            return 3
        if args.deadline and time.time() + expected_seconds(durations, suite) > args.deadline:
            logging.info("next task (%s %d) would not finish by the deadline; stopping", suite, task_id)
            CURRENT.unlink(missing_ok=True)
            return 3
        if suite not in suites:
            suites[suite] = benchmark.get_benchmark_dict()[suite]()
        task_suite = suites[suite]
        CURRENT.write_text(json.dumps({"suite": suite, "task_id": task_id, "pid": os.getpid(), "started": time.time()}))

        t0 = time.time()
        set_seed_everywhere(seed)
        env = None
        try:
            task = task_suite.get_task(task_id)
            mapped = instruction_map[suite, task_id]
            if task.name != mapped["task_name"]:
                raise ValueError(f"Task-name mismatch for {suite}/{task_id}")
            if mapped["category"] == "Language_Instructions":
                instruction = task.language  # Preserve the intentional language perturbation.
            else:
                if task.language != mapped["official_instruction"]:
                    raise ValueError(f"Official instruction mismatch for {suite}/{task_id}")
                instruction = mapped["clean_instruction"]
            initial_states = task_suite.get_task_init_states(task_id)
            # The bddl file name carries LIBERO-Plus's camera / robot-state / noise settings.
            env = OffScreenRenderEnv(
                bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                camera_heights=adapter.image_size, camera_widths=adapter.image_size,
            )
            env.seed(seed)
            success, frames = run_episode(env, adapter, instruction, initial_states[0], TASK_MAX_STEPS[suite])
        except Exception as error:  # counted like a hang: retried once, then skipped
            logging.exception("%s task=%d raised", suite, task_id)
            append_result({"suite": suite, "task_id": task_id, "hung": True, "error": repr(error)[:300]})
            continue
        finally:
            if env is not None:
                env.close()
        seconds = time.time() - t0
        if save_video:
            path = KIT / "videos" / suite / f"task{task_id:04d}_{'success' if success else 'failure'}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(path, frames, fps=20)
        append_result({"suite": suite, "task_id": task_id, "success": bool(success), "seconds": round(seconds, 1)})
        durations[suite].append(seconds)
        logging.info("%s task=%d success=%s (%.0fs)", suite, task_id, success, seconds)

        # LIBERO-Plus leaks open asset files (thousands per Objects Layout task); restart well before the limit.
        open_fds = len(os.listdir("/proc/self/fd"))
        threshold = int(os.environ.get("FD_RESTART", 0)) or min(50000, resource.getrlimit(resource.RLIMIT_NOFILE)[0] // 4)
        if open_fds > threshold:
            logging.info("%d open file handles; restarting the evaluator to release them", open_fds)
            CURRENT.unlink(missing_ok=True)
            return 2

    CURRENT.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
