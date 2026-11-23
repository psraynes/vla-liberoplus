"""Separate, resumable standard LIBERO 4-suite × 10-task × 50-state evaluation."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from smolvla_adapter import Adapter


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = {"libero_spatial": 280, "libero_object": 280,
             "libero_goal": 300, "libero_10": 520}
WAIT_STEPS = 10
DUMMY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def evaluate_episode(env, adapter, instruction, init_state, max_steps):
    env.reset()
    obs = env.set_init_state(init_state)
    adapter.reset()
    for _ in range(WAIT_STEPS):
        obs, _, _, _ = env.step(DUMMY)
    for step in range(max_steps):
        action = adapter.predict(obs, instruction)
        obs, _, _, _ = env.step(action[0].tolist())
        if env.check_success():
            return True, step + 1
    return False, max_steps


def load_existing(path):
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = (row["suite"], row["task_id"], row["episode"])
                if key in done:
                    raise ValueError(f"Duplicate existing result: {key}")
                done.add(key)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--task-id", type=int, default=None,
                        help="Pilot: run only this task ID in each suite")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50:
        raise ValueError("Use 1–50 fixed initial states per task")

    path = Path(benchmark.__file__).resolve()
    if "LIBERO-standard" not in str(path):
        raise RuntimeError(f"Expected original LIBERO checkout, imported {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.jsonl"
    done = load_existing(result_path)
    manifest_path = out / "manifest.json"
    manifest = {
        "model": os.environ["SMOLVLA_MODEL"], "seed_rule": "7 + episode",
        "episodes_per_task": args.episodes, "task_id_filter": args.task_id,
        "action_steps": 10, "libero_import": str(path),
        "libero_git_commit": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
        "suite_max_steps": MAX_STEPS,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Existing run manifest differs; choose a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    suites = {name: benchmark.get_benchmark_dict()[name]() for name in SUITES}
    for name, suite in suites.items():
        if suite.get_num_tasks() != 10:
            raise RuntimeError(f"Expected 10 original tasks in {name}; got {suite.get_num_tasks()}")
    adapter = Adapter()
    if adapter.policy.config.n_action_steps != 10:
        raise RuntimeError("Expected ten executed actions per model query")

    for suite_name, suite in suites.items():
        selected = [args.task_id] if args.task_id is not None else range(10)
        for task_id in selected:
            if not 0 <= task_id < 10:
                raise ValueError(task_id)
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            if len(init_states) < args.episodes:
                raise RuntimeError(f"Only {len(init_states)} states for {suite_name}/{task_id}")
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            for episode in range(args.episodes):
                key = (suite_name, task_id, episode)
                if key in done:
                    continue
                seed = 7 + episode
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                env = None
                try:
                    env = OffScreenRenderEnv(
                        bddl_file_name=str(bddl),
                        camera_heights=adapter.image_size,
                        camera_widths=adapter.image_size,
                    )
                    env.seed(seed)
                    success, steps = evaluate_episode(
                        env, adapter, task.language, init_states[episode], MAX_STEPS[suite_name]
                    )
                finally:
                    if env is not None:
                        env.close()
                row = {"suite": suite_name, "task_id": task_id, "episode": episode,
                       "seed": seed, "success": bool(success), "steps": steps,
                       "instruction": task.language, "task_name": task.name}
                with result_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                done.add(key)
                print(f"{suite_name}/{task_id}/{episode}: success={success} steps={steps}", flush=True)
    expected = len(SUITES) * (1 if args.task_id is not None else 10) * args.episodes
    if len(done) != expected:
        raise RuntimeError(f"Completed {len(done)}/{expected} episodes")
    print(f"PASS: {expected} original LIBERO episodes complete at {result_path}")


if __name__ == "__main__":
    main()
