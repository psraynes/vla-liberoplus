#!/usr/bin/env python3
"""Record successful, non-evaluation LIBERO+ teacher rollouts as labeled episodes.

Teacher adapter contract: image_size=256; reset(); predict(obs, instruction) ->
one or more *unnormalized* relative 7-D actions ready for env.step().
"""

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from build_teacher_plan import heldout_pairs


MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def axis_angle_xyzw(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("Invalid end-effector quaternion")
    w = float(np.clip(q[3], -1, 1))
    den = np.sqrt(max(0.0, 1.0 - w * w))
    if den < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(q[:3] * (2 * np.arccos(w) / den), dtype=np.float32)


def observation_row(obs: dict, action: np.ndarray) -> dict:
    state = np.concatenate((
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        axis_angle_xyzw(obs["robot0_eef_quat"]),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    ))
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Invalid 8-D robot state")
    images = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        image = np.ascontiguousarray(obs[key][::-1, ::-1])
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError(f"Unexpected {key} image shape/dtype")
        images.append(image)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Teacher action is not a finite 7-D LIBERO action")
    return {"camera1": images[0], "camera2": images[1], "state": state, "action": action.copy()}


def rollout(env, adapter, instruction: str, initial_state: np.ndarray, max_steps: int):
    env.reset()
    obs = env.set_init_state(initial_state)
    adapter.reset()
    queue = []
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY_ACTION.tolist())
    rows = []
    for _ in range(max_steps):
        if not queue:
            proposed = np.asarray(adapter.predict(obs, instruction), dtype=np.float32)
            if proposed.ndim == 1:
                proposed = proposed[None, :]
            if proposed.ndim != 2 or proposed.shape[1] != 7 or len(proposed) == 0:
                raise ValueError(f"Teacher adapter returned {proposed.shape}; expected N x 7 actions")
            queue = list(proposed)
        action = np.asarray(queue.pop(0), dtype=np.float32)
        rows.append(observation_row(obs, action))  # pre-action observation, action passed to env.step
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True, rows
    return False, rows


def save_episode(path: Path, rows: list[dict], instruction: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as target:
        np.savez_compressed(
            target,
            camera1=np.stack([row["camera1"] for row in rows]),
            camera2=np.stack([row["camera2"] for row in rows]),
            state=np.stack([row["state"] for row in rows]),
            action=np.stack([row["action"] for row in rows]),
            instruction=np.asarray(instruction),
        )
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--paired-plan", type=Path, required=True)
    parser.add_argument("--adapter", required=True, help="Import path module:Class")
    parser.add_argument("--adapter-path", type=Path, help="Directory containing teacher adapter code")
    parser.add_argument("--teacher-checkpoint", required=True, help="Exact model checkpoint/revision for provenance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attempts-per-task", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0)
    args = parser.parse_args()
    if args.attempts_per_task < 1 or args.max_tasks < 0:
        parser.error("Attempts must be positive; max-tasks must be nonnegative")
    if args.adapter_path:
        sys.path.insert(0, str(args.adapter_path.resolve()))
    module_name, class_name = args.adapter.split(":", 1)

    heldout = heldout_pairs(args.paired_plan)
    with args.plan.open(encoding="utf-8", newline="") as source:
        plan = list(csv.DictReader(source, delimiter="\t"))
    if not plan or len({(r["suite"], r["task_id"]) for r in plan}) != len(plan):
        raise ValueError("Collection plan is empty or has duplicate task variants")
    for row in plan:
        key = row["suite"], int(row["task_id"])
        if key in heldout or row["category"] not in ("Camera Viewpoints", "Sensor Noise"):
            raise ValueError(f"Unsafe collection plan row: {row}")
    if args.max_tasks:
        plan = plan[: args.max_tasks]

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_info = {
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "paired_plan_sha256": hashlib.sha256(args.paired_plan.read_bytes()).hexdigest(),
        "teacher_checkpoint": args.teacher_checkpoint,
        "seed": args.seed,
        "attempts_per_task": args.attempts_per_task,
        "max_tasks": args.max_tasks,
    }
    info_path = output / "run_info.json"
    if info_path.exists():
        if json.loads(info_path.read_text(encoding="utf-8")) != run_info:
            raise ValueError("Collection provenance changed; refusing to resume into an existing output")
    else:
        info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    manifest = output / "attempts.jsonl"
    attempted = set()
    if manifest.exists():
        with manifest.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    record = json.loads(line)
                    if record.get("success") is True:
                        episode = (output / record["episode_file"]).resolve()
                        if not episode.is_relative_to(output) or not episode.is_file():
                            raise FileNotFoundError(f"Recorded successful episode is missing: {episode}")
                    attempted.add((record["suite"], int(record["task_id"]), int(record["seed"])))

    adapter = getattr(importlib.import_module(module_name), class_name)()
    if adapter.image_size != 256:
        raise ValueError("Teacher adapter must request 256-pixel cameras")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suites = {}
    for row in plan:
        suite, task_id = row["suite"], int(row["task_id"])
        if suite not in suites:
            suites[suite] = benchmark.get_benchmark_dict()[suite]()
        task_suite = suites[suite]
        for attempt in range(args.attempts_per_task):
            seed = args.seed + attempt
            if (suite, task_id, seed) in attempted:
                continue
            seed_everything(seed)
            env = None
            started = time.time()
            record = {"suite": suite, "task_id": task_id, "category": row["category"],
                      "difficulty": int(row["difficulty"]), "split": row["split"],
                      "seed": seed, "teacher_checkpoint": args.teacher_checkpoint}
            try:
                task = task_suite.get_task(task_id)
                initial_state = task_suite.get_task_init_states(task_id)[0]
                env = OffScreenRenderEnv(
                    bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                    camera_heights=256, camera_widths=256,
                )
                env.seed(seed)
                success, rows = rollout(env, adapter, task.language, initial_state, MAX_STEPS[suite])
                record.update(success=bool(success), steps=len(rows), instruction=task.language,
                              max_abs_action=float(max(np.max(np.abs(r["action"])) for r in rows)))
                if success:
                    path = output / "episodes" / f"{suite}_task{task_id:04d}_seed{seed:04d}.npz"
                    save_episode(path, rows, task.language)
                    record["episode_file"] = str(path.relative_to(output))
            except Exception as error:
                record.update(success=False, steps=0, error=repr(error)[:500])
            finally:
                if env is not None:
                    env.close()
            record["seconds"] = round(time.time() - started, 1)
            with manifest.open("a", encoding="utf-8") as target:
                target.write(json.dumps(record) + "\n")
                target.flush()
                os.fsync(target.fileno())
            attempted.add((suite, task_id, seed))
            print(f"{suite}/{task_id} seed={seed} success={record['success']} steps={record['steps']}", flush=True)
            # LIBERO+ can retain asset file handles across task variants. The job
            # launcher restarts this process and resumes from attempts.jsonl.
            if os.path.isdir("/proc/self/fd"):
                import resource
                open_fds = len(os.listdir("/proc/self/fd"))
                threshold = min(50000, resource.getrlimit(resource.RLIMIT_NOFILE)[0] // 4)
                if open_fds > threshold:
                    print(f"Restart requested after {open_fds} open file handles", flush=True)
                    return 2
    print(f"Attempt log: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
