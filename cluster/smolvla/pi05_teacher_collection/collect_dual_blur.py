#!/usr/bin/env python3
"""Collect PI05 trajectories with sigma-4 blur in both cameras."""

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

import cv2
import numpy as np

from build_teacher_plan import heldout_pairs
from collect_teacher_rollouts import (DUMMY_ACTION, MAX_STEPS, observation_row,
                                      save_episode, seed_everything)


def rollout(env, adapter, instruction: str, initial_state: np.ndarray, max_steps: int):
    """The environment blurs agentview; blur wrist before both prediction and capture."""
    env.reset()
    obs = env.set_init_state(initial_state)
    adapter.reset()
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY_ACTION.tolist())
    queue = []
    rows = []
    for _ in range(max_steps):
        policy_obs = dict(obs)
        wrist = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)
        policy_obs["robot0_eye_in_hand_image"] = cv2.GaussianBlur(wrist, (0, 0), sigmaX=4.0)
        if not queue:
            proposed = np.asarray(adapter.predict(policy_obs, instruction), dtype=np.float32)
            if proposed.ndim == 1:
                proposed = proposed[None, :]
            if proposed.ndim != 2 or proposed.shape[1] != 7 or len(proposed) == 0:
                raise ValueError(f"Teacher returned {proposed.shape}; expected N x 7 actions")
            queue = list(proposed)
        action = np.asarray(queue.pop(0), dtype=np.float32)
        rows.append(observation_row(policy_obs, action))
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True, rows
    return False, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--paired-plan", type=Path, required=True)
    parser.add_argument("--adapter", default="pi05_adapter:Adapter")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    with args.plan.open(encoding="utf-8", newline="") as source:
        plan = list(csv.DictReader(source, delimiter="\t"))
    if not plan or len({(r["suite"], int(r["task_id"])) for r in plan}) != len(plan):
        raise ValueError("Plan is empty or duplicates a source variant")
    heldout = heldout_pairs(args.paired_plan)
    if any((r["suite"], int(r["task_id"])) in heldout for r in plan):
        raise ValueError("A held-out evaluation variant entered the plan")
    recipe_ids = {r["recipe"] for r in plan}
    if len(recipe_ids) != 1:
        raise ValueError(f"Expected one recipe per run, got {recipe_ids}")
    recipe = recipe_ids.pop()
    if recipe != "new_view_dual_blur" or any(
        not r["synthetic_task_name"].endswith("_noise_14") for r in plan
    ):
        raise ValueError(f"Unknown recipe {recipe}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = {"plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "paired_plan_sha256": hashlib.sha256(args.paired_plan.read_bytes()).hexdigest(),
            "teacher_checkpoint": args.teacher_checkpoint, "seed": args.seed, "recipe": recipe,
            "agent_blur_sigma": 4.0, "wrist_blur_sigma": 4.0}
    info_path = output / "run_info.json"
    if info_path.exists():
        if json.loads(info_path.read_text(encoding="utf-8")) != info:
            raise ValueError("Provenance changed; refusing to resume")
    else:
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    manifest = output / "attempts.jsonl"
    attempted = set()
    if manifest.exists():
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    key = (record["suite"], int(record["task_id"]), int(record["seed"]))
                    if key in attempted:
                        raise ValueError(f"Duplicate recorded attempt: {key}")
                    if record.get("success") is True and not (output / record["episode_file"]).is_file():
                        raise FileNotFoundError(f"Missing saved episode: {record['episode_file']}")
                    attempted.add(key)

    module, cls = args.adapter.split(":", 1)
    adapter = getattr(importlib.import_module(module), cls)()
    if adapter.image_size != 256:
        raise ValueError("Teacher adapter must use 256-pixel cameras")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suites = {}
    for row in plan:
        suite, task_id = row["suite"], int(row["task_id"])
        key = (suite, task_id, args.seed)
        if key in attempted:
            continue
        if suite not in suites:
            suites[suite] = benchmark.get_benchmark_dict()[suite]()
        task_suite = suites[suite]
        seed_everything(args.seed)
        start = time.time()
        env = None
        record = {"suite": suite, "task_id": task_id, "category": row["category"],
                  "source_category": row["source_category"],
                  "difficulty": int(row["difficulty"]), "split": row["split"],
                  "seed": args.seed, "teacher_checkpoint": args.teacher_checkpoint,
                  "recipe": recipe, "synthetic_task_name": row["synthetic_task_name"],
                  "agent_blur_sigma": 4.0, "wrist_blur_sigma": 4.0}
        try:
            task = task_suite.get_task(task_id)
            if Path(task.bddl_file).stem != row["task_name"]:
                raise ValueError("Task name differs from frozen expansion plan")
            source_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            base_name = row["task_name"].split("_view_", 1)[0]
            base_path = source_path.with_name(base_name + ".bddl")
            if not base_path.is_file():
                raise FileNotFoundError(f"Missing base BDDL: {base_path}")
            synthetic_path = source_path.with_name(row["synthetic_task_name"] + ".bddl")
            instruction = row["synthetic_task_name"].replace("_", " ")
            env = OffScreenRenderEnv(bddl_file_name=str(synthetic_path), camera_heights=256, camera_widths=256)
            env.seed(args.seed)
            initial_state = task_suite.get_task_init_states(task_id)[0]
            success, frames = rollout(env, adapter, instruction, initial_state, MAX_STEPS[suite])
            record.update(success=bool(success), steps=len(frames), instruction=instruction,
                          synthetic_bddl=str(synthetic_path),
                          max_abs_action=float(max(abs(float(v)) for frame in frames for v in frame["action"])))
            if success:
                episode = output / "episodes" / f"{suite}_task{task_id:04d}_{recipe}_seed{args.seed:04d}.npz"
                save_episode(episode, frames, instruction)
                record["episode_file"] = str(episode.relative_to(output))
        except Exception as error:
            record.update(success=False, steps=0, error=repr(error)[:500])
        finally:
            if env is not None:
                env.close()
        record["seconds"] = round(time.time() - start, 1)
        with manifest.open("a", encoding="utf-8") as target:
            target.write(json.dumps(record) + "\n")
            target.flush()
            os.fsync(target.fileno())
        attempted.add(key)
        print(f"{recipe} {suite}/{task_id} success={record['success']} steps={record['steps']}", flush=True)
        if os.path.isdir("/proc/self/fd"):
            import resource
            threshold = min(50000, resource.getrlimit(resource.RLIMIT_NOFILE)[0] // 4)
            if len(os.listdir("/proc/self/fd")) > threshold:
                print("File-descriptor restart requested", flush=True)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
