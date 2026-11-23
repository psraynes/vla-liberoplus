#!/usr/bin/env python3
"""Convert successful teacher rollouts into a new local LeRobot v3 dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from build_teacher_plan import heldout_pairs


FEATURES = {
    "observation.images.camera1": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "observation.images.camera2": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": [8], "names": [f"state_{i}" for i in range(8)]},
    "action": {"dtype": "float32", "shape": [7], "names": [f"action_{i}" for i in range(7)]},
}


def check_episode(source: Path, expected_steps: int):
    with np.load(source, allow_pickle=False) as data:
        cam1, cam2 = data["camera1"], data["camera2"]
        state, action = data["state"], data["action"]
        instruction = str(data["instruction"].item())
    n = len(action)
    if n != expected_steps or n < 1:
        raise ValueError(f"Frame/action count mismatch in {source}")
    if any(image.shape != (n, 256, 256, 3) or image.dtype != np.uint8 for image in (cam1, cam2)):
        raise ValueError(f"Invalid camera array in {source}")
    if state.shape != (n, 8) or action.shape != (n, 7):
        raise ValueError(f"Invalid state/action array in {source}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"Nonfinite state or action values in {source}")
    if not instruction:
        raise ValueError(f"Missing task instruction in {source}")
    return cam1, cam2, state.astype(np.float32), action.astype(np.float32), instruction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--paired-plan", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    attempts_root = args.attempt_root.resolve(strict=True)
    output = args.dataset_root.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite an existing dataset path: {output}")
    if output == attempts_root or output.is_relative_to(attempts_root):
        raise ValueError("Output cannot be inside the raw rollout directory")
    heldout = heldout_pairs(args.paired_plan)
    with (attempts_root / "attempts.jsonl").open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    selected = [r for r in records if r.get("success") is True and r.get("split") == args.split]
    if not selected:
        raise ValueError(f"No successful {args.split} episodes")
    for record in selected:
        pair = record["suite"], int(record["task_id"])
        if pair in heldout:
            raise ValueError(f"Held-out evaluation variant in collected data: {pair}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=20, features=FEATURES, root=output,
        robot_type="panda", use_videos=True,
    )
    provenance = output / "collection_manifest.jsonl"
    with provenance.open("w", encoding="utf-8") as target:
        for record in selected:
            path = (attempts_root / record["episode_file"]).resolve(strict=True)
            if not path.is_relative_to(attempts_root):
                raise ValueError(f"Episode path escapes raw rollout directory: {path}")
            cam1, cam2, state, action, instruction = check_episode(path, int(record["steps"]))
            episode_index = dataset.num_episodes
            for i in range(len(action)):
                dataset.add_frame({
                    "observation.images.camera1": cam1[i],
                    "observation.images.camera2": cam2[i],
                    "observation.state": state[i],
                    "action": action[i],
                    "task": instruction,
                })
            dataset.save_episode()
            target.write(json.dumps({"episode_index": episode_index, **record}, sort_keys=True) + "\n")
            target.flush()
            print(f"Saved episode {episode_index}: {record['suite']}/{record['task_id']}", flush=True)
    dataset.finalize()
    print(f"Finalized {dataset.num_episodes} episodes at {output}")


if __name__ == "__main__":
    main()
