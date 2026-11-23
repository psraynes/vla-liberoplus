#!/usr/bin/env python3
"""Audit and convert the frozen v2 + v3 PI05 successes to a new LeRobot dataset."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_teacher_plan import heldout_pairs


SOURCES = (
    ("v2_original", "collection_full_v2", 1804, 1412),
    ("new_view", "expansion_v3_full_new_view", 1804, 1744),
    ("new_view_blur", "expansion_v3_full_new_view_blur", 1804, 1746),
    ("new_view_dual_blur", "expansion_v3_full_new_view_dual_blur", 1804, 788),
    ("new_view_blur_retry", "expansion_v3_retry_new_view_blur_object_720", 1, 1),
)
EXPECTED = {"train": 5105, "validation": 586}
FEATURES = {
    "observation.images.camera1": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "observation.images.camera2": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": [8], "names": [f"state_{i}" for i in range(8)]},
    "action": {"dtype": "float32", "shape": [7], "names": [f"action_{i}" for i in range(7)]},
}
TEACHER = "lerobot/pi05_libero_finetuned_v044@7152f13e34c6c115d808aea0e5d08aca74eac14b"
RETRY_KEY = ("libero_object", 720, 7)


def load_sources(raw_base: Path, paired_plan: Path):
    heldout = heldout_pairs(paired_plan)
    paired_hash = hashlib.sha256(paired_plan.read_bytes()).hexdigest()
    selected = []
    source_split = {}
    synthetic_names = set()
    episode_paths = set()
    mild_error = None
    retry = None
    summary = Counter()
    for label, dirname, expected_attempts, expected_successes in SOURCES:
        root = (raw_base / dirname).resolve(strict=True)
        info = json.loads((root / "run_info.json").read_text(encoding="utf-8"))
        if info.get("paired_plan_sha256") != paired_hash or info.get("teacher_checkpoint") != TEACHER or info.get("seed") != 7:
            raise ValueError(f"Source provenance mismatch: {root}")
        records = [json.loads(line) for line in (root / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != expected_attempts:
            raise ValueError(f"Wrong attempt count in {root}: {len(records)}")
        keys = set()
        for record in records:
            key = record["suite"], int(record["task_id"]), int(record["seed"])
            if key in keys or key[:2] in heldout:
                raise ValueError(f"Duplicate or held-out source: {root} {key}")
            keys.add(key)
            if record.get("teacher_checkpoint") != TEACHER or record.get("split") not in EXPECTED:
                raise ValueError(f"Wrong teacher or split in {root}: {key}")
            prior = source_split.setdefault(key[:2], record["split"])
            if prior != record["split"]:
                raise ValueError(f"Training/validation source leakage: {key}")
            if label != "v2_original":
                expected_recipe = "new_view_blur" if label == "new_view_blur_retry" else label
                if record.get("recipe") != expected_recipe or not record.get("synthetic_task_name"):
                    raise ValueError(f"Wrong expansion recipe in {root}: {key}")
                if record.get("success") is True:
                    synthetic = (record["suite"], record["synthetic_task_name"])
                    if synthetic in synthetic_names:
                        raise ValueError(f"Duplicate synthetic setting: {synthetic}")
                    synthetic_names.add(synthetic)
            if "error" in record:
                if label != "new_view_blur" or key != RETRY_KEY or mild_error is not None:
                    raise ValueError(f"Unexpected error in {root}: {key}: {record['error']}")
                mild_error = record
                continue
            if record.get("success") is not True:
                continue
            episode = (root / record["episode_file"]).resolve(strict=True)
            if not episode.is_relative_to(root) or not episode.is_file():
                raise ValueError(f"Missing or escaped episode: {episode}")
            if episode in episode_paths:
                raise ValueError(f"Duplicate saved episode path: {episode}")
            episode_paths.add(episode)
            if label == "new_view_blur_retry":
                if key != RETRY_KEY:
                    raise ValueError(f"Unexpected retry key: {key}")
                retry = record
            selected.append((label, root, record, episode))
            summary[(label, record["split"])] += 1
        if sum(summary[(label, split)] for split in EXPECTED) != expected_successes:
            raise ValueError(f"Wrong success count in {root}")
    if mild_error is None or retry is None:
        raise ValueError("Missing the recorded mild-blur error or successful one-task retry")
    if retry["synthetic_task_name"] != mild_error["synthetic_task_name"] or retry["split"] != mild_error["split"]:
        raise ValueError("Retry does not match the original failed save")
    for split, expected in EXPECTED.items():
        actual = sum(count for (_, part), count in summary.items() if part == split)
        if actual != expected:
            raise ValueError(f"Wrong {split} count: {actual} != {expected}")
    return selected, summary


def read_episode(path: Path, steps: int):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        cam1, cam2 = data["camera1"], data["camera2"]
        state, action = data["state"], data["action"]
        instruction = str(data["instruction"].item())
    n = len(action)
    if n != steps or n < 1 or state.shape != (n, 8) or action.shape != (n, 7):
        raise ValueError(f"Invalid frame/state/action count: {path}")
    if any(x.shape != (n, 256, 256, 3) or x.dtype != np.uint8 for x in (cam1, cam2)):
        raise ValueError(f"Invalid image array: {path}")
    if not np.isfinite(state).all() or not np.isfinite(action).all() or not instruction:
        raise ValueError(f"Nonfinite data or missing instruction: {path}")
    return cam1, cam2, state.astype(np.float32), action.astype(np.float32), instruction


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-base", type=Path, required=True)
    p.add_argument("--paired-plan", type=Path, required=True)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--split", choices=tuple(EXPECTED))
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--repo-id")
    args = p.parse_args()
    selected, summary = load_sources(args.raw_base, args.paired_plan)
    for (label, split), count in sorted(summary.items()):
        print(f"{label} / {split}: {count}")
    print(f"PASS: {len(selected)} successful episodes; 5,105 train, 586 validation; frozen evaluation excluded")
    if args.audit_only:
        return
    if not args.split or not args.dataset_root or not args.repo_id:
        p.error("--split, --dataset-root and --repo-id are required for conversion")
    output = args.dataset_root.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing dataset: {output}")
    if output.is_relative_to(args.raw_base.resolve()):
        raise ValueError("Output may not be inside the raw collection directory")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=20, features=FEATURES,
                                    root=output, robot_type="panda", use_videos=True)
    manifest = output / "collection_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as target:
        for label, root, record, episode in selected:
            if record["split"] != args.split:
                continue
            cam1, cam2, state, action, instruction = read_episode(episode, int(record["steps"]))
            if instruction != record.get("instruction"):
                raise ValueError(f"Saved instruction differs from attempt record: {episode}")
            episode_index = dataset.num_episodes
            for i in range(len(action)):
                dataset.add_frame({"observation.images.camera1": cam1[i],
                                   "observation.images.camera2": cam2[i],
                                   "observation.state": state[i], "action": action[i],
                                   "task": instruction})
            dataset.save_episode()
            target.write(json.dumps({"episode_index": episode_index, "source_run": root.name,
                                     "collection_recipe": label, **record}, sort_keys=True) + "\n")
            target.flush()
            print(f"Saved {args.split} episode {episode_index}: {label} {record['suite']}/{record['task_id']}", flush=True)
    dataset.finalize()
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("total_episodes") != EXPECTED[args.split] or dataset.num_episodes != EXPECTED[args.split]:
        raise ValueError("Finalized episode count differs from the audited source count")
    for key, expected in FEATURES.items():
        actual = info.get("features", {}).get(key, {})
        if actual.get("dtype") != expected["dtype"] or actual.get("shape") != expected["shape"]:
            raise ValueError(f"Finalized feature differs from source schema: {key}: {actual}")
    print(f"PASS: finalized {dataset.num_episodes} {args.split} episodes at {output}")


if __name__ == "__main__":
    main()
