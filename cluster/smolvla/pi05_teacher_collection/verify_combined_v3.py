#!/usr/bin/env python3
"""Verify finalized combined PI05 datasets before training SmolVLA."""

import argparse
import json
from collections import Counter
from pathlib import Path

from build_teacher_plan import heldout_pairs


EXPECTED = {
    "train": {"v2_original": 1263, "new_view": 1569, "new_view_blur": 1572,
              "new_view_dual_blur": 700, "new_view_blur_retry": 1},
    "validation": {"v2_original": 149, "new_view": 175, "new_view_blur": 174,
                   "new_view_dual_blur": 88},
}
FEATURES = {
    "observation.images.camera1": ("video", [256, 256, 3]),
    "observation.images.camera2": ("video", [256, 256, 3]),
    "observation.state": ("float32", [8]),
    "action": ("float32", [7]),
}


def check(root: Path, split: str, heldout: set[tuple[str, int]]):
    root = root.resolve(strict=True)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Wrong LeRobot dataset version: {root}")
    for key, (dtype, shape) in FEATURES.items():
        feature = info.get("features", {}).get(key, {})
        if feature.get("dtype") != dtype or feature.get("shape") != shape:
            raise ValueError(f"Wrong feature {key}: {feature}")
    with (root / "collection_manifest.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    expected = EXPECTED[split]
    if len(records) != sum(expected.values()) or info.get("total_episodes") != len(records):
        raise ValueError(f"Wrong {split} count: {len(records)}, metadata={info.get('total_episodes')}")
    if int(info.get("total_frames", 0)) <= 0:
        raise ValueError(f"Missing frame count: {root}")
    recipes = Counter()
    source_split = {}
    episodes = set()
    for record in records:
        index = int(record["episode_index"])
        if index in episodes:
            raise ValueError(f"Duplicate episode index: {index}")
        episodes.add(index)
        pair = record["suite"], int(record["task_id"])
        if pair in heldout or record.get("success") is not True or record.get("split") != split:
            raise ValueError(f"Held-out or invalid record: {pair}")
        if not record.get("teacher_checkpoint") or not record.get("instruction"):
            raise ValueError(f"Missing episode provenance: {pair}")
        source_split[pair] = split
        recipes[record["collection_recipe"]] += 1
    if episodes != set(range(len(records))) or recipes != Counter(expected):
        raise ValueError(f"Episode sequence or recipe counts differ: {recipes}")
    print(f"PASS {split}: {len(records)} episodes; {info['total_frames']} frames; recipes={dict(recipes)}")
    return source_split, int(info["total_frames"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-base", type=Path, required=True)
    p.add_argument("--paired-plan", type=Path, required=True)
    args = p.parse_args()
    heldout = heldout_pairs(args.paired_plan)
    datasets = {}
    for split in EXPECTED:
        root = args.dataset_base / f"pi05_camera_sensor_combined_v3_{split}"
        datasets[split] = check(root, split, heldout)
    if set(datasets["train"][0]) & set(datasets["validation"][0]):
        raise ValueError("Source variants overlap train and validation")
    print("PASS: both datasets finalized; source splits disjoint; 4,295-task evaluation untouched")


if __name__ == "__main__":
    main()
