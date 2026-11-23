#!/usr/bin/env python3
"""Fail closed if a teacher dataset contains a paired evaluation variant."""

import argparse
import json
from pathlib import Path

from build_teacher_plan import heldout_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--paired-plan", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve(strict=True)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Expected a finalized LeRobot v3.0 dataset")
    expected = {
        "observation.images.camera1": ("video", [256, 256, 3]),
        "observation.images.camera2": ("video", [256, 256, 3]),
        "observation.state": ("float32", [8]),
        "action": ("float32", [7]),
    }
    for key, (dtype, shape) in expected.items():
        feature = info.get("features", {}).get(key, {})
        if feature.get("dtype") != dtype or feature.get("shape") != shape:
            raise ValueError(f"Unexpected dataset feature {key}: {feature}")

    manifest = root / "collection_manifest.jsonl"
    heldout = heldout_pairs(args.paired_plan)
    records = []
    with manifest.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError("No collected teacher episodes")
    if len(records) != info.get("total_episodes"):
        raise ValueError("Episode count differs between LeRobot metadata and provenance manifest")
    teacher_ids = set()
    for record in records:
        pair = (record["suite"], int(record["task_id"]))
        if pair in heldout:
            raise ValueError(f"Evaluation variant found in training data: {pair}")
        if record.get("split") != "train" or record.get("success") is not True:
            raise ValueError(f"Non-training or unsuccessful episode in dataset: {pair}")
        if record.get("category") not in ("Camera Viewpoints", "Sensor Noise"):
            raise ValueError(f"Unexpected perturbation category: {record.get('category')}")
        if int(record.get("steps", 0)) < 1:
            raise ValueError(f"Empty episode: {pair}")
        teacher_ids.add(record.get("teacher_checkpoint"))
    if None in teacher_ids or "" in teacher_ids:
        raise ValueError("Teacher checkpoint provenance missing")
    print(f"PASS: {len(records)} successful training episodes, {len(teacher_ids)} teacher checkpoint(s), zero paired evaluation variants")


if __name__ == "__main__":
    main()
