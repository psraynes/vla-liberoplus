#!/usr/bin/env python3
"""Check converted PI05 datasets before SmolVLA training."""

import argparse
import json
from pathlib import Path

from build_teacher_plan import heldout_pairs


EXPECTED = {"train": 1263, "validation": 149}
FEATURES = {
    "observation.images.camera1": ("video", [256, 256, 3]),
    "observation.images.camera2": ("video", [256, 256, 3]),
    "observation.state": ("float32", [8]),
    "action": ("float32", [7]),
}


def check(root: Path, split: str, heldout: set[tuple[str, int]]) -> set[tuple[str, int]]:
    root = root.resolve(strict=True)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0 metadata: {root}")
    for name, (dtype, shape) in FEATURES.items():
        feature = info.get("features", {}).get(name, {})
        if feature.get("dtype") != dtype or feature.get("shape") != shape:
            raise ValueError(f"Unexpected feature {name}: {feature}")
    records = [json.loads(line) for line in (root / "collection_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != EXPECTED[split] or info.get("total_episodes") != len(records):
        raise ValueError(f"Wrong {split} episode count: manifest={len(records)}, metadata={info.get('total_episodes')}")
    pairs = set()
    for record in records:
        pair = record["suite"], int(record["task_id"])
        if pair in pairs or pair in heldout:
            raise ValueError(f"Duplicate or held-out variant: {pair}")
        if record.get("success") is not True or record.get("split") != split:
            raise ValueError(f"Non-success or wrong split: {pair}")
        if record.get("category") not in ("Camera Viewpoints", "Sensor Noise"):
            raise ValueError(f"Wrong category: {pair}")
        if not record.get("teacher_checkpoint") or not record.get("instruction"):
            raise ValueError(f"Missing teacher or instruction provenance: {pair}")
        pairs.add(pair)
    print(f"PASS {split}: {len(pairs)} successful episodes; no held-out variants")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-base", type=Path, required=True)
    parser.add_argument("--paired-plan", type=Path, required=True)
    args = parser.parse_args()
    heldout = heldout_pairs(args.paired_plan)
    train = check(args.dataset_base / "pi05_camera_sensor_full_train_v2", "train", heldout)
    validation = check(args.dataset_base / "pi05_camera_sensor_full_validation_v2", "validation", heldout)
    if train & validation:
        raise ValueError("Train and validation variants overlap")
    print("PASS: train/validation disjoint; frozen 4,295-task evaluation untouched")


if __name__ == "__main__":
    main()
