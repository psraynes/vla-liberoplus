#!/usr/bin/env python3
"""Create a paired stronger-blur plan from the frozen new-view plans."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_teacher_plan import heldout_pairs


FIELDS = ("order", "suite", "task_id", "category", "source_category",
          "difficulty", "split", "task_name", "recipe", "synthetic_task_name")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--view-plans", type=Path, required=True)
    p.add_argument("--classification", type=Path, required=True)
    p.add_argument("--paired-plan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    classification = json.loads(args.classification.read_text(encoding="utf-8"))
    all_names = {(suite, item["name"]) for suite, items in classification.items() for item in items}
    if len(all_names) != 10030:
        raise ValueError("Expected 10,030 distinct released benchmark names")
    heldout = heldout_pairs(args.paired_plan)
    output = args.output_dir
    targets = [output / f"expansion_{mode}_new_view_dual_blur.tsv" for mode in ("pilot", "full")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Dual-blur plan exists; refusing to overwrite it")
    plans = {}
    for mode, expected in (("pilot", 30), ("full", 1804)):
        source = args.view_plans / f"expansion_{mode}_new_view.tsv"
        with source.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(rows) != expected or len({(r["suite"], int(r["task_id"])) for r in rows}) != expected:
            raise ValueError(f"Invalid {mode} source view plan")
        converted = []
        for row in rows:
            pair = row["suite"], int(row["task_id"])
            if pair in heldout or row["recipe"] != "new_view":
                raise ValueError(f"Held-out or unexpected source row: {pair}")
            old = row["synthetic_task_name"]
            if not old.endswith("_initstate_0"):
                raise ValueError(f"Unexpected source suffix: {old}")
            name = old + "_noise_14"  # LIBERO+ gaussian sigma=4 on agentview
            if (row["suite"], name) in all_names:
                raise ValueError(f"New setting collides with benchmark: {name}")
            converted.append({**row, "category": "Camera+Dual Sensor Blur",
                              "recipe": "new_view_dual_blur", "synthetic_task_name": name})
        plans[mode] = converted
    pilot_pairs = {(r["suite"], r["task_id"], r["synthetic_task_name"]) for r in plans["pilot"]}
    full_pairs = {(r["suite"], r["task_id"], r["synthetic_task_name"]) for r in plans["full"]}
    if not pilot_pairs <= full_pairs or len({(r["suite"], r["synthetic_task_name"]) for r in plans["full"]}) != 1804:
        raise ValueError("Pilot is not a subset or full plan has duplicate settings")
    output.mkdir(parents=True, exist_ok=True)
    for mode, target in zip(("pilot", "full"), targets):
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(plans[mode])
        print(f"{target}: {len(plans[mode])} attempts; sha256={hashlib.sha256(target.read_bytes()).hexdigest()}")
    print("Matched source IDs, camera poses, splits, and seed with new_view; added sigma-4 agent and wrist blur.")


if __name__ == "__main__":
    main()
