#!/usr/bin/env python3
"""Derive spatial dual-camera blur plans from frozen new-view plans."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_teacher_plan import heldout_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-plans", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--paired-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    released = json.loads(args.classification.read_text(encoding="utf-8"))
    released_names = {(suite, item["name"]) for suite, items in released.items() for item in items}
    heldout = heldout_pairs(args.paired_plan)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; refusing to overwrite plans")

    outputs = {}
    for scope in ("pilot", "full"):
        source = args.spatial_plans / f"spatial_{scope}_new_view.tsv"
        with source.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = reader.fieldnames
            rows = list(reader)
        if not fields or len(rows) != (15 if scope == "pilot" else 181):
            raise ValueError(f"Unexpected spatial {scope} plan")
        for row in rows:
            key = (row["suite"], int(row["task_id"]))
            name = row["synthetic_task_name"]
            if (key in heldout or row["suite"] != "libero_spatial"
                    or row["source_category"] != "Light Conditions"
                    or row["recipe"] != "new_view"
                    or not name.startswith(row["base_task_name"] + "_view_")
                    or not name.endswith("_initstate_0")):
                raise ValueError(f"Unexpected source row: {row}")
            row["recipe"] = "new_view_dual_blur"
            row["category"] = "Camera+Dual Sensor Blur"
            row["synthetic_task_name"] = name + "_noise_14"
            if (row["suite"], row["synthetic_task_name"]) in released_names:
                raise ValueError(f"Synthetic name is released: {row['synthetic_task_name']}")
        if len({(r["suite"], r["synthetic_task_name"]) for r in rows}) != len(rows):
            raise ValueError("Duplicate synthetic names")
        outputs[scope] = (fields, rows)

    pilot_keys = {(r["suite"], r["task_id"], r["synthetic_task_name"])
                  for r in outputs["pilot"][1]}
    full_keys = {(r["suite"], r["task_id"], r["synthetic_task_name"])
                 for r in outputs["full"][1]}
    if not pilot_keys <= full_keys:
        raise ValueError("Pilot is not a subset of full plan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scope, (fields, rows) in outputs.items():
        target = args.output_dir / f"spatial_{scope}_new_view_dual_blur.tsv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{target}: {len(rows)} attempts; sha256={hashlib.sha256(target.read_bytes()).hexdigest()}")
    print("Same spatial source IDs, camera poses, seed and split; sigma-4 blur in both cameras")


if __name__ == "__main__":
    main()
