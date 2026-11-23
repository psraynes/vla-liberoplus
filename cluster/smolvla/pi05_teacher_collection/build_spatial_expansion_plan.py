#!/usr/bin/env python3
"""Plan new spatial camera settings from non-held-out light variants."""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from build_expansion_plan import RECIPES, candidate_views, suffix
from build_teacher_plan import heldout_pairs


FIELDS = ("order", "suite", "task_id", "category", "source_category",
          "difficulty", "split", "task_name", "base_task_name", "recipe",
          "synthetic_task_name")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classification", type=Path, required=True)
    p.add_argument("--paired-plan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    classification = json.loads(args.classification.read_text(encoding="utf-8"))
    if sum(map(len, classification.values())) != 10030 or len(classification["libero_spatial"]) != 2402:
        raise ValueError("Unexpected released LIBERO+ classification size")
    heldout = heldout_pairs(args.paired_plan)
    all_names = {(suite, item["name"]) for suite, items in classification.items() for item in items}
    if len(all_names) != 10030:
        raise ValueError("Duplicate released task names")
    candidates = []
    for task_id, item in enumerate(classification["libero_spatial"]):
        if item["id"] != task_id + 1:
            raise ValueError(f"Classification ID mismatch at spatial/{task_id}")
        if item["category"] != "Light Conditions" or ("libero_spatial", task_id) in heldout:
            continue
        name = item["name"]
        if "_light_" not in name or not name.rsplit("_light_", 1)[1].isdigit():
            raise ValueError(f"Cannot identify canonical base BDDL for {name}")
        base = name.rsplit("_light_", 1)[0]
        candidates.append({"suite": "libero_spatial", "task_id": task_id,
                           "source_category": "Light Conditions", "difficulty": item["difficulty_level"],
                           "task_name": name, "base_task_name": base})
    if len(candidates) != 181:
        raise ValueError(f"Expected 181 non-held-out spatial light variants, found {len(candidates)}")

    # Assign train/validation by source variant and retain that assignment for both recipes.
    groups = defaultdict(list)
    for row in candidates:
        groups[row["base_task_name"]].append(row)
    rng = random.Random(42)
    for group in groups.values():
        rng.shuffle(group)
        n_val = max(1, round(0.1 * len(group)))
        for index, row in enumerate(group):
            row["split"] = "validation" if index < n_val else "train"
    rng.shuffle(candidates)
    pilot = [row for row in candidates if row["split"] == "validation"][:3]
    pilot += [row for row in candidates if row["split"] == "train"][:12]
    if len(pilot) != 15:
        raise ValueError("Insufficient pilot examples")

    used = set()
    for row in candidates:
        for view in candidate_views():
            names = {recipe: row["base_task_name"] + suffix(view, noise)
                     for recipe, noise in RECIPES.items()}
            if all(("libero_spatial", name) not in all_names and
                   ("libero_spatial", name) not in used for name in names.values()):
                row["synthetic_names"] = names
                used.update(("libero_spatial", name) for name in names.values())
                break
        else:
            raise ValueError(f"No unused camera setting for {row['task_name']}")
    if len(used) != 2 * len(candidates):
        raise AssertionError("Synthetic spatial settings are not unique")

    output = args.output_dir
    targets = [output / f"spatial_{mode}_{recipe}.tsv"
               for mode in ("pilot", "full") for recipe in RECIPES]
    if any(target.exists() for target in targets):
        raise FileExistsError("Spatial plan already exists; refusing to replace it")
    output.mkdir(parents=True, exist_ok=True)
    for mode, rows in (("pilot", pilot), ("full", candidates)):
        for recipe in RECIPES:
            target = output / f"spatial_{mode}_{recipe}.tsv"
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
                writer.writeheader()
                for order, row in enumerate(rows):
                    writer.writerow({key: (order if key == "order" else
                                           "Camera Viewpoints" if key == "category" and recipe == "new_view" else
                                           "Camera+Sensor Noise" if key == "category" else
                                           recipe if key == "recipe" else
                                           row["synthetic_names"][recipe] if key == "synthetic_task_name" else
                                           row[key]) for key in FIELDS})
            print(f"{target}: {len(rows)} attempts; sha256={hashlib.sha256(target.read_bytes()).hexdigest()}")
    print("362 unique synthetic spatial settings; 181 non-held-out spatial light source variants")
    print("Both recipes inherit each source variant's split; no released task name is reused.")


if __name__ == "__main__":
    main()
