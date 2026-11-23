#!/usr/bin/env python3
"""Plan new simulated camera/noise settings outside every released LIBERO+ variant."""

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

from build_teacher_plan import heldout_pairs


RECIPES = {"new_view": None, "new_view_blur": 12}
FIELDS = ("order", "suite", "task_id", "category", "source_category", "difficulty", "split", "task_name", "recipe", "synthetic_task_name")


def read_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def candidate_views():
    options = itertools.product(range(3, 60, 2), (0, 2, 4, 6, 8), (100, 95, 105))
    return sorted(options, key=lambda x: (abs(x[0]) + 2 * abs(x[1]) + abs(x[2] - 100), x))


def suffix(view: tuple[int, int, int], noise: int | None) -> str:
    horizon, vertical, scale = view
    result = f"_view_{horizon}_{vertical}_{scale}_0_0_initstate_0"
    return result if noise is None else result + f"_noise_{noise}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-plan", type=Path, required=True)
    parser.add_argument("--pilot-plan", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--paired-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    full, pilot = read_tsv(args.full_plan), read_tsv(args.pilot_plan)
    if len(full) != 1804 or len(pilot) != 30:
        raise ValueError(f"Unexpected source plan counts: full={len(full)}, pilot={len(pilot)}")
    heldout = heldout_pairs(args.paired_plan)
    classification = json.loads(args.classification.read_text(encoding="utf-8"))
    all_names = {(suite, item["name"]) for suite, items in classification.items() for item in items}
    if len(all_names) != 10030:
        raise ValueError("Expected 10,030 unique original benchmark names")
    full_keys = {(r["suite"], int(r["task_id"])) for r in full}
    pilot_keys = {(r["suite"], int(r["task_id"])) for r in pilot}
    if len(full_keys) != len(full) or len(pilot_keys) != len(pilot) or not pilot_keys <= full_keys:
        raise ValueError("Duplicate source variant or pilot is not a subset of full plan")

    views = candidate_views()
    used_names: set[tuple[str, str]] = set()
    assigned: dict[tuple[str, int], dict[str, str]] = {}
    view_count: Counter[str] = Counter()
    for row in full:
        suite, task_id = row["suite"], int(row["task_id"])
        source_name = row["task_name"]
        if (suite, task_id) in heldout or (suite, source_name) not in all_names:
            raise ValueError(f"Source variant is held out or unknown: {suite}/{task_id}")
        if row["category"] not in ("Camera Viewpoints", "Sensor Noise"):
            raise ValueError(f"Unexpected source category: {row['category']}")
        if "_view_" not in source_name or "_initstate_0" not in source_name:
            raise ValueError(f"Source variant cannot be safely remapped: {source_name}")
        base_name = source_name.split("_view_", 1)[0]
        for view in views:
            names = {recipe: base_name + suffix(view, noise) for recipe, noise in RECIPES.items()}
            if all((suite, name) not in all_names and (suite, name) not in used_names for name in names.values()):
                assigned[(suite, task_id)] = names
                used_names.update((suite, name) for name in names.values())
                view_count[suite] += 1
                break
        else:
            raise ValueError(f"No unused camera configuration for {suite}/{base_name}")
    if len(used_names) != 2 * len(full):
        raise AssertionError("Synthetic settings are not unique")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / f"expansion_{mode}_{recipe}.tsv" for mode in ("pilot", "full") for recipe in RECIPES]
    if any(path.exists() for path in targets):
        raise FileExistsError("An expansion plan already exists; refusing to replace it")

    for mode, source_rows in (("pilot", pilot), ("full", full)):
        for recipe in RECIPES:
            rows = []
            for row in source_rows:
                key = row["suite"], int(row["task_id"])
                rows.append({**row, "source_category": row["category"],
                             "category": "Camera Viewpoints" if recipe == "new_view" else "Camera+Sensor Noise",
                             "recipe": recipe, "synthetic_task_name": assigned[key][recipe]})
            target = output / f"expansion_{mode}_{recipe}.tsv"
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            print(f"{target}: {len(rows)} attempts; sha256={hashlib.sha256(target.read_bytes()).hexdigest()}")
    print(f"{len(used_names)} distinct synthetic settings; source counts by suite: {dict(view_count)}")
    print("All source task IDs are outside the 4,295 held-out pairs; every synthetic setting is outside all 10,030 released benchmark names.")


if __name__ == "__main__":
    main()
