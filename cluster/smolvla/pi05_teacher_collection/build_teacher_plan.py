#!/usr/bin/env python3
"""Select teacher collection variants while excluding the paired evaluation set."""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = ("Camera Viewpoints", "Sensor Noise")


def heldout_pairs(path: Path) -> set[tuple[str, int]]:
    pairs = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Bad paired plan row: {line!r}")
            key = (fields[1], int(fields[2]))
            if key in pairs:
                raise ValueError(f"Duplicate paired evaluation task: {key}")
            pairs.add(key)
    if len(pairs) != 4295:
        raise ValueError(f"Expected 4,295 paired evaluation tasks; found {len(pairs)}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--paired-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("teacher_collection_plan.tsv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--pilot-per-cell", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    if args.pilot_per_cell < 1:
        parser.error("--pilot-per-cell must be positive")

    heldout = heldout_pairs(args.paired_plan)
    classification = json.loads(args.classification.read_text(encoding="utf-8"))
    if any(suite not in classification for suite in SUITES):
        raise ValueError("Classification is missing a benchmark suite")

    cells = defaultdict(list)
    all_pairs = set()
    for suite in SUITES:
        for task_id, item in enumerate(classification[suite]):
            if item.get("id") != task_id + 1:
                raise ValueError(f"Classification ID/order mismatch: {suite}/{task_id}")
            key = (suite, task_id)
            all_pairs.add(key)
            category = item["category"]
            if category in CATEGORIES and key not in heldout:
                cells[(suite, category)].append(
                    {"suite": suite, "task_id": task_id, "category": category,
                     "difficulty": item["difficulty_level"], "task_name": item["name"]}
                )
    if len(all_pairs) != 10030:
        raise ValueError(f"Expected 10,030 classified variants; found {len(all_pairs)}")
    if not heldout <= all_pairs:
        raise ValueError("Paired evaluation plan refers to unknown variants")

    rng = random.Random(args.seed)
    for cell in cells.values():
        rng.shuffle(cell)
        n_val = max(1, round(len(cell) * args.validation_fraction))
        for i, row in enumerate(cell):
            row["split"] = "validation" if i < n_val else "train"
        rng.shuffle(cell)  # Interleave splits in the full plan.

    ordered_cells = [(suite, category) for suite in SUITES for category in CATEGORIES]
    rows = [row for group in zip_longest(*(cells[key] for key in ordered_cells))
            for row in group if row is not None]
    pilot_cells = []
    skipped_cells = []
    for key in ordered_cells:
        cell = cells[key]
        if not cell:
            skipped_cells.append(key)
            continue
        validation = [row for row in cell if row["split"] == "validation"]
        train = [row for row in cell if row["split"] == "train"]
        if not validation:
            raise ValueError(f"Missing validation candidates for nonempty cell {key}")
        pilot_cells.append(validation[:1] + train[: args.pilot_per_cell - 1])
    pilot = [row for group in zip_longest(*pilot_cells) for row in group if row is not None]
    if not rows:
        raise ValueError("No eligible camera or sensor-noise variants remain")
    if any((row["suite"], row["task_id"]) in heldout for row in rows):
        raise AssertionError("A held-out evaluation task entered the collection plan")

    fields = ("order", "suite", "task_id", "category", "difficulty", "split", "task_name")
    def write(path: Path, selected: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for order, row in enumerate(selected):
                writer.writerow({"order": order, **row})

    write(args.output, rows)
    pilot_path = args.output.with_name(args.output.stem + "_pilot.tsv")
    write(pilot_path, pilot)
    counts = Counter((row["suite"], row["category"], row["split"]) for row in rows)
    print(f"Held out: {len(heldout)} exact paired evaluation variants")
    print(f"Teacher collection candidates: {len(rows)}; pilot candidates: {len(pilot)}")
    for suite, category in skipped_cells:
        print(f"  {suite} / {category}: no eligible tasks after held-out exclusion")
    for cell, count in sorted(counts.items()):
        print(f"  {' / '.join(cell)}: {count}")
    print(f"Saved {args.output} and {pilot_path}")


if __name__ == "__main__":
    main()
