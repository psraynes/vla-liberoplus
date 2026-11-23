#!/usr/bin/env python3
"""Freeze clean instructions and the affected subset of the 10,030-task plan."""

import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

LANGUAGE = "Language_Instructions"


def normalize_category(value):
    return value.replace(" ", "_")


def original_language(base_name):
    # LIBERO's grab_language_from_filename for an unperturbed task name.
    if base_name[0].isupper():
        match = re.search(r"SCENE\d+_", base_name)
        if not match:
            raise ValueError(f"Unknown LIBERO-10 scene prefix: {base_name}")
        base_name = base_name[match.end():]
    return base_name.replace("_", " ")


def filename_language(task_name):
    # LIBERO+'s grab_language_from_filename for a non-Language variant.
    if task_name[0].isupper():
        match = re.search(r"SCENE\d+_", task_name)
        if not match:
            raise ValueError(f"Unknown scene prefix: {task_name}")
        task_name = task_name[match.end():]
    return task_name.replace("_", " ")


def read_plan(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            order, suite, task_id, category, round_id = line.split("\t")
            rows.append((order, suite, int(task_id), category, round_id))
    if len(rows) != 10030 or len({(r[1], r[2]) for r in rows}) != 10030:
        raise ValueError("Expected 10,030 unique tasks in the frozen full plan")
    return rows


def write_plan(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# order\tsuite\ttask_id\tcategory\tround\n")
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--full-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    classification = json.loads(args.classification.read_text(encoding="utf-8"))
    full = read_plan(args.full_plan)
    reference = list(csv.DictReader(args.reference.open(encoding="utf-8", newline="")))
    if len(reference) != 4295:
        raise ValueError("Expected 4,295 teammate reference rows")
    bases = collections.defaultdict(set)
    for row in reference:
        bases[row["suite"]].add(row["base_libero_task"])
        task = classification[row["suite"]][int(row["task_id"])]
        if task["name"] != row["liberoplus_task_name"]:
            raise ValueError(f"Teammate task ID/name mismatch: {row['suite']}/{row['task_id']}")
    if {suite: len(names) for suite, names in bases.items()} != {
        suite: 10 for suite in classification
    }:
        raise ValueError("Expected 10 canonical tasks per LIBERO suite")

    records = []
    affected = []
    counts = collections.Counter()
    for plan_row in full:
        order, suite, task_id, category, _ = plan_row
        task = classification[suite][task_id]
        if normalize_category(task["category"]) != category:
            raise ValueError(f"Plan/classification category mismatch: {suite}/{task_id}")
        name = task["name"]
        matches = [base for base in bases[suite] if name == base or name.startswith(base + "_")]
        if len(matches) != 1:
            raise ValueError(f"Expected one base task match for {suite}/{task_id}: {matches}")
        base = matches[0]
        if category == LANGUAGE:
            # LIBERO+ gets this instruction from BDDL. It is a language-perturbation task.
            official = ""
            clean = ""
        else:
            if "_language_" in name:
                raise ValueError(f"Unexpected language marker outside Language category: {suite}/{task_id}")
            official = filename_language(name)
            clean = original_language(base)
            if official == clean:
                raise ValueError(f"Non-Language instruction was already clean: {suite}/{task_id}")
            affected.append(plan_row)
        counts[category] += 1
        records.append((order, suite, task_id, category, name, base, official, clean))
    if len(affected) != 8493 or counts[LANGUAGE] != 1537:
        raise ValueError(f"Unexpected affected/Language counts: {len(affected)}/{counts[LANGUAGE]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "instruction_map.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("order", "suite", "task_id", "category", "task_name", "base_task_name",
                         "official_instruction", "clean_instruction"))
        writer.writerows(records)
    for worker in (0, 1):
        write_plan(args.output_dir / f"affected_worker{worker}.tsv", affected[worker::2])
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(f"{path.name}: {hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"PASS: {len(affected)} affected tasks; {counts[LANGUAGE]} Language tasks unchanged; 40 base tasks")


if __name__ == "__main__":
    main()
