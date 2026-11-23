#!/usr/bin/env python3
"""Freeze the 5,735 tasks absent from the exact 4,295-task comparison."""

import argparse
import hashlib
from pathlib import Path


def read_plan(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        order, suite, task_id, category, round_no = line.split("\t")
        rows.append((int(order), suite, int(task_id), category, int(round_no)))
    if len({(r[1], r[2]) for r in rows}) != len(rows):
        raise ValueError(f"Duplicate suite/task in {path}")
    return rows


def write_plan(path, rows):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# order\tsuite\ttask_id\tcategory\tround\n")
        for index, (_, suite, task_id, category, round_no) in enumerate(rows):
            handle.write(f"{index}\t{suite}\t{task_id}\t{category}\t{round_no}\n")
    print(f"{path.name}: {len(rows)} rows, sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-plan", type=Path, required=True)
    parser.add_argument("--paired-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    full = read_plan(args.full_plan)
    paired = read_plan(args.paired_plan)
    if len(full) != 10030 or len(paired) != 4295:
        raise ValueError("Expected full 10,030 and paired 4,295 plans")
    full_by_key = {(suite, task_id): (category, round_no) for _, suite, task_id, category, round_no in full}
    for _, suite, task_id, category, round_no in paired:
        if full_by_key.get((suite, task_id)) != (category, round_no):
            raise ValueError(f"Paired row differs from full plan: {suite}/{task_id}")
    paired_keys = {(suite, task_id) for _, suite, task_id, _, _ in paired}
    remaining = [row for row in full if (row[1], row[2]) not in paired_keys]
    if len(remaining) != 5735:
        raise ValueError(f"Expected 5,735 remaining tasks; got {len(remaining)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plan(args.output_dir / "full_plan_10030.tsv", full)
    for worker in (0, 1):
        write_plan(args.output_dir / f"remaining_worker{worker}.tsv", remaining[worker::2])
    print("PASS: paired + remaining covers all 10,030 unique tasks")


if __name__ == "__main__":
    main()
