#!/usr/bin/env python3
"""Summarize teacher collection attempts and check the held-out boundary."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_teacher_plan import heldout_pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--paired-plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    with args.plan.open(encoding="utf-8", newline="") as source:
        plan = list(csv.DictReader(source, delimiter="\t"))
    planned = {(r["suite"], int(r["task_id"])) for r in plan}
    if len(planned) != len(plan):
        raise ValueError("Collection plan has duplicate variants")
    heldout = heldout_pairs(args.paired_plan)
    if planned & heldout:
        raise ValueError("Collection plan overlaps held-out evaluation")
    manifest = args.run / "attempts.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()] if manifest.exists() else []
    seen = set()
    summary = Counter()
    for record in records:
        pair = (record["suite"], int(record["task_id"]))
        key = (*pair, int(record["seed"]))
        if pair not in planned or pair in heldout or key in seen:
            raise ValueError(f"Invalid, held-out, or duplicate attempt: {key}")
        seen.add(key)
        kind = "success" if record.get("success") is True else "error" if "error" in record else "failure"
        summary[(record["category"], record["split"], kind)] += 1
        if kind == "success":
            episode = (args.run / record["episode_file"]).resolve()
            if not episode.is_relative_to(args.run.resolve()) or not episode.is_file():
                raise FileNotFoundError(f"Missing successful episode: {episode}")
    successes = sum(v for (_, _, kind), v in summary.items() if kind == "success")
    errors = sum(v for (_, _, kind), v in summary.items() if kind == "error")
    print(f"{args.run.name}: {len(records)}/{len(plan)} attempted; {successes} successful episodes; {errors} errors; {len(plan)-len(records)} remaining")
    action_maxima = [float(r["max_abs_action"]) for r in records if "max_abs_action" in r]
    if action_maxima:
        print(f"Largest absolute action component issued: {max(action_maxima):.3f}")
    for key, count in sorted(summary.items()):
        print(f"  {' / '.join(key)}: {count}")
    return 0 if len(records) == len(plan) and errors == 0 and successes > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
