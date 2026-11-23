#!/usr/bin/env python3
"""Report paired rollout outcomes; label incomplete results explicitly."""
import collections
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARMS = ("base", "noise", "clean")


def read_results(arm):
    results = {}
    errors = collections.Counter()
    for path in sorted((ROOT / "results" / arm).glob("results*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            key = (record["suite"], int(record["task_id"]))
            if record.get("hung"):
                errors[key] += 1
            elif "success" in record:
                if key in results:
                    raise ValueError(f"Duplicate completed record in {arm}: {key}")
                results[key] = bool(record["success"])
    return results, errors


def main():
    plan_file = ROOT / "results" / "base" / "plan.tsv"
    plan = []
    for line in plan_file.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        order, suite, task_id, category, _round = line.split("\t")
        plan.append((int(order), suite, int(task_id), category))
    if len(plan) != 4295 or len({(s, t) for _, s, t, _ in plan}) != 4295:
        raise ValueError("Expected 4295 unique task pairs")
    data = {arm: read_results(arm) for arm in ARMS}
    for arm in ARMS:
        outcomes, errors = data[arm]
        print(f"{arm}: {len(outcomes)}/4295 complete; {sum(errors.values())} error records")

    rows = []
    for order, suite, task_id, category in plan:
        key = (suite, task_id)
        rows.append({"order": order, "suite": suite, "task_id": task_id, "category": category,
                     **{arm: data[arm][0].get(key) for arm in ARMS}})
    out = ROOT / "paired_rollouts.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("order", "suite", "task_id", "category", *ARMS))
        writer.writeheader()
        writer.writerows(rows)
    common = [row for row in rows if all(row[arm] is not None for arm in ARMS)]
    label = "FINAL" if len(common) == 4295 else "PARTIAL"
    print(f"{label} common exact-task rollouts: {len(common)}/4295")
    if common:
        for category in sorted({row[3] for row in plan}):
            subset = [row for row in common if row["category"] == category]
            if subset:
                rates = "  ".join(f"{arm}={100 * sum(row[arm] for row in subset) / len(subset):.1f}%"
                                  for arm in ARMS)
                print(f"{category}: n={len(subset)}  {rates}")
        for arm in ARMS:
            print(f"all {arm}: {100 * sum(row[arm] for row in common) / len(common):.2f}%")
        for other in ("base", "clean"):
            noise_only = sum(row["noise"] and not row[other] for row in common)
            other_only = sum(row[other] and not row["noise"] for row in common)
            print(f"noise vs {other}: noise-only successes={noise_only}; {other}-only successes={other_only}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
