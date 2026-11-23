#!/usr/bin/env python3
"""Check one 2,867/2,868-task extension shard without changing results."""

import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    plan = []
    for line in (run / "plan.tsv").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            row = line.split("\t")
            plan.append((row[1], int(row[2])))
    expected = set(plan)
    if len(plan) not in (2867, 2868) or len(expected) != len(plan):
        raise ValueError("Expected a unique 2,867/2,868-task remaining shard")
    done = {}
    errors = collections.Counter()
    for path in sorted(run.glob("results*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record["suite"], int(record["task_id"]))
                if key not in expected:
                    raise ValueError(f"Unexpected result {key}")
                if record.get("hung"):
                    errors[key] += 1
                elif "success" in record:
                    if key in done:
                        raise ValueError(f"Duplicate completed result {key}")
                    done[key] = bool(record["success"])
                else:
                    raise ValueError(f"Malformed result {key}")
    missing = expected - done.keys()
    skipped = [key for key in missing if errors[key] >= 2]
    print(f"{run.name}: {len(done)}/{len(plan)} complete; {len(missing)} remaining; "
          f"{len(skipped)} skipped after two errors; {sum(errors.values())} error records")
    if skipped:
        print("Skipped:", skipped[:20])
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
