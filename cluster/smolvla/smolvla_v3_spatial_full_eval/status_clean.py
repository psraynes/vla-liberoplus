#!/usr/bin/env python3
"""Check an isolated clean-instruction worker without touching its records."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    run = Path(sys.argv[1])
    plan = {(row[1], int(row[2])) for line in (run / "plan.tsv").read_text().splitlines()
            if line and not line.startswith("#") for row in [line.split("\t")]}
    done = {}
    errors = Counter()
    for path in sorted(run.glob("results*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["suite"], int(record["task_id"]))
            if key not in plan:
                raise ValueError(f"Result outside plan: {key}")
            if record.get("hung"):
                errors[key] += 1
            elif "success" in record:
                if key in done:
                    raise ValueError(f"Duplicate completed result: {key}")
                done[key] = bool(record["success"])
            else:
                raise ValueError(f"Malformed result: {key}")
    skipped = sum(key not in done and errors[key] >= 2 for key in plan)
    print(f"{run.name}: {len(done)}/{len(plan)} complete; {len(plan)-len(done)} remaining; "
          f"{skipped} skipped after two errors; {sum(errors.values())} error records")
    return 0 if len(done) == len(plan) else 1


if __name__ == "__main__":
    raise SystemExit(main())
