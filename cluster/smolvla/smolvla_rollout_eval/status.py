#!/usr/bin/env python3
"""Check complete unique results for one isolated LIBERO+ rollout run."""
import collections
import json
import sys
from pathlib import Path


def main() -> int:
    run = Path(sys.argv[1])
    plan = [(row[1], int(row[2])) for line in (run / "plan.tsv").read_text().splitlines()
            if line and not line.startswith("#") for row in [line.split("\t")]]
    expected = set(plan)
    if len(plan) != 4295 or len(expected) != 4295:
        raise ValueError("Plan must have 4295 unique suite/task pairs")
    done = {}
    hangs = collections.Counter()
    for path in sorted(run.glob("results*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            key = (record["suite"], int(record["task_id"]))
            if key not in expected:
                raise ValueError(f"Unexpected result key: {key}")
            if record.get("hung"):
                hangs[key] += 1
            elif "success" in record:
                if key in done:
                    raise ValueError(f"Duplicate completed result: {key}")
                done[key] = bool(record["success"])
    incomplete = [key for key in plan if key not in done]
    skipped = [key for key in incomplete if hangs[key] >= 2]
    print(f"{run.name}: {len(done)}/4295 complete; {len(incomplete)} remaining; "
          f"{len(skipped)} skipped after two errors; {sum(hangs.values())} error records")
    if skipped:
        print("Skipped:", skipped[:20])
    return 0 if not incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
