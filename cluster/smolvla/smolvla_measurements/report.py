"""Report only complete standard-LIBERO runs and completed profile JSONs."""
import argparse
import json
from collections import Counter
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def standard(root, arm):
    path = root / f"standard_full_{arm}" / "results.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    keys = [(r["suite"], int(r["task_id"]), int(r["episode"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate standard episodes for {arm}")
    expected = {(s, tid, episode) for s in SUITES for tid in range(10) for episode in range(50)}
    counts = Counter((r["suite"] for r in rows if r["success"]))
    if set(keys) != expected:
        print(f"{arm}: PARTIAL standard LIBERO {len(keys)}/2000 episodes")
        return None
    rates = {s: 100 * counts[s] / 500 for s in SUITES}
    rates["mean"] = sum(rates.values()) / 4
    print(f"{arm}: STANDARD FINAL 2000/2000 episodes; " +
          " ".join(f"{k}={v:.2f}%" for k, v in rates.items()))
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    for arm in ("base", "noise", "clean", "teacher_v2", "combined_v3"):
        standard(args.results, arm)
        paths = sorted(args.results.glob(f"profile_{arm}_*.json"))
        if paths:
            p = paths[-1]
            row = json.loads(p.read_text(encoding="utf-8"))
            print(f"{arm}: PROFILE {row['gpu']} n={row['measured_queries']} "
                  f"mean={row['latency_ms_mean']:.2f}ms/query "
                  f"model={row['model_chunk_ms_mean']:.2f}ms/query "
                  f"per_action={row['ms_per_executed_action_mean']:.2f}ms "
                  f"peak_torch_reserved={row['peak_torch_reserved_mib']:.0f}MiB "
                  f"({p})")


if __name__ == "__main__":
    main()
