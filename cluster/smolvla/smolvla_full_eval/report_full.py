#!/usr/bin/env python3
"""Report exact 10,030-task base and combined-v3 SmolVLA results."""

import collections
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARMS = ("base", "combined_v3")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = (
    ("Camera_Viewpoints", "Camera"), ("Robot_Initial_States", "Robot"),
    ("Language_Instructions", "Language"), ("Light_Conditions", "Light"),
    ("Background_Textures", "Background"), ("Sensor_Noise", "Noise"),
    ("Objects_Layout", "Layout"),
)


def read_plan(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            order, suite, task_id, category, _round = line.split("\t")
            rows.append((int(order), suite, int(task_id), category))
    return rows


def read_results(run):
    outcomes = {}
    errors = collections.Counter()
    for path in sorted(run.glob("results*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record["suite"], int(record["task_id"]))
                if record.get("hung"):
                    errors[key] += 1
                elif "success" in record:
                    if key in outcomes:
                        raise ValueError(f"Duplicate completed result: {key} in {run}")
                    outcomes[key] = bool(record["success"])
                else:
                    raise ValueError(f"Malformed result: {key} in {run}")
    return outcomes, errors


def print_table(headers, rows):
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    print("  ".join(str(v).rjust(w) for v, w in zip(headers, widths)))
    for row in rows:
        print("  ".join(str(v).rjust(w) for v, w in zip(row, widths)))


def load_arm(full, arm):
    paired = ROOT.parent / f"smolvla_rollout_eval/results/{arm}"
    runs = (paired, *(ROOT / f"results/{arm}_remaining_w{w}" for w in (0, 1)))
    expected = {(suite, task_id) for _, suite, task_id, _ in full}
    paired_manifest = (paired / "run_manifest.txt").read_text(encoding="utf-8").splitlines()
    shared = [line for line in paired_manifest if line.startswith(("model=", "seed=", "action_steps=", "save_video="))]
    if len(shared) != 4:
        raise ValueError("Original paired run manifest is incomplete")
    outcomes = {}
    total_errors = 0
    for run, count in zip(runs, (4295, 2868, 2867)):
        manifest = (run / "run_manifest.txt").read_text(encoding="utf-8").splitlines()
        if any(line not in manifest for line in shared):
            raise ValueError(f"Run settings differ from paired evaluation: {run}")
        plan = read_plan(run / "plan.tsv")
        if len(plan) != count:
            raise ValueError(f"Unexpected plan size for {run}")
        keys = {(suite, task_id) for _, suite, task_id, _ in plan}
        if len(keys) != count or keys & outcomes.keys():
            raise ValueError(f"Duplicate or overlapping plan for {run}")
        observed, errors = read_results(run)
        if observed.keys() != keys:
            raise ValueError(f"{run}: {len(observed)}/{len(keys)} completed; report requires all tasks")
        if any(key not in keys for key in errors):
            raise ValueError(f"Error record outside plan in {run}")
        total_errors += sum(errors.values())
        outcomes.update(observed)
    if outcomes.keys() != expected:
        raise ValueError(f"{arm} runs do not cover the full plan")
    by_key = {(suite, task_id): category for _, suite, task_id, category in full}
    for run in runs:
        for _, suite, task_id, category in read_plan(run / "plan.tsv"):
            if by_key[suite, task_id] != category:
                raise ValueError(f"Category differs from full plan for {suite}/{task_id}")
    return outcomes, total_errors


def main():
    full = read_plan(ROOT / "plans/full_plan_10030.tsv")
    if len(full) != 10030 or len({(s, t) for _, s, t, _ in full}) != 10030:
        raise ValueError("Expected 10,030 unique full-plan tasks")
    data = {arm: load_arm(full, arm) for arm in ARMS}
    export = ROOT / "smolvla_base_vs_combined_v3_full_10030.csv"
    with export.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("order", "suite", "task_id", "category", *ARMS))
        for order, suite, task_id, category in full:
            writer.writerow((order, suite, task_id, category,
                             *(int(data[arm][0][suite, task_id]) for arm in ARMS)))

    headers = ["", *[name for _, name in CATEGORIES], "Total"]
    print("SmolVLA full LIBERO+ exact paired evaluation: 10,030/10,030 tasks per model")
    counts = []
    for suite in (*SUITES, "all suites"):
        current = [(s, t, c) for _, s, t, c in full if suite == "all suites" or s == suite]
        cells = [[(s, t) for s, t, c in current if c == category] for category, _ in CATEGORIES]
        counts.append([suite, *[len(cell) for cell in cells], len(current)])
    for arm in ARMS:
        outcomes, errors = data[arm]
        print(f"\n{arm} success rate % ({errors} recovered error records)")
        body = []
        for suite in (*SUITES, "all suites"):
            current = [(s, t, c) for _, s, t, c in full if suite == "all suites" or s == suite]
            cells = [[(s, t) for s, t, c in current if c == category] for category, _ in CATEGORIES]
            def rate(keys):
                return f"{100 * sum(outcomes[key] for key in keys) / len(keys):.1f}" if keys else "-"
            body.append([suite, *[rate(cell) for cell in cells], rate([(s, t) for s, t, _ in current])])
        print_table(headers, body)
    print("\nEpisodes per cell")
    print_table(headers, counts)
    base, fine = (data[arm][0] for arm in ARMS)
    gains = sum(fine[key] and not base[key] for key in base)
    losses = sum(base[key] and not fine[key] for key in base)
    delta = 100 * (sum(fine.values()) - sum(base.values())) / len(base)
    print(f"\nOverall: base {sum(base.values())}/10030 = {100 * sum(base.values()) / len(base):.2f}%; "
          f"combined-v3 {sum(fine.values())}/10030 = {100 * sum(fine.values()) / len(fine):.2f}%")
    print(f"Paired change: {delta:+.2f} percentage points; combined-v3-only successes={gains}; base-only successes={losses}")
    print("These are full benchmark rates; the earlier 4,295-task results are a paired subset.")
    print(f"Saved {export}")


if __name__ == "__main__":
    main()
