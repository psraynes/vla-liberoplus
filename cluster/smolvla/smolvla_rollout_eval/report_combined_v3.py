#!/usr/bin/env python3
"""Exact-task LIBERO+ tables for four SmolVLA checkpoints on the frozen 4,295 plan."""

import csv
from pathlib import Path

from report import read_results


ROOT = Path(__file__).resolve().parent
ARMS = ("base", "noise", "clean", "combined_v3")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = (
    ("Camera_Viewpoints", "Camera"),
    ("Robot_Initial_States", "Robot"),
    ("Language_Instructions", "Language"),
    ("Light_Conditions", "Light"),
    ("Background_Textures", "Background"),
    ("Sensor_Noise", "Noise"),
    ("Objects_Layout", "Layout"),
)


def load_plan():
    plan = []
    with (ROOT / "results/base/plan.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            order, suite, task_id, category, _round = line.rstrip("\n").split("\t")
            plan.append((int(order), suite, int(task_id), category))
    if len(plan) != 4295 or len({(s, t) for _, s, t, _ in plan}) != 4295:
        raise ValueError("Expected 4,295 unique suite/task pairs")
    if {s for _, s, _, _ in plan} != set(SUITES):
        raise ValueError("Unexpected suite in frozen plan")
    if {c for _, _, _, c in plan} != {c for c, _ in CATEGORIES}:
        raise ValueError("Unexpected category in frozen plan")
    return plan


def table(headers, body, widths=None):
    widths = widths or [max(len(str(row[i])) for row in [headers, *body]) for i in range(len(headers))]
    lines = ["  ".join(str(value).rjust(width) for value, width in zip(headers, widths))]
    lines.extend("  ".join(str(value).rjust(width) for value, width in zip(row, widths)) for row in body)
    return "\n".join(lines)


def rate(rows, arm):
    return f"{100 * sum(row[arm] for row in rows) / len(rows):.1f}" if rows else "-"


def exact_rate(rows, arm):
    return f"{100 * sum(row[arm] for row in rows) / len(rows):.2f}%"


def main():
    plan = load_plan()
    expected = {(suite, task_id) for _, suite, task_id, _ in plan}
    data = {}
    for arm in ARMS:
        results, errors = read_results(arm)
        if set(results) != expected or sum(errors.values()):
            raise ValueError(f"{arm}: {len(results)}/4295 complete; {sum(errors.values())} error records")
        data[arm] = results
    rows = [{"order": order, "suite": suite, "task_id": task_id,
             "category": category, **{arm: int(data[arm][suite, task_id]) for arm in ARMS}}
            for order, suite, task_id, category in plan]

    csv_path = ROOT / "paired_rollouts_four_models.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("order", "suite", "task_id", "category", *ARMS))
        writer.writeheader()
        writer.writerows(rows)

    names = [label for _, label in CATEGORIES]
    headings = ["", *names, "Total"]
    lines = ["SmolVLA LIBERO+ exact paired evaluation", "4,295/4,295 identical tasks per model; seed 7; no errors", ""]
    for arm in ARMS:
        lines += [f"{arm} success rate %", table(headings, [
            [suite, *[rate([r for r in rows if (suite == "all suites" or r["suite"] == suite)
                           and r["category"] == category], arm)
                      for category, _ in CATEGORIES],
             rate([r for r in rows if suite == "all suites" or r["suite"] == suite], arm)]
            for suite in (*SUITES, "all suites")
        ]), ""]

    lines += ["Episodes per cell (same tasks for every model)", table(headings, [
        [suite, *[sum((suite == "all suites" or r["suite"] == suite) and r["category"] == category
                      for r in rows) for category, _ in CATEGORIES],
         sum(suite == "all suites" or r["suite"] == suite for r in rows)]
        for suite in (*SUITES, "all suites")
    ]), ""]

    lines += ["Exact paired comparison (percentage points)",
              table(["Comparison", "Base", "Other", "Change", "Other-only", "Base-only"], [
                  [arm, exact_rate(rows, "base"), exact_rate(rows, arm),
                   f"{100 * (sum(r[arm] for r in rows) - sum(r['base'] for r in rows)) / len(rows):+.2f}",
                   sum(r[arm] and not r["base"] for r in rows),
                   sum(r["base"] and not r[arm] for r in rows)]
                  for arm in ARMS if arm != "base"
              ]), ""]
    lines += ["Note: These are raw rates on the frozen 4,295-task subset, not a weighted 10,030-task benchmark score.",
              f"Per-task CSV: {csv_path}"]
    report = "\n".join(lines) + "\n"
    text_path = ROOT / "combined_v3_results_table.txt"
    text_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Text report: {text_path}")


if __name__ == "__main__":
    main()
