#!/usr/bin/env python3
"""Pair the clean-instruction rollouts with frozen official-instruction results."""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("base", "combined_v3")
CATEGORIES = ("Camera_Viewpoints", "Robot_Initial_States", "Language_Instructions",
              "Light_Conditions", "Background_Textures", "Sensor_Noise", "Objects_Layout")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def read_clean(arm):
    outcomes = {}
    errors = Counter()
    for worker in (0, 1):
        run = ROOT / "results" / f"{arm}_clean_w{worker}"
        plan = {(row[1], int(row[2])) for line in (run / "plan.tsv").read_text().splitlines()
                if line and not line.startswith("#") for row in [line.split("\t")]}
        manifest = (run / "run_manifest.txt").read_text().splitlines()
        if f"arm={arm}_clean_w{worker}" not in manifest or "instruction_mode=clean" not in manifest:
            raise ValueError(f"Unexpected manifest for {run}")
        found = {}
        for path in sorted(run.glob("results*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record["suite"], int(record["task_id"]))
                if key not in plan:
                    raise ValueError(f"Result outside {run} plan: {key}")
                if record.get("hung"):
                    errors[key] += 1
                elif "success" in record:
                    if key in found:
                        raise ValueError(f"Duplicate completed result: {key}")
                    found[key] = int(bool(record["success"]))
                else:
                    raise ValueError(f"Malformed result: {key}")
        if found.keys() != plan:
            raise ValueError(f"{run}: only {len(found)}/{len(plan)} complete")
        if outcomes.keys() & found.keys():
            raise ValueError(f"Overlapping clean workers: {run}")
        outcomes.update(found)
    return outcomes, sum(errors.values())


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    official_path = ROOT.parent / "smolvla_full_eval/smolvla_base_vs_combined_v3_full_10030.csv"
    official = list(csv.DictReader(official_path.open(encoding="utf-8", newline="")))
    mapping = list(csv.DictReader((ROOT / "frozen/instruction_map.csv").open(encoding="utf-8", newline="")))
    if len(official) != 10030 or len(mapping) != 10030:
        raise ValueError("Expected complete full benchmark and instruction map")
    clean = {arm: read_clean(arm) for arm in ARMS}
    expected_affected = {(r["suite"], int(r["task_id"])) for r in mapping
                         if r["category"] != "Language_Instructions"}
    for arm in ARMS:
        if clean[arm][0].keys() != expected_affected:
            raise ValueError(f"Incomplete affected-task coverage for {arm}")

    rows = []
    for baseline, instruction in zip(official, mapping):
        if (baseline["order"], baseline["suite"], baseline["task_id"], baseline["category"]) != (
            instruction["order"], instruction["suite"], instruction["task_id"], instruction["category"]
        ):
            raise ValueError("Official result and instruction map order differ")
        key = (baseline["suite"], int(baseline["task_id"]))
        row = {name: baseline[name] for name in ("order", "suite", "task_id", "category")}
        row.update({name: instruction[name] for name in
                    ("task_name", "base_task_name", "official_instruction", "clean_instruction")})
        for arm in ARMS:
            row[f"{arm}_official"] = int(baseline[arm])
            row[f"{arm}_clean_instruction"] = (clean[arm][0][key] if key in expected_affected
                                                else int(baseline[arm]))
        row["language_task_reused"] = int(key not in expected_affected)
        rows.append(row)

    columns = ("order", "suite", "task_id", "category", "task_name", "base_task_name",
               "official_instruction", "clean_instruction", "language_task_reused",
               "base_official", "base_clean_instruction", "combined_v3_official",
               "combined_v3_clean_instruction")
    full_path = ROOT / "smolvla_clean_instruction_full_10030_per_task.csv"
    write_csv(full_path, rows, columns)
    paired_path = ROOT.parent / "eval_kit/paired_plan_4295.tsv"
    if not paired_path.exists():
        paired_path = ROOT.parent / "smolvla_turing_setup/eval_kit/paired_plan_4295.tsv"
    paired_keys = [(row[1], row[2]) for line in paired_path.read_text().splitlines()
                   if line and not line.startswith("#") for row in [line.split("\t")]]
    if len(paired_keys) != 4295 or len(set(paired_keys)) != 4295:
        raise ValueError("Expected 4,295 unique paired keys")
    lookup = {(r["suite"], r["task_id"]): r for r in rows}
    subset = [lookup[key] for key in paired_keys]
    subset_path = ROOT / "smolvla_clean_instruction_paired_4295_per_task.csv"
    write_csv(subset_path, subset, columns)

    lines = ["SmolVLA clean-instruction ablation, seed 7, one trial per affected task",
             "10,030 task outcomes per model: 8,493 re-evaluated with cleaned text; "
             "1,537 Language tasks retain their original outcomes.", ""]
    for arm in ARMS:
        lines.append(f"{arm} ({clean[arm][1]} recovered error records)")
        lines.append("suite/category                     n  official   clean    change")
        for suite in (*SUITES, "all suites"):
            for category in (*CATEGORIES, "ALL"):
                cell = [r for r in rows if (suite == "all suites" or r["suite"] == suite)
                        and (category == "ALL" or r["category"] == category)]
                if not cell:
                    continue
                official_rate = 100 * sum(r[f"{arm}_official"] for r in cell) / len(cell)
                clean_rate = 100 * sum(r[f"{arm}_clean_instruction"] for r in cell) / len(cell)
                lines.append(f"{suite}/{category:<28} {len(cell):>5}  {official_rate:>7.2f}% "
                             f"{clean_rate:>7.2f}% {clean_rate-official_rate:>+8.2f}")
        gains = sum(r[f"{arm}_clean_instruction"] and not r[f"{arm}_official"] for r in rows)
        losses = sum(r[f"{arm}_official"] and not r[f"{arm}_clean_instruction"] for r in rows)
        lines.append(f"Clean-only successes={gains}; official-only successes={losses}")
        lines.append("")
    lines += [f"Full per-task CSV: {full_path}", f"Paired-subset per-task CSV: {subset_path}",
              "The 'clean' training arm from earlier work means no noise augmentation; it is separate from this text ablation."]
    report_path = ROOT / "clean_instruction_results.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
