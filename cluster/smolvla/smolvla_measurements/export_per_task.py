"""Export final per-task SmolVLA results without modifying source result files."""

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path


PLUS_ARMS = ("base", "noise", "clean", "combined_v3")
STANDARD_ARMS = (*PLUS_ARMS[:3], "teacher_v2", PLUS_ARMS[3])
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def read_wide(path, arms, expected):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"order", "suite", "task_id", "category", *arms}
        if not required <= set(reader.fieldnames or ()):  # fail before accepting any result
            raise ValueError(f"Missing columns in {path}: {required - set(reader.fieldnames or ())}")
        rows = list(reader)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} tasks in {path}; got {len(rows)}")
    by_key = {}
    orders = set()
    for row in rows:
        key = (row["suite"], int(row["task_id"]))
        order = int(row["order"])
        if key in by_key or order in orders or row["suite"] not in SUITES:
            raise ValueError(f"Duplicate/invalid task or order in {path}: {key}")
        orders.add(order)
        for arm in arms:
            if row[arm] not in ("0", "1"):
                raise ValueError(f"Invalid {arm} success at {key}: {row[arm]}")
        by_key[key] = row
    return rows, by_key


def read_manifest(path):
    pairs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            pairs[key] = value
    for key, expected in (("seed", "7"), ("action_steps", "10")):
        if pairs.get(key) != expected:
            raise ValueError(f"{path}: expected {key}={expected}")
    if not pairs.get("model"):
        raise ValueError(f"{path}: missing model")
    return pairs


def read_standard(root, arm):
    run = root / "smolvla_measurements" / "results" / f"standard_full_{arm}"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("episodes_per_task") != 50 or manifest.get("task_id_filter") is not None:
        raise ValueError(f"{run}: not a 50-trial full standard run")
    if manifest.get("action_steps") != 10 or manifest.get("seed_rule") != "7 + episode":
        raise ValueError(f"{run}: unexpected action or seed protocol")
    result_path = run / "results.jsonl"
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    expected = {(suite, task_id, episode) for suite in SUITES
                for task_id in range(10) for episode in range(50)}
    observed = {}
    for row in rows:
        key = (row["suite"], int(row["task_id"]), int(row["episode"]))
        if key in observed:
            raise ValueError(f"{arm}: duplicate standard episode {key}")
        if type(row.get("success")) is not bool or row.get("seed") != 7 + key[2]:
            raise ValueError(f"{arm}: malformed standard episode {key}")
        observed[key] = row
    if set(observed) != expected:
        raise ValueError(f"{arm}: {len(observed)}/2000 standard episodes complete")
    return manifest, observed, result_path


def read_profile(root, arm, model):
    candidates = sorted((root / "smolvla_measurements" / "results").glob(f"profile_{arm}_*.json"))
    if not candidates:
        raise FileNotFoundError(f"Missing {arm} profile JSON")
    path = candidates[-1]
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("model") != model or row.get("measured_queries") != 100:
        raise ValueError(f"{path}: profile/model mismatch")
    if row.get("actions_executed_per_query") != 10:
        raise ValueError(f"{path}: expected ten actions per query")
    for key in ("latency_ms_mean", "model_chunk_ms_mean", "ms_per_executed_action_mean",
                "peak_torch_allocated_mib", "peak_torch_reserved_mib"):
        if not isinstance(row.get(key), (float, int)) or row[key] <= 0:
            raise ValueError(f"{path}: invalid {key}")
    return row, path


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.home() / "smolvla_turing")
    args = parser.parse_args()
    root = args.root.resolve()
    paired_path = root / "smolvla_rollout_eval" / "paired_rollouts_four_models.csv"
    full_path = root / "smolvla_full_eval" / "smolvla_base_vs_combined_v3_full_10030.csv"
    paired, paired_by_key = read_wide(paired_path, PLUS_ARMS, 4295)
    full, full_by_key = read_wide(full_path, ("base", "combined_v3"), 10030)
    if not set(paired_by_key) <= set(full_by_key):
        raise ValueError("4,295 paired tasks are not a subset of full LIBERO+")
    for key, row in paired_by_key.items():
        other = full_by_key[key]
        for column in ("suite", "task_id", "category", "base", "combined_v3"):
            if row[column] != other[column]:
                raise ValueError(f"Full/paired mismatch for {key}, column {column}")

    plus_models = {}
    for arm in PLUS_ARMS:
        manifest = read_manifest(root / "smolvla_rollout_eval" / "results" / arm / "run_manifest.txt")
        plus_models[arm] = manifest["model"]
    for arm in ("base", "combined_v3"):
        manifest = read_manifest(root / "smolvla_full_eval" / "results" /
                                 f"{arm}_remaining_w0" / "run_manifest.txt")
        if manifest["model"] != plus_models[arm]:
            raise ValueError(f"Model path differs between full and paired runs: {arm}")

    standard = {}
    profiles = {}
    for arm in STANDARD_ARMS:
        manifest, episodes, source = read_standard(root, arm)
        if arm in plus_models and manifest["model"] != plus_models[arm]:
            raise ValueError(f"Standard/Plus model path mismatch: {arm}")
        standard[arm] = (manifest, episodes, source)
        profiles[arm] = read_profile(root, arm, manifest["model"])
    for arm in STANDARD_ARMS:
        for key, row in standard[arm][1].items():
            base_row = standard["base"][1][key]
            if (row["task_name"], row["instruction"]) != (base_row["task_name"], base_row["instruction"]):
                raise ValueError(f"Standard task differs across models: {arm}/{key}")

    out = root / "smolvla_measurements" / "final_per_task_exports_v1"
    out.mkdir(parents=False, exist_ok=False)
    plus_fields = ("benchmark", "model", "checkpoint", "suite", "task_id", "category",
                   "success", "seed", "actions_per_query", "in_frozen_4295")
    plus_rows = []
    for row in full:
        key = (row["suite"], int(row["task_id"]))
        for arm in ("base", "combined_v3"):
            plus_rows.append(dict(zip(plus_fields, (
                "LIBERO+ official", arm, plus_models[arm], key[0], key[1], row["category"],
                int(row[arm]), 7, 10, int(key in paired_by_key)
            ))))
    for row in paired:
        for arm in ("noise", "clean"):
            plus_rows.append(dict(zip(plus_fields, (
                "LIBERO+ official", arm, plus_models[arm], row["suite"], int(row["task_id"]),
                row["category"], int(row[arm]), 7, 10, 1
            ))))
    if len(plus_rows) != 2 * 10030 + 2 * 4295:
        raise AssertionError(len(plus_rows))
    write_csv(out / "liberoplus_all_models_per_task.csv", plus_fields, plus_rows)
    for arm in PLUS_ARMS:
        write_csv(out / f"liberoplus_{arm}_per_task.csv", plus_fields,
                  (r for r in plus_rows if r["model"] == arm))

    episode_fields = ("benchmark", "model", "checkpoint", "suite", "task_id", "task_name",
                      "instruction", "episode", "seed", "success", "steps", "actions_per_query")
    episode_rows = []
    task_rows = []
    task_fields = ("benchmark", "model", "checkpoint", "suite", "task_id", "task_name",
                   "instruction", "successes", "trials", "success_rate_pct")
    for arm in STANDARD_ARMS:
        manifest, episodes, _ = standard[arm]
        for key in sorted(episodes):
            row = episodes[key]
            episode_rows.append(dict(zip(episode_fields, (
                "standard LIBERO", arm, manifest["model"], key[0], key[1], row["task_name"],
                row["instruction"], key[2], row["seed"], int(row["success"]), row["steps"], 10
            ))))
        for suite in SUITES:
            for task_id in range(10):
                selected = [episodes[suite, task_id, episode] for episode in range(50)]
                successes = sum(row["success"] for row in selected)
                task_rows.append(dict(zip(task_fields, (
                    "standard LIBERO", arm, manifest["model"], suite, task_id,
                    selected[0]["task_name"], selected[0]["instruction"],
                    successes, 50, round(100 * successes / 50, 2)
                ))))
    write_csv(out / "standard_all_models_per_episode.csv", episode_fields, episode_rows)
    write_csv(out / "standard_all_models_per_task.csv", task_fields, task_rows)
    for arm in STANDARD_ARMS:
        write_csv(out / f"standard_{arm}_per_task.csv", task_fields,
                  (r for r in task_rows if r["model"] == arm))

    profile_fields = ("model", "checkpoint", "gpu", "measured_queries", "actions_per_query",
                      "latency_ms_mean", "latency_ms_median", "latency_ms_p95",
                      "model_chunk_ms_mean", "ms_per_executed_action_mean",
                      "peak_torch_allocated_mib", "peak_torch_reserved_mib", "source_json")
    profile_rows = []
    for arm in STANDARD_ARMS:
        row, path = profiles[arm]
        profile_rows.append(dict(zip(profile_fields, (
            arm, row["model"], row["gpu"], row["measured_queries"],
            row["actions_executed_per_query"], row["latency_ms_mean"],
            row["latency_ms_median"], row["latency_ms_p95"],
            row["model_chunk_ms_mean"], row["ms_per_executed_action_mean"],
            row["peak_torch_allocated_mib"], row["peak_torch_reserved_mib"], str(path)
        ))))
    write_csv(out / "model_profiles.csv", profile_fields, profile_rows)

    for source in (paired_path, full_path):
        shutil.copy2(source, out / source.name)
    (out / "README.txt").write_text(
        "Per-task SmolVLA results for the capstone report.\n"
        "LIBERO+ base and combined_v3: 10,030 tasks each; noise and clean: 4,295 each.\n"
        "teacher_v2 has no completed LIBERO+ rollout evaluation.\n"
        "Standard LIBERO: 40 tasks x 50 episodes for each of five models.\n"
        "success is 0/1; standard per-task success_rate_pct aggregates 50 episodes.\n"
        "in_frozen_4295 permits exact-subset filtering of full base/combined_v3 rows.\n"
        "Profiles use one fixed observation on an allocated GPU; peak_torch_* are PyTorch allocator metrics.\n"
        "Source JSONL, manifests, and profile JSON remain in their original Turing directories.\n",
        encoding="utf-8",
    )
    manifest = {"files": []}
    for path in sorted(out.iterdir()):
        if path.is_file():
            manifest["files"].append({"name": path.name, "bytes": path.stat().st_size,
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (out / "MANIFEST_SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive = out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as target:
        target.add(out, arcname=out.name)
    print(f"PASS LIBERO+ task rows={len(plus_rows)}; standard episodes={len(episode_rows)}; "
          f"standard task/model rows={len(task_rows)}; profiles={len(profile_rows)}")
    print(f"Saved {archive}")


if __name__ == "__main__":
    main()
