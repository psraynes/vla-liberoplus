"""Measure fresh SmolVLA action-chunk queries on a fixed real LIBERO+ observation."""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "smolvla_turing" / "eval_kit"))

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from smolvla_adapter import Adapter


def gpu_snapshot():
    cmd = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
           "--format=csv,noheader,nounits"]
    return subprocess.check_output(cmd, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 5 or args.queries < 20:
        raise ValueError("Use at least 5 warmup and 20 measured queries")
    if os.getenv("SMOLVLA_PROFILE_CSV"):
        raise RuntimeError("Disable the model's CSV profiling hook for end-to-end timing")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)

    model = os.environ["SMOLVLA_MODEL"]
    adapter = Adapter()
    if adapter.policy.config.n_action_steps != 10:
        raise RuntimeError("Expected ten executed actions per query")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=adapter.image_size, camera_widths=adapter.image_size,
    )
    try:
        env.seed(7)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.warmup):
            adapter.reset()  # force a new chunk, rather than consuming cached actions
            adapter.predict(obs, task.language)
        torch.cuda.synchronize()
        timings = []
        for _ in range(args.queries):
            adapter.reset()
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            adapter.predict(obs, task.language)
            torch.cuda.synchronize()
            timings.append((time.perf_counter_ns() - start) / 1_000_000)
        report = {
            "model": model, "job_id": os.getenv("SLURM_JOB_ID"),
            "gpu": torch.cuda.get_device_name(), "gpu_snapshot_after": gpu_snapshot(),
            "source_task": "libero_spatial/0, fixed post-settle observation, seed 7",
            "protocol": "fresh action chunk per query; adapter preprocessing + model + postprocessing + host copy; CUDA synchronized; simulator excluded",
            "warmup_queries": args.warmup, "measured_queries": args.queries,
            "actions_executed_per_query": adapter.policy.config.n_action_steps,
            "latency_ms_mean": statistics.mean(timings),
            "latency_ms_median": statistics.median(timings),
            "latency_ms_p95": float(np.percentile(timings, 95)),
            "ms_per_executed_action_mean": statistics.mean(timings) / adapter.policy.config.n_action_steps,
            "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_torch_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "raw_latency_ms": timings,
        }
        model_csv = args.output.with_suffix(".model_queries.csv")
        if model_csv.exists():
            raise FileExistsError(model_csv)
        os.environ["SMOLVLA_PROFILE_CSV"] = str(model_csv)
        try:
            for _ in range(args.queries):
                adapter.reset()
                adapter.predict(obs, task.language)
        finally:
            os.environ.pop("SMOLVLA_PROFILE_CSV", None)
        if not model_csv.exists():
            raise RuntimeError("Pinned SmolVLA model profiling hook did not write CSV")
        with model_csv.open(newline="", encoding="utf-8") as stream:
            model_rows = list(csv.DictReader(stream))
        if len(model_rows) != args.queries:
            raise RuntimeError(f"Expected {args.queries} model queries; got {len(model_rows)}")
        model_ms = [float(row["latency_ms"]) for row in model_rows]
        report["model_chunk_ms_mean"] = statistics.mean(model_ms)
        report["model_chunk_ms_median"] = statistics.median(model_ms)
        report["model_chunk_ms_p95"] = float(np.percentile(model_ms, 95))
        report["model_profile_csv"] = str(model_csv)
    finally:
        env.close()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "raw_latency_ms"}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
