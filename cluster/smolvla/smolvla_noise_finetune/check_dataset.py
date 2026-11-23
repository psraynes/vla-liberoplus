#!/usr/bin/env python3
"""Check the pinned demonstration dataset before a SmolVLA training job."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    path = Path(hf_hub_download("lerobot/libero", "meta/info.json", repo_type="dataset", revision=args.revision))
    info = json.loads(path.read_text())
    if info.get("codebase_version") != "v3.0" or info.get("fps") != 10.0:
        raise ValueError(f"Unexpected dataset version/fps: {info.get('codebase_version')}, {info.get('fps')}")
    features = info.get("features", {})
    expected = {
        "observation.images.image": ("video", [256, 256, 3]),
        "observation.images.image2": ("video", [256, 256, 3]),
        "observation.state": ("float32", [8]),
        "action": ("float32", [7]),
        "task_index": ("int64", [1]),
    }
    for name, (dtype, shape) in expected.items():
        feature = features.get(name, {})
        if feature.get("dtype") != dtype or feature.get("shape") != shape:
            raise ValueError(f"Unexpected feature {name}: {feature}")
    if info.get("total_episodes", 0) < 1000 or info.get("total_tasks") != 40:
        raise ValueError("Unexpected LIBERO episode/task count")
    print(f"PASS: lerobot/libero@{args.revision}, {info['total_episodes']} episodes, 40 tasks, 2 cameras, 8D state, 7D action")


if __name__ == "__main__":
    main()
