#!/usr/bin/env python3
"""Save a first/middle/last contact sheet from one collected teacher episode."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    records = [json.loads(line) for line in (run / "attempts.jsonl").read_text().splitlines() if line.strip()]
    success = next((record for record in records if record.get("success") is True), None)
    if success is None:
        raise ValueError("No successful teacher episode to preview")
    path = (run / success["episode_file"]).resolve(strict=True)
    if not path.is_relative_to(run):
        raise ValueError("Episode path escapes collection directory")
    with np.load(path, allow_pickle=False) as data:
        first, middle, last = 0, len(data["action"]) // 2, len(data["action"]) - 1
        frames = [np.concatenate((data["camera1"][i], data["camera2"][i]), axis=1)
                  for i in (first, middle, last)]
    sheet = np.concatenate(frames, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.output, sheet)
    print(f"Saved {args.output} from {success['suite']}/{success['task_id']} at first/middle/last steps")


if __name__ == "__main__":
    main()
