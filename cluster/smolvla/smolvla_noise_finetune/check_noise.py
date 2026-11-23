#!/usr/bin/env python3
"""Fast fail-closed preflight for the exact noise recipe and captured trainer."""

import json
from pathlib import Path

import torch

from lerobot.transforms.transforms import ImageTransformsConfig
from train_noise import UInt8CompatibleImageTransforms


def main():
    recipe = json.loads(Path("noise_recipe.json").read_text())
    assert recipe["name"] == "smolvla_sensor_noise_v1"
    assert recipe["max_num_transforms"] == 1
    transforms = recipe["transforms"]
    assert set(transforms) == {"clean", "gaussian_sensor", "motion_blur", "compression"}
    assert all(entry["weight"] > 0 for entry in transforms.values())

    config = ImageTransformsConfig(enable=True, max_num_transforms=1)
    for name in ("gaussian_sensor", "motion_blur", "compression"):
        from lerobot.transforms.transforms import ImageTransformConfig

        entry = transforms[name]
        config.tfs = {name: ImageTransformConfig(**entry)}
        transform = UInt8CompatibleImageTransforms(config)
        y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
        source = torch.stack([((x * 17 + y * 29 + c * 41) % 256).to(torch.uint8) for c in range(3)])
        torch.manual_seed(17)
        result = transform(source)
        if result.dtype != torch.uint8 or result.shape != source.shape:
            raise AssertionError(f"{name}: wrong image dtype/shape")
        changed = (result != source).count_nonzero().item()
        if changed < 1:
            raise AssertionError(f"{name}: transform did not change an image")
        print(f"PASS {name}: {changed} / {source.numel()} pixels changed")

    print("PASS: all sensor-noise transforms modify uint8 camera input")


if __name__ == "__main__":
    main()
