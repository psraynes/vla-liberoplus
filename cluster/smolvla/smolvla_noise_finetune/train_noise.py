#!/usr/bin/env python3
"""Run the captured LeRobot trainer with effective uint8 camera augmentation.

The captured LeRobot training factory requests uint8 video frames, but its custom
GaussianNoise/MotionBlur/JPEGCompression transforms only modify floating tensors.
Convert only at the transform boundary, then return uint8 as the trainer expects.
This changes no actions, states, model code, or validation images.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from lerobot.datasets import factory
from lerobot.transforms.transforms import ImageTransforms
from split_compat import ensure_episode_tasks


class UInt8CompatibleImageTransforms(ImageTransforms):
    def __init__(self, cfg):
        recipe_path = os.environ.get("SMOLVLA_NOISE_RECIPE")
        if recipe_path:
            expected = json.loads(Path(recipe_path).read_text())["transforms"]
            actual = {
                name: {"weight": entry.weight, "type": entry.type, "kwargs": entry.kwargs}
                for name, entry in cfg.tfs.items()
            }
            if json.loads(json.dumps(actual)) != expected:
                raise ValueError("Parsed training transform configuration differs from noise_recipe.json")
            if not cfg.enable or cfg.max_num_transforms != 1:
                raise ValueError("Noise training must enable exactly one image transform per frame")
        super().__init__(cfg)

    def forward(self, *inputs):
        was_uint8 = [isinstance(value, torch.Tensor) and value.dtype == torch.uint8 for value in inputs]
        converted = [value.float().div(255.0) if flag else value for value, flag in zip(inputs, was_uint8)]
        result = super().forward(*converted)

        def restore(value, index):
            if not was_uint8[index]:
                return value
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise TypeError("Image transform did not return a floating tensor")
            return value.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)

        if len(inputs) == 1:
            return restore(result, 0)
        if not isinstance(result, (tuple, list)) or len(result) != len(inputs):
            raise TypeError("Image transform changed the number of camera inputs")
        return type(result)(restore(value, index) for index, value in enumerate(result))


def main():
    if "--dataset.image_transforms.enable=true" in sys.argv and not os.environ.get("SMOLVLA_NOISE_RECIPE"):
        raise RuntimeError("Noise training requires SMOLVLA_NOISE_RECIPE for config verification")
    factory.ImageTransforms = UInt8CompatibleImageTransforms
    original_make_dataset = factory.make_dataset

    def make_dataset_with_compatible_episode_metadata(cfg):
        dataset = original_make_dataset(cfg)
        if cfg.dataset.eval_split > 0:
            ensure_episode_tasks(dataset)
        return dataset

    factory.make_dataset = make_dataset_with_compatible_episode_metadata
    from lerobot.scripts.lerobot_train import main as lerobot_main

    lerobot_main()


if __name__ == "__main__":
    main()
