"""Reproduce the captured LeRobot split failure and check the metadata repair."""

import __future__
import ast
import logging
import math
import tarfile
from pathlib import Path
from types import SimpleNamespace

from split_compat import ensure_episode_tasks


class EpisodeTable:
    def __init__(self, columns):
        self.columns = columns
        self.column_names = list(columns)

    def __getitem__(self, name):
        if name not in self.columns:
            raise ValueError(f"Column '{name}' doesn't exist.")
        return self.columns[name]

    def __len__(self):
        return len(self.columns["dataset_from_index"])

    def add_column(self, name, values):
        return EpisodeTable({**self.columns, name: values})


class Scalar:
    def __init__(self, value):
        self.value = value

    def as_py(self):
        return self.value


class Frames:
    column_names = ["task_index"]

    def __init__(self):
        self.values = [0] * 8 + [1] * 8
        self.data = self

    def __len__(self):
        return len(self.values)

    def column(self, name):
        assert name == "task_index"
        return [Scalar(value) for value in self.values]


def main():
    archive = Path(__file__).resolve().parent.parent / "smolvla_turing_setup" / "captured" / "lerobot_source.tar.gz"
    with tarfile.open(archive) as source:
        code = source.extractfile("lerobot/src/lerobot/datasets/factory.py").read().decode()
    fn = next(node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)
              and node.name == "make_train_eval_datasets")
    namespace = {
        "math": math,
        "logging": logging,
        "resolve_delta_timestamps": lambda *args: None,
        "LeRobotDataset": lambda *args, **kwargs: SimpleNamespace(episodes=kwargs["episodes"]),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "factory.py", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)

    dataset = SimpleNamespace(
        episodes=None, num_episodes=8,
        meta=SimpleNamespace(episodes=EpisodeTable({"dataset_from_index": list(range(0, 16, 2)),
                                                   "dataset_to_index": list(range(2, 18, 2))}),
                             total_tasks=2),
        hf_dataset=Frames(),
    )
    cfg = SimpleNamespace(dataset=SimpleNamespace(eval_split=0.25, repo_id="lerobot/libero", root=None,
                                                  revision="pinned", video_backend="torchcodec",
                                                  image_transforms=SimpleNamespace(enable=False),
                                                  use_imagenet_stats=False),
                          trainable_config=None, tolerance_s=0.0001)
    namespace["make_dataset"] = lambda unused: dataset
    try:
        namespace["make_train_eval_datasets"](cfg)
    except ValueError as error:
        assert "Column 'tasks' doesn't exist" in str(error), error
        print("RED: captured LeRobot split reproduces the pilot's missing tasks-column error")
    else:
        raise AssertionError("Original split unexpectedly accepted metadata without tasks")

    namespace["make_dataset"] = lambda unused: ensure_episode_tasks(dataset)
    train, validation = namespace["make_train_eval_datasets"](cfg)
    assert train.episodes == [0, 1, 2, 4, 5, 6], train.episodes
    assert validation.episodes == [3, 7], validation.episodes
    assert not set(train.episodes) & set(validation.episodes)
    print("GREEN: task-stratified split has 6 training and 2 validation episodes, disjoint")


if __name__ == "__main__":
    main()
