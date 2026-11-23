"""Adapt LIBERO v3 episode metadata to this captured LeRobot split implementation.

The trainer's make_train_eval_datasets() groups episodes by a `tasks` column.
The pinned public LIBERO release can omit that column even though each frame has
`task_index`. Add an in-memory column derived from the first frame of each
episode. This leaves downloaded files, training examples and split logic intact.
"""


def ensure_episode_tasks(dataset):
    episodes = dataset.meta.episodes
    if "tasks" in episodes.column_names:
        return dataset
    if not {"dataset_from_index", "dataset_to_index"} <= set(episodes.column_names):
        raise ValueError("Cannot derive episode tasks: frame boundaries are missing")

    frames = dataset.hf_dataset
    if "task_index" not in frames.column_names:
        raise ValueError("Cannot derive episode tasks: frame task_index is missing")
    task_index_column = frames.data.column("task_index")
    task_count = dataset.meta.total_tasks
    labels = []
    for start, stop in zip(episodes["dataset_from_index"], episodes["dataset_to_index"], strict=True):
        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= len(frames):
            raise ValueError(f"Invalid episode frame interval [{start}, {stop})")
        scalar = task_index_column[start]
        task_index = int(scalar.as_py() if hasattr(scalar, "as_py") else scalar)
        end_scalar = task_index_column[stop - 1]
        end_task_index = int(end_scalar.as_py() if hasattr(end_scalar, "as_py") else end_scalar)
        if end_task_index != task_index:
            raise ValueError(f"Task index changes within episode [{start}, {stop})")
        if not 0 <= task_index < task_count:
            raise ValueError(f"Invalid frame task_index {task_index} at episode start {start}")
        labels.append([str(task_index)])
    if len(labels) != len(episodes):
        raise ValueError("Episode task derivation changed the episode count")
    dataset.meta.episodes = episodes.add_column("tasks", labels)
    print(f"Added in-memory task labels for {len(labels)} episodes; split remains stratified by task", flush=True)
    return dataset
