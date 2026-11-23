"""Load config.sh into os.environ for the Python scripts (shell scripts source it directly)."""

import os
import subprocess
from pathlib import Path

KIT = Path(__file__).resolve().parent


def load() -> dict:
    out = subprocess.run(
        ["bash", "-c", f"set -a; source '{KIT / 'config.sh'}'; env -0"],
        check=True, capture_output=True,
    ).stdout.decode()
    values = dict(item.split("=", 1) for item in out.split("\0") if "=" in item)
    os.environ.update(values)
    return values


def classification_path() -> Path:
    return Path(os.environ["LIBERO_PLUS_ROOT"]) / "libero" / "libero" / "benchmark" / "task_classification.json"
