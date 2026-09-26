"""Workspace layout.

Everything taskdistill writes at runtime lives under the workspace, ``$TASKDISTILL_HOME``
(default ``./.taskdistill``): the capture/serve store, the response cache, the spend ledger,
the pricing snapshot, curated data and training runs.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HOME = "./.taskdistill"


def home() -> Path:
    """Return the workspace directory, creating it if needed."""
    path = Path(os.environ.get("TASKDISTILL_HOME") or DEFAULT_HOME).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def task_home(task: str) -> Path:
    path = home() / task
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(task: str) -> Path:
    path = task_home(task) / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runs_dir(task: str) -> Path:
    path = task_home(task) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def eval_dir(task: str) -> Path:
    path = task_home(task) / "eval"
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_path() -> Path:
    return home() / "store.sqlite"


def ledger_path() -> Path:
    return home() / "ledger.sqlite"


def cache_path() -> Path:
    return home() / "cache.sqlite"


def pricing_path() -> Path:
    return home() / "pricing.json"


def datasets_dir() -> Path:
    path = home() / "_datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_home(path: Path | str) -> str:
    """Render a path relative to the workspace so reports never carry absolute paths."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(home()))
    except ValueError:
        try:
            return str(p.relative_to(Path.cwd()))
        except ValueError:
            return p.name
