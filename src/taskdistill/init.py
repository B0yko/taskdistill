"""``taskdistill init``: scaffold ``tasks/<task>/`` from the packaged templates.

Templates live in ``taskdistill/_data/templates/<type>/`` and are read with ``importlib.resources``,
so scaffolding works from an installed wheel as well as from a source checkout.
"""

from __future__ import annotations

import contextlib
import os
import re
import secrets
import tempfile
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from taskdistill.config import ConfigError, load_task

PLACEHOLDER = "__TASK__"
TASK_TYPES = ("classification", "extraction")
TASK_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class TaskExistsError(ConfigError, FileExistsError):
    """``tasks/<task>/`` already exists and overwriting was not requested."""


def templates_root() -> Traversable:
    return resources.files("taskdistill") / "_data" / "templates"


def template_files(task_type: str) -> dict[str, str]:
    """File name -> text of the packaged template for ``task_type``, sorted by name."""
    if task_type not in TASK_TYPES:
        raise ConfigError(f"type: unknown task type '{task_type}'; expected one of {', '.join(TASK_TYPES)}")
    directory = templates_root() / task_type
    files = {
        entry.name: entry.read_text(encoding="utf-8")
        for entry in directory.iterdir()
        if entry.is_file() and not entry.name.startswith(".")
    }
    if "task.yaml" not in files:
        raise ConfigError(f"the packaged {task_type} template has no task.yaml")
    return dict(sorted(files.items()))


def render_template(name: str, task_type: str) -> dict[str, str]:
    """The template files for ``task_type`` with every ``__TASK__`` replaced by ``name`` (literal replacement)."""
    return {file: text.replace(PLACEHOLDER, name) for file, text in template_files(task_type).items()}


def _os_reason(exc: OSError) -> str:
    return exc.strerror or str(exc)


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _check_destination(target: Path, files: dict[str, str], force: bool) -> None:
    """Refuse anything in the way of ``target`` and its files before the template is validated or written."""
    for ancestor in target.parents:
        if _exists(ancestor):
            if not ancestor.is_dir():
                raise ConfigError(f"{ancestor} exists and is not a directory")
            break
    if not _exists(target):
        return
    if not target.is_dir():
        raise ConfigError(f"{target} exists and is not a directory")
    if not force:
        raise TaskExistsError(f"{target} already exists; use --force to overwrite its template files")
    for file in files:
        path = target / file
        if _exists(path) and not path.is_file():
            raise ConfigError(f"{path} exists and is not a file")


def _validate(name: str, files: dict[str, str], target: Path) -> None:
    """Load the rendered files with ``load_task`` from a scratch copy, so a failure writes nothing."""
    try:
        with tempfile.TemporaryDirectory(prefix="taskdistill-init-") as tmp:
            staged = Path(tmp) / name
            staged.mkdir()
            for file, text in files.items():
                (staged / file).write_text(text, encoding="utf-8")
            try:
                load_task(str(staged))
            except ConfigError as exc:
                message = str(exc).replace(str(staged / "task.yaml"), str(target / "task.yaml"))
                raise ConfigError(message) from exc
    except OSError as exc:
        raise ConfigError(f"cannot stage the {name} template for validation: {_os_reason(exc)}") from exc


def _write_new(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, which must not exist yet."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_files(target: Path, files: dict[str, str]) -> None:
    """Write every file to a hidden temporary name first, then move each into place.

    A failure while writing leaves the existing files untouched and removes the temporary ones.
    """
    staged: list[tuple[Path, Path]] = []
    try:
        for file, text in files.items():
            tmp = target / f".{file}.{secrets.token_hex(8)}.tmp"
            staged.append((tmp, target / file))
            _write_new(tmp, text)
        for tmp, final in staged:
            tmp.replace(final)
    finally:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)


def init_task(name: str, task_type: str, dest_root: Path | None = None, force: bool = False) -> Path:
    """Create ``<dest_root>/tasks/<name>/`` (``dest_root`` defaults to the current directory) from a template.

    The rendered spec is validated with :func:`taskdistill.config.load_task` before anything is written.
    An existing directory is refused unless ``force``; with ``force`` only the template's files are
    replaced (a symlink among them is replaced, not written through) and any other file is kept.
    Every problem, including one reported by the filesystem, raises
    :class:`~taskdistill.config.ConfigError`. Returns the task directory.
    """
    if not TASK_NAME.fullmatch(name):
        raise ConfigError(
            f"task: '{name}' is not a valid task name; use 1-64 letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    files = render_template(name, task_type)
    target = (Path.cwd() if dest_root is None else Path(dest_root)) / "tasks" / name
    try:
        _check_destination(target, files, force)
    except TaskExistsError:
        raise
    except OSError as exc:
        raise ConfigError(f"{target}: cannot inspect the destination: {_os_reason(exc)}") from exc
    _validate(name, files, target)
    created = False
    try:
        created = not _exists(target)
        target.mkdir(parents=True, exist_ok=True)
        _write_files(target, files)
    except OSError as exc:
        if created:
            with contextlib.suppress(OSError):
                target.rmdir()
        raise ConfigError(f"{target}: cannot write task files: {_os_reason(exc)}") from exc
    return target
