"""Packaging: build the wheel, install it into a clean venv outside the source tree and run the CLI from there.

The wheel is built with ``uv build --wheel`` and installed with ``uv pip install`` into a fresh Python 3.12 venv
(``uv`` is taken from ``$UV``, then ``PATH``, then ``~/.local/bin/uv``; the tests skip without it). Installing
fetches the runtime dependencies from PyPI; the ``torch`` extra is not installed. Every command runs from an empty
directory outside the repository with ``TASKDISTILL_HOME`` inside it and no API key in the environment, so the demo
uses the teacher recording packaged under ``taskdistill/_data``. The only other download is the student tokenizer
from the Hugging Face Hub, which the curate length filter needs. On macOS arm64 the wheel pulls in mlx and mlx-lm;
on Linux it does not.

Dependency versions: by default the install is constrained to the versions in ``uv.lock`` (exported with
``uv export --frozen``), so a given commit resolves the same dependencies today and next month. With
``TASKDISTILL_PACKAGING_DEPS=latest`` the constraints are dropped and uv resolves the newest releases the
``pyproject.toml`` ranges allow, which is what a fresh ``pip install`` or ``uvx`` user gets; a failure in that mode
can come from an upstream release rather than from this repository.

Package data must be read through ``importlib.resources``, not from paths built on ``__file__``. An installed wheel
is unpacked into real files, where both work, so one test also imports ``taskdistill`` straight from the zipped
wheel (``sys.path`` pointing at the ``.whl``) and runs ``init`` and the packaged-spec and recording lookups there.

The default suite (and CI) runs ``taskdistill init`` and ``taskdistill demo invoices --profile quick --until curate``
from the installed wheel. The full quick demo from the installed wheel (training, evaluation, the cascade server and
the report) needs Metal, so it runs in the local ``-m mlx`` suite::

    .venv/bin/python -m pytest tests/test_packaging.py -m mlx -q -p no:cacheprovider
"""

from __future__ import annotations

import email.parser
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pytest
import yaml

import taskdistill

pytestmark = [pytest.mark.network, pytest.mark.slow, pytest.mark.timeout(900)]

REPO = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO / "src" / "taskdistill"
VERSION = taskdistill.__version__
DIST_INFO = f"taskdistill-{VERSION}.dist-info"
DEMOS = ("banking77", "invoices")
DEMO_STAGES = ("data", "capture", "curate", "train", "eval", "serve", "report")
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

EXPECTED_DATA = (
    "taskdistill/_data/pricing_snapshot.json",
    "taskdistill/_data/demos/banking77/teacher_recording.jsonl.gz",
    "taskdistill/_data/demos/invoices/teacher_recording.jsonl.gz",
    "taskdistill/_data/tasks/banking77/task.yaml",
    "taskdistill/_data/tasks/banking77/teacher_prompt.md",
    "taskdistill/_data/tasks/banking77/labels.txt",
    "taskdistill/_data/tasks/invoices/task.yaml",
    "taskdistill/_data/tasks/invoices/teacher_prompt.md",
    "taskdistill/_data/tasks/invoices/schema.json",
    "taskdistill/_data/templates/classification/task.yaml",
    "taskdistill/_data/templates/classification/teacher_prompt.md",
    "taskdistill/_data/templates/classification/labels.txt",
    "taskdistill/_data/templates/extraction/task.yaml",
    "taskdistill/_data/templates/extraction/teacher_prompt.md",
    "taskdistill/_data/templates/extraction/schema.json",
)
FORBIDDEN_PARTS = {"tests", "reports", "examples", "scripts", "docs", "__pycache__", ".taskdistill", "fixtures"}
FORBIDDEN_NAMES = {"conftest.py", "tiny_model.py", "pyproject.toml", "uv.lock", ".env"}

DEPS_MODE = os.environ.get("TASKDISTILL_PACKAGING_DEPS", "locked")

SCRUBBED_ENV = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV", "CONDA_PREFIX"}
SCRUBBED_PREFIXES = ("TASKDISTILL_", "OPENROUTER_")

PROBE = """
import importlib.metadata, importlib.util, json, platform, sys
from importlib import resources
import taskdistill
data = resources.files("taskdistill") / "_data"
print(json.dumps({
    "file": taskdistill.__file__,
    "version": taskdistill.__version__,
    "python": list(sys.version_info[:2]),
    "prefix": sys.prefix,
    "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
    "modules": {m: importlib.util.find_spec(m) is not None for m in ("torch", "peft", "mlx", "mlx_lm", "pytest")},
    "distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    "recordings": sorted(d.name for d in (data / "demos").iterdir() if (d / "teacher_recording.jsonl.gz").is_file()),
    "templates": sorted(d.name for d in (data / "templates").iterdir() if (d / "task.yaml").is_file()),
}))
"""

LOAD_SPEC = """
import json
from taskdistill.config import load_task
spec = load_task("tickets")
print(json.dumps({"task": spec.task, "type": spec.type, "labels": spec.labels, "source": spec.source}))
"""

ZIPPED = """
import json, os, sys
from pathlib import Path
wheel = sys.argv[1]
sys.path.insert(0, wheel)
from importlib import resources
import taskdistill
import taskdistill.cli
from taskdistill.config import load_task
from taskdistill.teacher.factory import find_recording
from taskdistill.teacher.replay import load_recording
sys.argv = ["taskdistill", "init", "t2", "--type", "extraction"]
try:
    taskdistill.cli.main()
except SystemExit as exc:
    code = exc.code
else:
    code = 0
bundled = load_task("invoices")
scaffolded = load_task("t2")
found = find_recording("invoices")
recording = load_recording(found)
inside = wheel + os.sep
print(json.dumps({
    "init_exit": code,
    "package_file": taskdistill.__file__,
    "outside_wheel": sorted(
        name for name, module in sys.modules.items()
        if name.split(".")[0] == "taskdistill" and not str(getattr(module, "__file__", "")).startswith(inside)
    ),
    "resources_is_path": isinstance(resources.files("taskdistill"), Path),
    "bundled": {
        "source": bundled.source, "type": bundled.type, "schema": bool(bundled.json_schema),
        "prompt_sha256": bundled.teacher_prompt_sha256,
    },
    "scaffolded": {"task": scaffolded.task, "type": scaffolded.type, "schema": bool(scaffolded.json_schema)},
    "recording": {
        "path": str(found), "is_path": isinstance(found, Path), "records": len(recording),
        "task": recording.manifest.get("task"), "prompt_sha256": recording.manifest.get("teacher_prompt_sha256"),
    },
}))
"""


# ------------------------------------------------------------------------------------------------ helpers
def _uv() -> str | None:
    candidates = [os.environ.get("UV"), shutil.which("uv"), str(Path.home() / ".local" / "bin" / "uv")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _clean_env(**extra: str) -> dict[str, str]:
    """The current environment without API keys, taskdistill settings or anything that points Python elsewhere."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(SCRUBBED_PREFIXES) and key not in SCRUBBED_ENV
    }
    env.update(extra)
    return env


def _run(
    argv: Sequence[str | Path], *, cwd: Path, env: Mapping[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    command = [str(arg) for arg in argv]
    try:
        proc = subprocess.run(
            command, cwd=cwd, env=dict(env), capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        output = "\n".join(str(part)[-6000:] for part in (exc.stdout, exc.stderr) if part)
        pytest.fail(f"{' '.join(command)} did not finish within {timeout:.0f} s\n{output}")
    assert proc.returncode == 0, (
        f"{' '.join(command)} exited with {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-6000:]}\n--- stderr ---\n{proc.stderr[-6000:]}"
    )
    return proc


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_versions(constraints: Path) -> dict[str, set[str]]:
    """Distribution name -> the versions ``uv.lock`` pins for it (one per resolution fork)."""
    pins: dict[str, set[str]] = {}
    for line in constraints.read_text(encoding="utf-8").splitlines():
        requirement = line.split(";", 1)[0].strip()
        if not requirement or requirement.startswith(("#", "-")) or "==" not in requirement:
            continue
        name, version = requirement.split("==", 1)
        pins.setdefault(_canonical(name.split("[", 1)[0]), set()).add(version.strip())
    return pins


def _outside_repo(path: Path) -> Path:
    resolved = path.resolve()
    assert not resolved.is_relative_to(REPO), f"{resolved} must be outside the source tree {REPO}"
    return resolved


@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENROUTER_API_KEY", "TASKDISTILL_TEACHER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


# ----------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def uv_bin() -> str:
    found = _uv()
    if found is None:
        pytest.skip("uv is not available ($UV, PATH or ~/.local/bin/uv)")
    return found


@pytest.fixture(scope="module")
def wheel(uv_bin: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = _outside_repo(tmp_path_factory.mktemp("dist"))
    _run([uv_bin, "build", "--wheel", "--out-dir", out, REPO], cwd=REPO, env=_clean_env(), timeout=300)
    wheels = sorted(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel in {out}, found {[w.name for w in wheels]}"
    assert wheels[0].name == f"taskdistill-{VERSION}-py3-none-any.whl"
    return wheels[0]


@pytest.fixture(scope="module")
def constraints(uv_bin: str, tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    """The runtime dependency versions from ``uv.lock``, or None with ``TASKDISTILL_PACKAGING_DEPS=latest``."""
    if DEPS_MODE == "latest":
        return None
    assert DEPS_MODE == "locked", f"TASKDISTILL_PACKAGING_DEPS must be 'locked' or 'latest', got {DEPS_MODE!r}"
    out = _outside_repo(tmp_path_factory.mktemp("lock")) / "constraints.txt"
    _run(
        [uv_bin, "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project", "--quiet", "-o", out],
        cwd=REPO,
        env=_clean_env(),
        timeout=120,
    )
    assert _locked_versions(out), f"no pinned versions in {out}"
    return out


@pytest.fixture(scope="module")
def venv(uv_bin: str, wheel: Path, constraints: Path | None, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh Python 3.12 venv, outside the repository, with only the wheel and its runtime dependencies."""
    root = _outside_repo(tmp_path_factory.mktemp("install"))
    target = root / "venv"
    env = _clean_env()
    _run([uv_bin, "venv", "--python", "3.12", target], cwd=root, env=env, timeout=180)
    python = target / "bin" / "python"
    pinned: list[str | Path] = [] if constraints is None else ["--constraint", constraints]
    _run([uv_bin, "pip", "install", "--python", python, wheel, *pinned], cwd=root, env=env, timeout=480)
    assert (target / "bin" / "taskdistill").is_file()
    return target


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """An empty working directory outside the repository."""
    work = _outside_repo(tmp_path) / "work"
    work.mkdir()
    return work


def _cli_env(venv: Path, workdir: Path) -> dict[str, str]:
    return _clean_env(
        TASKDISTILL_HOME=str(workdir / "workspace"),
        PATH=f"{venv / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        TRANSFORMERS_VERBOSITY="error",
        NO_COLOR="1",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------------- wheel
def test_wheel_contains_the_package_data_and_nothing_from_the_repository(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        packaged = {name: zf.read(name) for name in names if name.startswith("taskdistill/_data/")}
        metadata = email.parser.Parser().parsestr(zf.read(f"{DIST_INFO}/METADATA").decode("utf-8"))
        entry_points = zf.read(f"{DIST_INFO}/entry_points.txt").decode("utf-8")

    for name in EXPECTED_DATA:
        assert name in names, f"{name} is missing from the wheel"
    recordings = sorted(n for n in names if fnmatch(n, "taskdistill/_data/demos/*/teacher_recording.jsonl.gz"))
    assert recordings == [f"taskdistill/_data/demos/{demo}/teacher_recording.jsonl.gz" for demo in DEMOS]

    source_data = {
        f"taskdistill/{rel.as_posix()}": (PACKAGE_SRC / rel).read_bytes()
        for rel in sorted(path.relative_to(PACKAGE_SRC) for path in (PACKAGE_SRC / "_data").rglob("*"))
        if (PACKAGE_SRC / rel).is_file() and not any(p.startswith(".") or p == "__pycache__" for p in rel.parts)
    }
    assert set(packaged) == set(source_data)
    for name, content in source_data.items():
        assert hashlib.sha256(packaged[name]).digest() == hashlib.sha256(content).digest(), name

    source_modules = {
        f"taskdistill/{rel.as_posix()}"
        for rel in (path.relative_to(PACKAGE_SRC) for path in PACKAGE_SRC.rglob("*.py"))
        if "__pycache__" not in rel.parts
    }
    wheel_modules = {name for name in names if name.endswith(".py")}
    assert wheel_modules == source_modules

    assert {name.split("/", 1)[0] for name in names} == {"taskdistill", DIST_INFO}
    for name in names:
        parts = name.split("/")
        assert not FORBIDDEN_PARTS.intersection(parts), f"{name} does not belong in the wheel"
        assert parts[-1] not in FORBIDDEN_NAMES, f"{name} does not belong in the wheel"
        assert not name.endswith((".pyc", ".pyo", ".sqlite", ".safetensors")), f"{name} does not belong in the wheel"

    assert metadata["Name"] == "taskdistill"
    assert metadata["Version"] == VERSION
    assert metadata["Requires-Python"] == ">=3.12"
    requires = metadata.get_all("Requires-Dist") or []
    for heavy in ("torch", "peft"):
        lines = [r for r in requires if r.split(";")[0].strip().startswith(heavy)]
        assert lines, f"{heavy} is not declared"
        assert all("extra == 'torch'" in r or 'extra == "torch"' in r for r in lines), lines
    mlx_lines = [r for r in requires if r.startswith(("mlx==", "mlx-lm=="))]
    assert len(mlx_lines) == 2
    assert all("sys_platform == 'darwin'" in r and "platform_machine == 'arm64'" in r for r in mlx_lines), mlx_lines
    assert "torch" in (metadata.get_all("Provides-Extra") or [])
    assert "taskdistill = taskdistill.cli:main" in entry_points


# --------------------------------------------------------------------------------------- installed wheel
def test_installed_package_is_the_wheel_without_the_torch_extra(
    venv: Path, workdir: Path, constraints: Path | None
) -> None:
    env = _cli_env(venv, workdir)
    version = _run([venv / "bin" / "taskdistill", "--version"], cwd=workdir, env=env, timeout=120)
    assert version.stdout.strip() == f"taskdistill {VERSION}"

    probe = _run([venv / "bin" / "python", "-I", "-c", PROBE], cwd=workdir, env=env, timeout=120)
    info = json.loads(probe.stdout.strip().splitlines()[-1])
    assert Path(info["prefix"]).resolve() == venv.resolve()
    module = Path(info["file"]).resolve()
    assert module.is_relative_to(venv.resolve()), f"taskdistill imported from {module}, not the venv"
    assert "site-packages" in module.parts
    assert not module.is_relative_to(REPO)
    assert info["version"] == VERSION
    assert info["python"] == [3, 12]
    assert info["recordings"] == list(DEMOS)
    assert info["templates"] == ["classification", "extraction"]
    modules = info["modules"]
    assert modules["torch"] is False, "torch is an extra and must not be installed by default"
    assert modules["peft"] is False
    assert modules["pytest"] is False, "development dependencies leaked into the install"
    assert modules["mlx"] is info["apple_silicon"]
    assert modules["mlx_lm"] is info["apple_silicon"]

    if constraints is not None:
        pins = _locked_versions(constraints)
        installed = {_canonical(name): version for name, version in info["distributions"].items()}
        installed.pop("taskdistill")
        assert installed, "no dependencies were installed"
        unlocked = sorted(name for name in installed if name not in pins)
        assert not unlocked, f"installed distributions that uv.lock does not pin: {unlocked}"
        drifted = {
            name: (version, sorted(pins[name])) for name, version in installed.items() if version not in pins[name]
        }
        assert not drifted, f"installed versions differ from uv.lock: {drifted}"


def test_package_data_loads_from_the_zipped_wheel(venv: Path, wheel: Path, workdir: Path) -> None:
    """``importlib.resources`` reads package data from inside the ``.whl``; ``__file__``-relative paths cannot."""
    env = _cli_env(venv, workdir)
    proc = _run([venv / "bin" / "python", "-I", "-c", ZIPPED, wheel], cwd=workdir, env=env, timeout=180)
    info = json.loads(proc.stdout.strip().splitlines()[-1])

    inside = f"{wheel}{os.sep}"
    assert info["package_file"].startswith(inside), info["package_file"]
    assert info["outside_wheel"] == [], f"imported from outside the wheel: {info['outside_wheel']}"
    assert info["resources_is_path"] is False, "importlib.resources should see the zip archive, not a directory"

    assert info["init_exit"] in (0, None)
    directory = workdir / "tasks" / "t2"
    assert sorted(p.name for p in directory.iterdir()) == ["schema.json", "task.yaml", "teacher_prompt.md"]
    assert info["scaffolded"] == {"task": "t2", "type": "extraction", "schema": True}

    bundled = info["bundled"]
    assert bundled["source"] == "tasks/invoices/task.yaml", "the invoices spec should come from the package data"
    assert bundled["type"] == "extraction"
    assert bundled["schema"] is True

    recording = info["recording"]
    assert recording["is_path"] is False
    assert recording["path"] == f"{inside}taskdistill/_data/demos/invoices/teacher_recording.jsonl.gz"
    assert recording["task"] == "invoices"
    assert recording["records"] > 0
    assert recording["prompt_sha256"] == bundled["prompt_sha256"]
    assert not (workdir / "workspace" / "invoices").exists()


def test_init_from_the_installed_wheel(venv: Path, workdir: Path) -> None:
    env = _cli_env(venv, workdir)
    proc = _run(
        [venv / "bin" / "taskdistill", "init", "tickets", "--type", "classification"], cwd=workdir, env=env, timeout=120
    )
    assert "created tasks/tickets/" in proc.stdout

    directory = workdir / "tasks" / "tickets"
    spec_file = directory / "task.yaml"
    assert spec_file.is_file()
    assert sorted(p.name for p in directory.iterdir()) == ["labels.txt", "task.yaml", "teacher_prompt.md"]
    raw = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
    assert raw["task"] == "tickets"
    assert raw["type"] == "classification"
    assert "__TASK__" not in spec_file.read_text(encoding="utf-8")

    loaded = _run([venv / "bin" / "python", "-I", "-c", LOAD_SPEC], cwd=workdir, env=env, timeout=120)
    spec = json.loads(loaded.stdout.strip().splitlines()[-1])
    labels = [ln.strip() for ln in (directory / "labels.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert spec["task"] == "tickets"
    assert spec["type"] == "classification"
    assert spec["labels"] == labels
    assert len(labels) >= 2
    assert Path(spec["source"]).resolve() == spec_file.resolve()


def test_replay_demo_until_curate_from_the_installed_wheel(venv: Path, workdir: Path) -> None:
    env = _cli_env(venv, workdir)
    proc = _run(
        [venv / "bin" / "taskdistill", "demo", "invoices", "--profile", "quick", "--until", "curate"],
        cwd=workdir,
        env=env,
        timeout=420,
    )
    assert "teacher replay" in proc.stdout
    assert "[3/7] curate" in proc.stdout
    assert "[4/7]" not in proc.stdout

    workspace = workdir / "workspace"
    task_home = workspace / "invoices"
    data = task_home / "data"
    assert not (workdir / ".taskdistill").exists(), "the workspace must follow TASKDISTILL_HOME"

    train = _read_jsonl(data / "train.jsonl")
    assert train
    assert [m["role"] for m in train[0]["messages"]] == ["system", "user", "assistant"]
    assert json.loads(train[0]["messages"][-1]["content"])
    for split in ("valid", "test"):
        assert _read_jsonl(data / f"{split}.jsonl")

    stats = json.loads((data / "curate_stats.json").read_text(encoding="utf-8"))
    assert stats["task"] == "invoices"
    assert stats["teacher"]["mode"] == "replay"
    labelling = stats["labelling"]
    assert labelling["mode"] == "replay"
    assert labelling["live"] == 0
    assert labelling["cost_usd"] == 0
    assert labelling["replayed"] == labelling["labelled"] > 0
    assert all(stats["splits"][split] > 0 for split in ("train", "valid", "test"))
    assert stats["splits"]["train"] == len(train)
    assert stats["leakage"]["ok"] is True
    assert stats["length"]["tokenizer"] == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    assert stats["length"]["splits"]["train"]["kept"] == len(train)

    timing = json.loads((task_home / "demo_timing_quick.json").read_text(encoding="utf-8"))
    assert timing["mode"] == "replay"
    assert list(timing["stages_s"]) == ["data", "capture", "curate"]
    assert not any((task_home / "runs").glob("*"))
    assert not (workdir / "request.json").exists()
    assert not (workdir / "reports").exists()


@pytest.mark.mlx
@pytest.mark.timeout(3600)
@pytest.mark.skipif(not APPLE_SILICON, reason="the full quick demo trains with MLX (Apple Silicon only)")
def test_full_quick_demo_from_the_installed_wheel(venv: Path, workdir: Path) -> None:
    env = _cli_env(venv, workdir)
    proc = _run(
        [venv / "bin" / "taskdistill", "demo", "invoices", "--profile", "quick"], cwd=workdir, env=env, timeout=3300
    )
    assert "teacher replay" in proc.stdout
    assert "routes:" in proc.stdout

    task_home = workdir / "workspace" / "invoices"
    timing = json.loads((task_home / "demo_timing_quick.json").read_text(encoding="utf-8"))
    assert timing["mode"] == "replay"
    assert list(timing["stages_s"]) == list(DEMO_STAGES)
    assert any((task_home / "runs").iterdir())

    reports = workdir / "reports" / "invoices"
    report = json.loads((reports / "report.json").read_text(encoding="utf-8"))
    assert report["task"] == "invoices"
    assert report["selected_run"]["run_id"]
    assert (reports / "report.md").read_text(encoding="utf-8").strip()
    assert json.loads((workdir / "request.json").read_text(encoding="utf-8"))["messages"]
