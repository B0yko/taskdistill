from __future__ import annotations

import errno
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

import taskdistill
from taskdistill import init as init_module
from taskdistill.config import ConfigError, TaskSpec, load_task
from taskdistill.init import PLACEHOLDER, TaskExistsError, init_task, template_files, templates_root
from taskdistill.tasks.classification import normalise_label
from taskdistill.tasks.extraction import canonical_output, validate

ENV_VARS = ("TASKDISTILL_TEACHER_BASE_URL", "TASKDISTILL_TEACHER_MODEL", "TASKDISTILL_TEACHER_API_KEY")
CLASSIFICATION_FILES = ["labels.txt", "task.yaml", "teacher_prompt.md"]
EXTRACTION_FILES = ["schema.json", "task.yaml", "teacher_prompt.md"]
TEMPLATE_LABELS = ["billing", "delivery", "returns", "other"]
TEMPLATE_FIELDS = ["order_id", "order_date", "total_amount"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _snapshot(directory: Path) -> dict[str, bytes | str]:
    """File name -> bytes, with ``"<dir>"`` / ``"-> target"`` for directories and symlinks."""
    out: dict[str, bytes | str] = {}
    for p in sorted(directory.iterdir()):
        if p.is_symlink():
            out[p.name] = f"-> {p.readlink()}"
        elif p.is_dir():
            out[p.name] = "<dir>"
        else:
            out[p.name] = p.read_bytes()
    return out


needs_permissions = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs POSIX permissions that apply to the user"
)


@pytest.fixture
def locked(tmp_path: Path) -> Iterator[Path]:
    """An existing directory that is restored to mode 0o755 after the test."""
    directory = tmp_path / "locked"
    directory.mkdir()
    try:
        yield directory
    finally:
        directory.chmod(0o755)


# --- scaffolding ----------------------------------------------------------------------------------------------------


def test_init_classification(tmp_path: Path) -> None:
    directory = init_task("support", "classification", dest_root=tmp_path)
    assert directory == tmp_path / "tasks" / "support"
    assert _names(directory) == CLASSIFICATION_FILES

    spec = load_task(str(directory))
    assert (spec.task, spec.type) == ("support", "classification")
    assert (spec.labels_file, spec.schema_file) == ("labels.txt", None)
    assert spec.labels == TEMPLATE_LABELS
    assert spec.teacher.base_url == "https://openrouter.ai/api/v1"
    assert spec.teacher.api_key_env == "TASKDISTILL_TEACHER_API_KEY"
    assert spec.teacher.model == "openai/gpt-4o-mini"
    assert (spec.teacher.temperature, spec.teacher.max_tokens, spec.teacher.response_format) == (0.0, 24, None)
    assert spec.teacher.extra_body == {}
    assert spec.teacher.provider is None
    assert spec.student.base_model == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    assert spec.student.system_prompt == "Classify the message into exactly one label."
    assert (spec.student.max_tokens, spec.student.response_template) == (16, None)
    assert (spec.train.profile, spec.train.lora_rank, spec.train.lora_layers) == ("full", 16, "all")
    assert (spec.train.learning_rate, spec.train.batch_size, spec.train.epochs) == (1.0e-4, 8, 2)
    assert (spec.train.max_seq_len, spec.train.seed) == (512, 13)
    assert spec.curate.split.stratify is True
    assert spec.curate.split.predefined == "meta.split"
    assert spec.curate.pii.kinds == ["email", "phone", "iban", "card", "ipv4", "ssn"]
    cascade = spec.cascade
    assert (cascade.reference, cascade.metric, cascade.target, cascade.max_drop) == ("teacher", "agreement", 0.97, None)
    assert (cascade.on_teacher_error, cascade.escalation_response) == ("student", "canonical")
    assert spec.cost.amortisation_enabled is False
    assert spec.budget.usd_cap is None


def test_init_extraction(tmp_path: Path) -> None:
    directory = init_task("orders", "extraction", dest_root=tmp_path)
    assert directory == tmp_path / "tasks" / "orders"
    assert _names(directory) == EXTRACTION_FILES

    spec = load_task(str(directory))
    assert (spec.task, spec.type, spec.labels_file, spec.schema_file) == ("orders", "extraction", None, "schema.json")
    assert spec.labels == []
    assert spec.schema_fields == TEMPLATE_FIELDS
    assert spec.teacher.max_tokens == 256
    assert spec.student.max_tokens == 256
    assert spec.student.system_prompt == "Extract the fields as JSON matching the schema."
    assert spec.train.max_seq_len == 1024
    assert spec.curate.split.stratify is False
    assert (spec.cascade.reference, spec.cascade.metric, spec.cascade.target) == ("teacher", "agreement", 0.97)


def test_extraction_template_schema_works_with_the_extraction_helpers(tmp_path: Path) -> None:
    spec = load_task(str(init_task("orders", "extraction", dest_root=tmp_path)))
    assert spec.json_schema is not None
    good: dict[str, Any] = {"total_amount": 42.5, "order_date": "2026-09-01", "order_id": "A-10442"}
    assert validate(good, spec.json_schema) == []
    expected = '{"order_id":"A-10442","order_date":"2026-09-01","total_amount":42.5}'
    assert canonical_output(good, spec.json_schema) == expected
    assert validate({**good, "total_amount": None}, spec.json_schema) == []
    assert validate({**good, "order_date": "01/09/2026"}, spec.json_schema) != []
    assert validate({"order_id": "A-10442", "order_date": "2026-09-01"}, spec.json_schema) != []
    assert validate({**good, "currency": "EUR"}, spec.json_schema) != []


def test_default_destination_is_the_current_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    directory = init_task("support", "classification")
    assert directory.resolve() == (tmp_path / "tasks" / "support").resolve()
    spec = load_task("support")
    assert spec.task == "support"
    assert Path(spec.source).resolve() == (tmp_path / "tasks" / "support" / "task.yaml").resolve()


def test_tasks_directory_may_already_hold_other_tasks(tmp_path: Path) -> None:
    init_task("first", "classification", dest_root=tmp_path)
    init_task("second", "extraction", dest_root=tmp_path)
    assert _names(tmp_path / "tasks") == ["first", "second"]


# --- name substitution ----------------------------------------------------------------------------------------------


def test_task_name_is_substituted_everywhere(tmp_path: Path) -> None:
    for task_type, name in (("classification", "support-routing"), ("extraction", "order.fields_v2")):
        directory = init_task(name, task_type, dest_root=tmp_path)
        for path in directory.iterdir():
            assert PLACEHOLDER not in path.read_text(encoding="utf-8"), path.name
        assert (directory / "task.yaml").read_text(encoding="utf-8").splitlines()[0].startswith(f'task: "{name}"')
        assert load_task(str(directory)).task == name


@pytest.mark.parametrize("name", ["123", "true", "null", "yes", "Off", "2026-09-26", "0x1F", "1_000", "017", "1.5"])
def test_names_yaml_would_read_as_other_types_stay_strings(tmp_path: Path, name: str) -> None:
    directory = init_task(name, "classification", dest_root=tmp_path)
    assert load_task(str(directory)).task == name


def test_name_of_64_characters_is_accepted(tmp_path: Path) -> None:
    name = "a" * 64
    assert init_task(name, "classification", dest_root=tmp_path).name == name


def test_other_template_text_is_copied_verbatim(tmp_path: Path) -> None:
    directory = init_task("support", "classification", dest_root=tmp_path)
    for file, text in template_files("classification").items():
        assert (directory / file).read_text(encoding="utf-8") == text.replace(PLACEHOLDER, "support")
    assert (directory / "labels.txt").read_text(encoding="utf-8") == "billing\ndelivery\nreturns\nother\n"


# --- environment ----------------------------------------------------------------------------------------------------


def test_env_references_are_kept_in_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = init_task("support", "classification", dest_root=tmp_path)
    text = (directory / "task.yaml").read_text(encoding="utf-8")
    assert "model: ${TASKDISTILL_TEACHER_MODEL:-openai/gpt-4o-mini}" in text
    assert "base_url: ${TASKDISTILL_TEACHER_BASE_URL:-https://openrouter.ai/api/v1}" in text

    monkeypatch.setenv("TASKDISTILL_TEACHER_MODEL", "vendor/other-model")
    monkeypatch.setenv("TASKDISTILL_TEACHER_BASE_URL", "http://127.0.0.1:9000/v1")
    spec = load_task(str(directory))
    assert spec.teacher.model == "vendor/other-model"
    assert spec.teacher.base_url == "http://127.0.0.1:9000/v1"


def test_validation_failure_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKDISTILL_TEACHER_MODEL", "<pick a model>")
    with pytest.raises(ConfigError) as info:
        init_task("support", "classification", dest_root=tmp_path)
    message = str(info.value)
    assert "teacher.model: teacher.model must be a model slug" in message
    assert str(tmp_path / "tasks" / "support" / "task.yaml") in message
    assert "taskdistill-init-" not in message
    assert not (tmp_path / "tasks").exists()


def test_validation_failure_with_force_keeps_existing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = init_task("support", "classification", dest_root=tmp_path)
    (directory / "labels.txt").write_text("urgent\nroutine\n", encoding="utf-8")
    before = _snapshot(directory)
    monkeypatch.setenv("TASKDISTILL_TEACHER_MODEL", "<pick a model>")
    with pytest.raises(ConfigError, match=r"teacher\.model"):
        init_task("support", "classification", dest_root=tmp_path, force=True)
    assert _snapshot(directory) == before


# --- overwriting ----------------------------------------------------------------------------------------------------


def test_existing_task_is_not_overwritten(tmp_path: Path) -> None:
    directory = init_task("support", "classification", dest_root=tmp_path)
    (directory / "labels.txt").write_text("urgent\nroutine\n", encoding="utf-8")
    before = _snapshot(directory)
    with pytest.raises(TaskExistsError) as info:
        init_task("support", "classification", dest_root=tmp_path)
    assert "already exists" in str(info.value)
    assert "--force" in str(info.value)
    assert isinstance(info.value, FileExistsError)
    assert isinstance(info.value, ConfigError)
    assert _snapshot(directory) == before


def test_existing_empty_directory_is_also_refused(tmp_path: Path) -> None:
    (tmp_path / "tasks" / "support").mkdir(parents=True)
    with pytest.raises(TaskExistsError):
        init_task("support", "extraction", dest_root=tmp_path)
    assert _names(tmp_path / "tasks" / "support") == []


def test_force_overwrites_template_files_and_keeps_others(tmp_path: Path) -> None:
    directory = init_task("support", "classification", dest_root=tmp_path)
    (directory / "labels.txt").write_text("urgent\nroutine\n", encoding="utf-8")
    (directory / "task.yaml").write_text("task: broken\n", encoding="utf-8")
    (directory / "notes.md").write_text("keep me\n", encoding="utf-8")

    assert init_task("support", "classification", dest_root=tmp_path, force=True) == directory
    assert (directory / "labels.txt").read_text(encoding="utf-8") == "billing\ndelivery\nreturns\nother\n"
    assert (directory / "notes.md").read_text(encoding="utf-8") == "keep me\n"
    assert load_task(str(directory)).labels == TEMPLATE_LABELS


def test_force_can_switch_the_task_type(tmp_path: Path) -> None:
    directory = init_task("support", "classification", dest_root=tmp_path)
    init_task("support", "extraction", dest_root=tmp_path, force=True)
    assert _names(directory) == sorted({*CLASSIFICATION_FILES, *EXTRACTION_FILES})
    spec = load_task(str(directory))
    assert spec.type == "extraction"
    assert spec.schema_fields == TEMPLATE_FIELDS


def test_force_on_a_missing_directory_just_creates_it(tmp_path: Path) -> None:
    directory = init_task("support", "extraction", dest_root=tmp_path, force=True)
    assert _names(directory) == EXTRACTION_FILES


@pytest.mark.parametrize("force", [False, True])
def test_a_file_in_the_way_is_refused(tmp_path: Path, force: bool) -> None:
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "support").write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="exists and is not a directory"):
        init_task("support", "classification", dest_root=tmp_path, force=force)
    assert (tmp_path / "tasks" / "support").read_text(encoding="utf-8") == "not a directory\n"


@pytest.mark.parametrize("force", [False, True])
def test_a_file_named_tasks_is_refused(tmp_path: Path, force: bool) -> None:
    (tmp_path / "tasks").write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        init_task("support", "classification", dest_root=tmp_path, force=force)
    assert str(info.value) == f"{tmp_path / 'tasks'} exists and is not a directory"
    assert (tmp_path / "tasks").read_text(encoding="utf-8") == "not a directory\n"


@pytest.mark.parametrize("force", [False, True])
def test_a_destination_root_that_is_a_file_is_refused(tmp_path: Path, force: bool) -> None:
    root = tmp_path / "root"
    root.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        init_task("support", "extraction", dest_root=root, force=force)
    assert str(info.value) == f"{root} exists and is not a directory"
    assert _names(tmp_path) == ["root"]


def test_a_dangling_tasks_symlink_is_refused(tmp_path: Path) -> None:
    (tmp_path / "tasks").symlink_to(tmp_path / "missing")
    with pytest.raises(ConfigError, match="exists and is not a directory"):
        init_task("support", "classification", dest_root=tmp_path)
    assert not (tmp_path / "missing").exists()


def test_a_symlinked_tasks_directory_is_followed(tmp_path: Path) -> None:
    (tmp_path / "shared").mkdir()
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "tasks").symlink_to(tmp_path / "shared")
    init_task("support", "classification", dest_root=tmp_path / "project")
    assert _names(tmp_path / "shared" / "support") == CLASSIFICATION_FILES


def test_a_directory_in_place_of_a_template_file_is_refused_before_writing(tmp_path: Path) -> None:
    directory = init_task("support", "classification", dest_root=tmp_path)
    (directory / "labels.txt").write_text("urgent\nroutine\n", encoding="utf-8")
    (directory / "task.yaml").write_text("task: custom\n", encoding="utf-8")
    (directory / "teacher_prompt.md").unlink()
    (directory / "teacher_prompt.md").mkdir()
    before = _snapshot(directory)
    with pytest.raises(ConfigError) as info:
        init_task("support", "classification", dest_root=tmp_path, force=True)
    assert str(info.value) == f"{directory / 'teacher_prompt.md'} exists and is not a file"
    assert _snapshot(directory) == before
    assert (directory / "labels.txt").read_text(encoding="utf-8") == "urgent\nroutine\n"


def test_a_dangling_symlink_in_place_of_a_template_file_is_refused(tmp_path: Path) -> None:
    directory = init_task("support", "extraction", dest_root=tmp_path)
    (directory / "schema.json").unlink()
    (directory / "schema.json").symlink_to(tmp_path / "missing.json")
    before = _snapshot(directory)
    with pytest.raises(ConfigError, match=r"schema\.json exists and is not a file$"):
        init_task("support", "extraction", dest_root=tmp_path, force=True)
    assert _snapshot(directory) == before
    assert not (tmp_path / "missing.json").exists()


def test_force_replaces_a_symlinked_template_file_without_writing_through_it(tmp_path: Path) -> None:
    shared = tmp_path / "shared-labels.txt"
    shared.write_text("urgent\nroutine\n", encoding="utf-8")
    directory = init_task("support", "classification", dest_root=tmp_path)
    (directory / "labels.txt").unlink()
    (directory / "labels.txt").symlink_to(shared)
    init_task("support", "classification", dest_root=tmp_path, force=True)
    assert shared.read_text(encoding="utf-8") == "urgent\nroutine\n"
    assert not (directory / "labels.txt").is_symlink()
    assert (directory / "labels.txt").read_text(encoding="utf-8") == "billing\ndelivery\nreturns\nother\n"


@needs_permissions
def test_a_read_only_destination_is_a_config_error(locked: Path) -> None:
    locked.chmod(0o555)
    with pytest.raises(ConfigError) as info:
        init_task("support", "classification", dest_root=locked)
    target = locked / "tasks" / "support"
    assert str(info.value) == f"{target}: cannot write task files: Permission denied"
    assert _names(locked) == []


@needs_permissions
def test_an_unsearchable_destination_is_a_config_error(locked: Path) -> None:
    locked.chmod(0o000)
    with pytest.raises(ConfigError) as info:
        init_task("support", "classification", dest_root=locked)
    target = locked / "tasks" / "support"
    assert str(info.value) == f"{target}: cannot inspect the destination: Permission denied"


@needs_permissions
def test_a_read_only_task_directory_with_force_is_a_config_error(tmp_path: Path) -> None:
    directory = init_task("support", "classification", dest_root=tmp_path)
    (directory / "labels.txt").write_text("urgent\nroutine\n", encoding="utf-8")
    before = _snapshot(directory)
    directory.chmod(0o555)
    try:
        with pytest.raises(ConfigError, match=r": cannot write task files: Permission denied$"):
            init_task("support", "classification", dest_root=tmp_path, force=True)
    finally:
        directory.chmod(0o755)
    assert _snapshot(directory) == before


def _fail_on_call(monkeypatch: pytest.MonkeyPatch, fail_at: int) -> list[Path]:
    """Make the ``fail_at``-th file write raise ENOSPC after writing part of the file."""
    calls: list[Path] = []
    real = init_module._write_new

    def flaky(path: Path, text: str) -> None:
        calls.append(path)
        if len(calls) == fail_at:
            real(path, text[:5])
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        real(path, text)

    monkeypatch.setattr(init_module, "_write_new", flaky)
    return calls


def test_a_failed_write_with_force_leaves_the_old_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = init_task("support", "classification", dest_root=tmp_path)
    (directory / "labels.txt").write_text("urgent\nroutine\n", encoding="utf-8")
    (directory / "notes.md").write_text("keep me\n", encoding="utf-8")
    before = _snapshot(directory)
    calls = _fail_on_call(monkeypatch, fail_at=3)
    with pytest.raises(ConfigError) as info:
        init_task("support", "classification", dest_root=tmp_path, force=True)
    assert str(info.value) == f"{directory}: cannot write task files: {os.strerror(errno.ENOSPC)}"
    assert len(calls) == 3
    assert all(path.parent == directory and path.name.startswith(".") for path in calls)
    assert _snapshot(directory) == before


def test_a_failed_write_removes_the_new_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_call(monkeypatch, fail_at=2)
    with pytest.raises(ConfigError, match="cannot write task files"):
        init_task("support", "extraction", dest_root=tmp_path)
    assert _names(tmp_path / "tasks") == []

    monkeypatch.undo()
    directory = init_task("support", "extraction", dest_root=tmp_path)
    assert _names(directory) == EXTRACTION_FILES


def test_force_leaves_no_temporary_files(tmp_path: Path) -> None:
    directory = init_task("support", "classification", dest_root=tmp_path)
    (directory / "notes.md").write_text("keep me\n", encoding="utf-8")
    init_task("support", "classification", dest_root=tmp_path, force=True)
    assert _names(directory) == sorted([*CLASSIFICATION_FILES, "notes.md"])


def test_an_unusable_temporary_directory_is_a_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "missing"))
    with pytest.raises(ConfigError, match=r"^cannot stage the support template for validation: "):
        init_task("support", "classification", dest_root=tmp_path / "work")
    assert not (tmp_path / "work").exists()


# --- bad arguments --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["", "../escape", "a/b", "a\\b", "-dash", ".hidden", "..", "has space", "tab\tname", "x" * 65, "naïve", "a:b"],
)
def test_invalid_names_are_refused_before_writing(tmp_path: Path, name: str) -> None:
    with pytest.raises(ConfigError, match=r"^task: "):
        init_task(name, "classification", dest_root=tmp_path)
    assert _names(tmp_path) == []


@pytest.mark.parametrize("task_type", ["regression", "Classification", "", "templates"])
def test_unknown_type_is_refused_before_writing(tmp_path: Path, task_type: str) -> None:
    with pytest.raises(ConfigError, match=r"^type: unknown task type") as info:
        init_task("support", task_type, dest_root=tmp_path)
    assert "classification, extraction" in str(info.value)
    assert _names(tmp_path) == []


@pytest.mark.parametrize("name", ["a", "A1", "a.b_c-d", "9" * 64, "", "-a", ".a", "a b", "a/b", "x" * 65, "é"])
def test_init_names_agree_with_the_spec_validator(name: str) -> None:
    raw = {
        "task": name,
        "type": "classification",
        "labels_file": "labels.txt",
        "teacher": {"model": "vendor/model-a"},
        "student": {"system_prompt": "p"},
        "cascade": {"target": 0.9},
    }
    try:
        TaskSpec.model_validate(raw)
        spec_accepts = True
    except ValidationError:
        spec_accepts = False
    assert bool(init_module.TASK_NAME.fullmatch(name)) is spec_accepts


# --- packaged templates ---------------------------------------------------------------------------------------------


def test_templates_are_package_resources() -> None:
    root = templates_root()
    assert str(root) == str(resources.files("taskdistill") / "_data" / "templates")
    assert sorted(entry.name for entry in root.iterdir() if entry.is_dir()) == ["classification", "extraction"]
    assert list(template_files("classification")) == CLASSIFICATION_FILES
    assert list(template_files("extraction")) == EXTRACTION_FILES


def test_placeholder_appears_once_in_each_task_yaml() -> None:
    for task_type in ("classification", "extraction"):
        text = template_files(task_type)["task.yaml"]
        assert text.count(PLACEHOLDER) == 1
        assert text.startswith(f'task: "{PLACEHOLDER}"')


def _model_keys(model: type[BaseModel], prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for name, field in model.model_fields.items():
        if field.exclude:
            continue
        key = prefix + (field.alias or name)
        keys.add(key)
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            keys |= _model_keys(annotation, key + ".")
    return keys


def _yaml_keys(tree: dict[str, Any], prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for name, value in tree.items():
        key = prefix + name
        keys.add(key)
        if isinstance(value, dict) and key != "teacher.extra_body":
            keys |= _yaml_keys(value, key + ".")
    return keys


@pytest.mark.parametrize("task_type", ["classification", "extraction"])
def test_templates_spell_out_every_spec_key(task_type: str) -> None:
    tree = yaml.safe_load(template_files(task_type)["task.yaml"])
    assert _yaml_keys(tree) == _model_keys(TaskSpec)


@pytest.mark.parametrize("task_type", ["classification", "extraction"])
def test_every_template_key_has_a_trailing_comment(task_type: str) -> None:
    key_line = re.compile(r"^\s*[A-Za-z_]+:")
    lines = [line for line in template_files(task_type)["task.yaml"].splitlines() if key_line.match(line)]
    assert len(lines) == len(_model_keys(TaskSpec))
    for line in lines:
        assert re.search(r"\s#\s\S", line), line


def _comment(text: str, key: str) -> str:
    match = re.search(rf"^\s*{key}: \S+\s+# (.+)$", text, re.MULTILINE)
    assert match is not None, key
    return match.group(1)


def test_only_the_extraction_template_suggests_json_mode() -> None:
    classification = template_files("classification")["task.yaml"]
    extraction = template_files("extraction")["task.yaml"]
    assert "json_object" not in classification
    assert _comment(classification, "response_format") == "keep null: the teacher must answer with a bare label"
    assert "{type: json_object}" in _comment(extraction, "response_format")
    # A JSON-mode answer is not a label: every teacher output would be dropped.
    assert normalise_label('{"label": "billing"}', TEMPLATE_LABELS) is None
    assert normalise_label("billing", TEMPLATE_LABELS) == "billing"


@pytest.mark.parametrize(("task_type", "metric"), [("classification", "accuracy"), ("extraction", "field_f1")])
def test_the_documented_max_drop_alternative_loads(tmp_path: Path, task_type: str, metric: str) -> None:
    directory = init_task("support", task_type, dest_root=tmp_path)
    text = (directory / "task.yaml").read_text(encoding="utf-8")
    comment = _comment(text, "target")
    assert comment.startswith("or ")
    alternative = yaml.safe_load("{" + comment.removeprefix("or ") + "}")
    assert alternative == {"target": None, "max_drop": 0.01, "reference": "gold", "metric": metric}

    tree = yaml.safe_load(text)
    tree["cascade"].update(alternative)
    (directory / "task.yaml").write_text(yaml.safe_dump(tree, sort_keys=False), encoding="utf-8")
    cascade = load_task(str(directory)).cascade
    assert (cascade.target, cascade.max_drop, cascade.reference, cascade.metric) == (None, 0.01, "gold", metric)

    tree["cascade"]["metric"] = "agreement"
    (directory / "task.yaml").write_text(yaml.safe_dump(tree, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"cascade\.metric 'agreement' is measured against the teacher"):
        load_task(str(directory))


def test_classification_prompt_lists_every_label() -> None:
    files = template_files("classification")
    labels = [line.strip() for line in files["labels.txt"].splitlines() if line.strip()]
    assert labels == TEMPLATE_LABELS
    prompt_lines = files["teacher_prompt.md"].splitlines()
    for label in labels:
        assert label in prompt_lines


def test_extraction_prompt_names_every_schema_field() -> None:
    files = template_files("extraction")
    schema = json.loads(files["schema.json"])
    assert list(schema["properties"]) == TEMPLATE_FIELDS
    assert schema["required"] == TEMPLATE_FIELDS
    for field in TEMPLATE_FIELDS:
        assert f"- {field} (" in files["teacher_prompt.md"]


def test_init_reads_templates_through_a_traversable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "templates.zip"
    source = Path(str(templates_root()))
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                zf.write(path, f"templates/{path.relative_to(source).as_posix()}")
    expected = init_task("support", "extraction", dest_root=tmp_path / "from-files")

    with zipfile.ZipFile(archive) as zf:
        monkeypatch.setattr(init_module, "templates_root", lambda: zipfile.Path(zf, "templates/"))
        directory = init_task("support", "extraction", dest_root=tmp_path / "from-zip")
    assert _snapshot(directory) == _snapshot(expected)


def test_init_works_when_the_package_is_imported_from_a_zip(tmp_path: Path) -> None:
    package = Path(taskdistill.__file__).parent
    archive = tmp_path / "taskdistill.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(package.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, f"taskdistill/{path.relative_to(package).as_posix()}")
    work = tmp_path / "work"
    work.mkdir()
    script = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(archive)!r})
        from importlib import resources
        import taskdistill
        from taskdistill.config import load_task
        from taskdistill.init import init_task
        directory = init_task("support", "classification")
        spec = load_task("support")
        print(json.dumps({{
            "module": taskdistill.__file__,
            "files_type": type(resources.files("taskdistill")).__name__,
            "directory": str(directory),
            "labels": spec.labels,
        }}))
        """
    )
    env = {k: v for k, v in os.environ.items() if k not in ENV_VARS and k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=work, env=env, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["module"].startswith(str(archive))
    assert out["files_type"] != "PosixPath"
    assert Path(out["directory"]).resolve() == (work / "tasks" / "support").resolve()
    assert out["labels"] == TEMPLATE_LABELS
