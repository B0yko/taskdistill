from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from taskdistill import config
from taskdistill.config import ConfigError, TaskSpec, TeacherSpec, expand_env, load_task

ENV_VARS = (
    "TASKDISTILL_TEACHER_BASE_URL",
    "TASKDISTILL_TEACHER_API_KEY",
    "TASKDISTILL_TEACHER_MODEL",
    "OPENROUTER_API_KEY",
    "TD_TEST_A",
    "TD_TEST_B",
    "TD_TEST_UNSET",
    "TD_TEST_TEAM_KEY",
)

PROMPT = "Pick one label.\n"
LABELS = "alpha\n\n  beta  \ngamma\n"
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "zeta_id": {"type": "string"},
        "amount": {"type": ["number", "null"]},
        "issued_on": {"type": "string", "format": "date"},
    },
    "required": ["zeta_id", "amount", "issued_on"],
}

CLASSIFICATION: dict[str, Any] = {
    "task": "tickets",
    "type": "classification",
    "labels_file": "labels.txt",
    "teacher": {"model": "vendor/model-a"},
    "student": {"system_prompt": "Classify the message."},
    "cascade": {"target": 0.97},
}
EXTRACTION: dict[str, Any] = {
    "task": "orders",
    "type": "extraction",
    "schema_file": "schema.json",
    "teacher": {"model": "vendor/model-a", "max_tokens": 256},
    "student": {"system_prompt": "Extract the fields as JSON.", "max_tokens": 256},
    "cascade": {"target": 0.97},
}
CLASSIFICATION_FILES = {"labels.txt": LABELS, "teacher_prompt.md": PROMPT}
EXTRACTION_FILES = {"schema.json": json.dumps(SCHEMA), "teacher_prompt.md": PROMPT}

DELETE = object()


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clean current directory with no bundled specs and none of the teacher variables set."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    bundled = tmp_path / "_bundled"
    bundled.mkdir()
    monkeypatch.setattr(config, "packaged_tasks", lambda: bundled)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def changed(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Deep copy of ``base`` with dotted keys set (or removed with ``DELETE``)."""
    out = copy.deepcopy(base)
    for dotted, value in changes.items():
        *parents, leaf = dotted.split(".")
        node = out
        for part in parents:
            node = node.setdefault(part, {})
        if value is DELETE:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return out


def write_spec_dir(directory: Path, spec: dict[str, Any] | str, files: dict[str, str] | None = None) -> Path:
    """Write ``task.yaml`` plus its files (the default ones for the spec's type) into ``directory``."""
    if files is None:
        is_extraction = isinstance(spec, dict) and spec.get("type") == "extraction"
        files = EXTRACTION_FILES if is_extraction else CLASSIFICATION_FILES
    directory.mkdir(parents=True, exist_ok=True)
    text = spec if isinstance(spec, str) else yaml.safe_dump(spec, sort_keys=False)
    (directory / "task.yaml").write_text(text, encoding="utf-8")
    for file, content in files.items():
        (directory / file).write_text(content, encoding="utf-8")
    return directory


def write_task(root: Path, spec: dict[str, Any] | str, files: dict[str, str] | None = None) -> Path:
    """Write ``<root>/tasks/<task>/``."""
    name = spec["task"] if isinstance(spec, dict) else "tickets"
    return write_spec_dir(root / "tasks" / name, spec, files)


def load_error(root: Path, spec: dict[str, Any] | str, files: dict[str, str] | None = None) -> str:
    write_task(root, spec, files)
    name = spec["task"] if isinstance(spec, dict) else "tickets"
    with pytest.raises(ConfigError) as info:
        load_task(name)
    return str(info.value)


# --- ${ENV:-default} expansion ------------------------------------------------------------------------------------


def test_default_is_used_when_the_variable_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TD_TEST_A", raising=False)
    assert expand_env("${TD_TEST_A:-fallback}") == "fallback"


def test_default_is_used_when_the_variable_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TD_TEST_A", "")
    assert expand_env("${TD_TEST_A:-fallback}") == "fallback"


def test_set_variable_wins_over_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TD_TEST_A", "vendor/model-b")
    assert expand_env("${TD_TEST_A:-fallback}") == "vendor/model-b"
    assert expand_env("${TD_TEST_A}") == "vendor/model-b"


def test_empty_default_expands_to_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TD_TEST_A", raising=False)
    assert expand_env("x${TD_TEST_A:-}y") == "xy"


def test_default_may_contain_colons_and_slashes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TD_TEST_A", raising=False)
    assert expand_env("${TD_TEST_A:-https://openrouter.ai/api/v1}") == "https://openrouter.ai/api/v1"


def test_several_references_in_one_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TD_TEST_A", raising=False)
    monkeypatch.delenv("TD_TEST_B", raising=False)
    assert expand_env("https://${TD_TEST_A:-example.com}/v${TD_TEST_B:-1}") == "https://example.com/v1"
    monkeypatch.setenv("TD_TEST_A", "api.example.org")
    assert expand_env("https://${TD_TEST_A:-example.com}/v${TD_TEST_B:-1}") == "https://api.example.org/v1"


def test_expanded_values_are_not_expanded_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TD_TEST_A", "${TD_TEST_B:-nested}")
    assert expand_env("${TD_TEST_A}") == "${TD_TEST_B:-nested}"


def test_text_that_is_not_a_reference_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TD_TEST_A", "set")
    assert expand_env("$TD_TEST_A") == "$TD_TEST_A"
    assert expand_env("${1TD}") == "${1TD}"
    assert expand_env("${TD TEST}") == "${TD TEST}"
    assert expand_env("${TD_TEST_A:default}") == "${TD_TEST_A:default}"


def test_non_strings_and_keys_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TD_TEST_A", "set")
    tree = {"n": 3, "f": 0.5, "b": True, "none": None, "${TD_TEST_A}": "${TD_TEST_A}", "list": [1, "${TD_TEST_A}"]}
    assert expand_env(tree) == {"n": 3, "f": 0.5, "b": True, "none": None, "${TD_TEST_A}": "set", "list": [1, "set"]}


def test_unset_variable_without_default_names_the_key_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TD_TEST_UNSET", raising=False)
    with pytest.raises(ConfigError) as info:
        expand_env({"teacher": {"model": "${TD_TEST_UNSET}"}})
    message = str(info.value)
    assert message.startswith("teacher.model: ")
    assert "TD_TEST_UNSET is not set and has no default" in message


def test_unset_variable_inside_a_list_names_the_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TD_TEST_UNSET", raising=False)
    with pytest.raises(ConfigError, match=r"^curate\.pii\.kinds\[1\]: "):
        expand_env({"curate": {"pii": {"kinds": ["email", "${TD_TEST_UNSET}"]}}})


def test_empty_variable_without_default_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TD_TEST_A", "")
    with pytest.raises(ConfigError, match=r"^model: environment variable TD_TEST_A"):
        expand_env({"model": "${TD_TEST_A}"})


def test_load_task_reports_the_key_of_an_unset_variable(workdir: Path) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, {"teacher.model": "${TD_TEST_UNSET}"}))
    assert "teacher.model" in message
    assert "TD_TEST_UNSET" in message


def test_load_task_expands_env_before_validation(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = changed(
        CLASSIFICATION,
        {
            "teacher.base_url": "${TASKDISTILL_TEACHER_BASE_URL:-https://openrouter.ai/api/v1}",
            "teacher.model": "${TASKDISTILL_TEACHER_MODEL:-vendor/default-model}",
            "teacher.max_tokens": "${TD_TEST_A:-32}",
        },
    )
    write_task(workdir, spec)
    loaded = load_task("tickets")
    assert loaded.teacher.base_url == "https://openrouter.ai/api/v1"
    assert loaded.teacher.model == "vendor/default-model"
    assert loaded.teacher.max_tokens == 32

    monkeypatch.setenv("TASKDISTILL_TEACHER_BASE_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.setenv("TASKDISTILL_TEACHER_MODEL", "vendor/override")
    monkeypatch.setenv("TD_TEST_A", "48")
    loaded = load_task("tickets")
    assert loaded.teacher.base_url == "http://127.0.0.1:9000/v1"
    assert loaded.teacher.model == "vendor/override"
    assert loaded.teacher.max_tokens == 48


# --- loading valid specs -------------------------------------------------------------------------------------------


def test_classification_spec_loads_with_its_files(workdir: Path) -> None:
    directory = write_task(workdir, CLASSIFICATION)
    spec = load_task("tickets")
    assert spec.task == "tickets"
    assert spec.type == "classification"
    assert spec.labels == ["alpha", "beta", "gamma"]
    assert spec.teacher_prompt == "Pick one label.\n"
    assert spec.teacher_prompt_sha256 == "0139451d3353661dd78c093eb2063509523ff24a6223bea3ec8afbc50f46b6e1"
    assert spec.json_schema is None
    assert spec.schema_fields == []
    assert Path(spec.source).resolve() == (directory / "task.yaml").resolve()


def test_teacher_prompt_hash_is_sha256_of_the_utf8_text(workdir: Path) -> None:
    write_task(workdir, CLASSIFICATION, {"labels.txt": LABELS, "teacher_prompt.md": "abc"})
    spec = load_task("tickets")
    # SHA-256("abc"), the FIPS 180-2 test vector.
    assert spec.teacher_prompt_sha256 == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_defaults_fill_every_optional_key(workdir: Path) -> None:
    write_task(workdir, CLASSIFICATION)
    spec = load_task("tickets")
    assert spec.labels_file == "labels.txt"
    assert spec.schema_file is None
    assert (spec.input.from_, spec.input.regex) == ("last_user_message", None)
    teacher = spec.teacher
    assert teacher.base_url == "https://openrouter.ai/api/v1"
    assert teacher.api_key_env == "TASKDISTILL_TEACHER_API_KEY"
    assert teacher.prompt_file == "teacher_prompt.md"
    assert (teacher.temperature, teacher.max_tokens) == (0.0, 256)
    assert (teacher.response_format, teacher.extra_body) == (None, {})
    assert spec.student.base_model == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    assert (spec.student.max_tokens, spec.student.response_template) == (16, None)
    train = spec.train
    assert (train.profile, train.lora_rank, train.lora_layers, train.learning_rate) == ("full", 16, "all", 1.0e-4)
    assert (train.batch_size, train.epochs, train.max_seq_len, train.seed) == (8, 2, 512, 13)
    assert (spec.curate.dedupe.exact, spec.curate.dedupe.near_dup_jaccard) == (True, 0.9)
    assert spec.curate.pii.enabled is True
    assert spec.curate.pii.kinds == ["email", "phone", "iban", "card", "ipv4", "ssn"]
    split = spec.curate.split
    assert (split.predefined, split.val, split.test, split.stratify, split.group_by, split.seed) == (
        "meta.split",
        0.1,
        0.1,
        True,
        None,
        13,
    )
    cascade = spec.cascade
    assert (cascade.reference, cascade.metric, cascade.target, cascade.max_drop) == ("teacher", "agreement", 0.97, None)
    assert (cascade.on_teacher_error, cascade.escalation_response) == ("student", "canonical")
    cost = spec.cost
    assert (cost.local_watts, cost.usd_per_kwh, cost.hardware_usd, cost.amortisation_hours) == (20.0, 0.30, 0.0, 0.0)
    assert spec.budget.usd_cap is None


def test_extraction_spec_loads_schema_in_property_order(workdir: Path) -> None:
    write_task(workdir, EXTRACTION)
    spec = load_task("orders")
    assert spec.type == "extraction"
    assert spec.json_schema == SCHEMA
    assert spec.schema_fields == ["zeta_id", "amount", "issued_on"]
    assert spec.labels == []


def test_loaded_fields_are_not_part_of_the_dump(workdir: Path) -> None:
    write_task(workdir, CLASSIFICATION)
    dumped = load_task("tickets").model_dump(by_alias=True)
    assert not {"labels", "json_schema", "teacher_prompt", "source"} & set(dumped)
    assert dumped["input"]["from"] == "last_user_message"


def test_custom_file_names_are_honoured(workdir: Path) -> None:
    spec = changed(CLASSIFICATION, {"labels_file": "intents.txt", "teacher.prompt_file": "prompt.txt"})
    write_task(workdir, spec, {"intents.txt": "yes\nno\n", "prompt.txt": "Answer yes or no."})
    loaded = load_task("tickets")
    assert loaded.labels == ["yes", "no"]
    assert loaded.teacher_prompt == "Answer yes or no."


# --- errors name the offending key ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"cascade.target": 1.5}, ": cascade.target: Input should be less than or equal to 1"),
        ({"cascade.target": -0.1}, ": cascade.target: Input should be greater than or equal to 0"),
        ({"cascade.target": DELETE, "cascade.max_drop": 2}, ": cascade.max_drop: Input should be less than or equal"),
        ({"cascade.on_teacher_error": "retry"}, ": cascade.on_teacher_error: Input should be 'student' or 'error'"),
        ({"cascade.escalation_response": "json"}, ": cascade.escalation_response: "),
        ({"train.lora_rank": 0}, ": train.lora_rank: Input should be greater than 0"),
        ({"train.lora_layers": "some"}, ": train.lora_layers"),
        ({"train.profile": "medium"}, ": train.profile: Input should be 'quick' or 'full'"),
        ({"train.learning_rate": 0}, ": train.learning_rate: Input should be greater than 0"),
        ({"train.epochs": -1}, ": train.epochs: "),
        ({"teacher.max_tokens": 0}, ": teacher.max_tokens: Input should be greater than 0"),
        ({"teacher.model": "<slug chosen in the bake-off>"}, ": teacher.model: teacher.model must be a model slug"),
        ({"teacher.model": ""}, ": teacher.model: teacher.model must be a model slug"),
        ({"teacher.extra_body": {"stream": True, "model": "x"}}, ": teacher.extra_body: teacher.extra_body must not"),
        ({"student.max_tokens": 0}, ": student.max_tokens: "),
        ({"student.response_template": '{"intent": "{name}"}'}, ": student.response_template: student.response_"),
        ({"type": "regression"}, ": type: Input should be 'classification' or 'extraction'"),
        ({"task": "bad name!"}, ": task: task must be 1-64 characters"),
        ({"task": "-leading"}, ": task: task must be 1-64 characters"),
        ({"input.from": "stdin"}, ": input.from: Input should be 'last_user_message' or 'regex'"),
        ({"curate.dedupe.near_dup_jaccard": 1.5}, ": curate.dedupe.near_dup_jaccard: "),
        (
            {"curate.dedupe.near_dup_jaccard": 0.3},
            ": curate.dedupe.near_dup_jaccard: Input should be greater than or equal to 0.5",
        ),
        ({"curate.pii.kinds": ["email", "passport"]}, ": curate.pii.kinds.1: Input should be 'email'"),
        ({"curate.split.val": 1.0}, ": curate.split.val: Input should be less than 1"),
        ({"curate.split.predefined": "split"}, "curate.split.predefined must name a meta field"),
        ({"curate.split.group_by": "customer"}, "curate.split.group_by must name a meta field"),
        ({"curate.split.val": 0.6, "curate.split.test": 0.4}, "curate.split.val + curate.split.test must be < 1"),
        ({"cost.local_watts": -1}, ": cost.local_watts: Input should be greater than or equal to 0"),
        ({"cost.usd_per_kwh": -0.1}, ": cost.usd_per_kwh: "),
        ({"budget.usd_cap": 0}, ": budget.usd_cap: Input should be greater than 0"),
        ({"student.system_prompt": DELETE}, ": student.system_prompt: Field required"),
        ({"teacher": DELETE}, ": teacher: Field required"),
        ({"cascade": DELETE}, ": cascade: Field required"),
    ],
)
def test_validation_errors_name_the_offending_key(workdir: Path, changes: dict[str, Any], expected: str) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, changes))
    assert expected in message


def test_error_lines_start_with_the_spec_file(workdir: Path) -> None:
    directory = write_task(workdir, changed(CLASSIFICATION, {"train.lora_rank": 0}))
    with pytest.raises(ConfigError) as info:
        load_task("tickets")
    assert str(info.value) == f"{directory / 'task.yaml'}: train.lora_rank: Input should be greater than 0"


def test_every_error_is_reported_on_its_own_line(workdir: Path) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, {"train.lora_rank": 0, "cascade.target": 2}))
    lines = message.splitlines()
    assert len(lines) == 2
    assert ": train.lora_rank: " in lines[0]
    assert ": cascade.target: " in lines[1]


@pytest.mark.parametrize(
    ("changes", "key"),
    [
        ({"temperature": 0}, "temperature"),
        ({"teacher.temprature": 0}, "teacher.temprature"),
        ({"curate.split.stratified": True}, "curate.split.stratified"),
        ({"cascade.threshold": 0.5}, "cascade.threshold"),
        ({"cost.gpu_usd": 1}, "cost.gpu_usd"),
    ],
)
def test_unknown_keys_are_forbidden(workdir: Path, changes: dict[str, Any], key: str) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, changes))
    assert f": {key}: Extra inputs are not permitted" in message


# --- input.from / input.regex ---------------------------------------------------------------------------------------


def test_regex_input_needs_a_regex(workdir: Path) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, {"input.from": "regex"}))
    assert ": input: input.regex is required when input.from is 'regex'" in message


@pytest.mark.parametrize("regex", [r"Message: (.*)", r"Message: (?P<text>.*)"])
def test_regex_without_the_named_group_input_is_rejected(workdir: Path, regex: str) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, {"input.from": "regex", "input.regex": regex}))
    assert "input.regex must contain a named group 'input'" in message


def test_regex_that_does_not_compile_is_rejected(workdir: Path) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, {"input.from": "regex", "input.regex": "(?P<input>"}))
    assert "input.regex does not compile" in message


def test_regex_with_the_named_group_loads(workdir: Path) -> None:
    regex = r"Message:\s*(?P<input>.*)"
    write_task(workdir, changed(CLASSIFICATION, {"input.from": "regex", "input.regex": regex}))
    spec = load_task("tickets")
    assert (spec.input.from_, spec.input.regex) == ("regex", regex)


# --- cascade --------------------------------------------------------------------------------------------------------


def test_cascade_needs_target_or_max_drop(workdir: Path) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, {"cascade.target": DELETE}))
    assert ": cascade: set exactly one of cascade.target and cascade.max_drop" in message


def test_cascade_rejects_both_target_and_max_drop(workdir: Path) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, {"cascade.max_drop": 0.01}))
    assert ": cascade: set exactly one of cascade.target and cascade.max_drop" in message


def test_cascade_accepts_max_drop_alone(workdir: Path) -> None:
    changes = {"cascade.target": DELETE, "cascade.max_drop": 0.01, "cascade.reference": "gold"}
    write_task(workdir, changed(CLASSIFICATION, {**changes, "cascade.metric": "accuracy"}))
    cascade = load_task("tickets").cascade
    assert (cascade.target, cascade.max_drop, cascade.reference, cascade.metric) == (None, 0.01, "gold", "accuracy")


def test_cascade_accepts_an_explicit_null_for_the_unused_one(workdir: Path) -> None:
    write_task(workdir, changed(CLASSIFICATION, {"cascade.max_drop": None}))
    assert load_task("tickets").cascade.target == 0.97


def test_agreement_is_measured_against_the_teacher(workdir: Path) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, {"cascade.reference": "gold"}))
    assert ": cascade: cascade.metric 'agreement' is measured against the teacher; use reference: teacher" in message


@pytest.mark.parametrize("metric", ["accuracy", "macro_f1"])
def test_classification_metrics_with_gold_reference_load(workdir: Path, metric: str) -> None:
    write_task(workdir, changed(CLASSIFICATION, {"cascade.reference": "gold", "cascade.metric": metric}))
    assert load_task("tickets").cascade.metric == metric


def test_field_f1_is_for_extraction_only(workdir: Path) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, {"cascade.metric": "field_f1"}))
    assert ": (top level): cascade.metric field_f1 applies to extraction tasks only" in message


@pytest.mark.parametrize("metric", ["accuracy", "macro_f1"])
def test_classification_metrics_are_rejected_for_extraction(workdir: Path, metric: str) -> None:
    message = load_error(workdir, changed(EXTRACTION, {"cascade.metric": metric}))
    assert f": (top level): cascade.metric {metric} applies to classification tasks only" in message


def test_field_f1_loads_for_extraction(workdir: Path) -> None:
    write_task(workdir, changed(EXTRACTION, {"cascade.reference": "gold", "cascade.metric": "field_f1"}))
    assert load_task("orders").cascade.metric == "field_f1"


def test_classification_needs_a_labels_file(workdir: Path) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, {"labels_file": DELETE}))
    assert ": (top level): labels_file is required for type: classification" in message


def test_extraction_needs_a_schema_file(workdir: Path) -> None:
    message = load_error(workdir, changed(EXTRACTION, {"schema_file": None}))
    assert ": (top level): schema_file is required for type: extraction" in message


# --- cost amortisation ----------------------------------------------------------------------------------------------


def test_hardware_price_without_amortisation_hours_fails(workdir: Path) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, {"cost.hardware_usd": 1500}))
    assert ": cost: cost.hardware_usd and cost.amortisation_hours must both be > 0" in message


def test_amortisation_hours_without_hardware_price_fails(workdir: Path) -> None:
    message = load_error(workdir, changed(CLASSIFICATION, {"cost.amortisation_hours": 8760}))
    assert ": cost: cost.hardware_usd and cost.amortisation_hours must both be > 0" in message


def test_amortisation_with_both_values_positive(workdir: Path) -> None:
    write_task(workdir, changed(CLASSIFICATION, {"cost.hardware_usd": 1500, "cost.amortisation_hours": 8760}))
    cost = load_task("tickets").cost
    assert (cost.hardware_usd, cost.amortisation_hours, cost.amortisation_enabled) == (1500.0, 8760.0, True)


def test_amortisation_off_with_both_values_zero(workdir: Path) -> None:
    write_task(workdir, changed(CLASSIFICATION, {"cost.hardware_usd": 0, "cost.amortisation_hours": 0}))
    assert load_task("tickets").cost.amortisation_enabled is False


# --- labels, prompt and schema files -------------------------------------------------------------------------------


def test_duplicate_labels_are_rejected(workdir: Path) -> None:
    message = load_error(workdir, CLASSIFICATION, {"labels.txt": "alpha\nbeta\nalpha\n", "teacher_prompt.md": PROMPT})
    assert message.endswith(": labels_file: duplicate labels")


def test_labels_that_differ_only_in_surrounding_space_are_duplicates(workdir: Path) -> None:
    files = {"labels.txt": "alpha\n  alpha \nbeta\n", "teacher_prompt.md": PROMPT}
    message = load_error(workdir, CLASSIFICATION, files)
    assert "labels_file: duplicate labels" in message


@pytest.mark.parametrize(
    ("labels", "first", "second"),
    [
        ("Billing\nbilling\nother\n", "Billing", "billing"),
        ("card_arrival\ncard arrival\nother\n", "card_arrival", "card arrival"),
        ("card-arrival\nother\nCard_Arrival\n", "card-arrival", "Card_Arrival"),
    ],
)
def test_labels_that_normalise_to_the_same_key_are_duplicates(
    workdir: Path, labels: str, first: str, second: str
) -> None:
    message = load_error(workdir, CLASSIFICATION, {"labels.txt": labels, "teacher_prompt.md": PROMPT})
    assert ": labels_file: " in message
    assert repr(first) in message
    assert repr(second) in message


def test_a_single_label_is_not_enough(workdir: Path) -> None:
    message = load_error(workdir, CLASSIFICATION, {"labels.txt": "alpha\n\n", "teacher_prompt.md": PROMPT})
    assert "labels_file: need at least 2 labels" in message


def test_missing_labels_file(workdir: Path) -> None:
    message = load_error(workdir, CLASSIFICATION, {"teacher_prompt.md": PROMPT})
    assert "labels_file: file not found: labels.txt" in message


def test_missing_teacher_prompt(workdir: Path) -> None:
    message = load_error(workdir, CLASSIFICATION, {"labels.txt": LABELS})
    assert "teacher.prompt_file: file not found: teacher_prompt.md" in message


def test_missing_schema_file(workdir: Path) -> None:
    message = load_error(workdir, EXTRACTION, {"teacher_prompt.md": PROMPT})
    assert "schema_file: file not found: schema.json" in message


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "items": {"type": "string"}},
        {"type": "object"},
        {"type": "object", "properties": ["a", "b"]},
        {"properties": {"a": {"type": "string"}}},
    ],
)
def test_schema_must_be_an_object_schema_with_properties(workdir: Path, schema: dict[str, Any]) -> None:
    message = load_error(workdir, EXTRACTION, {"schema.json": json.dumps(schema), "teacher_prompt.md": PROMPT})
    assert "schema_file: must be a JSON Schema with type object and properties" in message


def test_schema_that_is_not_json(workdir: Path) -> None:
    message = load_error(workdir, EXTRACTION, {"schema.json": "{type: object}", "teacher_prompt.md": PROMPT})
    assert "schema_file: invalid JSON" in message


@pytest.mark.parametrize("text", ["[1, 2]", "null", '"object"', "3"])
def test_schema_json_that_is_not_an_object_is_a_config_error(workdir: Path, text: str) -> None:
    message = load_error(workdir, EXTRACTION, {"schema.json": text, "teacher_prompt.md": PROMPT})
    assert "schema_file: must be a JSON Schema with type object and properties" in message


@pytest.mark.parametrize(
    ("base", "key", "value"),
    [
        (EXTRACTION, "labels", ["a", "b"]),
        (CLASSIFICATION, "json_schema", {"type": "object", "properties": {"a": {"type": "string"}}}),
        (CLASSIFICATION, "teacher_prompt", "Answer yes."),
        (EXTRACTION, "source", "elsewhere.yaml"),
    ],
    ids=["labels-on-extraction", "json_schema-on-classification", "teacher_prompt", "source"],
)
def test_loaded_fields_are_not_accepted_as_yaml_keys(workdir: Path, base: dict[str, Any], key: str, value: Any) -> None:
    message = load_error(workdir, changed(base, {key: value}))
    assert f": {key}: Extra inputs are not permitted" in message


# --- YAML shape -----------------------------------------------------------------------------------------------------


def test_invalid_yaml(workdir: Path) -> None:
    message = load_error(workdir, "task: [unclosed\n")
    assert "invalid YAML" in message


@pytest.mark.parametrize("text", ["- a\n- b\n", "", "just a string\n"])
def test_top_level_must_be_a_mapping(workdir: Path, text: str) -> None:
    message = load_error(workdir, text)
    assert message.endswith(": expected a mapping at the top level")


def test_flow_style_mappings_from_the_documented_spec_load(workdir: Path) -> None:
    text = """\
task: tickets
type: classification
labels_file: labels.txt
teacher:
  model: vendor/model-a
  extra_body: {provider: {order: [provider-a], allow_fallbacks: false}}
student:
  system_prompt: "Classify the message."
curate:
  dedupe: {exact: true, near_dup_jaccard: 0.9}
  pii: {enabled: true, kinds: [email, phone]}
  split: {predefined: meta.split, val: 0.1, test: 0.1, stratify: true, group_by: null, seed: 13}
cascade:
  target: 0.97
cost: {local_watts: 20, usd_per_kwh: 0.30, hardware_usd: 0, amortisation_hours: 0}
budget: {usd_cap: null}
"""
    write_task(workdir, text)
    spec = load_task("tickets")
    assert spec.teacher.provider == "provider-a"
    assert spec.curate.pii.kinds == ["email", "phone"]


# --- teacher helpers ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra_body", "provider"),
    [
        ({"provider": {"order": ["provider-a", "provider-b"], "allow_fallbacks": False}}, "provider-a"),
        ({"provider": {"order": ["provider-b"]}}, "provider-b"),
        ({}, None),
        ({"provider": {"order": []}}, None),
        ({"provider": {"sort": "price"}}, None),
        ({"provider": "provider-a"}, None),
        ({"provider": {"order": "provider-a"}}, None),
        ({"reasoning": {"enabled": False}}, None),
    ],
)
def test_provider_is_the_first_entry_of_provider_order(extra_body: dict[str, Any], provider: str | None) -> None:
    assert TeacherSpec(model="vendor/model-a", extra_body=extra_body).provider == provider


def test_api_key_prefers_the_configured_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKDISTILL_TEACHER_API_KEY", "key-primary")
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-fallback")
    assert TeacherSpec(model="vendor/model-a").api_key() == "key-primary"


def test_api_key_falls_back_to_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TASKDISTILL_TEACHER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-fallback")
    assert TeacherSpec(model="vendor/model-a").api_key() == "key-fallback"


def test_empty_api_key_falls_back_to_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKDISTILL_TEACHER_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-fallback")
    assert TeacherSpec(model="vendor/model-a").api_key() == "key-fallback"


def test_no_api_key_gives_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TASKDISTILL_TEACHER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    assert TeacherSpec(model="vendor/model-a").api_key() is None


def test_custom_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKDISTILL_TEACHER_API_KEY", "key-primary")
    monkeypatch.setenv("TD_TEST_TEAM_KEY", "key-team")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    teacher = TeacherSpec(model="vendor/model-a", api_key_env="TD_TEST_TEAM_KEY")
    assert teacher.api_key() == "key-team"
    monkeypatch.delenv("TD_TEST_TEAM_KEY")
    assert teacher.api_key() is None
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-fallback")
    assert teacher.api_key() == "key-fallback"


def test_task_spec_can_be_built_in_code() -> None:
    spec = TaskSpec.model_validate(CLASSIFICATION)
    assert spec.labels == []
    assert spec.teacher_prompt_sha256 == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# --- resolving a task name ------------------------------------------------------------------------------------------


def test_local_spec_wins_over_a_bundled_one(workdir: Path, tmp_path: Path) -> None:
    bundled = tmp_path / "_bundled" / "demo"
    write_spec_dir(bundled, changed(CLASSIFICATION, {"task": "demo", "teacher.model": "vendor/bundled"}))
    spec = load_task("demo")
    assert spec.teacher.model == "vendor/bundled"
    assert spec.source == str(bundled / "task.yaml")

    local = write_task(workdir, changed(CLASSIFICATION, {"task": "demo", "teacher.model": "vendor/local"}))
    spec = load_task("demo")
    assert spec.teacher.model == "vendor/local"
    assert Path(spec.source).resolve() == (local / "task.yaml").resolve()


def test_directory_without_task_yaml_falls_through_to_the_bundled_spec(workdir: Path, tmp_path: Path) -> None:
    bundled = write_spec_dir(tmp_path / "_bundled" / "demo", changed(CLASSIFICATION, {"task": "demo"}))
    (workdir / "tasks" / "demo").mkdir(parents=True)
    assert load_task("demo").source == str(bundled / "task.yaml")


def test_an_explicit_directory_wins(workdir: Path, tmp_path: Path) -> None:
    write_task(workdir, changed(CLASSIFICATION, {"teacher.model": "vendor/local"}))
    elsewhere = write_task(tmp_path / "elsewhere", changed(CLASSIFICATION, {"teacher.model": "vendor/explicit"}))
    assert load_task(str(elsewhere)).teacher.model == "vendor/explicit"
    assert load_task("tickets").teacher.model == "vendor/local"


def test_unknown_task_points_to_init(workdir: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_task("nope")
    message = str(info.value)
    assert "task 'nope' not found" in message
    assert "taskdistill init nope" in message


def test_bundled_spec_is_read_through_a_traversable(
    workdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("tasks/demo/task.yaml", yaml.safe_dump(changed(CLASSIFICATION, {"task": "demo"})))
        for file, content in CLASSIFICATION_FILES.items():
            zf.writestr(f"tasks/demo/{file}", content)
    with zipfile.ZipFile(archive) as zf:
        monkeypatch.setattr(config, "packaged_tasks", lambda: zipfile.Path(zf, "tasks/"))
        spec = load_task("demo")
    assert spec.labels == ["alpha", "beta", "gamma"]
    assert spec.source == "tasks/demo/task.yaml"


def test_packaged_tasks_is_inside_the_package() -> None:
    root = config.packaged_tasks()
    assert root.name == "tasks"
    assert root.is_dir()
