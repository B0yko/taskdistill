"""Task specification (``tasks/<task>/task.yaml``), validated with Pydantic.

String values may reference environment variables as ``${NAME}`` or ``${NAME:-default}``
(the default is used when the variable is unset or empty, as in POSIX shells).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
PII_KINDS = ("email", "phone", "iban", "card", "ipv4", "ssn")
PiiKind = Literal["email", "phone", "iban", "card", "ipv4", "ssn"]
_DEFAULT_PII_KINDS: tuple[PiiKind, ...] = ("email", "phone", "iban", "card", "ipv4", "ssn")
#: Attributes filled from the files a spec points to; the YAML may not set them.
LOADED_FIELDS = ("labels", "json_schema", "teacher_prompt", "source")


class ConfigError(ValueError):
    """A task spec could not be loaded; the message names the offending key."""


def expand_env(value: Any, *, where: str = "") -> Any:
    """Recursively expand ``${NAME}`` / ``${NAME:-default}`` in every string of a parsed YAML tree."""
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            env = os.environ.get(name)
            if env:
                return env
            if default is not None:
                return default
            raise ConfigError(f"{where or 'value'}: environment variable {name} is not set and has no default")

        return ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: expand_env(v, where=f"{where}.{k}" if where else str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, where=f"{where}[{i}]") for i, v in enumerate(value)]
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class InputSpec(_Strict):
    from_: Literal["last_user_message", "regex"] = Field(default="last_user_message", alias="from")
    regex: str | None = None

    @model_validator(mode="after")
    def _check_regex(self) -> InputSpec:
        if self.from_ == "regex":
            if not self.regex:
                raise ValueError("input.regex is required when input.from is 'regex'")
            try:
                compiled = re.compile(self.regex, re.DOTALL)
            except re.error as exc:
                raise ValueError(f"input.regex does not compile: {exc}") from exc
            if "input" not in compiled.groupindex:
                raise ValueError("input.regex must contain a named group 'input', e.g. (?P<input>...)")
        return self


class TeacherSpec(_Strict):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "TASKDISTILL_TEACHER_API_KEY"
    model: str
    prompt_file: str = "teacher_prompt.md"
    temperature: float = 0.0
    max_tokens: int = Field(default=256, gt=0)
    response_format: dict[str, Any] | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def _model_slug(cls, v: str) -> str:
        if not v or "<" in v:
            raise ValueError("teacher.model must be a model slug")
        return v

    @field_validator("extra_body")
    @classmethod
    def _extra_body_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        reserved = {"model", "messages", "stream", "n", "tools"}
        clash = sorted(reserved & set(v))
        if clash:
            raise ValueError(f"teacher.extra_body must not set {', '.join(clash)}")
        return v

    @property
    def provider(self) -> str | None:
        """The pinned upstream provider (first entry of ``extra_body.provider.order``), if any."""
        provider = self.extra_body.get("provider")
        if isinstance(provider, dict):
            order = provider.get("order")
            if isinstance(order, list) and order:
                return str(order[0])
        return None

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or os.environ.get("OPENROUTER_API_KEY") or None


class StudentSpec(_Strict):
    base_model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    system_prompt: str
    max_tokens: int = Field(default=16, gt=0)
    response_template: str | None = None

    @field_validator("response_template")
    @classmethod
    def _template_has_label(cls, v: str | None) -> str | None:
        if v is not None and "{label}" not in v:
            raise ValueError("student.response_template must contain the literal placeholder {label}")
        return v


class TrainSpec(_Strict):
    profile: Literal["quick", "full"] = "full"
    lora_rank: int = Field(default=16, gt=0)
    lora_layers: Annotated[int, Field(gt=0)] | Literal["all"] = "all"
    learning_rate: float = Field(default=1.0e-4, gt=0)
    batch_size: int = Field(default=8, gt=0)
    epochs: float = Field(default=2, gt=0)
    max_seq_len: int = Field(default=512, gt=0)
    seed: int = 13


class DedupeSpec(_Strict):
    exact: bool = True
    near_dup_jaccard: float = Field(default=0.9, ge=0.5, le=1)


class PiiSpec(_Strict):
    enabled: bool = True
    kinds: list[PiiKind] = Field(default_factory=lambda: list(_DEFAULT_PII_KINDS))


class SplitSpec(_Strict):
    predefined: str | None = "meta.split"
    val: float = Field(default=0.1, ge=0, lt=1)
    test: float = Field(default=0.1, ge=0, lt=1)
    stratify: bool = True
    group_by: str | None = None
    seed: int = 13

    @model_validator(mode="after")
    def _fractions(self) -> SplitSpec:
        if self.val + self.test >= 1:
            raise ValueError("curate.split.val + curate.split.test must be < 1")
        for name in ("predefined", "group_by"):
            value = getattr(self, name)
            if value is not None and not value.startswith("meta."):
                raise ValueError(f"curate.split.{name} must name a meta field, e.g. meta.split")
        return self


class CurateSpec(_Strict):
    dedupe: DedupeSpec = Field(default_factory=DedupeSpec)
    pii: PiiSpec = Field(default_factory=PiiSpec)
    split: SplitSpec = Field(default_factory=SplitSpec)


class CascadeSpec(_Strict):
    reference: Literal["teacher", "gold"] = "teacher"
    metric: Literal["agreement", "accuracy", "macro_f1", "field_f1"] = "agreement"
    target: float | None = Field(default=None, ge=0, le=1)
    max_drop: float | None = Field(default=None, ge=0, le=1)
    on_teacher_error: Literal["student", "error"] = "student"
    escalation_response: Literal["canonical", "raw"] = "canonical"

    @model_validator(mode="after")
    def _one_target(self) -> CascadeSpec:
        if (self.target is None) == (self.max_drop is None):
            raise ValueError("set exactly one of cascade.target and cascade.max_drop")
        if self.metric == "agreement" and self.reference != "teacher":
            raise ValueError("cascade.metric 'agreement' is measured against the teacher; use reference: teacher")
        return self


class CostSpec(_Strict):
    local_watts: float = Field(default=20.0, ge=0)
    usd_per_kwh: float = Field(default=0.30, ge=0)
    hardware_usd: float = Field(default=0.0, ge=0)
    amortisation_hours: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _amortisation(self) -> CostSpec:
        if (self.hardware_usd > 0) != (self.amortisation_hours > 0):
            raise ValueError(
                "cost.hardware_usd and cost.amortisation_hours must both be > 0 to enable hardware "
                "amortisation, or both 0 to disable it"
            )
        return self

    @property
    def amortisation_enabled(self) -> bool:
        return self.hardware_usd > 0 and self.amortisation_hours > 0


class BudgetSpec(_Strict):
    usd_cap: float | None = Field(default=None, gt=0)


class TaskSpec(_Strict):
    task: str
    type: Literal["classification", "extraction"]
    labels_file: str | None = None
    schema_file: str | None = None
    input: InputSpec = Field(default_factory=InputSpec)
    teacher: TeacherSpec
    student: StudentSpec
    train: TrainSpec = Field(default_factory=TrainSpec)
    curate: CurateSpec = Field(default_factory=CurateSpec)
    cascade: CascadeSpec
    cost: CostSpec = Field(default_factory=CostSpec)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)

    # Loaded from the files the spec points to (not part of the YAML).
    labels: list[str] = Field(default_factory=list, exclude=True)
    json_schema: dict[str, Any] | None = Field(default=None, exclude=True)
    teacher_prompt: str = Field(default="", exclude=True)
    source: str = Field(default="", exclude=True)

    @field_validator("task")
    @classmethod
    def _task_name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", v):
            raise ValueError("task must be 1-64 characters of letters, digits, '.', '_' or '-'")
        return v

    @model_validator(mode="after")
    def _type_files(self) -> TaskSpec:
        if self.type == "classification":
            if not self.labels_file:
                raise ValueError("labels_file is required for type: classification")
            if self.cascade.metric == "field_f1":
                raise ValueError("cascade.metric field_f1 applies to extraction tasks only")
        else:
            if not self.schema_file:
                raise ValueError("schema_file is required for type: extraction")
            if self.cascade.metric in ("accuracy", "macro_f1"):
                raise ValueError(f"cascade.metric {self.cascade.metric} applies to classification tasks only")
        return self

    @property
    def teacher_prompt_sha256(self) -> str:
        return hashlib.sha256(self.teacher_prompt.encode("utf-8")).hexdigest()

    @property
    def schema_fields(self) -> list[str]:
        if not self.json_schema:
            return []
        return list(self.json_schema.get("properties", {}))


def _format_validation_error(exc: ValidationError, source: str) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"] if part != "__root__")
        msg = err["msg"].removeprefix("Value error, ")
        lines.append(f"{source}: {loc or '(top level)'}: {msg}")
    return "\n".join(lines)


def packaged_tasks() -> Traversable:
    return resources.files("taskdistill") / "_data" / "tasks"


def resolve_task_dir(task: str) -> Traversable | Path:
    """Find a task spec directory.

    Lookup order: an explicit directory path; ``./tasks/<task>/``; a bundled demo spec.
    """
    candidate = Path(task)
    if (candidate / "task.yaml").is_file():
        return candidate
    local = Path.cwd() / "tasks" / task
    if (local / "task.yaml").is_file():
        return local
    bundled = packaged_tasks() / task
    if bundled.is_dir() and (bundled / "task.yaml").is_file():
        return bundled
    raise ConfigError(
        f"task '{task}' not found: expected ./tasks/{task}/task.yaml "
        f"(create one with `taskdistill init {task} --type classification|extraction`)"
    )


def load_task(task: str) -> TaskSpec:
    """Load, expand and validate ``task.yaml`` plus the prompt, labels and schema it references."""
    directory = resolve_task_dir(task)
    spec_file = directory / "task.yaml"
    source = f"tasks/{task}/task.yaml" if not isinstance(directory, Path) else str(spec_file)
    try:
        raw = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: expected a mapping at the top level")
    for key in LOADED_FIELDS:
        if key in raw:
            raise ConfigError(f"{source}: {key}: Extra inputs are not permitted (filled from the files the spec names)")
    raw = expand_env(raw)
    try:
        spec = TaskSpec.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, source)) from exc

    def read(name: str, key: str) -> str:
        path = directory / name
        if not path.is_file():
            raise ConfigError(f"{source}: {key}: file not found: {name}")
        return path.read_text(encoding="utf-8")

    spec.source = source
    spec.teacher_prompt = read(spec.teacher.prompt_file, "teacher.prompt_file")
    if spec.type == "classification":
        assert spec.labels_file is not None
        labels = [ln.strip() for ln in read(spec.labels_file, "labels_file").splitlines() if ln.strip()]
        if len(labels) < 2:
            raise ConfigError(f"{source}: labels_file: need at least 2 labels")
        if len(set(labels)) != len(labels):
            raise ConfigError(f"{source}: labels_file: duplicate labels")
        from taskdistill.tasks.classification import label_key

        seen: dict[str, str] = {}
        for label in labels:
            other = seen.setdefault(label_key(label), label)
            if other != label:
                raise ConfigError(
                    f"{source}: labels_file: labels {other!r} and {label!r} normalise to the same key; "
                    "teacher outputs could not be told apart"
                )
        spec.labels = labels
    else:
        assert spec.schema_file is not None
        try:
            schema = json.loads(read(spec.schema_file, "schema_file"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{source}: schema_file: invalid JSON: {exc}") from exc
        if (
            not isinstance(schema, dict)
            or schema.get("type") != "object"
            or not isinstance(schema.get("properties"), dict)
        ):
            raise ConfigError(f"{source}: schema_file: must be a JSON Schema with type object and properties")
        spec.json_schema = schema
    return spec
