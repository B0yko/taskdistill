"""Typed evaluation splits.

Every selection step (model, checkpoint, run, threshold, isotonic fit, baseline tuning) accepts only
a :class:`ValidationSplit`. :class:`TestSplit` is a distinct class, not an alias or a subclass, and
passing one to a selection function raises ``TypeError`` at runtime.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, ParamSpec, Self, TypeVar

TaskType = Literal["classification", "extraction"]


@dataclass(frozen=True)
class EvalRecord:
    """One example of a split, optionally with a system's prediction attached."""

    id: str
    input: str
    gold: Any = None
    teacher: Any = None
    pred: Any = None
    confidence: float | None = None
    alt_pred: Any = None
    alt_confidence: float | None = None
    group: str | None = None
    traits: tuple[str, ...] = ()
    latency_ms: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


class _Split:
    name: ClassVar[str] = ""

    def __init__(self, records: Sequence[EvalRecord], task_type: TaskType) -> None:
        self.records: tuple[EvalRecord, ...] = tuple(records)
        self.task_type: TaskType = task_type

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[EvalRecord]:
        return iter(self.records)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self.records)}, task_type={self.task_type!r})"

    def replace_records(self, records: Sequence[EvalRecord]) -> Self:
        """A new split of the same class with different records."""
        return type(self)(records, self.task_type)

    def with_predictions(
        self,
        preds: Sequence[Any],
        confidences: Sequence[float | None],
        *,
        latencies_ms: Sequence[float | None] | None = None,
        alt_preds: Sequence[Any] | None = None,
        alt_confidences: Sequence[float | None] | None = None,
    ) -> Self:
        n = len(self.records)
        if len(preds) != n or len(confidences) != n:
            raise ValueError("predictions must align with the split's records")
        out = []
        for i, rec in enumerate(self.records):
            out.append(
                dataclasses.replace(
                    rec,
                    pred=preds[i],
                    confidence=confidences[i],
                    latency_ms=None if latencies_ms is None else latencies_ms[i],
                    alt_pred=None if alt_preds is None else alt_preds[i],
                    alt_confidence=None if alt_confidences is None else alt_confidences[i],
                )
            )
        return self.replace_records(out)

    @property
    def has_gold(self) -> bool:
        return any(r.gold is not None for r in self.records)

    @property
    def has_teacher(self) -> bool:
        return any(r.teacher is not None for r in self.records)


class ValidationSplit(_Split):
    """The only split selection functions may read."""

    name: ClassVar[str] = "valid"


class TestSplit(_Split):
    """The held-out split; reported, never used for any choice."""

    __test__ = False  # not a pytest test class
    name: ClassVar[str] = "test"


def require_validation(value: Any, *, argument: str = "split") -> None:
    """Raise TypeError unless ``value`` is a ValidationSplit (or a mapping/sequence of them)."""
    if isinstance(value, ValidationSplit):
        return
    if isinstance(value, Mapping):
        if not value:
            raise TypeError(f"{argument}: expected ValidationSplit values, got an empty mapping")
        for key, item in value.items():
            require_validation(item, argument=f"{argument}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        if not value:
            raise TypeError(f"{argument}: expected ValidationSplit items, got an empty sequence")
        for i, item in enumerate(value):
            require_validation(item, argument=f"{argument}[{i}]")
        return
    kind = type(value).__name__
    raise TypeError(f"{argument}: selection reads validation data only; got {kind}, expected ValidationSplit")


P = ParamSpec("P")
R = TypeVar("R")


def validation_only(*params: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator: the named parameters must be ValidationSplit (checked at call time)."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(fn)
        missing = [p for p in params if p not in signature.parameters]
        if missing:
            raise ValueError(f"{fn.__name__} has no parameter(s) {missing}")

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = signature.bind(*args, **kwargs)
            for name in params:
                require_validation(bound.arguments.get(name), argument=f"{fn.__name__}({name})")
            return fn(*args, **kwargs)

        wrapper.__validation_only__ = params  # type: ignore[attr-defined]
        return wrapper

    return decorate
