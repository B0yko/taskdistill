"""Curate stage 9: teacher labelling of examples that have no teacher output.

Each request is built by ``build_teacher_request`` from the RAW extracted input, so its key equals the
application's own request, a cascade escalation and the recording (ADR 0004); the student still trains on the
scrubbed text. Before anything beyond a sample is sent, the cost is projected from the first ``projection_sample``
uncached requests and confirmed (``--yes`` above $0.50). The source bounds concurrency; a replay miss or a budget
refusal stops the batch and propagates.

An invalid answer drops a train or valid example. A test example is kept with no teacher value, so the test split
does not depend on the teacher's answers and the teacher's invalid answers count against it in evaluation. A test
example normalisation already kept without a teacher value (its recorded outputs were invalid or had no majority,
``teacher_origin = "invalid"``) is never relabelled.
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from taskdistill.config import TaskSpec
from taskdistill.curate.merge import SPLITS, CurateError, Example
from taskdistill.curate.normalise import normalise_output
from taskdistill.teacher.base import TeacherResult, TeacherSource
from taskdistill.teacher.client import confirm_spend, latency_stats, project_cost
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request

DEFAULT_PROJECTION_SAMPLE = 50

NO_TEACHER = (
    "{n} example(s) have no teacher output and no teacher is available to label them: set a teacher API key "
    "(TASKDISTILL_TEACHER_API_KEY or OPENROUTER_API_KEY), provide a teacher recording, or import outputs with "
    "--format pairs"
)


async def _complete_all(teacher: TeacherSource, bodies: Sequence[dict[str, Any]]) -> list[TeacherResult]:
    """Complete every body concurrently; on the first failure cancel the rest and re-raise it."""
    tasks = [asyncio.ensure_future(teacher.complete(body)) for body in bodies]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _cached_flags(teacher: TeacherSource, bodies: Sequence[dict[str, Any]]) -> list[bool]:
    is_cached = getattr(teacher, "is_cached", None)
    if not callable(is_cached):
        return [False] * len(bodies)
    return [bool(is_cached(body)) for body in bodies]


def _projection_line(projection: dict[str, Any], mode: str) -> str:
    projected = projection.get("projected_usd")
    amount = "unknown" if projected is None else f"${projected:.4f}"
    mean = projection.get("mean_usd")
    per = "" if mean is None else f", mean ${mean:.6f}/request"
    return (
        f"      projected teacher cost ({mode}): {amount} for {projection['n_total']} uncached request(s) "
        f"(sample of {projection['sample_n']}{per})"
    )


async def _label(
    teacher: TeacherSource,
    bodies: list[dict[str, Any]],
    *,
    yes: bool,
    projection_sample: int,
    log: Callable[[str], None],
) -> tuple[list[TeacherResult], dict[str, Any]]:
    try:
        cached = _cached_flags(teacher, bodies)
        uncached = [i for i, flag in enumerate(cached) if not flag]
        sample = uncached[: max(projection_sample, 0)]
        results: list[TeacherResult | None] = [None] * len(bodies)
        for i, result in zip(sample, await _complete_all(teacher, [bodies[i] for i in sample]), strict=True):
            results[i] = result
        projection = project_cost([r for r in results if r is not None], len(uncached))
        log(_projection_line(projection, teacher.mode))
        if len(uncached) > len(sample):
            confirm_spend(projection["projected_usd"], yes)
        rest = [i for i, result in enumerate(results) if result is None]
        for i, result in zip(rest, await _complete_all(teacher, [bodies[i] for i in rest]), strict=True):
            results[i] = result
        return [r for r in results if r is not None], projection
    finally:
        # Connections belong to this event loop, which ends with the batch; a live teacher reopens them lazily.
        await teacher.aclose()


def recorded_cost(result: TeacherResult) -> float | None:
    """The cost a replayed or cached answer carried when it was first produced (``usage.cost``), if recorded."""
    cost = result.usage.get("cost") if isinstance(result.usage, dict) else None
    if isinstance(cost, int | float) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
        return float(cost)
    return float(result.cost_usd) if result.cost_usd > 0 else None


def _iso_date(timestamps: Sequence[float]) -> str | None:
    stamps = [t for t in timestamps if isinstance(t, int | float) and t > 0]
    if not stamps:
        return None
    return datetime.fromtimestamp(max(stamps), UTC).date().isoformat()


def label_examples(
    spec: TaskSpec,
    examples: Sequence[Example],
    teacher: TeacherSource | None,
    *,
    yes: bool = False,
    max_usd: float | None = None,
    projection_sample: int = DEFAULT_PROJECTION_SAMPLE,
    log: Callable[[str], None] = print,
) -> tuple[list[Example], dict[str, Any], list[str]]:
    """Label every example without a teacher value (except test rows already kept as ``teacher_origin = "invalid"``);
    return the kept examples, the counts and the request keys.

    Newly labelled examples get the normalised teacher value (``teacher_origin = "labelled"``). An invalid answer
    drops a train or valid example; a test example is kept without a teacher value (``teacher_origin =
    "invalid"``). ``cost_usd`` is this run's live spend; ``recorded_cost_usd`` is what the replayed and cached
    answers cost when they were produced. ``max_usd`` is only recorded: the caller sets the run cap on the
    teacher. The batch runs in its own event loop and releases the teacher's connections (``aclose``) before that
    loop ends.
    """
    pending = [ex for ex in examples if ex.teacher is None and ex.teacher_origin != "invalid"]
    stats: dict[str, Any] = {
        "requested": len(pending),
        "mode": teacher.mode if teacher is not None else None,
        "max_usd": max_usd,
        "projection": None,
        "live": 0,
        "cached": 0,
        "replayed": 0,
        "cost_usd": 0.0,
        "recorded_cost_usd": None,
        "providers": {},
        "truncated": 0,
        "invalid": 0,
        "invalid_by_split": dict.fromkeys(SPLITS, 0),
        "kept_invalid_test": 0,
        "labelled": 0,
        "date": None,
        "latency": None,
        "concurrency": getattr(teacher, "concurrency", None),
    }
    if not pending:
        return list(examples), stats, []
    if teacher is None:
        raise CurateError(NO_TEACHER.format(n=len(pending)))

    bodies = [build_teacher_request(spec, ex.raw_input) for ex in pending]
    keys = [request_key(body) for body in bodies]
    results, projection = asyncio.run(_label(teacher, bodies, yes=yes, projection_sample=projection_sample, log=log))

    sources = Counter(r.source for r in results)
    dropped: set[int] = set()
    invalid: Counter[str] = Counter()
    for ex, result in zip(pending, results, strict=True):
        value, _ = normalise_output(spec, result.output or "")
        if value is None:
            invalid[str(ex.split)] += 1
            if ex.split == "test":
                ex.teacher_origin = "invalid"
            else:
                dropped.add(id(ex))
            continue
        ex.teacher = value
        ex.teacher_origin = "labelled"
    kept = [ex for ex in examples if id(ex) not in dropped]
    recorded = [cost for r in results if r.source != "live" and (cost := recorded_cost(r)) is not None]
    stats.update(
        projection=projection,
        live=sources["live"],
        cached=sources["cache"],
        replayed=sources["replay"],
        cost_usd=sum(float(r.cost_usd) for r in results if r.source == "live"),
        recorded_cost_usd=sum(recorded) if recorded else None,
        providers=dict(sorted(Counter(r.provider or "unknown" for r in results).items())),
        truncated=sum(1 for r in results if r.truncated),
        invalid=sum(invalid.values()),
        invalid_by_split={name: invalid[name] for name in SPLITS},
        kept_invalid_test=invalid["test"],
        labelled=len(pending) - sum(invalid.values()),
        date=_iso_date([r.created for r in results]),
        latency=latency_stats(results),
    )
    return kept, stats, sorted(set(keys))
