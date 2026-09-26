"""Build a teacher recording from the response cache and the captured traffic (must-have 13).

The records are the teacher answers for a set of request keys: the task's captured requests, the keys curate
sent for labelling, and any extra request bodies (smoke-test queries, the README request). An answer comes from
the response cache, else from the store's capture of that request (the capture proxy and ``capture --import
--format openai`` keep the teacher's response body, not a cache row). Either counts only when it was produced
under the spec's routing and reasoning settings (the cache context of the request), and a request body must have
been built for the spec's teacher model and settings, so a recording never holds outputs from another model or
provider under the spec's manifest. Records hold the output and its metadata, never inputs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from taskdistill import paths
from taskdistill.config import TaskSpec
from taskdistill.store import CaptureRow, Store
from taskdistill.teacher.base import TeacherResult
from taskdistill.teacher.cache import CONTEXT_EXCLUDED, ResponseCache, request_context
from taskdistill.teacher.client import message_text
from taskdistill.teacher.factory import load_pricing
from taskdistill.teacher.pricing import PricingError
from taskdistill.teacher.replay import RecordingError, expected_manifest, load_recording, write_recording
from taskdistill.teacher.request_key import canonical_json, request_key
from taskdistill.teacher.requests import build_teacher_request

LABELLING_KEYS = "labelling_keys.txt"
#: How many missing keys an error message lists.
SHOWN_MISSING = 5


def spec_context(spec: TaskSpec) -> str:
    """The cache context of every request :func:`build_teacher_request` makes for ``spec`` (input-independent)."""
    return request_context(build_teacher_request(spec, ""))


def _context_fields(body: Mapping[str, Any]) -> dict[str, bytes]:
    return {k: canonical_json(v) for k, v in body.items() if k not in CONTEXT_EXCLUDED and v is not None}


def check_request(spec: TaskSpec, body: Mapping[str, Any], context: str | None = None) -> None:
    """:class:`RecordingError` unless ``body`` asks the spec's teacher model under the spec's settings.

    A body for another provider or reasoning setting has the same request key as the spec's request for the same
    input, so recording it would replay another provider's output as the spec's teacher.
    """
    where = spec.source or spec.task
    key = request_key(body)
    model = body.get("model")
    if model != spec.teacher.model:
        raise RecordingError(
            f"request {key[:12]} is for model {model!r}, but the recording is for teacher.model "
            f"{spec.teacher.model!r} of {where}"
        )
    if request_context(body) == (context if context is not None else spec_context(spec)):
        return
    wanted, given = _context_fields(build_teacher_request(spec, "")), _context_fields(body)
    differing = sorted(name for name in wanted.keys() | given.keys() if wanted.get(name) != given.get(name))
    raise RecordingError(
        f"request {key[:12]} was built with other {', '.join(differing) or 'settings'} than teacher.extra_body of "
        f"{where}; a recording holds only requests made under the spec's settings (its manifest records them)"
    )


def wanted_keys(spec: TaskSpec, keys: Iterable[str] | None, requests: Iterable[Mapping[str, Any]] | None) -> list[str]:
    """The request keys to record, in order of first appearance: those of ``requests`` (each checked with
    :func:`check_request`), then the bare ``keys``."""
    context = spec_context(spec)
    wanted: list[str] = []
    for body in requests or ():
        check_request(spec, body, context)
        wanted.append(request_key(body))
    wanted.extend(key.strip() for key in keys or () if key.strip())
    return list(dict.fromkeys(wanted))


def collect_records(
    cache: ResponseCache, wanted: Iterable[str], context: str
) -> tuple[list[TeacherResult], list[str], int]:
    """Cached results for ``wanted`` under ``context``, the keys that are missing, and how many of those are cached
    under another context (another provider or reasoning setting)."""
    keys = sorted(set(wanted))
    found = {result.key: result for result in cache.iter_results(keys, context)}
    missing = [key for key in keys if key not in found]
    elsewhere = sum(1 for key in missing if key in cache)
    return [found[key] for key in sorted(found)], missing, elsewhere


def _captured_requests(spec: TaskSpec, store: Store) -> Iterator[tuple[str, dict[str, Any], CaptureRow]]:
    """``(request key, request body, row)`` of the task's successful, non-streamed captures for the spec's teacher
    model, in capture order. The key is recomputed from the stored body, so it is the current request key."""
    for row in store.iter_captures(spec.task):
        if not row.captured or row.status is None or not 200 <= row.status < 300:
            continue
        try:
            body = json.loads(row.request_body or "")
        except ValueError:
            continue
        if not isinstance(body, dict) or body.get("model") != spec.teacher.model:
            continue
        try:
            key = request_key(body)
        except ValueError:
            continue
        yield key, body, row


def _captured_result(key: str, row: CaptureRow) -> TeacherResult | None:
    """The teacher answer a capture row holds (a ``chat.completion`` with a message), or None."""
    try:
        response = json.loads(row.response_body or "")
    except ValueError:
        return None
    if not isinstance(response, dict) or response.get("error") is not None:
        return None
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), Mapping):
        return None
    if choice.get("error") is not None or choice.get("finish_reason") == "error":
        return None
    usage = response.get("usage")
    provider = response.get("provider")
    finish_reason = choice.get("finish_reason")
    return TeacherResult(
        key=key,
        output=message_text(choice["message"].get("content")),
        response=response,
        usage=dict(usage) if isinstance(usage, dict) else {},
        latency_ms=row.latency_ms,
        provider=provider if isinstance(provider, str) else None,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        created=row.ts,
        source="cache",
    )


def captured_records(
    spec: TaskSpec, store: Store, wanted: Iterable[str], context: str
) -> tuple[list[TeacherResult], list[str], set[str]]:
    """The captured answers for ``wanted`` made under ``context`` (the first capture of a key wins), the keys
    without one, and which of those were captured under another context."""
    keys = set(wanted)
    found: dict[str, TeacherResult] = {}
    elsewhere: set[str] = set()
    for key, body, row in _captured_requests(spec, store):
        if key not in keys or key in found:
            continue
        try:
            same_settings = request_context(body) == context
        except ValueError:
            continue
        if not same_settings:
            elsewhere.add(key)
            continue
        result = _captured_result(key, row)
        if result is not None:
            found[key] = result
    missing = sorted(keys - found.keys())
    return [found[key] for key in sorted(found)], missing, elsewhere & set(missing)


def _workspace_store() -> Store | None:
    path = paths.store_path()
    return Store(path) if path.is_file() else None


def _default_pricing_date() -> str:
    try:
        return load_pricing().date
    except PricingError as exc:
        raise RecordingError(
            f"the recording manifest needs the pricing snapshot date, but the snapshot cannot be read ({exc}); "
            "run `taskdistill pricing refresh` or pass the snapshot date explicitly"
        ) from exc


def build_recording(
    spec: TaskSpec,
    *,
    keys: list[str] | None = None,
    requests: list[dict[str, Any]] | None = None,
    out: Path,
    cache: ResponseCache | None = None,
    pricing_date: str | None = None,
    store: Store | None = None,
) -> dict[str, Any]:
    """Write the recording of ``keys`` and ``requests`` to ``out``; return its manifest.

    Each answer comes from the response cache, else from the task's captured traffic in ``store`` (default: the
    workspace store, when there is one). Request bodies must have been built for the spec's model and settings
    (:func:`check_request`), and every wanted key must be answered under the spec's settings; otherwise
    :class:`RecordingError` says what differs, or how many keys are missing and names the first few. The manifest
    is :func:`expected_manifest` plus the record count and the time of the newest record, so the same cache and
    store give the same bytes. ``pricing_date`` defaults to the date of the pricing snapshot in use; an unreadable
    snapshot is an error.
    """
    wanted = wanted_keys(spec, keys, requests)
    if not wanted:
        raise RecordingError(f"no request keys to record for task '{spec.task}'")
    source = cache if cache is not None else ResponseCache()
    context = spec_context(spec)
    results, missing, elsewhere = collect_records(source, wanted, context)
    captures = store if store is not None else _workspace_store()
    if missing and captures is not None:
        cached_elsewhere = {key for key in missing if key in source}
        captured, missing, captured_elsewhere = captured_records(spec, captures, missing, context)
        results = sorted([*results, *captured], key=lambda result: result.key)
        # a key can be both cached and captured under another setting: count it once
        elsewhere = len((cached_elsewhere | captured_elsewhere) & set(missing))
    if missing:
        shown = ", ".join(missing[:SHOWN_MISSING]) + (", ..." if len(missing) > SHOWN_MISSING else "")
        note = (
            f"; {elsewhere} of them are cached or captured under another provider or reasoning setting than "
            f"teacher.extra_body of {spec.source or spec.task}"
            if elsewhere
            else ""
        )
        raise RecordingError(
            f"{len(missing)} of {len(wanted)} request keys are not in the response cache or the task's captured "
            f"traffic ({shown}){note}. Label them with the live teacher first (curate or the demo with --live), "
            "then record again."
        )
    manifest = expected_manifest(spec, pricing_date if pricing_date is not None else _default_pricing_date())
    path = write_recording(out, manifest, results)
    return load_recording(path).manifest


def default_keys(spec: TaskSpec, store: Store) -> list[str]:
    """The task's captured request keys (successful, non-streamed, for the spec's teacher model) plus the keys in
    ``$TASKDISTILL_HOME/<task>/data/labelling_keys.txt``, deduplicated in order of first appearance."""
    keys = [key for key, _, _ in _captured_requests(spec, store)]
    labelling = paths.home() / spec.task / "data" / LABELLING_KEYS
    if labelling.is_file():
        keys.extend(line.strip() for line in labelling.read_text(encoding="utf-8").splitlines() if line.strip())
    return list(dict.fromkeys(keys))


def fill_missing(
    spec: TaskSpec,
    requests: Iterable[Mapping[str, Any]],
    *,
    yes: bool = False,
    max_usd: float | None = None,
    cache: ResponseCache | None = None,
    log: Any = print,
) -> int:
    """Ask the live teacher for every request whose answer is not cached yet (budgeted, then cached).

    A recording must hold every request a demo can send; inputs that one profile drops as near-duplicates
    before labelling can survive in the other profile. Returns the number of requests sent.
    """
    import asyncio

    from taskdistill.ledger import worst_case_cost
    from taskdistill.teacher.client import confirm_spend
    from taskdistill.teacher.factory import make_teacher, new_run_id

    cache = cache if cache is not None else ResponseCache()
    context = spec_context(spec)
    missing = [dict(body) for body in requests if cache.get(request_key(body), context) is None]
    if not missing:
        return 0
    price = load_pricing().price_for(spec.teacher.model, spec.teacher.provider)
    projected = sum(worst_case_cost(body, price) for body in missing)
    log(f"{len(missing)} request(s) not cached; worst-case projection ${projected:.4f}")
    confirm_spend(projected, yes)
    teacher = make_teacher(spec, mode="live", phase="record-fill", run_id=new_run_id("record"), run_cap=max_usd)

    async def run() -> None:
        tasks = [asyncio.ensure_future(teacher.complete(body)) for body in missing]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            # Stop the requests still waiting for a slot before the client closes: they were never sent and
            # must cost nothing (the ones on the wire settle at their worst case when cancelled).
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            await teacher.aclose()

    asyncio.run(run())
    return len(missing)
