"""``taskdistill bench``: replay test inputs through a running cascade server and measure it end to end.

Requests are the application's own (``build_teacher_request`` on the raw test inputs, in data order), sent one at a
time. The first ``warmup`` requests are excluded from the statistics and the timed requests use the inputs after
them. The machine's load and memory pressure are recorded before timing. A response that was escalated to a replayed
teacher (``x-taskdistill-teacher: replay``) fails the bench at once, so recorded answers never enter a live number.
The server must serve the bench's task. Spend is the ledger delta of the ``serve`` phase of the task in this workspace,
so the server must share it: when live teacher answers arrive but that delta stays at 0, the spend cannot be measured,
so the projection needs ``--yes``, ``--max-usd`` fails (it cannot be enforced) and ``spend_usd`` is null.
``--max-usd`` is checked before every request: the bench stops (during the warm-up: fails) when the spend so far plus
the most that request can cost would cross the cap. That bound is the worst case of one teacher call for the request's
body at the pricing snapshot's price for the teacher (what the server's ledger reserves), or the largest amount one
request has added to the spend so far when that is more or the snapshot has no price for the teacher.
The JSON is always written to ``$TASKDISTILL_HOME/<task>/bench/`` (where ``report`` finds it), and to ``out`` too.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import httpx
import numpy as np

from taskdistill import paths
from taskdistill.config import TaskSpec
from taskdistill.hardware import hardware_info, machine_state
from taskdistill.ledger import Ledger, worst_case_cost
from taskdistill.store import Store
from taskdistill.teacher.client import DEFAULT_SPEND_THRESHOLD, SpendNotConfirmed, confirm_spend, pinned_provider
from taskdistill.teacher.factory import load_pricing
from taskdistill.teacher.pricing import PricingError, PricingSnapshot
from taskdistill.teacher.requests import build_teacher_request

TOKEN_ENV = "TASKDISTILL_SERVER_TOKEN"
SERVE_PHASE = "serve"
#: Timed requests after the warm-up before the spend is projected.
PROJECTION_SAMPLE = 50
ROUTE_HEADER = "x-taskdistill-route"
REASON_HEADER = "x-taskdistill-reason"
CONFIDENCE_HEADER = "x-taskdistill-confidence"
TEACHER_HEADER = "x-taskdistill-teacher"
DEFAULT_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
#: How long to wait for a server whose model is still loading (/healthz answers 503).
READY_TIMEOUT_S = 120.0


class BenchError(click.ClickException):
    """The bench cannot run or its numbers would not be live measurements."""


def _base_url(url: str) -> str:
    base = url.strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    if not base.startswith(("http://", "https://")):
        raise BenchError(f"--url must be an http(s) URL, got {url!r}")
    return base


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def bench_inputs(spec: TaskSpec, store: Store) -> tuple[list[tuple[str, str]], int]:
    """``(input_hash, raw input)`` of the curated test split in data order, and how many had no raw input."""
    from taskdistill.curate.io import raw_inputs_by_hash

    meta = paths.home() / spec.task / "data" / "test.meta.jsonl"
    if not meta.is_file():
        raise BenchError(
            f"no curated test split for task '{spec.task}' (data/test.meta.jsonl); run `taskdistill curate` first"
        )
    hashes: list[str] = []
    with meta.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                hashes.append(str(json.loads(line).get("input_hash") or ""))
    raw = raw_inputs_by_hash(store, spec)
    inputs = [(h, raw[h]) for h in hashes if h and h in raw]
    return inputs, len(hashes) - len(inputs)


def _serve_spent(ledger: Ledger, task: str) -> float:
    return ledger.spent(task=task, phase=SERVE_PHASE)


def _healthz(
    client: httpx.Client, base: str, headers: Mapping[str, str], ready_timeout_s: float = READY_TIMEOUT_S
) -> dict[str, Any]:
    """The server's /healthz JSON, waiting while it answers 503 (model still loading)."""
    deadline = time.monotonic() + ready_timeout_s
    while True:
        try:
            resp = client.get(f"{base}/healthz", headers=dict(headers))
        except httpx.HTTPError as exc:
            raise BenchError(f"the server at {base} did not answer /healthz: {exc}") from exc
        if resp.status_code != 503 or time.monotonic() >= deadline:
            break
        time.sleep(1.0)
    if resp.status_code != 200:
        raise BenchError(f"the server at {base} answered /healthz with HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise BenchError(f"the server at {base} returned a /healthz body that is not JSON") from exc
    return data if isinstance(data, dict) else {}


def _send(client: httpx.Client, url: str, body: dict[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        resp = client.post(url, json=body, headers=dict(headers))
        resp.read()
    except httpx.HTTPError as exc:
        return {
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "status": None,
            "route": None,
            "reason": None,
            "confidence": None,
            "teacher": None,
            "error": type(exc).__name__,
        }
    latency = (time.perf_counter() - started) * 1000.0
    return {
        "latency_ms": latency,
        "status": resp.status_code,
        "route": resp.headers.get(ROUTE_HEADER),
        "reason": resp.headers.get(REASON_HEADER),
        "confidence": _float(resp.headers.get(CONFIDENCE_HEADER)),
        "teacher": resp.headers.get(TEACHER_HEADER),
        "error": None if resp.status_code == 200 else f"HTTP {resp.status_code}",
    }


def _live_answer(result: Mapping[str, Any]) -> bool:
    """A response the live teacher answered (these are what the server's ledger records spend for)."""
    return result.get("route") == "teacher" and str(result.get("teacher") or "").strip().lower() == "live"


def _unmeasured_reason(live_answers: int) -> str:
    return (
        f"{live_answers} request{'' if live_answers == 1 else 's'} answered by the live teacher, but no "
        f"'{SERVE_PHASE}' spend of this task in this workspace's ledger: the server runs with another "
        "TASKDISTILL_HOME (or its teacher reported no cost), so the spend cannot be measured"
    )


def _check_task(health: Mapping[str, Any], spec: TaskSpec, base: str) -> None:
    served = health.get("task")
    if served != spec.task:
        what = "did not report its task" if served is None else f"serves task '{served}'"
        raise BenchError(
            f"the server at {base} {what}, not '{spec.task}': start `taskdistill serve --task {spec.task}` "
            "and bench that"
        )


def _bench_path(task: str) -> Path:
    directory = paths.task_home(task) / "bench"
    stamp = _utc_stamp()
    target = directory / f"bench_{stamp}.json"
    suffix = 1
    while target.exists():
        suffix += 1
        target = directory / f"bench_{stamp}-{suffix}.json"
    return target


def _escalated(result: Mapping[str, Any]) -> bool:
    """A reason and a route other than ``student``: ``teacher``, ``student-fallback`` or a failed ``error``."""
    return bool(result.get("reason")) and result.get("route") not in (None, "student")


def _check_live(result: Mapping[str, Any], index: int) -> None:
    """Fail on any response the teacher was involved in (route other than ``student``) that came from a replay."""
    if result.get("route") != "student" and str(result.get("teacher") or "").strip().lower() == "replay":
        raise BenchError(
            f"request {index + 1} was escalated to a replayed teacher (x-taskdistill-teacher: replay): replayed "
            "answers never enter a live bench. Start `taskdistill serve` with a teacher API key and without --replay."
        )


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "n": len(values),
    }


def _nano(usd: float) -> int:
    return round(usd * 1e9)


class _SpendCap:
    """``--max-usd``, checked before every request: the spend so far plus the most the next request can cost.

    A request escalates to at most one teacher call, bounded by the worst case the server's ledger reserves for its
    body at the pricing snapshot's price. The bound is raised to the largest amount one request has added to the
    spend so far (a call whose timed-out attempts the server retried is charged for each of them), and is that
    amount alone when the snapshot cannot price the teacher.
    """

    def __init__(self, spec: TaskSpec, max_usd: float) -> None:
        self.max_usd = max_usd
        self.model = spec.teacher.model
        self.largest = 0.0
        self.pricing: PricingSnapshot | None = None
        #: Why the snapshot cannot bound a request (None when it can).
        self.unpriced: str | None = None
        try:
            pricing = load_pricing()
            pricing.price_for(self.model, spec.teacher.provider)
        except PricingError as exc:
            self.unpriced = f"the pricing snapshot is not readable ({exc})"
        except KeyError:
            self.unpriced = f"the pricing snapshot has no price for the teacher model {self.model}"
        else:
            self.pricing = pricing

    def observe(self, added: float) -> None:
        """Record what one request added to the spend."""
        self.largest = max(self.largest, added)

    def bound(self, body: Mapping[str, Any]) -> float:
        """The most sending ``body`` can add to the spend."""
        bound = self.largest
        if self.pricing is not None:
            price = self.pricing.price_for(self.model, pinned_provider(body))
            bound = max(bound, worst_case_cost(body, price))
        return bound

    def refuse(self, spent: float, body: Mapping[str, Any]) -> str | None:
        """Why ``body`` must not be sent with ``spent`` already spent (None when it fits under the cap)."""
        bound = self.bound(body)
        if spent < self.max_usd and _nano(spent) + math.ceil(bound * 1e9 - 1e-6) <= _nano(self.max_usd):
            return None
        return f"--max-usd {self.max_usd:g} would be crossed (${spent:.6f} spent, next request up to ${bound:.6f})"


def _command(url: str, spec: TaskSpec, n: int, warmup: int, yes: bool, max_usd: float | None) -> str:
    parts = ["taskdistill", "bench", "--url", url, "--task", spec.task, "--n", str(n), "--warmup", str(warmup)]
    if yes:
        parts.append("--yes")
    if max_usd is not None:
        parts += ["--max-usd", f"{max_usd:g}"]
    return " ".join(parts)


def run_bench(
    url: str,
    spec: TaskSpec,
    *,
    n: int = 300,
    warmup: int = 20,
    yes: bool = False,
    max_usd: float | None = None,
    token: str | None = None,
    out: Path | str | None = None,
    store: Store | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Run the bench against the server at ``url`` and write its JSON; return it.

    The JSON goes to ``$TASKDISTILL_HOME/<task>/bench/bench_<utc>.json`` and, when ``out`` is given, to ``out``
    too. Raises :class:`BenchError` when the server is unreachable or serves another task, there are too few test
    inputs, an escalated response came from a replayed teacher, or ``max_usd`` is set but the spend cannot be
    measured; ``SpendNotConfirmed`` when the projected (or unmeasurable) spend needs ``--yes``.
    """
    if n < 1:
        raise BenchError("--n must be at least 1")
    if warmup < 0:
        raise BenchError("--warmup must not be negative")
    if max_usd is not None and max_usd <= 0:
        raise BenchError("--max-usd must be positive")
    base = _base_url(url)
    token = token or os.environ.get(TOKEN_ENV) or None
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    notes: list[str] = []

    inputs, skipped = bench_inputs(spec, store if store is not None else Store())
    if skipped:
        notes.append(f"{skipped} test rows have no raw input in the store and were skipped")
    if len(inputs) <= warmup:
        raise BenchError(
            f"the test split has {len(inputs)} inputs with a raw input; the bench needs more than the {warmup} "
            "warm-up requests"
        )
    n_timed = min(n, len(inputs) - warmup)
    if n_timed < n:
        notes.append(f"only {len(inputs) - warmup} distinct test inputs after the warm-up set: n reduced to {n_timed}")

    own_client = client is None
    http = client if client is not None else httpx.Client(timeout=DEFAULT_TIMEOUT)
    try:
        health = _healthz(http, base, headers)
        _check_task(health, spec, base)
        threshold = _float(health.get("threshold"))
        mode = "student_only" if threshold == 0.0 else "cascade"
        if health.get("teacher") == "replay" and mode == "cascade":
            notes.append("the server's teacher is a replay: the bench fails at the first escalated request")
        endpoint = f"{base}/v1/chat/completions"
        ledger = Ledger()
        spent_start = _serve_spent(ledger, spec.task)
        sent = 0
        live_answers = 0
        cap = _SpendCap(spec, max_usd) if max_usd is not None else None
        if cap is not None and cap.unpriced is not None:
            notes.append(
                f"--max-usd bounds each request by the most one request has cost so far: {cap.unpriced}, so the "
                "first charged request is not bounded"
            )

        def spent_now() -> float:
            spent = _serve_spent(ledger, spec.task) - spent_start
            if max_usd is not None and live_answers and spent <= 0:
                raise BenchError(
                    f"--max-usd {max_usd:g} cannot be enforced: {_unmeasured_reason(live_answers)}. Run the bench "
                    "with the server's TASKDISTILL_HOME"
                )
            return spent

        def refused(body: Mapping[str, Any]) -> str | None:
            """Why the cap forbids sending ``body`` now (None without a cap or when it fits)."""
            return None if cap is None else cap.refuse(spent_now(), body)

        def send(body: dict[str, Any], index: int) -> dict[str, Any]:
            nonlocal sent, live_answers
            before = _serve_spent(ledger, spec.task) if cap is not None else 0.0
            result = _send(http, endpoint, body, headers)
            sent += 1
            if cap is not None:
                cap.observe(_serve_spent(ledger, spec.task) - before)
            _check_live(result, index)
            live_answers += _live_answer(result)
            return result

        for index, (_, raw) in enumerate(inputs[:warmup]):
            body = build_teacher_request(spec, raw)
            reason = refused(body)
            if reason is not None:
                raise BenchError(f"{reason}; stopped during the warm-up, after {sent} requests")
            send(body, index)

        state = machine_state()
        requests: list[dict[str, Any]] = []
        projection: dict[str, Any] | None = None
        stopped: str | None = None
        for offset, (input_hash, raw) in enumerate(inputs[warmup : warmup + n_timed]):
            body = build_teacher_request(spec, raw)
            reason = refused(body)
            if reason is not None:
                stopped = f"{reason}; stopped after {sent} requests"
                notes.append(stopped)
                break
            result = send(body, warmup + offset)
            requests.append({"i": offset, "input_hash": input_hash, **result})
            spent = spent_now()
            if projection is None and offset + 1 == PROJECTION_SAMPLE and n_timed > PROJECTION_SAMPLE:
                total = warmup + n_timed
                measurable = not (live_answers and spent <= 0)
                projection = {
                    "after_requests": sent,
                    "spent_usd": spent if measurable else None,
                    "projected_usd": spent / sent * total if measurable else None,
                    "total_requests": total,
                    "threshold_usd": DEFAULT_SPEND_THRESHOLD,
                    "unmeasured": None if measurable else _unmeasured_reason(live_answers),
                }
                if measurable:
                    confirm_spend(projection["projected_usd"], yes)
                elif not yes:
                    raise SpendNotConfirmed(
                        f"{_unmeasured_reason(live_answers)}; the spend of the remaining requests cannot be "
                        "projected. Run the bench with the server's TASKDISTILL_HOME, or re-run with --yes to confirm"
                    )
        state_after = machine_state()
    finally:
        if own_client:
            http.close()

    spend = _serve_spent(ledger, spec.task) - spent_start
    ok = [r for r in requests if r.get("status") == 200]
    answered = [r for r in requests if r.get("status") is not None]
    routes = Counter(str(r.get("route")) for r in answered if r.get("route"))
    reasons = Counter(str(r.get("reason")) for r in answered if r.get("reason"))
    escalated = [r for r in answered if _escalated(r)]
    failed_escalations = sum(1 for r in escalated if r.get("route") != "teacher")
    errors = len(requests) - len(ok)
    if requests and not ok:
        raise BenchError(f"every timed request failed (first error: {requests[0].get('error')})")
    spend_note: str | None = None
    if live_answers and spend <= 0:
        spend_note = _unmeasured_reason(live_answers)
        notes.append(f"spend not measured: {spend_note}")
    result_json: dict[str, Any] = {
        "task": spec.task,
        "url": base,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": _command(url, spec, n, warmup, yes, max_usd),
        "mode": mode,
        "threshold": threshold,
        "teacher": health.get("teacher"),
        "run_id": health.get("run_id") or health.get("run"),
        "healthz": health,
        "concurrency": 1,
        "warmup": warmup,
        "n_requested": n,
        "n": len(requests),
        "latency_ms": _summary([float(r["latency_ms"]) for r in ok]),
        "escalation_rate": len(escalated) / len(answered) if answered else None,
        "escalations": len(escalated),
        "failed_escalations": failed_escalations,
        "route_counts": dict(sorted(routes.items())),
        "reasons": dict(sorted(reasons.items())),
        "teacher_modes": dict(Counter(str(r["teacher"]) for r in ok if r.get("teacher"))),
        "errors": errors,
        "spend_usd": None if spend_note else spend,
        "spend_source": f"ledger delta, phase '{SERVE_PHASE}' of task '{spec.task}'",
        "spend_note": spend_note,
        "live_teacher_answers": live_answers,
        "projection": projection,
        "max_usd": max_usd,
        "stopped": stopped,
        "skipped_no_raw_input": skipped,
        "machine_state": state,
        "machine_state_after": state_after,
        "hardware": hardware_info(),
        "notes": notes,
        "requests": requests,
    }
    text = json.dumps(result_json, indent=2, ensure_ascii=False) + "\n"
    targets = [_bench_path(spec.task)] + ([Path(out)] if out is not None else [])
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return result_json
