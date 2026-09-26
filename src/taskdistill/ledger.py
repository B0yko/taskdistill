"""Spend ledger: reserve the worst case before every teacher call, settle it with the real charge afterwards.

The ledger is a SQLite file in the workspace (WAL mode). Amounts are stored as integer nano-dollars, so sums
are exact and float rounding can never let a reservation slip past a cap. A reservation is atomic across
threads (a lock) and across processes (``BEGIN IMMEDIATE``): the committed spend (settled amounts plus open
reservations) is read and the new reservation inserted inside one write transaction.

Three caps apply, and a call that would cross any of them is refused with :class:`BudgetExceeded`:

- the global cap ``TASKDISTILL_BUDGET_USD`` (default 5.00) over everything in the ledger;
- the optional per-task cap ``budget.usd_cap`` over the task's spend;
- the optional per-command cap ``--max-usd`` over the spend of one run.

A reservation must bound what the request sent can cost: :func:`reservation_cost` refuses (with
:class:`UnboundedCompletion`) a body that sets no completion limit when the model's maximum completion length is
not known either, instead of assuming a limit the teacher never receives.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from taskdistill import paths
from taskdistill.teacher.base import BudgetExceeded
from taskdistill.teacher.request_key import canonical_json

log = logging.getLogger("taskdistill.ledger")

ENV_BUDGET = "TASKDISTILL_BUDGET_USD"
DEFAULT_BUDGET_USD = 5.00
#: Completion bound a *projection* assumes for a request that sets no ``max_tokens``; never reserved for a call.
DEFAULT_MAX_TOKENS = 4096
#: Per-message allowance added to the prompt byte count (role, separators, chat-template tokens).
PER_MESSAGE_TOKENS = 16
NANO = 1_000_000_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    settled_ts REAL,
    task TEXT NOT NULL,
    phase TEXT NOT NULL,
    run_id TEXT NOT NULL,
    model TEXT,
    request_key TEXT,
    reserved_nusd INTEGER NOT NULL,
    actual_nusd INTEGER,
    status TEXT NOT NULL              -- open | settled | released
);
CREATE INDEX IF NOT EXISTS ledger_task ON ledger(task);
CREATE INDEX IF NOT EXISTS ledger_run ON ledger(run_id);
CREATE INDEX IF NOT EXISTS ledger_status ON ledger(status);
"""

#: Committed spend of a row: the settled amount, or the reserved amount while the reservation is open.
_COMMITTED = "COALESCE(SUM(CASE status WHEN 'settled' THEN actual_nusd WHEN 'open' THEN reserved_nusd ELSE 0 END), 0)"


class UnboundedCompletion(BudgetExceeded):
    """A call sets no completion limit and the model's maximum is unknown, so no worst case bounds its cost."""


class Price(Protocol):
    """USD per prompt token, per completion token and per request (``teacher.pricing.ModelPrice``)."""

    @property
    def prompt(self) -> float: ...

    @property
    def completion(self) -> float: ...

    @property
    def request(self) -> float: ...


def _to_nano_ceil(usd: float) -> int:
    return max(0, math.ceil(usd * NANO - 1e-6))


def _to_nano(usd: float) -> int:
    return max(0, round(usd * NANO))


def _usd(nano: int) -> float:
    return nano / NANO


def _fmt_cap(usd: float) -> str:
    return f"{usd:.2f}" if round(usd, 2) == usd else f"{usd:.6g}"


def _check_amount(usd: float, what: str) -> None:
    if not math.isfinite(usd) or usd < 0:
        raise ValueError(f"{what} must be a finite, non-negative USD amount, got {usd!r}")


def default_global_cap() -> float:
    """The global cap from ``TASKDISTILL_BUDGET_USD`` (default 5.00)."""
    raw = os.environ.get(ENV_BUDGET)
    if raw is None or not raw.strip():
        return DEFAULT_BUDGET_USD
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_BUDGET} must be a number of US dollars, got {raw!r}") from exc
    _check_amount(value, ENV_BUDGET)
    return value


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "") for part in content if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return ""


def _json_bytes(value: Any) -> int:
    return 0 if value is None else len(canonical_json(value))


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _utf8_bytes(text: str) -> int:
    """UTF-8 length of ``text``, a lone UTF-16 surrogate (which UTF-8 cannot encode) counted as 3 bytes.

    A client that cut a string inside a surrogate pair sends such a character as a JSON escape; however the
    provider decodes it (dropped, or U+FFFD, itself 3 bytes), it is never more than 3 tokens.
    """
    return len(text.encode("utf-8", errors="surrogatepass"))


def prompt_upper_bound(body: Mapping[str, Any]) -> int:
    """Prompt token upper bound: UTF-8 bytes of every message's content plus 16 per message.

    Tool definitions (``tools``, ``functions``) and assistant ``tool_calls``/``function_call`` are billed as
    prompt tokens too, so the UTF-8 bytes of their canonical JSON are added when a body carries them. Any string
    has a bound, a lone surrogate included; :class:`ValueError` only for NaN or an infinity in those JSON fields.
    """
    messages = body.get("messages") or []
    total = _json_bytes(body.get("tools")) + _json_bytes(body.get("functions"))
    for message in messages:
        if not isinstance(message, Mapping):
            total += PER_MESSAGE_TOKENS
            continue
        total += _utf8_bytes(_content_text(message.get("content"))) + PER_MESSAGE_TOKENS
        total += _json_bytes(message.get("tool_calls")) + _json_bytes(message.get("function_call"))
    return total


def completion_limit(body: Mapping[str, Any]) -> int | None:
    """The explicit per-choice completion limit: ``max_tokens``, else ``max_completion_tokens``, else None."""
    for field in ("max_tokens", "max_completion_tokens"):
        value = _positive_int(body.get(field))
        if value is not None:
            return value
    return None


def choice_count(body: Mapping[str, Any]) -> int:
    """Number of choices a body asks for (``n``, default 1)."""
    return _positive_int(body.get("n")) or 1


def completion_upper_bound(body: Mapping[str, Any], max_completion_tokens: int | None = None) -> int:
    """The per-choice completion limit times ``n``.

    The limit is ``max_tokens`` (or ``max_completion_tokens``); for a body without one, the model's
    ``max_completion_tokens`` when given, else :data:`DEFAULT_MAX_TOKENS` (a projection estimate only: a call
    is reserved through :func:`reservation_cost`, which never assumes it).
    """
    limit = completion_limit(body) or _positive_int(max_completion_tokens) or DEFAULT_MAX_TOKENS
    return limit * choice_count(body)


def worst_case_cost(body: Mapping[str, Any], price: Price, *, max_completion_tokens: int | None = None) -> float:
    """The most a chat-completions call can cost.

    Prompt bound x prompt price + completion bound (:func:`completion_upper_bound`) x completion price + the
    per-request fee. A token is never shorter than one UTF-8 byte, so the byte count bounds the prompt tokens of
    any byte-level BPE.
    """
    completion = completion_upper_bound(body, max_completion_tokens)
    return prompt_upper_bound(body) * price.prompt + completion * price.completion + price.request


def reservation_cost(
    body: Mapping[str, Any], price: Price, *, model: str | None = None, max_completion_tokens: int | None = None
) -> float:
    """The worst case to reserve before sending ``body`` exactly as given.

    Like :func:`worst_case_cost`, but a body that sets no completion limit is bounded only by the model's
    ``max_completion_tokens``: without it the teacher may generate any length, so the call is refused with
    :class:`UnboundedCompletion` rather than reserved at a guess.
    """
    if completion_limit(body) is None and _positive_int(max_completion_tokens) is None:
        raise UnboundedCompletion(
            f"teacher call refused (model {model}): the request sets no max_tokens and the pricing snapshot has no "
            "maximum completion length for the model, so no worst case bounds its cost; set max_tokens (or "
            "max_completion_tokens) in the request"
        )
    return worst_case_cost(body, price, max_completion_tokens=max_completion_tokens)


class Ledger:
    """Budgeted spend accounting shared by every teacher call of every process using the workspace."""

    def __init__(self, path: Path | str | None = None, global_cap: float | None = None) -> None:
        self.path = Path(path) if path is not None else paths.ledger_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.global_cap = default_global_cap() if global_cap is None else float(global_cap)
        _check_amount(self.global_cap, "global cap")
        self._lock = threading.Lock()
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout = 60000")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One ``BEGIN IMMEDIATE`` transaction under the thread lock: no other writer can interleave."""
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    @staticmethod
    def _committed(conn: sqlite3.Connection, where: str = "", params: tuple[Any, ...] = ()) -> int:
        row = conn.execute(f"SELECT {_COMMITTED} FROM ledger {where}", params).fetchone()
        return int(row[0])

    def reserve(
        self,
        amount: float,
        *,
        task: str,
        phase: str,
        run_id: str,
        model: str | None,
        run_cap: float | None = None,
        task_cap: float | None = None,
        request_key: str | None = None,
    ) -> int:
        """Reserve ``amount`` USD; return the reservation id or raise :class:`BudgetExceeded`."""
        _check_amount(amount, "reservation")
        for name, cap in (("run cap", run_cap), ("task cap", task_cap)):
            if cap is not None:
                _check_amount(cap, name)
        need = _to_nano_ceil(amount)
        with self._write() as conn:
            checks: list[tuple[str, float, int]] = [
                (f"global cap {ENV_BUDGET}={_fmt_cap(self.global_cap)}", self.global_cap, self._committed(conn))
            ]
            if task_cap is not None:
                committed = self._committed(conn, "WHERE task = ?", (task,))
                checks.append((f"task cap budget.usd_cap={_fmt_cap(task_cap)} (task '{task}')", task_cap, committed))
            if run_cap is not None:
                committed = self._committed(conn, "WHERE run_id = ?", (run_id,))
                checks.append((f"run cap --max-usd={_fmt_cap(run_cap)}", run_cap, committed))
            crossed = [
                f"{label} (committed ${_usd(committed):.6f})"
                for label, cap, committed in checks
                if committed + need > _to_nano(cap)
            ]
            if crossed:
                raise BudgetExceeded(
                    f"teacher call refused ({task}/{phase}, model {model}): its worst case ${_usd(need):.6f} "
                    f"would exceed the {'; the '.join(crossed)}"
                )
            cur = conn.execute(
                "INSERT INTO ledger (ts, task, phase, run_id, model, request_key, reserved_nusd, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'open')",
                (time.time(), task, phase, run_id, model, request_key, need),
            )
            return int(cur.lastrowid or 0)

    def _close(self, res_id: int, status: str, actual_nano: int) -> None:
        with self._write() as conn:
            row = conn.execute("SELECT status, reserved_nusd FROM ledger WHERE id = ?", (res_id,)).fetchone()
            if row is None:
                raise KeyError(f"no ledger reservation with id {res_id}")
            if row[0] != "open":
                raise ValueError(f"ledger reservation {res_id} is already {row[0]}")
            if status == "settled" and actual_nano > int(row[1]):
                log.warning(
                    "reservation %s settled at $%.6f, above its worst case $%.6f",
                    res_id,
                    _usd(actual_nano),
                    _usd(int(row[1])),
                )
            conn.execute(
                "UPDATE ledger SET status = ?, actual_nusd = ?, settled_ts = ? WHERE id = ?",
                (status, actual_nano, time.time(), res_id),
            )

    def settle(self, res_id: int, actual_usd: float) -> None:
        """Replace an open reservation by the call's actual charge."""
        _check_amount(actual_usd, "settlement")
        self._close(res_id, "settled", _to_nano(actual_usd))

    def release(self, res_id: int) -> None:
        """Drop an open reservation whose request provably cost nothing (an HTTP error response)."""
        self._close(res_id, "released", 0)

    def _where(self, task: str | None, phase: str | None, run_id: str | None) -> tuple[str, tuple[Any, ...]]:
        clauses, params = ["status = 'settled'"], []
        for column, value in (("task", task), ("phase", phase), ("run_id", run_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        return "WHERE " + " AND ".join(clauses), tuple(params)

    def spent(self, task: str | None = None, phase: str | None = None, run_id: str | None = None) -> float:
        """Settled spend in USD, optionally filtered."""
        where, params = self._where(task, phase, run_id)
        with self._connect() as conn:
            row = conn.execute(f"SELECT COALESCE(SUM(actual_nusd), 0) FROM ledger {where}", params).fetchone()
        return _usd(int(row[0]))

    def committed(self, task: str | None = None, run_id: str | None = None) -> float:
        """Settled spend plus open reservations in USD (what the caps are checked against)."""
        clauses, params = [], []
        for column, value in (("task", task), ("run_id", run_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            return _usd(self._committed(conn, where, tuple(params)))

    def open_reservations(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM ledger WHERE status = 'open'").fetchone()
        return int(row[0])

    def summary(self) -> dict[str, Any]:
        """Settled spend by task, phase and task/phase, the number of settled calls and the global cap."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT task, phase, COUNT(*), SUM(actual_nusd) FROM ledger WHERE status = 'settled' "
                "GROUP BY task, phase ORDER BY task, phase"
            ).fetchall()
            open_row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(reserved_nusd), 0) FROM ledger WHERE status = 'open'"
            ).fetchone()
            released = conn.execute("SELECT COUNT(*) FROM ledger WHERE status = 'released'").fetchone()
        by_task: dict[str, int] = {}
        by_phase: dict[str, int] = {}
        by_task_phase: dict[str, dict[str, float]] = {}
        calls = 0
        total = 0
        for task, phase, count, nano in rows:
            nano = int(nano or 0)
            calls += int(count)
            total += nano
            by_task[task] = by_task.get(task, 0) + nano
            by_phase[phase] = by_phase.get(phase, 0) + nano
            by_task_phase.setdefault(task, {})[phase] = _usd(nano)
        return {
            "total": _usd(total),
            "by_task": {k: _usd(v) for k, v in sorted(by_task.items())},
            "by_phase": {k: _usd(v) for k, v in sorted(by_phase.items())},
            "by_task_phase": by_task_phase,
            "calls": calls,
            "released": int(released[0]),
            "open_reservations": int(open_row[0]),
            "open_reserved": _usd(int(open_row[1])),
            "cap": self.global_cap,
        }

    def export_json(self, path: Path | str) -> dict[str, Any]:
        """Write :meth:`summary` (plus the export date) as JSON, e.g. ``reports/spend.json``."""
        data = {"exported": datetime.now(UTC).date().isoformat(), **self.summary()}
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return data
