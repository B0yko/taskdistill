"""SQLite store (WAL mode) for captured traffic, imported logs and served requests.

No table has a column for HTTP headers: the proxy and the server never persist any header.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taskdistill import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS captures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    ts REAL NOT NULL,
    source TEXT NOT NULL,             -- proxy | import
    request_key TEXT,
    request_body TEXT,
    response_body TEXT,
    status INTEGER,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    upstream_model TEXT,
    captured INTEGER NOT NULL,        -- 0 for streamed or failed requests
    error TEXT
);
CREATE INDEX IF NOT EXISTS captures_task ON captures(task);

CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    ts REAL NOT NULL,
    format TEXT NOT NULL,             -- pairs | inputs
    input TEXT NOT NULL,
    output TEXT,
    gold TEXT,                        -- JSON-encoded
    meta TEXT                         -- JSON-encoded object
);
CREATE INDEX IF NOT EXISTS imports_task ON imports(task);

CREATE TABLE IF NOT EXISTS served (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    ts REAL NOT NULL,
    route TEXT NOT NULL,              -- student | teacher | student-fallback | error
    reason TEXT,                      -- low_confidence | unsupported | input_unparsed
    confidence REAL,
    student_ms REAL,
    teacher_ms REAL,
    total_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    teacher_cost_usd REAL,
    teacher_mode TEXT,                -- live | replay
    status INTEGER,
    request_model TEXT
);
CREATE INDEX IF NOT EXISTS served_task_ts ON served(task, ts);
"""


@dataclass(frozen=True)
class CaptureRow:
    id: int
    task: str
    ts: float
    source: str
    request_key: str | None
    request_body: str | None
    response_body: str | None
    status: int | None
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    upstream_model: str | None
    captured: bool
    error: str | None


@dataclass(frozen=True)
class ImportRow:
    id: int
    task: str
    ts: float
    format: str
    input: str
    output: str | None
    gold: Any
    meta: dict[str, Any]


@dataclass(frozen=True)
class ServedRow:
    id: int
    task: str
    ts: float
    route: str
    reason: str | None
    confidence: float | None
    student_ms: float | None
    teacher_ms: float | None
    total_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    teacher_cost_usd: float | None
    teacher_mode: str | None
    status: int | None
    request_model: str | None


class Store:
    """Thread-safe access to the workspace store."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else paths.store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _insert(self, table: str, row: dict[str, Any]) -> int:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._lock, self._connect() as conn:
            cur = conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
            return int(cur.lastrowid or 0)

    def columns(self, table: str) -> list[str]:
        with self._connect() as conn:
            return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]

    # captures -----------------------------------------------------------------------------------
    def add_capture(
        self,
        *,
        task: str,
        source: str,
        request_key: str | None,
        request_body: str | None,
        response_body: str | None,
        status: int | None,
        latency_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = None,
        upstream_model: str | None = None,
        captured: bool = True,
        error: str | None = None,
    ) -> int:
        return self._insert(
            "captures",
            {
                "task": task,
                "ts": time.time(),
                "source": source,
                "request_key": request_key,
                "request_body": request_body,
                "response_body": response_body,
                "status": status,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
                "upstream_model": upstream_model,
                "captured": int(captured),
                "error": error,
            },
        )

    def iter_captures(self, task: str) -> Iterator[CaptureRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, task, ts, source, request_key, request_body, response_body, status, latency_ms, "
                "prompt_tokens, completion_tokens, cost_usd, upstream_model, captured, error "
                "FROM captures WHERE task = ? ORDER BY id",
                (task,),
            ).fetchall()
        for r in rows:
            values = list(r)
            values[13] = bool(values[13])
            yield CaptureRow(*values)

    def count_captures(self, task: str) -> dict[str, int]:
        with self._connect() as conn:
            total, captured = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(captured), 0) FROM captures WHERE task = ?", (task,)
            ).fetchone()
        return {"total": int(total), "captured": int(captured), "not_captured": int(total) - int(captured)}

    def add_captures(self, task: str, rows: list[dict[str, Any]]) -> int:
        """Insert many captures in one transaction; each row holds ``add_capture``'s keyword arguments."""
        now = time.time()
        cols = (
            "request_key", "request_body", "response_body", "status", "latency_ms", "prompt_tokens",
            "completion_tokens", "cost_usd", "upstream_model",
        )  # fmt: skip
        values = [
            (task, now, r["source"], *(r.get(c) for c in cols), int(r.get("captured", True)), r.get("error"))
            for r in rows
        ]
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO captures (task, ts, source, " + ", ".join(cols) + ", captured, error) "
                "VALUES (" + ", ".join("?" for _ in range(len(cols) + 5)) + ")",
                values,
            )
        return len(values)

    # imports ------------------------------------------------------------------------------------
    def add_imports(self, task: str, fmt: str, rows: list[dict[str, Any]]) -> int:
        now = time.time()
        values = [
            (
                task,
                now,
                fmt,
                r["input"],
                r.get("output"),
                None if r.get("gold") is None else json.dumps(r["gold"], ensure_ascii=False),
                json.dumps(r.get("meta") or {}, ensure_ascii=False, sort_keys=True),
            )
            for r in rows
        ]
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO imports (task, ts, format, input, output, gold, meta) VALUES (?, ?, ?, ?, ?, ?, ?)",
                values,
            )
        return len(values)

    def iter_imports(self, task: str) -> Iterator[ImportRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, task, ts, format, input, output, gold, meta FROM imports WHERE task = ? ORDER BY id",
                (task,),
            ).fetchall()
        for r in rows:
            yield ImportRow(
                id=r[0],
                task=r[1],
                ts=r[2],
                format=r[3],
                input=r[4],
                output=r[5],
                gold=None if r[6] is None else json.loads(r[6]),
                meta=json.loads(r[7]) if r[7] else {},
            )

    # served -------------------------------------------------------------------------------------
    def add_served(self, **row: Any) -> int:
        row.setdefault("ts", time.time())
        return self._insert("served", row)

    def iter_served(self, task: str, since: float | None = None) -> Iterator[ServedRow]:
        query = (
            "SELECT id, task, ts, route, reason, confidence, student_ms, teacher_ms, total_ms, prompt_tokens, "
            "completion_tokens, teacher_cost_usd, teacher_mode, status, request_model FROM served WHERE task = ?"
        )
        params: list[Any] = [task]
        if since is not None:
            query += " AND ts >= ?"
            params.append(since)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY id", params).fetchall()
        for r in rows:
            yield ServedRow(*r)
