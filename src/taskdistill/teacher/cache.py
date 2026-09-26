"""On-disk teacher response cache keyed by the request key, so re-runs of curate and the bake-off cost nothing.

Only curate, the bake-off and demo labelling use it; ``serve`` never reads it.

The request key covers the fields that determine the output for a given routing. Body fields outside it that
can still change the output (provider routing, reasoning controls, other extra body) are hashed into a
context stored next to each row: a lookup whose context differs is a miss, so changing the pinned provider or
the reasoning setting makes the next run call the teacher again instead of reusing the old outputs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from taskdistill import paths
from taskdistill.teacher.base import TeacherResult
from taskdistill.teacher.request_key import KEY_FIELDS, canonical_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    output TEXT,
    response TEXT NOT NULL,
    usage TEXT NOT NULL,
    latency_ms REAL,
    provider TEXT,
    finish_reason TEXT,
    truncated INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    context_sha256 TEXT
);
"""

_COLUMNS = "key, output, response, usage, latency_ms, provider, finish_reason, truncated, created, context_sha256"

#: Body fields left out of the context: the request key's own fields and those that never change the output.
CONTEXT_EXCLUDED = frozenset({*KEY_FIELDS, "stream", "stream_options", "user"})


def request_context(body: Mapping[str, Any], *, max_tokens_default: int | None = None) -> str:
    """SHA-256 of the canonical JSON of a body's output-changing fields outside the request key.

    ``max_tokens_default`` is the completion limit a client added to a body that set none; it bounds the
    output, so it is part of the context. Absent fields and explicit ``null`` are equivalent.
    """
    subset: dict[str, Any] = {k: v for k, v in body.items() if k not in CONTEXT_EXCLUDED and v is not None}
    if max_tokens_default is not None:
        subset = {"body": subset, "max_tokens_default": max_tokens_default}
    return hashlib.sha256(canonical_json(subset)).hexdigest()


def _row_to_result(row: tuple[Any, ...]) -> TeacherResult:
    return TeacherResult(
        key=row[0],
        output=row[1],
        response=json.loads(row[2]),
        usage=json.loads(row[3]),
        latency_ms=row[4],
        provider=row[5],
        finish_reason=row[6],
        created=row[8],
        source="cache",
        cost_usd=0.0,
        attempts=0,
        truncated=bool(row[7]),
        extra={"context_sha256": row[9]} if row[9] is not None else {},
    )


class ResponseCache:
    """SQLite (WAL) cache of successful teacher responses."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else paths.cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(responses)")}
            if "context_sha256" not in columns:
                conn.execute("ALTER TABLE responses ADD COLUMN context_sha256 TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, key: str, context: str | None = None) -> TeacherResult | None:
        """The cached result for ``key`` (source ``cache``, cost 0), or None.

        With ``context`` (see :func:`request_context`), a row stored under another context, or none, is a miss.
        """
        with self._connect() as conn:
            row = conn.execute(f"SELECT {_COLUMNS} FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None or (context is not None and row[9] != context):
            return None
        return _row_to_result(row)

    def put(self, result: TeacherResult, context: str | None = None) -> None:
        """Store ``result`` under its key, replacing any row there (including one of another context)."""
        values = (
            result.key,
            result.output,
            json.dumps(result.response, ensure_ascii=False, sort_keys=True),
            json.dumps(result.usage, ensure_ascii=False, sort_keys=True),
            result.latency_ms,
            result.provider,
            result.finish_reason,
            int(result.truncated),
            result.created,
            context,
        )
        with self._lock, self._connect() as conn:
            conn.execute(f"INSERT OR REPLACE INTO responses ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM responses WHERE key = ?", (key,)).fetchone() is not None

    def __len__(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def iter_results(self, keys: list[str] | None = None, context: str | None = None) -> Iterator[TeacherResult]:
        """Cached results ordered by key (all of them, or those of ``keys`` that are cached).

        With ``context``, rows stored under another context (or none) are skipped.
        """
        with self._connect() as conn:
            if keys is None:
                rows = conn.execute(f"SELECT {_COLUMNS} FROM responses ORDER BY key").fetchall()
            else:
                rows = []
                for key in sorted(set(keys)):
                    row = conn.execute(f"SELECT {_COLUMNS} FROM responses WHERE key = ?", (key,)).fetchone()
                    if row is not None:
                        rows.append(row)
        for row in rows:
            if context is None or row[9] == context:
                yield _row_to_result(row)
