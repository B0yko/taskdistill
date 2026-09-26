"""Import existing logs into the store (``taskdistill capture --import <file> --format openai|pairs|inputs``).

Formats, one JSON object per line (``.jsonl`` or gzip-compressed ``.jsonl.gz``; blank lines are skipped):

- ``openai``: ``{"request": {...}, "response": {...}}``, the format ``export`` writes. Stored as captures.
- ``pairs``: ``{"input": str, "output": str, "gold"?: any, "meta"?: object}``. ``output`` may also be a JSON
  object (an extraction result), which is stored as compact JSON.
- ``inputs``: ``{"input": str, "gold"?: any, "meta"?: object}``; the teacher labels these during curate.

``meta`` is kept as given, so a predefined split such as ``meta.split`` reaches curate. When set, the predefined
split field (``predefined``: the task's ``curate.split.predefined``, by default ``meta.split`` as in every template)
must hold train, valid or test, or an alias curate accepts (validation, val, dev): curate cannot place a row with any
other value, and an import row cannot be removed from the store once written. The whole file is validated before
anything is written, including that every string can be stored as UTF-8; the first bad line aborts the import with
its line number. ``pairs`` and ``inputs`` rows are written in one transaction, and so are ``openai`` rows when the
store offers ``add_captures``.

``completion_problem`` is the one shape check shared with the proxy and export: a pair the proxy stores as
captured is one export writes and import accepts.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict, get_args

from taskdistill.curate.merge import CurateError, canonical_split, meta_get, meta_keys
from taskdistill.store import Store
from taskdistill.teacher.request_key import request_key

Format = Literal["openai", "pairs", "inputs"]
FORMATS: tuple[str, ...] = get_args(Format)

_ALLOWED_KEYS: dict[str, tuple[str, ...]] = {
    "openai": ("request", "response"),
    "pairs": ("input", "output", "gold", "meta"),
    "inputs": ("input", "gold", "meta"),
}
_GZIP_MAGIC = b"\x1f\x8b"
UPSTREAM_ERROR_IN_2XX = "upstream error in a 2xx response"
UNPAIRED_SURROGATE = "contains an unpaired UTF-16 surrogate escape"


class ImportFormatError(ValueError):
    """A line of an import file does not match the declared format; the message names the line."""


class ImportCounts(TypedDict):
    read: int
    imported: int
    blank: int
    format: str


def _open_lines(path: Path) -> Iterator[tuple[int, bytes]]:
    with open(path, "rb") as fh:
        compressed = fh.read(2) == _GZIP_MAGIC
    if not compressed:
        with open(path, "rb") as fh:
            yield from enumerate(fh, start=1)
        return
    lineno = 0
    try:
        with gzip.open(path, "rb") as gz:
            for lineno, raw in enumerate(gz, start=1):
                yield lineno, raw
    except (OSError, EOFError) as exc:
        raise ImportFormatError(f"{path.name}:{lineno + 1}: cannot decompress gzip data ({exc})") from None


def _decode(raw: bytes, lineno: int, where: str) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ImportFormatError(f"{where}:{lineno}: not valid UTF-8 ({exc.reason} at byte {exc.start})") from None
    return text.removeprefix("\ufeff") if lineno == 1 else text


def utf8_safe(value: Any) -> bool:
    """False when a JSON value holds an unpaired surrogate (an escape like ``\\ud83d``) that UTF-8 cannot encode."""
    try:
        json.dumps(value, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def completion_problem(request: Any, response: Any) -> str | None:
    """Why a request/response pair is not a usable chat completion, or None when it is.

    The request needs a non-empty ``messages`` array. The response needs a non-empty ``choices`` array of
    objects and no upstream error: OpenRouter reports errors raised after generation started with HTTP 200, as a
    top-level ``error`` or as a choice with ``error`` or ``finish_reason: "error"``. Both bodies must be storable
    as UTF-8.
    """
    if not isinstance(request, Mapping):
        return '"request" must be a JSON object'
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return '"request.messages" must be a non-empty array'
    if not isinstance(response, Mapping):
        return '"response" must be a JSON object'
    if response.get("error") is not None:
        return UPSTREAM_ERROR_IN_2XX
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return '"response.choices" must be a non-empty array'
    for choice in choices:
        if not isinstance(choice, Mapping):
            return '"response.choices" must hold objects'
        if choice.get("error") is not None or choice.get("finish_reason") == "error":
            return UPSTREAM_ERROR_IN_2XX
    for name, body in (("request", request), ("response", response)):
        if not utf8_safe(body):
            return f'"{name}" {UNPAIRED_SURROGATE}'
    return None


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    return {dict: "object", list: "array", str: "string", bool: "boolean", int: "number", float: "number"}.get(
        type(value), type(value).__name__
    )


class _SplitField:
    """The predefined split field (a meta path such as ``meta.split``) and its keys inside ``meta``."""

    def __init__(self, path: str) -> None:
        try:
            self.keys = meta_keys(path)
        except CurateError as exc:
            raise ValueError(str(exc)) from None
        self.path = path


class _Line:
    """Validation helpers bound to one line, so every error carries its location."""

    def __init__(self, where: str, lineno: int, fmt: str) -> None:
        self.prefix = f"{where}:{lineno}"
        self.fmt = fmt

    def fail(self, message: str) -> ImportFormatError:
        return ImportFormatError(f"{self.prefix}: {message}")

    def parse(self, text: str) -> dict[str, Any]:
        try:
            obj = json.loads(text)
        except (ValueError, RecursionError) as exc:
            hint = " (expected JSON Lines: one object per line)" if text.lstrip().startswith("[") else ""
            raise self.fail(f"invalid JSON: {exc}{hint}") from None
        if not isinstance(obj, dict):
            raise self.fail(f"expected a JSON object, got {_type_name(obj)}")
        allowed = _ALLOWED_KEYS[self.fmt]
        unknown = sorted(set(obj) - set(allowed))
        if unknown:
            extra = " (put anything else in meta)" if "meta" in allowed else ""
            raise self.fail(
                f"unknown key(s) {', '.join(map(repr, unknown))} for format {self.fmt}; "
                f"allowed: {', '.join(allowed)}{extra}"
            )
        return obj

    def input_text(self, obj: Mapping[str, Any]) -> str:
        if "input" not in obj:
            raise self.fail('missing "input"')
        value = obj["input"]
        if not isinstance(value, str):
            raise self.fail(f'"input" must be a string, got {_type_name(value)}')
        if not value.strip():
            raise self.fail('"input" is empty')
        return value

    def output_text(self, obj: Mapping[str, Any]) -> str:
        if "output" not in obj:
            raise self.fail('missing "output" (use --format inputs for rows without an output)')
        value = obj["output"]
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if not isinstance(value, str):
            raise self.fail(f'"output" must be a string or a JSON object, got {_type_name(value)}')
        if not value.strip():
            raise self.fail('"output" is empty')
        return value

    def meta(self, obj: Mapping[str, Any], split: _SplitField | None = None) -> dict[str, Any]:
        value = obj.get("meta")
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise self.fail(f'"meta" must be a JSON object, got {_type_name(value)}')
        if split is not None:
            try:
                canonical_split(meta_get(value, split.keys), split.path)
            except CurateError as exc:
                raise self.fail(str(exc)) from None
        return value

    def body(self, obj: Mapping[str, Any], name: str) -> dict[str, Any]:
        if name not in obj:
            raise self.fail(f'missing "{name}"')
        value = obj[name]
        if not isinstance(value, dict):
            raise self.fail(f'"{name}" must be a JSON object, got {_type_name(value)}')
        return value


def _import_row(line: _Line, obj: Mapping[str, Any], fmt: str, split: _SplitField | None) -> dict[str, Any]:
    row: dict[str, Any] = {"input": line.input_text(obj)}
    row["output"] = line.output_text(obj) if fmt == "pairs" else None
    row["gold"] = obj.get("gold")
    row["meta"] = line.meta(obj, split)
    for name, value in row.items():
        if not utf8_safe(value):
            raise line.fail(f'"{name}" {UNPAIRED_SURROGATE}')
    return row


def _capture_row(line: _Line, obj: Mapping[str, Any]) -> dict[str, Any]:
    request = line.body(obj, "request")
    response = line.body(obj, "response")
    problem = completion_problem(request, response)
    if problem is not None:
        raise line.fail(problem)
    usage_value = response.get("usage")
    usage: dict[str, Any] = usage_value if isinstance(usage_value, dict) else {}
    model = response.get("model")

    def as_int(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    cost = usage.get("cost")
    try:
        key = request_key(request)
    except ValueError as exc:
        raise line.fail(f'"request" is not valid JSON: {exc}') from None
    return {
        "source": "import",
        "status": 200,
        "captured": True,
        "request_key": key,
        "request_body": json.dumps(request, ensure_ascii=False),
        "response_body": json.dumps(response, ensure_ascii=False),
        "prompt_tokens": as_int(usage.get("prompt_tokens")),
        "completion_tokens": as_int(usage.get("completion_tokens")),
        "cost_usd": float(cost) if isinstance(cost, int | float) and not isinstance(cost, bool) else None,
        "upstream_model": model if isinstance(model, str) else None,
    }


def _write_captures(store: Store, task: str, rows: list[dict[str, Any]]) -> int:
    """Write capture rows in one transaction when the store has ``add_captures(task, rows)``, else row by row."""
    bulk = getattr(store, "add_captures", None)
    if callable(bulk):
        return int(bulk(task, rows))
    for row in rows:
        store.add_capture(task=task, **row)
    return len(rows)


def import_file(
    store: Store, task: str, path: str | Path, fmt: str, predefined: str | None = "meta.split"
) -> ImportCounts:
    """Validate every line of ``path`` in format ``fmt``, then write the rows for ``task`` into ``store``.

    ``predefined`` is the meta path of the predefined split (the task's ``curate.split.predefined``; None skips the
    check): a ``pairs`` or ``inputs`` row whose value there is set must name train, valid or test.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown import format {fmt!r}; expected one of: {', '.join(FORMATS)}")
    split = _SplitField(predefined) if predefined else None
    path = Path(path)
    where = path.name
    rows: list[dict[str, Any]] = []
    blank = 0
    for lineno, raw in _open_lines(path):
        text = _decode(raw, lineno, where)
        if not text.strip():
            blank += 1
            continue
        line = _Line(where, lineno, fmt)
        obj = line.parse(text)
        rows.append(_capture_row(line, obj) if fmt == "openai" else _import_row(line, obj, fmt, split))

    if fmt == "openai":
        imported = _write_captures(store, task, rows) if rows else 0
    else:
        imported = store.add_imports(task, fmt, rows) if rows else 0
    return ImportCounts(read=len(rows), imported=imported, blank=blank, format=fmt)
