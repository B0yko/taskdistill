"""Extraction outputs: parse a model answer as JSON, validate it against the task's JSON Schema and
render it canonically (compact JSON, keys in schema property order)."""

from __future__ import annotations

import functools
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not valid JSON")


def _finite_float(literal: str) -> float:
    value = float(literal)
    if not math.isfinite(value):
        raise ValueError(f"number {literal} overflows a float")
    return value


_DECODER = json.JSONDecoder(parse_constant=_reject_constant, parse_float=_finite_float)


def _loads_object(text: str) -> dict[str, Any] | None:
    try:
        obj = _DECODER.decode(text)
    except (ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def parse_json_output(text: str) -> dict[str, Any] | None:
    """Parse a JSON object out of a model answer.

    Markdown code fences are stripped, then any text before the first ``{`` and after the last ``}``.
    If that span is not one JSON object (for example trailing prose that contains braces), the first
    complete object starting at the first ``{`` is used. Returns ``None`` when no object parses.
    """
    fenced = next((m.group(1) for m in _FENCE.finditer(text) if "{" in m.group(1)), None)
    body = fenced if fenced is not None else text
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        return None
    obj = _loads_object(body[start : end + 1])
    if obj is not None:
        return obj
    try:
        first, _ = _DECODER.raw_decode(body, start)
    except (ValueError, RecursionError):
        return None
    return first if isinstance(first, dict) else None


@functools.lru_cache(maxsize=32)
def _validator(schema_json: str) -> Draft202012Validator:
    schema = json.loads(schema_json)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _schema_json(schema: Mapping[str, Any]) -> str:
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def validate(obj: Any, schema: Mapping[str, Any]) -> list[str]:
    """Sorted JSON Schema (Draft 2020-12) errors of ``obj``, each as ``<json path>: <message>``.

    ``format`` is enforced (``"format": "date"`` requires an ISO ``YYYY-MM-DD`` date). Empty means valid.
    """
    validator = _validator(_schema_json(schema))
    return sorted(f"{err.json_path}: {err.message}" for err in validator.iter_errors(obj))


def _ordered(value: Any, schema: Any) -> Any:
    if not isinstance(schema, Mapping):
        return value
    if isinstance(value, dict):
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            return {name: _ordered(value[name], sub) for name, sub in properties.items() if name in value}
        return value
    if isinstance(value, list):
        items = schema.get("items")
        return [_ordered(item, items) for item in value] if isinstance(items, Mapping) else value
    return value


def canonical_output(obj: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
    """Compact JSON of ``obj`` with keys in schema property order; keys the schema does not name are dropped."""
    ordered = _ordered(dict(obj), schema)
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def normalise_extraction(text: str, schema: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Parse and validate a model answer: ``(object, canonical JSON)``, or ``(None, None)`` if either step fails."""
    obj = parse_json_output(text)
    if obj is None or validate(obj, schema):
        return None, None
    ordered: dict[str, Any] = _ordered(obj, schema)
    return ordered, json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
