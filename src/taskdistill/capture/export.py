"""Export captured traffic (``taskdistill capture --export <file.jsonl>``) in the ``openai`` import format.

A row is written only when ``completion_problem`` accepts it, the check ``import_file`` applies to each line, so
an exported file always imports.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

from taskdistill.capture.importer import completion_problem
from taskdistill.store import CaptureRow, Store

log = logging.getLogger(__name__)


def _exportable(row: CaptureRow) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not row.captured or row.status != 200 or not row.request_body or not row.response_body:
        return None
    try:
        request = json.loads(row.request_body)
        response = json.loads(row.response_body)
    except (ValueError, RecursionError):
        return None
    if not isinstance(request, dict) or not isinstance(response, dict):
        return None
    if completion_problem(request, response) is not None:
        return None
    return request, response


def export_file(store: Store, task: str, path: str | Path) -> int:
    """Write the captured rows of ``task`` that are complete chat completions as ``{"request", "response"}`` JSONL.

    A ``.gz`` path is gzip-compressed with an empty file name and mtime 0. Returns the number of rows written.
    """
    path = Path(path)
    lines: list[bytes] = []
    skipped = 0
    for row in store.iter_captures(task):
        pair = _exportable(row)
        if pair is None:
            if row.captured:
                skipped += 1
            continue
        request, response = pair
        lines.append(json.dumps({"request": request, "response": response}, ensure_ascii=False).encode("utf-8") + b"\n")
    if skipped:
        log.warning("export skipped %d captured row(s) of task %s that are not chat completions", skipped, task)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as fh:
        if path.suffix == ".gz":
            with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
                gz.writelines(lines)
        else:
            fh.writelines(lines)
    tmp.replace(path)
    return len(lines)
