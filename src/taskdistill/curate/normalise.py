"""Output normalisation shared by curate and canonical escalation in serve.

A raw model answer becomes ``(value, canonical text)``: the canonical label for classification, or the
parsed, schema-valid object and its compact JSON for extraction. ``(None, None)`` means the answer cannot
be normalised (curate drops and counts it; serve returns it raw and counts it).
"""

from __future__ import annotations

from typing import Any

from taskdistill.config import TaskSpec
from taskdistill.tasks.classification import normalise_label
from taskdistill.tasks.extraction import normalise_extraction


def normalise_output(spec: TaskSpec, text: str) -> tuple[Any, str | None]:
    """``(label, label)`` or ``(object, canonical JSON)`` for ``spec.type``; ``(None, None)`` if not normalisable.

    Raises ``ValueError`` when the spec has no labels or JSON Schema loaded.
    """
    if spec.type == "classification":
        if not spec.labels:
            raise ValueError(f"task {spec.task}: no labels loaded")
        label = normalise_label(text, spec.labels)
        return (label, label) if label is not None else (None, None)
    if not spec.json_schema:
        raise ValueError(f"task {spec.task}: no JSON Schema loaded")
    return normalise_extraction(text, spec.json_schema)
