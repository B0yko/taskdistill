"""Curate stage 10: the length filter.

Every example is tokenised the way training sees it: the student tokenizer's chat template over the system
prompt, the scrubbed input and the canonical teacher output. An example longer than ``train.max_seq_len`` is
dropped, never truncated, because the completion sits at the end of the sequence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from taskdistill.config import TaskSpec
from taskdistill.curate.merge import SPLITS, Example, target_text
from taskdistill.teacher.requests import build_student_messages


def model_label(name: str) -> str:
    """A model id as reports show it: a local directory is reduced to its name, so no absolute path is written."""
    path = Path(name).expanduser()
    return path.name if path.is_absolute() or name.startswith(".") else name


def training_messages(spec: TaskSpec, text: str, target: str) -> list[dict[str, str]]:
    """``[system, user, assistant]``: the student's short system prompt, the scrubbed input and the target."""
    return [*build_student_messages(spec, text), {"role": "assistant", "content": target}]


def example_messages(spec: TaskSpec, ex: Example) -> list[dict[str, str]]:
    """The chat written to ``data/``; a test row without a valid teacher answer has an empty assistant turn."""
    return training_messages(spec, ex.text, "" if ex.teacher is None else target_text(spec, ex.teacher))


def token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    """Tokens of the full chat (transformers 5: ``return_dict=False`` returns the id list)."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    return len(ids)


def length_stats(lengths: Sequence[int]) -> dict[str, Any]:
    if not lengths:
        return {"p50": None, "p95": None, "max": None}
    arr = np.asarray(lengths, dtype=float)
    return {
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "max": int(arr.max()),
    }


def length_filter(
    spec: TaskSpec, splits: Mapping[str, Sequence[Example]], tokenizer: Any
) -> tuple[dict[str, list[Example]], dict[str, Any]]:
    """Drop examples over ``spec.train.max_seq_len`` tokens; per split p50/p95/max (before the filter) and drops."""
    limit = spec.train.max_seq_len
    out: dict[str, list[Example]] = {}
    per_split: dict[str, Any] = {}
    for name in SPLITS:
        kept: list[Example] = []
        lengths: list[int] = []
        for ex in splits.get(name, []):
            ex.length = token_count(tokenizer, example_messages(spec, ex))
            lengths.append(ex.length)
            if ex.length <= limit:
                kept.append(ex)
        out[name] = kept
        per_split[name] = {
            "n": len(lengths),
            "kept": len(kept),
            "dropped": len(lengths) - len(kept),
            **length_stats(lengths),
            "max_kept": max((ex.length for ex in kept if ex.length is not None), default=None),
        }
    return out, {
        "max_seq_len": limit,
        "tokenizer": model_label(spec.student.base_model),
        "dropped": sum(stats["dropped"] for stats in per_split.values()),
        "splits": per_split,
    }
