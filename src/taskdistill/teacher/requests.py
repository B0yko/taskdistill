"""The one function that builds every request the "application" sends to the teacher.

The capture proxy traffic of the demos, curate's teacher labelling, the serve smoke test and the
README request all come from here, so their request keys match the recorded teacher outputs.
"""

from __future__ import annotations

import copy
from typing import Any

from taskdistill.config import TaskSpec


def build_teacher_request(spec: TaskSpec, input_text: str, *, include_extra_body: bool = True) -> dict[str, Any]:
    """Chat-completions body: the teacher prompt as system message, the raw input as user message."""
    body: dict[str, Any] = {
        "model": spec.teacher.model,
        "messages": [
            {"role": "system", "content": spec.teacher_prompt},
            {"role": "user", "content": input_text},
        ],
        "temperature": spec.teacher.temperature,
        "max_tokens": spec.teacher.max_tokens,
    }
    if spec.teacher.response_format is not None:
        body["response_format"] = copy.deepcopy(spec.teacher.response_format)
    if include_extra_body:
        for key, value in spec.teacher.extra_body.items():
            body.setdefault(key, copy.deepcopy(value))
    return body


def build_student_messages(spec: TaskSpec, input_text: str) -> list[dict[str, str]]:
    """The student's chat: its short system prompt plus the extracted input."""
    return [
        {"role": "system", "content": spec.student.system_prompt},
        {"role": "user", "content": input_text},
    ]
