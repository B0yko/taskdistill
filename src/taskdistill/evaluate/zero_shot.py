"""Prompts for the zero-shot baseline: the base model with no adapter and the task spelled out in the system prompt.

Classification lists every canonical label, one per line (decoding still runs under the label trie); extraction
includes the JSON Schema.
"""

from __future__ import annotations

import json

from taskdistill.config import TaskSpec

LABEL_INSTRUCTION = "Answer with exactly one label from the list."
SCHEMA_INSTRUCTION = "Extract the fields as a single JSON object matching this JSON Schema: {schema}. Output JSON only."


def zero_shot_system_prompt(spec: TaskSpec) -> str:
    """The student's system prompt plus the label list or the JSON Schema."""
    lead = spec.student.system_prompt.strip()
    if spec.type == "classification":
        if not spec.labels:
            raise ValueError(f"task {spec.task}: the zero-shot prompt needs the label set")
        labels = "\n".join(spec.labels)
        return f"{lead}\n\nLabels:\n{labels}\n\n{LABEL_INSTRUCTION}"
    if not spec.json_schema:
        raise ValueError(f"task {spec.task}: the zero-shot prompt needs the JSON Schema")
    schema = json.dumps(spec.json_schema, ensure_ascii=False)
    return f"{lead}\n\n{SCHEMA_INSTRUCTION.format(schema=schema)}"


def zero_shot_messages(spec: TaskSpec, input_text: str) -> list[dict[str, str]]:
    """Chat messages for the zero-shot baseline on one input."""
    return [
        {"role": "system", "content": zero_shot_system_prompt(spec)},
        {"role": "user", "content": input_text},
    ]
