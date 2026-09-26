"""Teacher bake-off: score candidate teachers on validation inputs that have gold labels (must-have 12).

``taskdistill teacher bakeoff --task <t> --models a,b,c --n 200`` sends the same validation inputs to every
candidate through the cached, budgeted teacher client and writes ``reports/bakeoff/<task>.json``.

- A candidate is ``slug`` or ``slug@provider-tag`` (e.g. ``deepseek/deepseek-v4-flash-0731@deepinfra/fp8``). A tag
  pins the OpenRouter provider: ``provider = {"order": [tag], "allow_fallbacks": false, "require_parameters": true,
  "data_collection": "deny"}``. ``:free`` variants are rejected: they are separate catalog entries with their own
  endpoints and limits, so their results say nothing about the paid model.
- Reasoning is disabled per request (``reasoning: {"enabled": false}``) for models whose supported parameters
  include ``reasoning``; pure instruct models never get the field. Models whose reasoning is mandatory are rejected.
  Model capabilities come from the pricing snapshot (the packaged one carries them), else from OpenRouter's
  ``GET /models`` and ``GET /models/{slug}/endpoints``. A candidate whose capabilities are unknown is refused
  unless the caller allows it, because its reasoning could not be disabled.
- JSON-mode support is reported per pinned endpoint; an unpinned candidate reports null, and when it wins the
  report says the provider that served it must be pinned before use.
- Inputs are the task's imported rows whose predefined split (``curate.split.predefined``) resolves to valid the
  way curate resolves it (the most protected split of all rows of an input wins, so an input with any test row
  is never used) and whose gold resolves by strict majority. They are ordered by a ``random.Random(13)`` shuffle
  of the input hashes; the first ``n`` are used.
- Responses are cached in ``$TASKDISTILL_HOME/bakeoff_cache.sqlite`` with one row per request key and routing
  context (:class:`BakeoffCache`): candidates that share a model but not a provider pin or reasoning setting send
  requests with the same key, and must neither evict each other's rows nor curate's labels in the shared cache.
- The first 50 requests of the first candidate are a sample: they project the cost of the whole bake-off (the
  other candidates at their snapshot prices), which needs ``--yes`` above $0.50 and must fit ``--max-usd``.
- A request that fails after the client's retries is scored as a wrong answer. HTTP 401 and 402 (bad key, no
  credits) stop the bake-off at once; so does a sample in which every request failed.
- Selection: among candidates with no truncated output, no reasoning tokens and at most 1% failed requests, the
  cheapest (cost per 1k successful requests) within 2 points of the best gold score. Accuracy for classification;
  for extraction both field micro-F1 and JSON validity must be within 2 points of their best.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from taskdistill import paths
from taskdistill.config import TaskSpec
from taskdistill.curate.dedupe import majority_vote
from taskdistill.curate.extract import input_hash
from taskdistill.evaluate.metrics import accuracy, extraction_scores
from taskdistill.store import Store
from taskdistill.tasks.classification import normalise_label
from taskdistill.tasks.extraction import normalise_extraction
from taskdistill.teacher.base import TeacherHTTPError, TeacherResult, TeacherSource, TeacherTimeout
from taskdistill.teacher.cache import ResponseCache
from taskdistill.teacher.client import confirm_spend, is_truncated, latency_stats, usage_cost
from taskdistill.teacher.factory import load_pricing, make_teacher, new_run_id, packaged_pricing, resolve_mode
from taskdistill.teacher.pricing import OPENROUTER_BASE_URL, ModelPrice, PricingSnapshot
from taskdistill.teacher.requests import build_teacher_request

PHASE = "bakeoff"
SEED = 13
SAMPLE_N = 50
#: Candidates within this many points (as a fraction) of the best score are eligible.
WITHIN = 0.02
#: A candidate with a larger share of failed requests is not eligible.
MAX_FAILURE_RATE = 0.01
_EPS = 1e-9
#: Accepted spellings of a predefined split value (as curate reads them) and how strongly each is protected.
SPLIT_ALIASES = {
    "train": "train",
    "valid": "valid",
    "validation": "valid",
    "val": "valid",
    "dev": "valid",
    "test": "test",
}
SPLIT_RANK = {"train": 0, "valid": 1, "test": 2}
INPUTS_NOTE = "valid split with gold, first n by seed 13"
DEFAULT_OUT_DIR = Path("reports/bakeoff")
BAKEOFF_CACHE_NAME = "bakeoff_cache.sqlite"
#: HTTP statuses after which no further request can succeed: the bake-off stops instead of scoring them.
FATAL_STATUSES = {
    401: "the teacher API key was refused; check it",
    402: "the key's credit limit or the account balance is exhausted (or the in-flight budget is full; retry later)",
}


class BakeoffError(ValueError):
    """The bake-off cannot run as requested; the message says why."""


class BakeoffCache(ResponseCache):
    """The response cache with one row per (request key, cache context), in a file of its own.

    The shared cache keeps one row per request key, and the key leaves out provider routing and reasoning, so two
    candidates of one model with different pins would replace each other's rows on every run (and replace curate's
    labels of the same inputs). Here each row is stored under a key derived from both. Only :meth:`get` and
    :meth:`put`, which the live client uses, translate keys; results carry the request key as usual.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        super().__init__(path if path is not None else paths.home() / BAKEOFF_CACHE_NAME)

    @staticmethod
    def row_key(key: str, context: str | None) -> str:
        return hashlib.sha256(f"{key}\n{context or ''}".encode()).hexdigest()

    def get(self, key: str, context: str | None = None) -> TeacherResult | None:
        hit = super().get(self.row_key(key, context), context)
        return None if hit is None else dataclasses.replace(hit, key=key)

    def put(self, result: TeacherResult, context: str | None = None) -> None:
        super().put(dataclasses.replace(result, key=self.row_key(result.key, context)), context)


# candidates ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    model: str
    tag: str | None = None

    @property
    def label(self) -> str:
        return f"{self.model}@{self.tag}" if self.tag else self.model


def parse_candidate(text: str) -> Candidate:
    """``slug`` or ``slug@provider-tag``; ``:free`` variants are rejected."""
    raw = text.strip()
    model, sep, tag = raw.partition("@")
    model, tag = model.strip(), tag.strip()
    if not model or (sep and not tag) or "@" in tag or any(ch.isspace() for ch in raw):
        raise BakeoffError(f"candidate {text!r} must be 'slug' or 'slug@provider-tag', e.g. vendor/model@deepinfra/fp8")
    if model.endswith(":free") or ":free:" in model:
        raise BakeoffError(
            f"candidate {model}: ':free' variants are rejected; they are separate catalog entries with their own "
            "endpoints and rate limits, so they cannot stand in for the paid model. Use the paid slug."
        )
    return Candidate(model, tag or None)


def parse_candidates(models: Sequence[str]) -> list[Candidate]:
    """Parse ``--models`` (entries may themselves be comma-separated); duplicates are an error."""
    candidates = [parse_candidate(part) for entry in models for part in entry.split(",") if part.strip()]
    if not candidates:
        raise BakeoffError("the bake-off needs at least one candidate model (--models a,b,c)")
    labels = [c.label for c in candidates]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise BakeoffError(f"duplicate candidates: {', '.join(duplicates)}")
    return candidates


def provider_pin(tag: str) -> dict[str, Any]:
    """The OpenRouter ``provider`` object that pins one endpoint tag."""
    return {"order": [tag], "allow_fallbacks": False, "require_parameters": True, "data_collection": "deny"}


# model capabilities ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelInfo:
    """What OpenRouter says a model and its endpoints accept."""

    model: str
    #: Union over the model's providers (``GET /models``); None when the listing does not say.
    supported_parameters: frozenset[str] | None
    #: The models-list ``reasoning`` object (``mandatory``, ``default_enabled``, ...); None for instruct models.
    reasoning: Mapping[str, Any] | None = None
    #: Endpoint tag -> supported parameters (None when the endpoint does not list them); None when unknown.
    endpoints: Mapping[str, frozenset[str] | None] | None = None

    @property
    def reasoning_mandatory(self) -> bool:
        return bool(self.reasoning and self.reasoning.get("mandatory"))

    def _matching(self, tag: str) -> list[frozenset[str] | None]:
        if not self.endpoints:
            return []
        wanted = tag.strip().lower()
        exact = [params for name, params in self.endpoints.items() if name.lower() == wanted]
        if exact or "/" in wanted:
            return exact
        return [params for name, params in self.endpoints.items() if name.lower().split("/", 1)[0] == wanted]

    def serves(self, tag: str) -> bool | None:
        """Whether an endpoint of the model matches ``tag`` (an exact tag or a base provider slug); None if unknown."""
        if self.endpoints is None:
            return None
        return bool(self._matching(tag))

    def parameters_for(self, tag: str) -> frozenset[str] | None:
        """Parameters every endpoint matching the pinned ``tag`` accepts; None when unknown."""
        matching = self._matching(tag)
        known = [params for params in matching if params is not None]
        if not matching or len(known) != len(matching):
            return None
        return frozenset.intersection(*known)

    @classmethod
    def from_parts(cls, model: str, entry: Mapping[str, Any]) -> ModelInfo:
        endpoints_raw = entry.get("endpoints")
        endpoints = (
            {str(tag): _parameters(params) for tag, params in endpoints_raw.items()}
            if isinstance(endpoints_raw, Mapping)
            else None
        )
        reasoning = entry.get("reasoning")
        return cls(
            model=model,
            supported_parameters=_parameters(entry.get("supported_parameters")),
            reasoning=reasoning if isinstance(reasoning, Mapping) else None,
            endpoints=endpoints,
        )


def _parameters(value: Any) -> frozenset[str] | None:
    """A ``supported_parameters`` list as a set; None when it is absent (unknown, not empty)."""
    return frozenset(map(str, value)) if isinstance(value, list | tuple | set | frozenset) else None


ModelInfoLookup = Callable[[str], ModelInfo | None]


def model_info_from_snapshot(data: Mapping[str, Any], model: str) -> ModelInfo | None:
    """Capabilities of ``model`` (an id or its dated canonical slug) from a pricing snapshot's ``capabilities``."""
    capabilities = data.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return None
    entry = capabilities.get(model)
    if entry is None:
        aliases = data.get("aliases")
        target = aliases.get(model) if isinstance(aliases, Mapping) else None
        entry = capabilities.get(target) if isinstance(target, str) else None
    return ModelInfo.from_parts(model, entry) if isinstance(entry, Mapping) else None


def snapshot_model_info(model: str) -> ModelInfo | None:
    """Capabilities from the workspace pricing snapshot when it carries them, else from the packaged snapshot."""
    sources: list[Callable[[], str]] = []
    local = paths.pricing_path()
    if local.is_file():
        sources.append(lambda: local.read_text(encoding="utf-8"))
    sources.append(lambda: packaged_pricing().read_text(encoding="utf-8"))
    for read in sources:
        try:
            data = json.loads(read())
        except (OSError, ValueError):
            continue
        if isinstance(data, Mapping):
            info = model_info_from_snapshot(data, model)
            if info is not None:
                return info
    return None


def fetch_model_info(
    model: str, base_url: str = OPENROUTER_BASE_URL, *, client: httpx.Client | None = None
) -> ModelInfo | None:
    """Capabilities from ``GET {base}/models`` and ``GET {base}/models/{id}/endpoints``; None when unavailable."""
    base = base_url.rstrip("/")
    own = client is None
    http = client if client is not None else httpx.Client(timeout=30.0)
    try:
        listing = http.get(f"{base}/models")
        if listing.status_code != 200:
            return None
        data = listing.json()
        rows = data.get("data") if isinstance(data, dict) else data
        entry = next(
            (m for m in rows or [] if isinstance(m, dict) and model in (m.get("id"), m.get("canonical_slug"))),
            None,
        )
        if entry is None:
            return None
        parts: dict[str, Any] = {
            "supported_parameters": entry.get("supported_parameters"),
            "reasoning": entry.get("reasoning"),
        }
        endpoints_resp = http.get(f"{base}/models/{entry.get('id')}/endpoints")
        if endpoints_resp.status_code == 200:
            payload = endpoints_resp.json()
            inner = payload.get("data", payload) if isinstance(payload, dict) else None
            listed = inner.get("endpoints") if isinstance(inner, dict) else None
            endpoints: dict[str, frozenset[str] | None] = {}
            for endpoint in listed if isinstance(listed, list) else []:
                if not isinstance(endpoint, dict):
                    continue
                tag = endpoint.get("tag") or endpoint.get("provider_name")
                if not isinstance(tag, str):
                    continue
                params = _parameters(endpoint.get("supported_parameters"))
                if tag in endpoints:
                    seen = endpoints[tag]
                    params = None if seen is None or params is None else seen & params
                endpoints[tag] = params
            parts["endpoints"] = {tag: None if p is None else sorted(p) for tag, p in endpoints.items()}
        return ModelInfo.from_parts(model, parts)
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        return None
    finally:
        if own:
            http.close()


def default_model_info(base_url: str) -> ModelInfoLookup:
    def lookup(model: str) -> ModelInfo | None:
        return snapshot_model_info(model) or fetch_model_info(model, base_url)

    return lookup


# inputs ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BakeoffInput:
    input_hash: str
    text: str
    gold: Any


def _meta_value(meta: Mapping[str, Any], path: str) -> Any:
    value: Any = meta
    for part in path.split(".")[1:]:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _gold_value(spec: TaskSpec, gold: Any) -> Any:
    """The gold label (classification) or schema-valid object (extraction), as curate normalises it; None when
    it is missing or invalid."""
    if gold is None:
        return None
    if spec.type == "classification":
        return normalise_label(gold, spec.labels) if isinstance(gold, str) else None
    if not spec.json_schema:
        return None
    if isinstance(gold, dict):
        try:
            gold = json.dumps(gold, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return None
    return normalise_extraction(gold, spec.json_schema)[0] if isinstance(gold, str) else None


def _split_name(value: Any) -> str | None:
    return SPLIT_ALIASES.get(value.strip().lower()) if isinstance(value, str) else None


def select_inputs(spec: TaskSpec, store: Store, n: int) -> list[BakeoffInput]:
    """Imported inputs whose split resolves to valid and whose gold resolves, one per input hash, in
    ``random.Random(13)`` order of the hashes.

    All import rows of an input count, as in curate: when their predefined splits disagree the most protected one
    wins (test > valid > train), so an input with any test row is left out; the gold is the strict majority of
    the rows' valid golds, else the input is left out.
    """
    split_path = spec.curate.split.predefined
    if split_path is None:
        raise BakeoffError(
            f"task {spec.task}: the bake-off scores validation inputs, but curate.split.predefined is not set"
        )
    texts: dict[str, str] = {}
    splits: dict[str, set[str]] = {}
    golds: dict[str, list[Any]] = {}
    for row in store.iter_imports(spec.task):
        if not row.input.strip():
            continue
        digest = input_hash(row.input)
        texts.setdefault(digest, row.input)
        name = _split_name(_meta_value(row.meta, split_path))
        if name is not None:
            splits.setdefault(digest, set()).add(name)
        gold = _gold_value(spec, row.gold)
        if gold is not None:
            golds.setdefault(digest, []).append(gold)
    by_hash: dict[str, BakeoffInput] = {}
    for digest, names in splits.items():
        if max(names, key=SPLIT_RANK.__getitem__) != "valid":
            continue
        ok, gold = majority_vote(golds.get(digest, []))
        if ok:
            by_hash[digest] = BakeoffInput(digest, texts[digest], gold)
    hashes = sorted(by_hash)
    random.Random(SEED).shuffle(hashes)
    return [by_hash[digest] for digest in hashes[:n]]


# planning --------------------------------------------------------------------------------------
@dataclass
class Plan:
    candidate: Candidate
    spec: TaskSpec
    price: ModelPrice
    disable_reasoning: bool
    json_mode: bool | None
    notes: list[str] = field(default_factory=list)


def candidate_spec(spec: TaskSpec, candidate: Candidate, prompt: str, *, disable_reasoning: bool) -> TaskSpec:
    """``spec`` with the candidate's model, the prompt, and the candidate's provider pin and reasoning setting.

    The spec's own ``provider`` and ``reasoning`` entries are dropped (they belong to its model); other extra body
    fields are kept.
    """
    extra = {k: copy.deepcopy(v) for k, v in spec.teacher.extra_body.items() if k not in ("provider", "reasoning")}
    if candidate.tag:
        extra["provider"] = provider_pin(candidate.tag)
    if disable_reasoning:
        extra["reasoning"] = {"enabled": False}
    out = spec.model_copy(deep=True)
    out.teacher = spec.teacher.model_copy(update={"model": candidate.model, "extra_body": extra}, deep=True)
    out.teacher_prompt = prompt
    return out


UNPINNED_NOTE = (
    "no provider pin: requests go to whichever provider OpenRouter picks, so the score may mix providers and "
    "JSON-mode support is not reported; pin a provider (slug@provider-tag) before using this teacher"
)


def plan_candidate(
    spec: TaskSpec,
    candidate: Candidate,
    prompt: str,
    info: ModelInfo | None,
    pricing: PricingSnapshot,
    *,
    allow_unknown: bool = False,
) -> Plan:
    """How ``candidate`` is asked: its price, provider pin, reasoning setting and JSON-mode support.

    Without known capabilities (neither the model's nor the pinned endpoint's supported parameters) the bake-off
    cannot tell whether reasoning must be disabled, so it refuses unless ``allow_unknown``.
    """
    try:
        price = pricing.price_for(candidate.model, candidate.tag)
    except KeyError as exc:
        raise BakeoffError(
            f"{candidate.label}: no price in the pricing snapshot of {pricing.date}; "
            "run `taskdistill pricing refresh` with this model"
        ) from exc
    notes: list[str] = [] if candidate.tag else [UNPINNED_NOTE]
    if info is not None and info.reasoning_mandatory:
        raise BakeoffError(
            f"{candidate.label}: reasoning is mandatory for this model and cannot be disabled; "
            "the teacher must be a non-reasoning model or have reasoning disabled per request"
        )
    if info is not None and candidate.tag and info.serves(candidate.tag) is False:
        known = ", ".join(sorted(info.endpoints or {}))
        raise BakeoffError(
            f"{candidate.label}: provider tag '{candidate.tag}' does not serve {candidate.model} ({known})"
        )
    pinned = info.parameters_for(candidate.tag) if info is not None and candidate.tag else None
    union = info.supported_parameters if info is not None else None
    known_params = union if union is not None else pinned
    if known_params is None:
        if not allow_unknown:
            raise BakeoffError(
                f"{candidate.label}: the model's supported parameters are unknown (not in the pricing snapshot, and "
                "GET /models did not list them or could not be reached), so the bake-off cannot tell whether "
                "reasoning must be disabled; a hybrid model would reason at its default and be cut off at "
                "max_tokens. Retry, check the slug, or allow unknown capabilities explicitly"
            )
        notes.append("model capabilities unknown: reasoning left at the model default, JSON-mode support unknown")
        return Plan(
            candidate, candidate_spec(spec, candidate, prompt, disable_reasoning=False), price, False, None, notes
        )
    model_reasons = "reasoning" in known_params
    disable = model_reasons and (pinned is None or "reasoning" in pinned)
    if model_reasons and not disable:
        notes.append("the pinned endpoint does not accept 'reasoning', so reasoning cannot be disabled there")
    json_mode = None if pinned is None else "response_format" in pinned
    if spec.teacher.response_format is not None and json_mode is False:
        raise BakeoffError(
            f"{candidate.label}: teacher.response_format is set but this endpoint does not support response_format; "
            "pin another provider or drop teacher.response_format"
        )
    return Plan(
        candidate, candidate_spec(spec, candidate, prompt, disable_reasoning=disable), price, disable, json_mode, notes
    )


# running ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Failure:
    """A request that failed after the client's retries (HTTP error or timeout); scored as a wrong answer."""

    error: str


Outcome = TeacherResult | Failure


async def _complete_all(teacher: TeacherSource, bodies: Sequence[dict[str, Any]]) -> list[Outcome]:
    """Every body's result, or its :class:`Failure`; HTTP 401/402 stop everything with :class:`BakeoffError`."""

    async def one(body: dict[str, Any]) -> Outcome:
        try:
            return await teacher.complete(body)
        except TeacherHTTPError as exc:
            if exc.status in FATAL_STATUSES:
                raise BakeoffError(
                    f"the bake-off stopped: {FATAL_STATUSES[exc.status]} ({exc}). Answers received so far are "
                    "cached, so a re-run does not pay for them again"
                ) from exc
            return Failure(str(exc))
        except TeacherTimeout as exc:
            return Failure(str(exc))

    tasks = [asyncio.ensure_future(one(body)) for body in bodies]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _reasoning_tokens(usage: Mapping[str, Any]) -> int:
    details = usage.get("completion_tokens_details")
    value = _int(details.get("reasoning_tokens")) if isinstance(details, Mapping) else None
    return value or 0


def project_bakeoff(sample: Sequence[Outcome], plans: Sequence[Plan], calls_left: Sequence[int]) -> dict[str, Any]:
    """Projected cost of the whole bake-off from the first candidate's sample.

    ``calls_left[i]`` counts the uncached calls still to make for ``plans[i]`` (for the first candidate, after the
    sample). The first candidate is projected at the mean cost of its live sample calls; the others (and the
    first when its sample was all cached) at the sample's mean token counts x their snapshot price.
    """
    results = [r for r in sample if isinstance(r, TeacherResult)]
    live = [float(r.cost_usd) for r in results if r.source == "live"]
    usages = [r.usage for r in results if _int(r.usage.get("prompt_tokens")) is not None]
    mean_prompt = sum(int(u["prompt_tokens"]) for u in usages) / len(usages) if usages else None
    mean_completion = sum(_int(u.get("completion_tokens")) or 0 for u in usages) / len(usages) if usages else None

    def by_tokens(price: ModelPrice) -> float | None:
        if mean_prompt is None or mean_completion is None:
            return None
        return mean_prompt * price.prompt + mean_completion * price.completion + price.request

    per_call: list[float | None] = []
    for i, plan in enumerate(plans):
        if i == 0 and live:
            per_call.append(sum(live) / len(live))
        else:
            per_call.append(by_tokens(plan.price))
    sample_usd = sum(live)
    remaining = 0.0
    projectable = True
    for cost, calls in zip(per_call, calls_left, strict=True):
        if calls == 0:
            continue
        if cost is None:
            projectable = False
            break
        remaining += cost * calls
    return {
        "sample_n": len(sample),
        "sample_live": len(live),
        "sample_failed": len(sample) - len(results),
        "sample_usd": sample_usd,
        "mean_prompt_tokens": mean_prompt,
        "mean_completion_tokens": mean_completion,
        "calls_left": int(sum(calls_left)),
        "projected_usd": sample_usd + remaining if projectable else None,
    }


# scoring ---------------------------------------------------------------------------------------
def score_outputs(spec: TaskSpec, outputs: Sequence[str | None], golds: Sequence[Any]) -> tuple[dict[str, Any], int]:
    """Gold metrics of the teacher outputs and the number of outputs that could not be normalised."""
    if spec.type == "classification":
        labels = [normalise_label(o, spec.labels) if o is not None else None for o in outputs]
        return {"accuracy": accuracy(list(golds), labels)}, sum(1 for label in labels if label is None)
    schema = spec.json_schema or {}
    objects = [normalise_extraction(o, schema)[0] if o is not None else None for o in outputs]
    scores = extraction_scores(list(golds), objects, spec.schema_fields)
    metrics = {
        name: scores[name] for name in ("field_micro_f1", "json_validity", "field_exact_match", "doc_exact_match")
    }
    return metrics, sum(1 for obj in objects if obj is None)


def primary_metrics(task_type: str) -> tuple[str, ...]:
    return ("accuracy",) if task_type == "classification" else ("field_micro_f1", "json_validity")


def allowed_failures(n: int) -> int:
    """How many failed requests out of ``n`` a candidate may have and stay eligible (at most 1%)."""
    return int(n * MAX_FAILURE_RATE + _EPS)


def summarise(
    spec: TaskSpec, plan: Plan, bodies: Sequence[Mapping[str, Any]], outcomes: Sequence[Outcome], golds: Sequence[Any]
) -> dict[str, Any]:
    """One candidate's row of the bake-off report.

    Failed requests count as wrong answers in the gold metrics; cost per 1k is over the successful requests,
    because a failed request costs nothing and says nothing about the price of an answer.
    """
    results = [o for o in outcomes if isinstance(o, TeacherResult)]
    failures = [o for o in outcomes if isinstance(o, Failure)]
    outputs = [o.output if isinstance(o, TeacherResult) else None for o in outcomes]
    metrics, invalid = score_outputs(spec, outputs, golds)
    n = len(outcomes)
    cost = sum(usage_cost(r.usage, plan.price) or 0.0 for r in results)
    truncated = sum(
        1
        for body, o in zip(bodies, outcomes, strict=True)
        if isinstance(o, TeacherResult) and (o.truncated or is_truncated(body, o.usage, o.finish_reason))
    )
    reasoning = sum(_reasoning_tokens(r.usage) for r in results)
    latency = latency_stats(results)
    candidate = plan.candidate
    return {
        "candidate": candidate.label,
        "model": candidate.model,
        "provider": candidate.tag,
        "n": n,
        "gold_score": metrics[primary_metrics(spec.type)[0]],
        "metrics": metrics,
        "invalid_outputs": invalid - len(failures),
        "errors": len(failures),
        "first_error": failures[0].error if failures else None,
        "cost_usd": cost,
        "cost_per_1k_usd": cost / len(results) * 1000 if results else None,
        "spent_usd": sum(float(r.cost_usd) for r in results if r.source == "live"),
        "latency_ms": {
            "p50": latency["p50_ms"],
            "p95": latency["p95_ms"],
            "mean": latency["mean_ms"],
            "n": latency["n"],
        },
        "truncated": truncated,
        "reasoning_tokens": reasoning,
        "reasoning_disabled": plan.disable_reasoning,
        "json_mode": plan.json_mode,
        "served_by": dict(sorted(Counter(r.provider or "unknown" for r in results).items())),
        "models_returned": dict(sorted(Counter(str(r.response.get("model") or "unknown") for r in results).items())),
        "cache_hits": sum(1 for r in results if r.source == "cache"),
        "tokens": {
            "prompt": sum(_int(r.usage.get("prompt_tokens")) or 0 for r in results),
            "completion": sum(_int(r.usage.get("completion_tokens")) or 0 for r in results),
        },
        "request": {
            "extra_body": copy.deepcopy(plan.spec.teacher.extra_body),
            "response_format": copy.deepcopy(plan.spec.teacher.response_format),
        },
        "checks": {
            "no_truncation": truncated == 0,
            "no_reasoning_tokens": reasoning == 0,
            "few_failures": len(failures) <= allowed_failures(n),
        },
        "notes": list(plan.notes),
    }


# selection -------------------------------------------------------------------------------------
REQUIRES = ("no truncated outputs", "no reasoning tokens", f"at most {MAX_FAILURE_RATE:.0%} failed requests")


def selection_rule(task_type: str) -> dict[str, Any]:
    metrics = primary_metrics(task_type)
    return {
        "metrics": list(metrics),
        "within": WITHIN,
        "choose": "lowest cost per 1k successful requests",
        "requires": list(REQUIRES),
        "max_failure_rate": MAX_FAILURE_RATE,
        "text": (
            f"cheapest candidate (cost per 1k requests) whose {' and '.join(metrics)} "
            f"{'are' if len(metrics) > 1 else 'is'} within {WITHIN:g} of the best, among candidates with "
            f"{', '.join(REQUIRES[:-1])} and {REQUIRES[-1]}"
        ),
    }


def _cost_key(row: Mapping[str, Any]) -> float:
    cost = row.get("cost_per_1k_usd")
    return float(cost) if isinstance(cost, int | float) else float("inf")


def select_candidate(rows: list[dict[str, Any]], task_type: str) -> dict[str, Any]:
    """Mark each row ``eligible`` and return the chosen candidate (``model`` None when none passes the checks).

    A chosen candidate without a provider pin carries ``pinned: false`` and a warning: its score may mix the
    providers OpenRouter routed to, and the provider must be pinned before it labels anything.
    """
    metrics = primary_metrics(task_type)
    passing = [i for i, row in enumerate(rows) if all(row["checks"].values())]
    for row in rows:
        row["eligible"] = False
    if not passing:
        return {
            "candidate": None,
            "model": None,
            "provider": None,
            "reason": f"no candidate passed the checks ({', '.join(REQUIRES)})",
        }
    best = {m: max(rows[i]["metrics"][m] for i in passing) for m in metrics}
    eligible = [i for i in passing if all(rows[i]["metrics"][m] >= best[m] - WITHIN - _EPS for m in metrics)]
    if eligible:
        pick = min(eligible, key=lambda i: (_cost_key(rows[i]), -rows[i]["gold_score"], i))
        best_text = ", ".join(f"{m} {best[m]:.3f}" for m in metrics)
        reason = f"cheapest of {len(eligible)} candidate(s) within {WITHIN:g} of the best ({best_text})"
    else:
        pick = max(passing, key=lambda i: (rows[i]["gold_score"], -i))
        reason = f"no candidate is within {WITHIN:g} of the best on every metric; chose the best {metrics[0]}"
    for i in eligible:
        rows[i]["eligible"] = True
    row = rows[pick]
    chosen: dict[str, Any] = {
        "candidate": row["candidate"],
        "model": row["model"],
        "provider": row["provider"],
        "pinned": row["provider"] is not None,
        "gold_score": row["gold_score"],
        "metrics": copy.deepcopy(row["metrics"]),
        "cost_per_1k_usd": row["cost_per_1k_usd"],
        "reason": reason,
        "teacher": {"model": row["model"], **copy.deepcopy(row["request"])},
    }
    if row["provider"] is None:
        served = row.get("served_by") or {}
        by = ", ".join(f"{name} ({count})" for name, count in served.items()) or "unknown providers"
        chosen["warning"] = (
            f"{row['candidate']} ran without a provider pin and was served by {by}; re-run it as "
            "slug@provider-tag and pin that provider in teacher.extra_body before labelling with it"
        )
    return chosen


# entry point -----------------------------------------------------------------------------------
def bakeoff_report_path(out_dir: Path | str, task: str, prompt_variant: Path | str | None = None) -> Path:
    """``<out_dir>/<task>.json``, or ``<out_dir>/<task>.<variant-stem>.json`` for a prompt-variant run."""
    name = f"{task}.{Path(prompt_variant).stem}.json" if prompt_variant is not None else f"{task}.json"
    return Path(out_dir) / name


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def _fmt_usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def run_bakeoff(
    spec: TaskSpec,
    *,
    models: list[str],
    n: int = 200,
    prompt_variant: Path | None = None,
    yes: bool = False,
    max_usd: float | None = None,
    store: Store,
    pricing: PricingSnapshot | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    teacher_factory: Callable[..., TeacherSource] | None = None,
    model_info: ModelInfoLookup | None = None,
    allow_unknown_capabilities: bool = False,
    cache: ResponseCache | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the bake-off and write its report; return the report.

    ``teacher_factory`` is called like :func:`make_teacher` (the default, which needs a teacher API key), with
    ``cache`` set to the bake-off's own cache (default :class:`BakeoffCache` in the workspace); a factory must use
    it, or the projection cannot tell which requests are cached. ``model_info`` looks up a model's capabilities
    (default: the pricing snapshot, else OpenRouter's model list); ``allow_unknown_capabilities`` runs candidates
    whose capabilities are unknown with reasoning at the model default instead of refusing them.
    """
    candidates = parse_candidates(models)
    if n < 1:
        raise BakeoffError("--n must be at least 1")
    if max_usd is not None and max_usd <= 0:
        raise BakeoffError("--max-usd must be positive")
    if spec.type == "classification" and not spec.labels:
        raise BakeoffError(f"task {spec.task}: no labels loaded")
    if spec.type == "extraction" and not spec.json_schema:
        raise BakeoffError(f"task {spec.task}: no JSON Schema loaded")
    factory = teacher_factory
    if factory is None:
        resolve_mode(spec, "live")
        factory = make_teacher
    prompt = spec.teacher_prompt
    variant_sha: str | None = None
    if prompt_variant is not None:
        try:
            prompt = Path(prompt_variant).read_text(encoding="utf-8")
        except OSError as exc:
            raise BakeoffError(f"cannot read the prompt variant {Path(prompt_variant).name}: {exc}") from exc
        variant_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    snapshot = pricing if pricing is not None else load_pricing()
    lookup = model_info if model_info is not None else default_model_info(spec.teacher.base_url)
    plans = [
        plan_candidate(spec, c, prompt, lookup(c.model), snapshot, allow_unknown=allow_unknown_capabilities)
        for c in candidates
    ]

    inputs = select_inputs(spec, store, n)
    if not inputs:
        raise BakeoffError(
            f"task {spec.task}: no imported validation inputs with gold labels; import them with "
            "`taskdistill capture --task <task> --import <file> --format inputs` (gold and meta.split = valid)"
        )
    if len(inputs) < n:
        log(f"[bakeoff] only {len(inputs)} validation inputs with gold (asked for {n})")
    golds = [item.gold for item in inputs]
    bodies = [[build_teacher_request(plan.spec, item.text) for item in inputs] for plan in plans]

    run_id = new_run_id(PHASE)
    teacher = factory(
        spec,
        mode="live",
        phase=PHASE,
        run_id=run_id,
        run_cap=max_usd,
        use_cache=True,
        pricing=snapshot,
        cache=cache if cache is not None else BakeoffCache(),
    )
    log(f"[bakeoff] {spec.task}: {len(plans)} candidate(s) x {len(inputs)} inputs ({INPUTS_NOTE}); run {run_id}")
    for plan in plans:
        for note in plan.notes:
            log(f"[bakeoff] {plan.candidate.label}: {note}")

    async def run_all() -> tuple[dict[str, Any], list[list[Outcome]]]:
        try:
            cached = getattr(teacher, "is_cached", None)

            def uncached(batch: Sequence[dict[str, Any]]) -> int:
                return sum(1 for body in batch if not (callable(cached) and cached(body)))

            first = plans[0].candidate.label
            sample_n = min(SAMPLE_N, len(inputs))
            sample = await _complete_all(teacher, bodies[0][:sample_n])
            failed = [o for o in sample if isinstance(o, Failure)]
            if len(failed) == len(sample):
                raise BakeoffError(
                    f"all {len(sample)} sample requests to {first} failed ({failed[0].error}); fix or drop that "
                    "candidate, or list another one first"
                )
            calls_left = [uncached(bodies[0][sample_n:])] + [uncached(batch) for batch in bodies[1:]]
            projection = project_bakeoff(sample, plans, calls_left)
            projected = projection["projected_usd"]
            log(
                f"[bakeoff] projected spend {_fmt_usd(projected)} (sample of {len(sample)} calls to {first}, "
                f"{len(failed)} failed; {projection['calls_left']} uncached calls left)"
            )
            if max_usd is not None and projected is not None and projected > max_usd + _EPS:
                raise BakeoffError(
                    f"projected spend ${projected:.4f} for the whole bake-off is above --max-usd ${max_usd:.2f} "
                    f"(sample of {len(sample)} calls to {first}, {projection['calls_left']} uncached calls left); "
                    "use fewer candidates or a smaller --n, or raise --max-usd"
                )
            confirm_spend(projected, yes)
            outcomes = [sample + await _complete_all(teacher, bodies[0][sample_n:])]
            for batch in bodies[1:]:
                outcomes.append(await _complete_all(teacher, batch))
            return projection, outcomes
        finally:
            await teacher.aclose()

    projection, outcomes = asyncio.run(run_all())
    rows = [summarise(spec, plan, bodies[i], outcomes[i], golds) for i, plan in enumerate(plans)]
    for row in rows:
        metric = primary_metrics(spec.type)
        scores = ", ".join(f"{m} {row['metrics'][m]:.3f}" for m in metric)
        log(
            f"[bakeoff] {row['candidate']}: {scores}, cost ${row['cost_usd']:.4f} "
            f"({_fmt_usd(row['cost_per_1k_usd'])}/1k), p50 {_fmt_ms(row['latency_ms']['p50'])}, "
            f"p95 {_fmt_ms(row['latency_ms']['p95'])}, truncated {row['truncated']}, errors {row['errors']}, "
            f"served by {', '.join(row['served_by']) or 'n/a'}"
        )
        if row["errors"]:
            log(
                f"[bakeoff] {row['candidate']}: {row['errors']} failed request(s), first: {row['first_error']}; "
                "failed requests are not cached, so a re-run retries only those"
            )
    chosen = select_candidate(rows, spec.type)
    report: dict[str, Any] = {
        "task": spec.task,
        "task_type": spec.type,
        "date": datetime.now(UTC).date().isoformat(),
        "n": len(inputs),
        "n_requested": n,
        "inputs": INPUTS_NOTE,
        "prompt_variant": variant_sha,
        "teacher_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "pricing_snapshot_date": snapshot.date,
        "run_id": run_id,
        "projection": projection,
        "spent_usd": sum(row["spent_usd"] for row in rows),
        "candidates": rows,
        "rule": selection_rule(spec.type),
        "chosen": chosen,
    }
    path = bakeoff_report_path(out_dir, spec.task, prompt_variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"[bakeoff] chosen: {chosen['candidate'] or 'none'} ({chosen['reason']}); wrote {_shown(path)}")
    if chosen.get("warning"):
        log(f"[bakeoff] warning: {chosen['warning']}")
    return report
