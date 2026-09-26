from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from taskdistill.config import TaskSpec
from taskdistill.curate.extract import input_hash
from taskdistill.ledger import Ledger
from taskdistill.store import Store
from taskdistill.teacher.bakeoff import (
    BAKEOFF_CACHE_NAME,
    BakeoffCache,
    BakeoffError,
    Candidate,
    ModelInfo,
    bakeoff_report_path,
    fetch_model_info,
    parse_candidate,
    parse_candidates,
    plan_candidate,
    provider_pin,
    run_bakeoff,
    select_candidate,
    select_inputs,
    snapshot_model_info,
)
from taskdistill.teacher.base import TeacherResult
from taskdistill.teacher.cache import ResponseCache, request_context
from taskdistill.teacher.client import LiveTeacher, SpendNotConfirmed
from taskdistill.teacher.factory import TeacherUnavailable
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request

BASE = "https://router.example.com/api/v1"
URL = f"{BASE}/chat/completions"
TASK = "demo-intents"
PROMPT = "Label the customer's message with one of the intents.\n"
LABELS = ["card_arrival", "lost_card", "card_payment", "refund_request"]
A, B, C = "vendor/model-a", "vendor/model-b", "vendor/model-c"
PRICES = {
    A: ModelPrice(prompt=4e-6, completion=8e-6),
    B: ModelPrice(prompt=2e-6, completion=4e-6),
    C: ModelPrice(prompt=1e-6, completion=2e-6),
}
PROVIDER_NAMES = {A: "Alpha", B: "Beta", C: "Gamma"}
PROMPT_TOKENS, COMPLETION_TOKENS = 120, 3
PARAMS = frozenset({"max_tokens", "temperature", "stop"})


def cost_of(model: str) -> float:
    return PRICES[model].cost(PROMPT_TOKENS, COMPLETION_TOKENS)


def gold_of(i: int) -> str:
    return LABELS[i % len(LABELS)]


def text_of(i: int) -> str:
    return f"query {i}: where is my card?"


def index_of(text: str) -> int:
    return int(text.split()[1].rstrip(":"))


def spoken(label: str) -> str:
    """How a teacher tends to answer: spaces instead of underscores, capitalised."""
    return label.replace("_", " ").capitalize()


def make_spec(**teacher: Any) -> TaskSpec:
    teacher_cfg: dict[str, Any] = {
        "base_url": BASE,
        "model": "vendor/incumbent",
        "max_tokens": 24,
        "extra_body": {
            "provider": {"order": ["incumbent/fp8"], "allow_fallbacks": False},
            "reasoning": {"enabled": False},
            "top_k": 1,
        },
    }
    teacher_cfg.update(teacher)
    spec = TaskSpec.model_validate(
        {
            "task": TASK,
            "type": "classification",
            "labels_file": "labels.txt",
            "teacher": teacher_cfg,
            "student": {"system_prompt": "Classify the message."},
            "cascade": {"target": 0.97},
        }
    )
    spec.teacher_prompt = PROMPT
    spec.labels = list(LABELS)
    spec.source = f"tasks/{TASK}/task.yaml"
    return spec


INFO = {
    A: ModelInfo(
        A,
        PARAMS | {"reasoning", "response_format"},
        {"mandatory": False, "default_enabled": True},
        {"alpha/fp8": PARAMS | {"reasoning", "response_format"}, "alpha/bf16": PARAMS},
    ),
    B: ModelInfo(B, PARAMS | {"response_format"}, None, {"beta/fp8": PARAMS | {"response_format"}}),
    C: ModelInfo(C, PARAMS | {"reasoning"}, {"mandatory": False}, {"gamma": PARAMS | {"reasoning"}}),
}


def info_lookup(model: str) -> ModelInfo | None:
    return INFO.get(model)


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "home"
    monkeypatch.setenv("TASKDISTILL_HOME", str(workspace))
    for name in ("TASKDISTILL_TEACHER_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return workspace


@pytest.fixture
def pricing() -> PricingSnapshot:
    return PricingSnapshot(
        date="2026-09-26",
        source="test",
        models={
            A: {"default": PRICES[A], "providers": {"alpha/fp8": PRICES[A]}},
            B: {"default": PRICES[B], "providers": {"beta/fp8": PRICES[B]}},
            C: {"default": PRICES[C], "providers": {"gamma": PRICES[C]}},
        },
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "store.sqlite")
    rows: list[dict[str, Any]] = [
        {"input": text_of(i), "gold": gold_of(i), "meta": {"split": "valid"}} for i in range(100)
    ]
    rows += [{"input": f"train {i}", "gold": gold_of(i), "meta": {"split": "train"}} for i in range(5)]
    rows += [{"input": f"test {i}", "gold": gold_of(i), "meta": {"split": "test"}} for i in range(5)]
    rows += [{"input": f"unlabelled {i}", "meta": {"split": "valid"}} for i in range(5)]
    store.add_imports(TASK, "inputs", rows)
    return store


class Teachers:
    """A teacher factory on a temporary ledger; uses the cache the bake-off passes and remembers how it was called."""

    def __init__(self, tmp_path: Path, pricing: PricingSnapshot) -> None:
        self.ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=5.0)
        self.pricing = pricing
        self.calls: list[dict[str, Any]] = []

    def __call__(self, spec: TaskSpec, **kw: Any) -> LiveTeacher:
        self.calls.append(kw)
        return LiveTeacher(
            spec.teacher.base_url,
            "sk-test",
            ledger=self.ledger,
            pricing=self.pricing,
            task=spec.task,
            phase=kw["phase"],
            run_id=kw["run_id"],
            cache=kw["cache"] if kw.get("use_cache", True) else None,
            run_cap=kw.get("run_cap"),
            rng=random.Random(7),
        )


@pytest.fixture
def teachers(tmp_path: Path, pricing: PricingSnapshot) -> Teachers:
    return Teachers(tmp_path, pricing)


Behaviour = Callable[[str, int], dict[str, Any]]


def answer(output: str | None, **extra: Any) -> dict[str, Any]:
    return {"output": output, **extra}


def always_right(gold: str, i: int) -> dict[str, Any]:
    return answer(spoken(gold))


def wrong_every(k: int) -> Behaviour:
    def behave(gold: str, i: int) -> dict[str, Any]:
        return answer(spoken(LABELS[(LABELS.index(gold) + 1) % len(LABELS)]) if i % k == 0 else gold)

    return behave


class FakeRouter:
    """Answers chat completions per model, recording every body it receives."""

    def __init__(self, behaviour: dict[str, Behaviour], costs: dict[str, float] | None = None) -> None:
        self.behaviour = behaviour
        self.costs = costs or {model: cost_of(model) for model in behaviour}
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        model = body["model"]
        text = body["messages"][1]["content"]
        i = index_of(text)
        spec = self.behaviour[model](gold_of(i), i)
        usage: dict[str, Any] = {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": spec.get("completion_tokens", COMPLETION_TOKENS),
            "cost": self.costs[model],
        }
        if "reasoning_tokens" in spec:
            usage["completion_tokens_details"] = {"reasoning_tokens": spec["reasoning_tokens"]}
        return httpx.Response(
            200,
            json={
                "id": f"gen-{i}",
                "object": "chat.completion",
                "created": 1790000000,
                "model": model + "-20260901",
                "provider": PROVIDER_NAMES[model],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": spec["output"]},
                        "finish_reason": spec.get("finish_reason", "stop"),
                    }
                ],
                "usage": usage,
            },
        )

    def of(self, model: str) -> list[dict[str, Any]]:
        return [b for b in self.bodies if b["model"] == model]


def bake(
    spec: TaskSpec,
    store: Store,
    teachers: Teachers,
    pricing: PricingSnapshot,
    router: FakeRouter,
    out_dir: Path,
    *,
    models: list[str],
    **kw: Any,
) -> tuple[dict[str, Any], respx.Route]:
    params: dict[str, Any] = {"n": 100, "yes": True, "model_info": info_lookup, "log": lambda _: None}
    params.update(kw)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(URL).mock(side_effect=router)
        report = run_bakeoff(
            spec, models=models, store=store, pricing=pricing, out_dir=out_dir, teacher_factory=teachers, **params
        )
    return report, route


# selection end to end --------------------------------------------------------------------------
def test_the_cheapest_candidate_within_two_points_is_chosen(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({A: always_right, B: wrong_every(100), C: wrong_every(10)})
    out_dir = tmp_path / "reports" / "bakeoff"
    report, route = bake(
        make_spec(), store, teachers, pricing, router, out_dir, models=[f"{A}@alpha/fp8", B, f"{C}@gamma"]
    )
    assert route.call_count == 300
    rows = {row["model"]: row for row in report["candidates"]}
    assert rows[A]["gold_score"] == 1.0
    assert rows[B]["gold_score"] == pytest.approx(0.99)
    assert rows[C]["gold_score"] == pytest.approx(0.90)
    assert [rows[m]["eligible"] for m in (A, B, C)] == [True, True, False]
    chosen = report["chosen"]
    assert (chosen["model"], chosen["provider"], chosen["candidate"]) == (B, None, B)
    assert chosen["teacher"]["model"] == B
    assert chosen["teacher"]["extra_body"] == {"top_k": 1}

    a = rows[A]
    assert a["n"] == 100 and a["invalid_outputs"] == 0 and a["errors"] == 0
    assert a["cost_usd"] == pytest.approx(100 * cost_of(A))
    assert a["cost_per_1k_usd"] == pytest.approx(1000 * cost_of(A))
    assert a["spent_usd"] == pytest.approx(100 * cost_of(A))
    assert a["latency_ms"]["n"] == 100 and a["latency_ms"]["p50"] is not None and a["latency_ms"]["p95"] is not None
    assert a["truncated"] == 0 and a["reasoning_tokens"] == 0
    assert a["served_by"] == {"Alpha": 100}
    assert a["models_returned"] == {A + "-20260901": 100}
    assert (a["json_mode"], rows[B]["json_mode"], rows[C]["json_mode"]) == (True, None, False)  # B is unpinned
    assert a["cache_hits"] == 0
    assert a["checks"] == {"no_truncation": True, "no_reasoning_tokens": True, "few_failures": True}
    assert chosen["pinned"] is False
    assert "served by Beta (100)" in chosen["warning"] and "slug@provider-tag" in chosen["warning"]
    assert any("no provider pin" in note for note in rows[B]["notes"])
    assert isinstance(teachers.calls[0]["cache"], BakeoffCache)

    assert report["task"] == TASK and report["n"] == 100
    assert report["inputs"] == "valid split with gold, first n by seed 13"
    assert report["prompt_variant"] is None
    assert report["rule"]["metrics"] == ["accuracy"] and report["rule"]["within"] == 0.02
    assert report["spent_usd"] == pytest.approx(100 * (cost_of(A) + cost_of(B) + cost_of(C)))
    assert teachers.calls[0]["phase"] == "bakeoff" and teachers.calls[0]["run_cap"] is None
    assert teachers.ledger.spent(task=TASK, phase="bakeoff") == pytest.approx(report["spent_usd"])

    path = out_dir / f"{TASK}.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert str(tmp_path) not in path.read_text(encoding="utf-8")


def test_free_variants_are_rejected_before_any_call(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({A: always_right})
    with pytest.raises(BakeoffError, match="':free' variants are rejected"):
        bake(make_spec(), store, teachers, pricing, router, tmp_path, models=[A, "vendor/model-a:free"])
    assert router.bodies == [] and teachers.calls == []
    with pytest.raises(BakeoffError, match=":free"):
        parse_candidate("vendor/model-a:free@alpha/fp8")
    assert not list(tmp_path.glob("*.json"))


def test_candidate_syntax() -> None:
    assert parse_candidate(" deepseek/deepseek-v4-flash-0731@deepinfra/fp8 ") == Candidate(
        "deepseek/deepseek-v4-flash-0731", "deepinfra/fp8"
    )
    assert parse_candidates(["a/x,b/y@beta", "c/z"]) == [Candidate("a/x"), Candidate("b/y", "beta"), Candidate("c/z")]
    for bad in ("", "a/x@", "@beta", "a/x@b@c", "a/ x"):
        with pytest.raises(BakeoffError):
            parse_candidates([bad])
    with pytest.raises(BakeoffError, match="duplicate candidates: a/x@beta"):
        parse_candidates(["a/x@beta", "a/x@beta"])


# request bodies --------------------------------------------------------------------------------
def test_reasoning_is_disabled_only_for_models_that_support_it_and_tags_pin_the_provider(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({A: always_right, B: always_right, C: always_right})
    report, _ = bake(make_spec(), store, teachers, pricing, router, tmp_path, models=[f"{A}@alpha/fp8", B, C], n=10)
    for body in router.of(A):
        assert body["reasoning"] == {"enabled": False}
        assert body["provider"] == {
            "order": ["alpha/fp8"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        }
        assert body["top_k"] == 1
        assert body["messages"][0] == {"role": "system", "content": PROMPT}
    for body in router.of(B):
        assert "reasoning" not in body  # a pure instruct model never gets the field
        assert "provider" not in body  # the spec's pin belongs to the spec's own model
    for body in router.of(C):
        assert body["reasoning"] == {"enabled": False}
        assert "provider" not in body
    rows = {row["model"]: row for row in report["candidates"]}
    assert [rows[m]["reasoning_disabled"] for m in (A, B, C)] == [True, False, True]
    assert rows[A]["request"]["extra_body"]["provider"] == provider_pin("alpha/fp8")


def test_an_endpoint_without_the_reasoning_parameter_does_not_get_it(pricing: PricingSnapshot) -> None:
    plan = plan_candidate(make_spec(), Candidate(A, "alpha/bf16"), PROMPT, INFO[A], pricing)
    assert not plan.disable_reasoning
    assert "reasoning" not in plan.spec.teacher.extra_body
    assert plan.json_mode is False
    assert plan.notes


def test_json_mode_follows_the_pinned_endpoint(pricing: PricingSnapshot) -> None:
    spec = make_spec()
    assert plan_candidate(spec, Candidate(A, "alpha/fp8"), PROMPT, INFO[A], pricing).json_mode is True
    assert plan_candidate(spec, Candidate(A, "ALPHA/FP8"), PROMPT, INFO[A], pricing).json_mode is True
    assert plan_candidate(spec, Candidate(A, "alpha"), PROMPT, INFO[A], pricing).json_mode is False  # bf16 lacks it
    unpinned = plan_candidate(spec, Candidate(A), PROMPT, INFO[A], pricing)
    assert unpinned.json_mode is None  # the serving provider is not known in advance
    assert unpinned.disable_reasoning  # the model's union says it reasons
    unknown = plan_candidate(spec, Candidate(A, "alpha/fp8"), PROMPT, None, pricing, allow_unknown=True)
    assert unknown.json_mode is None and not unknown.disable_reasoning
    assert any("capabilities unknown" in note for note in unknown.notes)
    no_endpoints = ModelInfo(A, PARAMS | {"reasoning"}, None, None)
    assert plan_candidate(spec, Candidate(A, "alpha/fp8"), PROMPT, no_endpoints, pricing).json_mode is None


def test_unusable_candidates_are_rejected_up_front(pricing: PricingSnapshot) -> None:
    spec = make_spec()
    mandatory = ModelInfo(A, PARAMS | {"reasoning"}, {"mandatory": True}, None)
    with pytest.raises(BakeoffError, match="reasoning is mandatory"):
        plan_candidate(spec, Candidate(A), PROMPT, mandatory, pricing)
    with pytest.raises(BakeoffError, match="provider tag 'delta' does not serve"):
        plan_candidate(spec, Candidate(A, "delta"), PROMPT, INFO[A], pricing)
    with pytest.raises(BakeoffError, match="no price"):
        plan_candidate(spec, Candidate("vendor/unpriced"), PROMPT, None, pricing)
    json_spec = make_spec(response_format={"type": "json_object"})
    with pytest.raises(BakeoffError, match="does not support response_format"):
        plan_candidate(json_spec, Candidate(C, "gamma"), PROMPT, INFO[C], pricing)
    with pytest.raises(BakeoffError, match="supported parameters are unknown"):
        plan_candidate(spec, Candidate(A, "alpha/fp8"), PROMPT, None, pricing)
    unlisted = ModelInfo(A, None, None, {"alpha/fp8": None})
    with pytest.raises(BakeoffError, match="supported parameters are unknown"):
        plan_candidate(spec, Candidate(A, "alpha/fp8"), PROMPT, unlisted, pricing)
    with pytest.raises(BakeoffError, match="supported parameters are unknown"):
        plan_candidate(spec, Candidate(A), PROMPT, unlisted, pricing)
    # The model-wide union is unknown but the pinned endpoint lists its parameters: that is enough.
    endpoint_only = ModelInfo(A, None, None, {"alpha/fp8": PARAMS | {"reasoning"}})
    plan = plan_candidate(spec, Candidate(A, "alpha/fp8"), PROMPT, endpoint_only, pricing)
    assert plan.disable_reasoning and plan.json_mode is False


def test_the_packaged_snapshot_knows_which_candidates_reason() -> None:
    flash = snapshot_model_info("deepseek/deepseek-v4-flash-0731")
    assert flash is not None
    assert "reasoning" in flash.supported_parameters and not flash.reasoning_mandatory
    params = flash.parameters_for("deepinfra/fp8")
    assert params is not None and {"reasoning", "response_format"} <= params
    assert snapshot_model_info("deepseek/deepseek-v4-flash-20260731") is not None  # canonical slug
    qwen = snapshot_model_info("qwen/qwen3-235b-a22b-2507")
    assert qwen is not None and "reasoning" not in qwen.supported_parameters and qwen.reasoning is None
    assert snapshot_model_info("vendor/unknown") is None


def test_model_info_can_be_fetched_from_the_models_list() -> None:
    models = {
        "data": [
            {
                "id": "vendor/hybrid",
                "canonical_slug": "vendor/hybrid-20260901",
                "supported_parameters": ["max_tokens", "reasoning", "response_format"],
                "reasoning": {"mandatory": False, "default_enabled": True},
            }
        ]
    }
    endpoints = {
        "data": {
            "id": "vendor/hybrid",
            "endpoints": [
                {"tag": "alpha/fp8", "supported_parameters": ["max_tokens", "reasoning", "response_format"]},
                {"tag": "beta", "supported_parameters": ["max_tokens"]},
            ],
        }
    }
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json=models))
        mock.get(f"{BASE}/models/vendor/hybrid/endpoints").mock(return_value=httpx.Response(200, json=endpoints))
        info = fetch_model_info("vendor/hybrid-20260901", BASE)
        assert fetch_model_info("vendor/missing", BASE) is None
    assert info is not None
    assert info.reasoning == {"mandatory": False, "default_enabled": True}
    assert info.parameters_for("alpha/fp8") == {"max_tokens", "reasoning", "response_format"}
    assert info.parameters_for("beta") == {"max_tokens"}
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{BASE}/models").mock(return_value=httpx.Response(503))
        assert fetch_model_info("vendor/hybrid", BASE) is None


def test_a_listing_without_supported_parameters_means_unknown_not_none() -> None:
    models = {"data": [{"id": "vendor/plain"}]}
    endpoints = {"data": {"endpoints": [{"tag": "alpha"}, {"tag": "beta", "supported_parameters": ["max_tokens"]}]}}
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json=models))
        mock.get(f"{BASE}/models/vendor/plain/endpoints").mock(return_value=httpx.Response(200, json=endpoints))
        info = fetch_model_info("vendor/plain", BASE)
    assert info is not None
    assert info.supported_parameters is None
    assert info.serves("alpha") is True
    assert info.parameters_for("alpha") is None
    assert info.parameters_for("beta") == {"max_tokens"}


# projection gate -------------------------------------------------------------------------------
def test_the_projection_gate_stops_after_the_sample(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({A: always_right, B: always_right}, costs={A: 0.01, B: 0.01})
    out_dir = tmp_path / "out"
    with pytest.raises(SpendNotConfirmed, match="--yes"):
        bake(make_spec(), store, teachers, pricing, router, out_dir, models=[A, B], yes=False)
    assert len(router.bodies) == 50
    assert {b["model"] for b in router.bodies} == {A}
    assert teachers.ledger.spent(task=TASK) == pytest.approx(0.5)
    assert teachers.ledger.open_reservations() == 0
    assert not out_dir.exists()


def test_the_projection_prices_every_candidate_from_the_snapshot(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({A: always_right, B: always_right}, costs={A: 1e-4, B: 3e-5})
    lines: list[str] = []
    report, route = bake(
        make_spec(), store, teachers, pricing, router, tmp_path, models=[A, B], yes=False, log=lines.append
    )
    assert route.call_count == 200
    projection = report["projection"]
    assert projection["sample_n"] == 50 and projection["sample_live"] == 50
    assert projection["calls_left"] == 150
    b_per_call = PROMPT_TOKENS * PRICES[B].prompt + COMPLETION_TOKENS * PRICES[B].completion
    expected = 50 * 1e-4 + 50 * 1e-4 + 100 * b_per_call
    assert projection["projected_usd"] == pytest.approx(expected)
    assert any("projected spend $" in line for line in lines)


# cache and latency -----------------------------------------------------------------------------
def test_cache_hits_are_excluded_from_latency(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({A: always_right})
    first, _ = bake(make_spec(), store, teachers, pricing, router, tmp_path, models=[A], n=10)
    assert first["candidates"][0]["latency_ms"]["n"] == 10
    second, route = bake(make_spec(), store, teachers, pricing, router, tmp_path, models=[A], n=20)
    assert route.call_count == 10
    row = second["candidates"][0]
    assert row["cache_hits"] == 10
    assert row["latency_ms"]["n"] == 10
    assert row["cost_usd"] == pytest.approx(20 * cost_of(A))  # the recorded usage.cost of every output
    assert row["spent_usd"] == pytest.approx(10 * cost_of(A))  # only this run's live calls
    third, route = bake(make_spec(), store, teachers, pricing, router, tmp_path, models=[A], n=20, yes=False)
    assert route.call_count == 0
    assert third["candidates"][0]["latency_ms"] == {"p50": None, "p95": None, "mean": None, "n": 0}
    assert third["projection"]["projected_usd"] == 0.0


# prompt variants -------------------------------------------------------------------------------
def test_a_prompt_variant_writes_a_separate_report(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({A: always_right})
    out_dir = tmp_path / "bakeoff"
    bake(make_spec(), store, teachers, pricing, router, out_dir, models=[A], n=10)
    base = out_dir / f"{TASK}.json"
    before = base.read_bytes()

    variant = tmp_path / "short.md"
    text = "Answer with the intent label only.\n"
    variant.write_text(text, encoding="utf-8")
    router.bodies.clear()
    report, _ = bake(make_spec(), store, teachers, pricing, router, out_dir, models=[A], n=10, prompt_variant=variant)
    assert base.read_bytes() == before
    path = bakeoff_report_path(out_dir, TASK, variant)
    assert path == out_dir / f"{TASK}.short.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert report["prompt_variant"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert report["teacher_prompt_sha256"] == report["prompt_variant"]
    assert len(router.bodies) == 10
    assert all(body["messages"][0]["content"] == text for body in router.bodies)


# inputs ----------------------------------------------------------------------------------------
def test_inputs_are_validation_rows_with_gold_deduplicated_and_seeded(tmp_path: Path) -> None:
    spec = make_spec()
    store = Store(tmp_path / "store.sqlite")
    rows: list[dict[str, Any]] = [
        {"input": text_of(i), "gold": gold_of(i), "meta": {"split": split}}
        for i, split in enumerate(["valid", "validation", "VAL", "dev", "valid", "valid"])
    ]
    rows += [
        {"input": "  query 0:   where is my card? ", "gold": gold_of(0), "meta": {"split": "valid"}},  # duplicate
        {"input": "query 6: not a label", "gold": "no_such_intent", "meta": {"split": "valid"}},
        {"input": "query 7: no gold", "meta": {"split": "valid"}},
        {"input": "query 8: training", "gold": gold_of(8), "meta": {"split": "train"}},
        {"input": "query 9: no split", "gold": gold_of(9), "meta": {}},
        {"input": "   ", "gold": gold_of(10), "meta": {"split": "valid"}},
        {"input": "query 11: spoken gold", "gold": "Card payment", "meta": {"split": "valid"}},
    ]
    store.add_imports(TASK, "inputs", rows)

    expected = sorted([*(input_hash(text_of(i)) for i in range(6)), input_hash("query 11: spoken gold")])
    random.Random(13).shuffle(expected)
    chosen = select_inputs(spec, store, 100)
    assert [item.input_hash for item in chosen] == expected
    by_hash = {item.input_hash: item for item in chosen}
    assert by_hash[input_hash(text_of(0))].text == text_of(0)
    assert by_hash[input_hash("query 11: spoken gold")].gold == "card_payment"
    assert [item.input_hash for item in select_inputs(spec, store, 3)] == expected[:3]

    nested = make_spec()
    nested.curate.split.predefined = "meta.origin.split"
    store.add_imports(TASK, "inputs", [{"input": "nested", "gold": gold_of(0), "meta": {"origin": {"split": "val"}}}])
    assert [item.text for item in select_inputs(nested, store, 10)] == ["nested"]

    unsplit = make_spec()
    unsplit.curate.split.predefined = None
    with pytest.raises(BakeoffError, match=r"curate\.split\.predefined"):
        select_inputs(unsplit, store, 10)


def test_no_validation_inputs_is_a_clear_error(tmp_path: Path, teachers: Teachers, pricing: PricingSnapshot) -> None:
    empty = Store(tmp_path / "empty.sqlite")
    router = FakeRouter({A: always_right})
    with pytest.raises(BakeoffError, match="no imported validation inputs with gold"):
        bake(make_spec(), empty, teachers, pricing, router, tmp_path, models=[A])
    assert router.bodies == []


def test_the_default_teacher_needs_an_api_key(tmp_path: Path, store: Store, pricing: PricingSnapshot) -> None:
    with pytest.raises(TeacherUnavailable, match="API key"):
        run_bakeoff(
            make_spec(), models=[A], store=store, pricing=pricing, out_dir=tmp_path, model_info=info_lookup, log=print
        )


# truncation, reasoning tokens and failures ----------------------------------------------------
def test_truncation_and_reasoning_tokens_make_a_candidate_ineligible(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    def thinks(gold: str, i: int) -> dict[str, Any]:
        if i % 25 == 0:
            return answer("", finish_reason="length", completion_tokens=24, reasoning_tokens=24)
        return answer(spoken(gold), reasoning_tokens=2)

    def over_limit(gold: str, i: int) -> dict[str, Any]:
        return answer(spoken(gold), completion_tokens=30) if i == 5 else always_right(gold, i)

    router = FakeRouter({A: always_right, B: over_limit, C: thinks})
    report, _ = bake(make_spec(), store, teachers, pricing, router, tmp_path, models=[A, B, C])
    rows = {row["model"]: row for row in report["candidates"]}
    assert rows[C]["truncated"] == 4
    assert rows[C]["reasoning_tokens"] == 4 * 24 + 96 * 2
    assert rows[C]["invalid_outputs"] == 4
    assert rows[C]["checks"] == {"no_truncation": False, "no_reasoning_tokens": False, "few_failures": True}
    assert rows[B]["truncated"] == 1  # more completion tokens than max_tokens
    assert rows[B]["checks"]["no_truncation"] is False
    assert [rows[m]["eligible"] for m in (A, B, C)] == [True, False, False]
    assert report["chosen"]["model"] == A


def test_failed_requests_count_as_wrong_answers(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({A: always_right})

    def flaky(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if index_of(body["messages"][1]["content"]) % 10 == 0:
            return httpx.Response(400, json={"error": {"code": 400, "message": "bad input"}})
        return router(request)

    with respx.mock(assert_all_called=False) as mock:
        mock.post(URL).mock(side_effect=flaky)
        report = run_bakeoff(
            make_spec(),
            models=[A],
            store=store,
            pricing=pricing,
            out_dir=tmp_path,
            teacher_factory=teachers,
            model_info=info_lookup,
            yes=True,
            log=lambda _: None,
        )
    row = report["candidates"][0]
    assert row["errors"] == 10
    assert row["invalid_outputs"] == 0
    assert row["gold_score"] == pytest.approx(0.9)
    assert "HTTP 400" in row["first_error"]
    assert row["cost_usd"] == pytest.approx(90 * cost_of(A))
    assert row["cost_per_1k_usd"] == pytest.approx(1000 * cost_of(A))  # per successful request
    assert row["checks"]["few_failures"] is False
    assert report["chosen"]["model"] is None and "failed requests" in report["chosen"]["reason"]


# selection rule ----------------------------------------------------------------------------------
def row(name: str, cost: float, ok: bool = True, **metrics: float) -> dict[str, Any]:
    primary = metrics.get("accuracy", metrics.get("field_micro_f1"))
    return {
        "candidate": name,
        "model": name,
        "provider": None,
        "gold_score": primary,
        "metrics": metrics,
        "cost_per_1k_usd": cost,
        "request": {"extra_body": {}, "response_format": None},
        "checks": {"no_truncation": ok, "no_reasoning_tokens": True, "few_failures": True},
    }


def test_classification_rule_is_the_cheapest_within_two_points() -> None:
    rows = [
        row("best", 3.0, accuracy=0.95),
        row("edge", 2.0, accuracy=0.93),  # exactly 2 points below: eligible
        row("cheap", 0.5, accuracy=0.9299),
        row("truncating", 0.1, ok=False, accuracy=0.99),  # fails the checks: neither best nor eligible
    ]
    chosen = select_candidate(rows, "classification")
    assert chosen["model"] == "edge"
    assert [r["eligible"] for r in rows] == [True, True, False, False]
    assert "accuracy 0.950" in chosen["reason"]

    tie = [row("x", 1.0, accuracy=0.90), row("y", 1.0, accuracy=0.91)]
    assert select_candidate(tie, "classification")["model"] == "y"


def test_extraction_rule_needs_field_f1_and_json_validity_within_two_points() -> None:
    rows = [
        row("best", 3.0, field_micro_f1=0.95, json_validity=1.0),
        row("sloppy", 1.0, field_micro_f1=0.94, json_validity=0.97),  # validity 3 points below the best
        row("steady", 2.0, field_micro_f1=0.935, json_validity=0.99),
    ]
    chosen = select_candidate(rows, "extraction")
    assert chosen["model"] == "steady"
    assert [r["eligible"] for r in rows] == [True, False, True]


def test_no_candidate_passing_the_checks_chooses_none() -> None:
    rows = [row("a", 1.0, ok=False, accuracy=0.9)]
    chosen = select_candidate(rows, "classification")
    assert chosen["model"] is None and "checks" in chosen["reason"]
    assert rows[0]["eligible"] is False


# extraction end to end -----------------------------------------------------------------------------
SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total_amount": {"type": "number"},
        "po_number": {"type": ["string", "null"]},
    },
    "required": ["invoice_number", "total_amount", "po_number"],
    "additionalProperties": False,
}


def invoice_gold(i: int) -> dict[str, Any]:
    return {"invoice_number": f"INV-{i:04d}", "total_amount": 100.0 + i, "po_number": None if i % 2 else f"PO-{i}"}


def test_extraction_bakeoff_scores_field_f1_and_json_validity(
    tmp_path: Path, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    spec = TaskSpec.model_validate(
        {
            "task": TASK,
            "type": "extraction",
            "schema_file": "schema.json",
            "teacher": {"base_url": BASE, "model": "vendor/incumbent", "max_tokens": 256},
            "student": {"system_prompt": "Extract the invoice fields as JSON."},
            "cascade": {"target": 0.97},
        }
    )
    spec.teacher_prompt = "Extract the fields.\n"
    spec.json_schema = SCHEMA
    store = Store(tmp_path / "store.sqlite")
    store.add_imports(
        TASK, "inputs", [{"input": text_of(i), "gold": invoice_gold(i), "meta": {"split": "valid"}} for i in range(20)]
    )

    def exact(gold: str, i: int) -> dict[str, Any]:
        return answer("```json\n" + json.dumps(invoice_gold(i)) + "\n```")

    def broken(gold: str, i: int) -> dict[str, Any]:
        if i % 10 == 0:
            return answer('{"invoice_number": "INV"')  # not JSON
        if i % 10 == 1:
            return answer(json.dumps({**invoice_gold(i), "total_amount": "lots"}))  # fails the schema
        return answer(json.dumps(invoice_gold(i)))

    router = FakeRouter({A: exact, B: broken})
    report, _ = bake(spec, store, teachers, pricing, router, tmp_path, models=[A, B], n=20)
    rows = {r["model"]: r for r in report["candidates"]}
    assert rows[A]["metrics"]["json_validity"] == 1.0
    assert rows[A]["metrics"]["field_micro_f1"] == 1.0
    assert rows[B]["metrics"]["json_validity"] == pytest.approx(0.8)
    assert rows[B]["invalid_outputs"] == 4
    tp = sum(1 for i in range(20) if i % 10 > 1 for v in invoice_gold(i).values() if v is not None)
    fn = sum(1 for i in range(20) if i % 10 <= 1 for v in invoice_gold(i).values() if v is not None)
    assert rows[B]["metrics"]["field_micro_f1"] == pytest.approx(2 * tp / (2 * tp + fn))
    assert rows[B]["gold_score"] == rows[B]["metrics"]["field_micro_f1"]
    assert report["rule"]["metrics"] == ["field_micro_f1", "json_validity"]
    assert report["chosen"]["model"] == A


# same model, several pins ------------------------------------------------------------------------
TWO_PINS = {
    A: ModelInfo(
        A,
        PARAMS | {"reasoning"},
        {"mandatory": False},
        {"alpha/fp8": PARAMS | {"reasoning"}, "alpha/bf16": PARAMS | {"reasoning"}},
    )
}


def stored(body: dict[str, Any], output: str) -> TeacherResult:
    """A teacher answer as the cache stores it."""
    return TeacherResult(
        key=request_key(body),
        output=output,
        response={},
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        latency_ms=1.0,
        provider="Alpha",
        finish_reason="stop",
        created=0.0,
        source="live",
    )


def test_two_pins_of_one_model_keep_their_own_cache_rows(tmp_path: Path, store: Store) -> None:
    price = ModelPrice(prompt=2.5e-5, completion=0.0)  # 120 prompt tokens -> $0.003 per call
    snapshot = PricingSnapshot(
        date="2026-09-26",
        source="test",
        models={A: {"default": price, "providers": {"alpha/fp8": price, "alpha/bf16": price}}},
    )
    teachers = Teachers(tmp_path, snapshot)
    router = FakeRouter({A: always_right}, costs={A: 0.003})
    models = [f"{A}@alpha/fp8", f"{A}@alpha/bf16"]
    first, route = bake(
        make_spec(), store, teachers, snapshot, router, tmp_path, models=models, model_info=TWO_PINS.get
    )
    assert route.call_count == 200
    assert first["projection"]["projected_usd"] == pytest.approx(first["spent_usd"]) == pytest.approx(0.6)
    fp8 = router.bodies[0]
    bf16 = next(
        b for b in router.bodies if b["provider"]["order"] == ["alpha/bf16"] and b["messages"] == fp8["messages"]
    )
    assert request_key(fp8) == request_key(bf16)  # the same key under two routing contexts

    router.bodies.clear()
    second, route = bake(
        make_spec(), store, teachers, snapshot, router, tmp_path, models=models, model_info=TWO_PINS.get, yes=False
    )
    assert route.call_count == 0
    assert second["projection"]["calls_left"] == 0 and second["projection"]["projected_usd"] == 0.0
    assert second["spent_usd"] == 0.0
    assert [row["cache_hits"] for row in second["candidates"]] == [100, 100]


def test_the_bakeoff_leaves_the_shared_response_cache_alone(
    home: Path, tmp_path: Path, store: Store, pricing: PricingSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    spec = make_spec(model=A, extra_body={"provider": {"order": ["alpha/fp8"], "allow_fallbacks": False}})
    shared = ResponseCache()  # the workspace cache curate labels into
    curated = [build_teacher_request(spec, text_of(i)) for i in range(100)]
    for body in curated:
        shared.put(stored(body, "card_arrival"), request_context(body))
    router = FakeRouter({A: always_right})
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(URL).mock(side_effect=router)
        report = run_bakeoff(
            spec,
            models=[f"{A}@alpha/fp8"],
            n=20,
            yes=True,
            store=store,
            pricing=pricing,
            out_dir=tmp_path / "out",
            model_info=info_lookup,
            log=lambda _: None,
        )
    assert route.call_count == 20  # the pin differs from the spec's, so curate's rows are not reused
    assert request_key(router.bodies[0]) in {request_key(b) for b in curated}
    for body in curated:
        hit = shared.get(request_key(body), request_context(body))
        assert hit is not None and hit.output == "card_arrival"
    assert (home / BAKEOFF_CACHE_NAME).is_file()
    assert report["candidates"][0]["gold_score"] == 1.0


# failures ----------------------------------------------------------------------------------------
def refusing(status: int, message: str, model: str | None = None, router: FakeRouter | None = None) -> Any:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if model is None or body["model"] == model:
            return httpx.Response(status, json={"error": {"code": status, "message": message}})
        assert router is not None
        return router(request)

    return respond


@pytest.mark.parametrize(("status", "hint"), [(401, "API key"), (402, "credit")])
def test_a_refused_key_or_exhausted_credits_stop_the_bakeoff(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot, status: int, hint: str
) -> None:
    out_dir = tmp_path / "out"
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(URL).mock(side_effect=refusing(status, "refused"))
        with pytest.raises(BakeoffError, match=rf"{hint}.*HTTP {status}"):
            run_bakeoff(
                make_spec(),
                models=[f"{A}@alpha/fp8", f"{B}@beta/fp8"],
                n=100,
                yes=True,
                store=store,
                pricing=pricing,
                out_dir=out_dir,
                teacher_factory=teachers,
                model_info=info_lookup,
                log=lambda _: None,
            )
    assert route.call_count <= 50  # stopped in the sample, not after 200 doomed calls
    assert not out_dir.exists()
    assert teachers.ledger.open_reservations() == 0


def test_credits_running_out_on_a_later_candidate_stop_the_run(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({A: always_right})
    out_dir = tmp_path / "out"
    with respx.mock(assert_all_called=False) as mock:
        mock.post(URL).mock(side_effect=refusing(402, "Insufficient credits", model=B, router=router))
        with pytest.raises(BakeoffError, match="HTTP 402"):
            run_bakeoff(
                make_spec(),
                models=[A, B],
                n=100,
                yes=True,
                store=store,
                pricing=pricing,
                out_dir=out_dir,
                teacher_factory=teachers,
                model_info=info_lookup,
                log=lambda _: None,
            )
    assert len(router.of(A)) == 100
    assert not out_dir.exists()
    router.bodies.clear()  # the re-run pays only for the candidate that was cut off
    report, route = bake(
        make_spec(), store, teachers, pricing, FakeRouter({A: always_right, B: always_right}), out_dir, models=[A, B]
    )
    assert route.call_count == 100
    assert report["candidates"][0]["cache_hits"] == 100


def test_a_sample_in_which_every_request_failed_stops_before_the_gate(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({B: always_right})
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(URL).mock(side_effect=refusing(400, "No endpoints found", model=A, router=router))
        with pytest.raises(BakeoffError, match=rf"all 50 sample requests to {A}@alpha/fp8 failed .*HTTP 400"):
            run_bakeoff(
                make_spec(),
                models=[f"{A}@alpha/fp8", B],
                n=100,
                yes=False,
                store=store,
                pricing=pricing,
                out_dir=tmp_path / "out",
                teacher_factory=teachers,
                model_info=info_lookup,
                log=lambda _: None,
            )
    assert route.call_count == 50 and router.bodies == []


def test_candidates_with_more_than_one_percent_failed_requests_are_not_chosen(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({A: always_right, B: always_right, C: always_right})

    def flaky(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        i = index_of(body["messages"][1]["content"])
        if (body["model"] == B and i in (0, 1)) or (body["model"] == C and i == 0) or body["model"] == "vendor/none":
            return httpx.Response(400, json={"error": {"code": 400, "message": "bad input"}})
        return router(request)

    with respx.mock(assert_all_called=False) as mock:
        mock.post(URL).mock(side_effect=flaky)
        report = run_bakeoff(
            make_spec(),
            models=[A, f"{B}@beta/fp8", f"{C}@gamma"],
            n=100,
            yes=True,
            store=store,
            pricing=pricing,
            out_dir=tmp_path,
            teacher_factory=teachers,
            model_info=info_lookup,
            log=lambda _: None,
        )
    rows = {row["model"]: row for row in report["candidates"]}
    assert (rows[B]["errors"], rows[C]["errors"]) == (2, 1)
    assert rows[B]["gold_score"] == pytest.approx(0.98) and rows[C]["gold_score"] == pytest.approx(0.99)
    assert rows[B]["checks"]["few_failures"] is False and rows[C]["checks"]["few_failures"] is True
    assert rows[B]["cost_per_1k_usd"] == pytest.approx(1000 * cost_of(B))
    assert [rows[m]["eligible"] for m in (A, B, C)] == [True, False, True]
    chosen = report["chosen"]
    assert chosen["candidate"] == f"{C}@gamma" and chosen["pinned"] is True and "warning" not in chosen
    assert report["rule"]["max_failure_rate"] == 0.01


# budget ------------------------------------------------------------------------------------------
def test_a_projection_above_max_usd_stops_after_the_sample(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({A: always_right, B: always_right}, costs={A: 0.01, B: 0.01})
    out_dir = tmp_path / "out"
    with pytest.raises(BakeoffError, match=r"projected spend \$1\.0252 .* above --max-usd \$0\.90"):
        bake(make_spec(), store, teachers, pricing, router, out_dir, models=[A, B], max_usd=0.9)
    assert len(router.bodies) == 50 and {b["model"] for b in router.bodies} == {A}
    assert teachers.calls[0]["run_cap"] == 0.9
    assert teachers.ledger.open_reservations() == 0
    assert not out_dir.exists()


# capabilities ------------------------------------------------------------------------------------
def test_unknown_capabilities_are_refused_unless_allowed_and_notes_are_logged_before_spending(
    tmp_path: Path, store: Store, teachers: Teachers, pricing: PricingSnapshot
) -> None:
    router = FakeRouter({A: always_right})
    with pytest.raises(BakeoffError, match="supported parameters are unknown"):
        bake(make_spec(), store, teachers, pricing, router, tmp_path, models=[A], model_info=lambda _: None)
    assert router.bodies == [] and teachers.calls == []

    lines: list[str] = []
    report, _ = bake(
        make_spec(),
        store,
        teachers,
        pricing,
        router,
        tmp_path,
        models=[A],
        n=10,
        model_info=lambda _: None,
        allow_unknown_capabilities=True,
        log=lines.append,
    )
    note = next(i for i, line in enumerate(lines) if "capabilities unknown" in line)
    projected = next(i for i, line in enumerate(lines) if "projected spend" in line)
    assert note < projected
    assert all("reasoning" not in body for body in router.bodies)
    assert report["candidates"][0]["json_mode"] is None


# input selection ---------------------------------------------------------------------------------
def test_inputs_resolve_split_and_gold_across_rows_like_curate(tmp_path: Path) -> None:
    store = Store(tmp_path / "store.sqlite")
    rows: list[dict[str, Any]] = [
        {"input": "in test too", "gold": LABELS[0], "meta": {"split": "valid"}},
        {"input": "in test too", "gold": LABELS[0], "meta": {"split": "test"}},  # test wins: never scored
        {"input": "in train too", "gold": LABELS[1], "meta": {"split": "train"}},
        {"input": "in train too", "meta": {"split": "valid"}},  # valid wins; the train row brings the gold
        {"input": "majority", "gold": LABELS[2], "meta": {"split": "valid"}},
        {"input": "majority", "gold": LABELS[2], "meta": {"split": "valid"}},
        {"input": "majority", "gold": LABELS[3], "meta": {"split": "valid"}},
        {"input": "tied", "gold": LABELS[0], "meta": {"split": "valid"}},
        {"input": "tied", "gold": LABELS[1], "meta": {"split": "valid"}},  # no strict majority: left out
    ]
    store.add_imports(TASK, "inputs", rows)
    chosen = {item.text: item.gold for item in select_inputs(make_spec(), store, 100)}
    assert chosen == {"in train too": LABELS[1], "majority": LABELS[2]}


def test_extraction_gold_must_satisfy_the_schema(tmp_path: Path) -> None:
    spec = TaskSpec.model_validate(
        {
            "task": TASK,
            "type": "extraction",
            "schema_file": "schema.json",
            "teacher": {"base_url": BASE, "model": "vendor/incumbent", "max_tokens": 256},
            "student": {"system_prompt": "Extract the invoice fields as JSON."},
            "cascade": {"target": 0.97},
        }
    )
    spec.json_schema = SCHEMA
    store = Store(tmp_path / "store.sqlite")
    store.add_imports(
        TASK,
        "inputs",
        [
            {"input": "good", "gold": invoice_gold(1), "meta": {"split": "valid"}},
            {"input": "bad", "gold": {**invoice_gold(2), "total_amount": "lots"}, "meta": {"split": "valid"}},
        ],
    )
    assert [(item.text, item.gold) for item in select_inputs(spec, store, 10)] == [("good", invoice_gold(1))]


def test_the_bakeoff_cache_keeps_one_row_per_key_and_context(tmp_path: Path) -> None:
    cache = BakeoffCache(tmp_path / "bakeoff.sqlite")
    body = build_teacher_request(make_spec(model=A), text_of(0))
    result = stored(body, "lost_card")
    cache.put(result, "ctx-a")
    cache.put(stored(body, "card_arrival"), "ctx-b")
    hit_a, hit_b = cache.get(result.key, "ctx-a"), cache.get(result.key, "ctx-b")
    assert hit_a is not None and hit_b is not None
    assert (hit_a.output, hit_b.output) == ("lost_card", "card_arrival")
    assert hit_a.key == hit_b.key == request_key(body)
    assert cache.get(result.key, "ctx-c") is None and len(cache) == 2
