"""Import (openai | pairs | inputs) and export of captured traffic, including the export -> import round trip."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from taskdistill.capture.export import export_file
from taskdistill.capture.importer import FORMATS, ImportFormatError, import_file
from taskdistill.capture.proxy import create_proxy_app
from taskdistill.store import Store
from taskdistill.teacher.request_key import request_key

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*httpx2.*")
    from fastapi.testclient import TestClient

UPSTREAM = "http://upstream.test/v1"


def _request(text: str, model: str = "example/teacher") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "system", "content": "Classify."}, {"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 24,
    }


def _response(label: str, *, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "example/teacher-0731",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": label}, "finish_reason": "stop"}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def _write(path: Path, lines: list[Any], *, gz: bool = False) -> Path:
    text = "".join((ln if isinstance(ln, str) else json.dumps(ln, ensure_ascii=False)) + "\n" for ln in lines)
    data = text.encode("utf-8")
    if gz:
        with open(path, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as out:
            out.write(data)
    else:
        path.write_bytes(data)
    return path


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store.sqlite")


# pairs and inputs ---------------------------------------------------------------------------------


def test_pairs_import_keeps_output_gold_and_meta_split(store: Store, tmp_path: Path) -> None:
    path = _write(
        tmp_path / "pairs.jsonl",
        [
            {
                "input": "Where is my card?",
                "output": "card_arrival",
                "gold": "card_arrival",
                "meta": {"split": "train"},
            },
            "",
            {"input": "My card was declined", "output": "declined_card_payment", "meta": {"split": "valid", "g": 3}},
            {"input": "Top up failed", "output": "top_up_failed", "gold": None, "meta": {"split": "test"}},
        ],
    )
    counts = import_file(store, "banking", path, "pairs")

    assert counts == {"read": 3, "imported": 3, "blank": 1, "format": "pairs"}
    rows = list(store.iter_imports("banking"))
    assert [(r.format, r.input, r.output, r.gold) for r in rows] == [
        ("pairs", "Where is my card?", "card_arrival", "card_arrival"),
        ("pairs", "My card was declined", "declined_card_payment", None),
        ("pairs", "Top up failed", "top_up_failed", None),
    ]
    assert [r.meta for r in rows] == [{"split": "train"}, {"g": 3, "split": "valid"}, {"split": "test"}]
    assert store.count_captures("banking")["total"] == 0


def test_inputs_import_has_no_output_and_keeps_structured_gold(store: Store, tmp_path: Path) -> None:
    gold = {"invoice_number": "INV-0042", "total": 1234.5, "due_date": None, "vendor": "Café Example Ltd"}
    path = _write(
        tmp_path / "inputs.jsonl",
        [
            {"input": "Invoice INV-0042 ...", "gold": gold, "meta": {"split": "test", "template": "t3"}},
            {"input": "Invoice INV-0043 ...", "meta": None},
            {"input": "Invoice INV-0044 ...", "gold": 7},
        ],
    )
    counts = import_file(store, "invoices", path, "inputs")

    assert counts == {"read": 3, "imported": 3, "blank": 0, "format": "inputs"}
    rows = list(store.iter_imports("invoices"))
    assert [r.output for r in rows] == [None, None, None]
    assert [r.gold for r in rows] == [gold, None, 7]
    assert [r.meta for r in rows] == [{"split": "test", "template": "t3"}, {}, {}]
    assert rows[0].format == "inputs"


def test_pairs_object_output_is_stored_as_compact_json(store: Store, tmp_path: Path) -> None:
    path = _write(tmp_path / "p.jsonl", [{"input": "Invoice ...", "output": {"vendor": "Café", "total": 12.5}}])
    import_file(store, "invoices", path, "pairs")
    [row] = store.iter_imports("invoices")
    assert row.output == '{"vendor":"Café","total":12.5}'


def test_gzip_bom_and_crlf_are_accepted(store: Store, tmp_path: Path) -> None:
    gz = _write(tmp_path / "rows.jsonl.gz", [{"input": "a"}, {"input": "b"}], gz=True)
    assert import_file(store, "t", gz, "inputs") == {"read": 2, "imported": 2, "blank": 0, "format": "inputs"}

    crlf = tmp_path / "crlf.jsonl"
    crlf.write_bytes(b'\xef\xbb\xbf{"input": "c"}\r\n\r\n{"input": "d"}\r\n')
    assert import_file(store, "t", crlf, "inputs") == {"read": 2, "imported": 2, "blank": 1, "format": "inputs"}
    assert [r.input for r in store.iter_imports("t")] == ["a", "b", "c", "d"]


def test_empty_file_imports_nothing(store: Store, tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    assert import_file(store, "t", path, "pairs") == {"read": 0, "imported": 0, "blank": 0, "format": "pairs"}


# openai -------------------------------------------------------------------------------------------


def test_openai_import_stores_captures(store: Store, tmp_path: Path) -> None:
    usage = {"prompt_tokens": 42, "completion_tokens": 3, "total_tokens": 45, "cost": 0.000123}
    first = {"request": _request("Where is my card?"), "response": _response("card_arrival", usage=usage)}
    second = {"request": _request("Card declined"), "response": _response("declined_card_payment")}
    path = _write(tmp_path / "log.jsonl", [first, second])

    counts = import_file(store, "banking", path, "openai")

    assert counts == {"read": 2, "imported": 2, "blank": 0, "format": "openai"}
    rows = list(store.iter_captures("banking"))
    assert [r.request_key for r in rows] == [request_key(first["request"]), request_key(second["request"])]
    a, b = rows
    assert (a.source, a.status, a.captured, a.error, a.latency_ms) == ("import", 200, True, None, None)
    assert json.loads(a.request_body or "") == first["request"]
    assert json.loads(a.response_body or "") == first["response"]
    assert (a.prompt_tokens, a.completion_tokens, a.upstream_model) == (42, 3, "example/teacher-0731")
    assert a.cost_usd == pytest.approx(0.000123)
    assert (b.prompt_tokens, b.completion_tokens, b.cost_usd) == (None, None, None)
    assert store.count_captures("banking") == {"total": 2, "captured": 2, "not_captured": 0}
    assert list(store.iter_imports("banking")) == []


# validation errors --------------------------------------------------------------------------------

GOOD_OPENAI = json.dumps({"request": _request("x"), "response": _response("y")})
# A choice ended by an upstream error after generation started (OpenRouter returns these with HTTP 200).
FINISH_ERROR_RESPONSE: dict[str, Any] = {
    **_response("card_"),
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "card_"},
            "finish_reason": "error",
            "error": {"code": 502, "message": "Provider disconnected"},
        }
    ],
}


@pytest.mark.parametrize(
    ("fmt", "lines", "expected"),
    [
        ("pairs", ['{"input": "a", "output": "b"}', '{"input": "a", "output": }'], "bad.jsonl:2: invalid JSON"),
        ("inputs", ["", '{"input": "a"}', "", "oops"], "bad.jsonl:4: invalid JSON"),
        ("inputs", ["[", '  {"input": "a"}', "]"], "bad.jsonl:1: invalid JSON"),
        ("inputs", ["[", '  {"input": "a"}', "]"], "expected JSON Lines"),
        ("inputs", ['["a", "b"]'], "bad.jsonl:1: expected a JSON object, got array"),
        ("pairs", ['{"output": "b"}'], 'bad.jsonl:1: missing "input"'),
        ("inputs", ['{"input": "   "}'], 'bad.jsonl:1: "input" is empty'),
        ("inputs", ['{"input": 5}'], '"input" must be a string, got number'),
        ("inputs", ['{"input": null}'], '"input" must be a string, got null'),
        ("inputs", ['{"input": "a", "meta": ["train"]}'], '"meta" must be a JSON object, got array'),
        ("pairs", ['{"input": "a", "output": "b", "meta": "train"}'], '"meta" must be a JSON object, got string'),
        ("pairs", ['{"input": "a"}'], 'missing "output"'),
        ("pairs", ['{"input": "a", "output": 3}'], '"output" must be a string or a JSON object, got number'),
        ("pairs", ['{"input": "a", "output": ""}'], '"output" is empty'),
        ("inputs", ['{"input": "a", "glod": "x"}'], "unknown key(s) 'glod' for format inputs"),
        ("inputs", ['{"input": "a", "output": "x"}'], "unknown key(s) 'output' for format inputs"),
        (
            "openai",
            [GOOD_OPENAI, '{"request": {"messages": [{"role": "user", "content": "x"}]}}'],
            ':2: missing "response"',
        ),
        ("openai", ['{"request": "x", "response": {}}'], '"request" must be a JSON object, got string'),
        (
            "openai",
            ['{"request": {"model": "m"}, "response": {"choices": [{}]}}'],
            '"request.messages" must be a non-empty',
        ),
        (
            "openai",
            ['{"request": {"messages": [{"role": "user", "content": "x"}]}, "response": {"choices": []}}'],
            '"response.choices" must be a non-empty array',
        ),
        ("openai", ['{"request": {}, "response": {}, "meta": {}}'], "unknown key(s) 'meta' for format openai"),
        (
            "openai",
            [json.dumps({"request": _request("x"), "response": {"error": {"code": 502, "message": "Provider error"}}})],
            "bad.jsonl:1: upstream error in a 2xx response",
        ),
        (
            "openai",
            [GOOD_OPENAI, json.dumps({"request": _request("x"), "response": FINISH_ERROR_RESPONSE})],
            "bad.jsonl:2: upstream error in a 2xx response",
        ),
        (
            "openai",
            ['{"request": {"messages": [{"role": "user", "content": "x"}]}, "response": {"choices": ["y"]}}'],
            '"response.choices" must hold objects',
        ),
        (
            "openai",
            [
                '{"request": {"model": "example/teacher", "messages": [{"role": "user", "content": "x"}], '
                '"temperature": NaN}, "response": ' + json.dumps(_response("y")) + "}"
            ],
            'bad.jsonl:1: "request" is not valid JSON',
        ),
        ("inputs", ['{"input": "cut \\ud83d"}'], 'bad.jsonl:1: "input" contains an unpaired UTF-16 surrogate escape'),
        ("pairs", ['{"input": "a", "output": {"v": "\\udc00"}}'], '"output" contains an unpaired UTF-16 surrogate'),
        ("pairs", ['{"input": "a", "output": "b", "gold": ["\\ud83d"]}'], '"gold" contains an unpaired UTF-16'),
        ("inputs", ['{"input": "a", "meta": {"\\ud83d": "x"}}'], '"meta" contains an unpaired UTF-16 surrogate'),
    ],
)
def test_bad_lines_are_rejected_with_line_numbers(
    fmt: str, lines: list[str], expected: str, store: Store, tmp_path: Path
) -> None:
    path = _write(tmp_path / "bad.jsonl", lines)
    with pytest.raises(ImportFormatError) as info:
        import_file(store, "t", path, fmt)
    assert expected in str(info.value)


def test_a_bad_line_imports_nothing(store: Store, tmp_path: Path) -> None:
    pairs = _write(tmp_path / "p.jsonl", [{"input": "a", "output": "b"}, {"input": "c", "output": "d"}, {"input": ""}])
    with pytest.raises(ImportFormatError, match=r"p\.jsonl:3"):
        import_file(store, "t", pairs, "pairs")
    openai = _write(tmp_path / "o.jsonl", [GOOD_OPENAI, "{"])
    with pytest.raises(ImportFormatError, match=r"o\.jsonl:2"):
        import_file(store, "t", openai, "openai")
    assert list(store.iter_imports("t")) == []
    assert store.count_captures("t")["total"] == 0


def _with_cut_emoji(line: str, text: str) -> str:
    """Replace the JSON string ``text`` with one cut inside a surrogate pair (``\\ud83d`` without its partner)."""
    assert f'"{text}"' in line
    return line.replace(f'"{text}"', '"cut \\ud83d"')


@pytest.mark.parametrize(("side", "text"), [("request", "x"), ("response", "y")])
def test_unpaired_surrogate_is_reported_with_its_line_and_nothing_is_written(
    side: str, text: str, store: Store, tmp_path: Path
) -> None:
    path = _write(tmp_path / "s.jsonl", [GOOD_OPENAI, GOOD_OPENAI, _with_cut_emoji(GOOD_OPENAI, text)])
    with pytest.raises(ImportFormatError) as info:
        import_file(store, "t", path, "openai")
    assert str(info.value) == f's.jsonl:3: "{side}" contains an unpaired UTF-16 surrogate escape'
    assert store.count_captures("t") == {"total": 0, "captured": 0, "not_captured": 0}

    pairs = _write(tmp_path / "p.jsonl", ['{"input": "a", "output": "b"}', '{"input": "cut \\ud83d", "output": "b"}'])
    with pytest.raises(ImportFormatError, match=r'^p\.jsonl:2: "input" contains an unpaired UTF-16 surrogate escape$'):
        import_file(store, "t", pairs, "pairs")
    assert list(store.iter_imports("t")) == []


def test_paired_surrogate_escapes_are_accepted(store: Store, tmp_path: Path) -> None:
    path = _write(tmp_path / "e.jsonl", ['{"input": "smile \\ud83d\\ude00", "meta": {"split": "test"}}'])
    assert import_file(store, "t", path, "inputs")["imported"] == 1
    [row] = store.iter_imports("t")
    assert row.input == "smile \U0001f600"


class _BulkStore(Store):
    """A store with a bulk capture insert: the importer must write every openai row through it in one call."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.bulk_calls: list[tuple[str, list[dict[str, Any]]]] = []

    def add_captures(self, task: str, rows: list[dict[str, Any]]) -> int:
        self.bulk_calls.append((task, rows))
        for row in rows:
            super().add_capture(task=task, **row)
        return len(rows)

    def add_capture(self, **_: Any) -> int:
        raise AssertionError("openai rows must be written with add_captures")


def test_openai_rows_use_the_bulk_insert_when_the_store_has_one(tmp_path: Path) -> None:
    store = _BulkStore(tmp_path / "store.sqlite")
    first = {"request": _request("Where is my card?"), "response": _response("card_arrival")}
    path = _write(tmp_path / "log.jsonl", [first, GOOD_OPENAI])

    assert import_file(store, "banking", path, "openai") == {"read": 2, "imported": 2, "blank": 0, "format": "openai"}
    [(task, rows)] = store.bulk_calls
    assert task == "banking"
    assert [(r["source"], r["status"], r["captured"]) for r in rows] == [("import", 200, True)] * 2
    assert [r["request_key"] for r in rows] == [request_key(first["request"]), request_key(_request("x"))]
    assert store.count_captures("banking") == {"total": 2, "captured": 2, "not_captured": 0}


def test_invalid_utf8_and_corrupt_gzip_are_reported(store: Store, tmp_path: Path) -> None:
    latin = tmp_path / "latin.jsonl"
    latin.write_bytes(b'{"input": "a"}\n{"input": "caf\xe9"}\n')
    with pytest.raises(ImportFormatError, match=r"latin\.jsonl:2: not valid UTF-8"):
        import_file(store, "t", latin, "inputs")

    good = _write(tmp_path / "good.jsonl.gz", [{"input": "a" * 200}], gz=True).read_bytes()
    broken = tmp_path / "broken.jsonl.gz"
    broken.write_bytes(good[: len(good) // 2])
    with pytest.raises(ImportFormatError, match=r"broken\.jsonl\.gz:1: cannot decompress gzip data"):
        import_file(store, "t", broken, "inputs")


def test_unknown_format_is_rejected(store: Store, tmp_path: Path) -> None:
    path = _write(tmp_path / "x.jsonl", [{"input": "a"}])
    with pytest.raises(ValueError, match="unknown import format 'csv'; expected one of: openai, pairs, inputs"):
        import_file(store, "t", path, "csv")
    assert FORMATS == ("openai", "pairs", "inputs")


# export -------------------------------------------------------------------------------------------


def _capture(store: Store, task: str, request: str | None, response: str | None, **kw: Any) -> None:
    fields: dict[str, Any] = {"status": 200, "captured": True, **kw}
    store.add_capture(
        task=task,
        source="proxy",
        request_key=request_key(json.loads(request)) if request and request.startswith("{") else None,
        request_body=request,
        response_body=response,
        **fields,
    )


def test_export_writes_only_captured_successful_rows(
    store: Store, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    good_req, good_resp = _request("Where is my card?"), _response("card_arrival", usage={"prompt_tokens": 9})
    _capture(store, "banking", json.dumps(good_req), json.dumps(good_resp))
    _capture(store, "banking", json.dumps(_request("streamed")), None, captured=False, error="stream: not captured")
    _capture(store, "banking", json.dumps(_request("limited")), None, status=429, captured=False)
    _capture(store, "banking", json.dumps(_request("odd")), json.dumps(_response("x")), status=201, captured=False)
    _capture(store, "banking", "not json", json.dumps(_response("x")))
    _capture(store, "banking", json.dumps(_request("no body")), None)
    _capture(store, "other", json.dumps(_request("other task")), json.dumps(_response("x")))

    out = tmp_path / "export.jsonl"
    with caplog.at_level(logging.WARNING, logger="taskdistill.capture.export"):
        count = export_file(store, "banking", out)

    assert count == 1
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln) for ln in lines] == [{"request": good_req, "response": good_resp}]
    assert any("skipped 2 captured row(s)" in r.getMessage() for r in caplog.records)


def test_export_never_writes_a_line_that_import_rejects(
    store: Store, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Rows flagged captured by an older proxy, before 2xx bodies were checked for being chat completions.
    good_req, good_resp = _request("Where is my card?"), _response("card_arrival")
    _capture(store, "banking", json.dumps(good_req), json.dumps(good_resp))
    _capture(store, "banking", json.dumps(_request("e1")), json.dumps({"error": {"code": 502, "message": "x"}}))
    _capture(store, "banking", json.dumps(_request("e2")), json.dumps(FINISH_ERROR_RESPONSE))
    _capture(store, "banking", json.dumps({"model": "m", "prompt": "no messages"}), json.dumps(_response("x")))
    _capture(store, "banking", json.dumps(_request("e3")), json.dumps({**_response("x"), "choices": []}))
    _capture(store, "banking", json.dumps(_request("e4")), _with_cut_emoji(json.dumps(_response("y")), "y"))
    store.add_capture(
        task="banking",
        source="proxy",
        request_key=None,
        request_body=_with_cut_emoji(json.dumps(_request("x")), "x"),
        response_body=json.dumps(_response("y")),
        status=200,
    )

    out = tmp_path / "export.jsonl"
    with caplog.at_level(logging.WARNING, logger="taskdistill.capture.export"):
        assert export_file(store, "banking", out) == 1
    assert [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()] == [
        {"request": good_req, "response": good_resp}
    ]
    assert any("skipped 6 captured row(s)" in r.getMessage() for r in caplog.records)
    target = Store(tmp_path / "target.sqlite")
    assert import_file(target, "banking", out, "openai") == {"read": 1, "imported": 1, "blank": 0, "format": "openai"}


def test_proxied_error_with_status_200_is_not_exported_and_the_export_imports(tmp_path: Path) -> None:
    source = Store(tmp_path / "a.sqlite")
    error_body = b'{"error":{"code":502,"message":"Provider returned error"}}'
    http = httpx.AsyncClient()
    with respx.mock(base_url=UPSTREAM) as router:
        router.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(200, content=json.dumps(_response("card_arrival")).encode()),
                httpx.Response(200, content=error_body),
                httpx.Response(200, content=json.dumps(_response("top_up_failed")).encode()),
            ]
        )
        with TestClient(create_proxy_app(source, UPSTREAM, client=http)) as proxy:
            for text in ("first", "second", "third"):
                assert proxy.post("/t/banking/v1/chat/completions", json=_request(text)).status_code == 200
    asyncio.run(http.aclose())
    assert source.count_captures("banking") == {"total": 3, "captured": 2, "not_captured": 1}

    out = tmp_path / "e.jsonl"
    assert export_file(source, "banking", out) == 2
    target = Store(tmp_path / "b.sqlite")
    assert import_file(target, "banking", out, "openai") == {"read": 2, "imported": 2, "blank": 0, "format": "openai"}
    assert [r.request_key for r in target.iter_captures("banking")] == [
        request_key(_request("first")),
        request_key(_request("third")),
    ]


def test_export_of_nothing_writes_an_empty_file(store: Store, tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "empty.jsonl"
    assert export_file(store, "banking", out) == 0
    assert out.read_bytes() == b""
    assert sorted(p.name for p in out.parent.iterdir()) == ["empty.jsonl"]


def test_export_gzip_has_no_name_and_zero_mtime(store: Store, tmp_path: Path) -> None:
    for i in range(3):
        _capture(store, "banking", json.dumps(_request(f"q{i}")), json.dumps(_response(f"l{i}")))
    plain, packed = tmp_path / "e.jsonl", tmp_path / "e.jsonl.gz"
    assert export_file(store, "banking", plain) == 3
    assert export_file(store, "banking", packed) == 3

    header = packed.read_bytes()[:10]
    assert header[:2] == b"\x1f\x8b"
    assert header[3] & 0x08 == 0  # FNAME flag unset
    assert header[4:8] == b"\x00\x00\x00\x00"  # MTIME
    assert gzip.decompress(packed.read_bytes()) == plain.read_bytes()


def test_export_import_round_trip(tmp_path: Path) -> None:
    source = Store(tmp_path / "a" / "store.sqlite")
    # Proxy-captured bodies keep the client's bytes: odd whitespace and key order.
    raw_request = (
        '{ "max_tokens": 24, "model" : "example/teacher",\n "messages": [{"role": "user", "content": "Café?"}] }'
    )
    raw_response = json.dumps(
        _response("card_arrival", usage={"prompt_tokens": 5, "completion_tokens": 2, "cost": 1e-05})
    )
    _capture(source, "banking", raw_request, raw_response)
    for i in range(3):
        _capture(source, "banking", json.dumps(_request(f"question {i}")), json.dumps(_response(f"label_{i}")))

    first = tmp_path / "first.jsonl.gz"
    assert export_file(source, "banking", first) == 4

    target = Store(tmp_path / "b" / "store.sqlite")
    assert import_file(target, "banking", first, "openai") == {"read": 4, "imported": 4, "blank": 0, "format": "openai"}

    before = list(source.iter_captures("banking"))
    after = list(target.iter_captures("banking"))
    assert [r.request_key for r in after] == [r.request_key for r in before]
    assert all(r.request_key is not None for r in after)
    for old, new in zip(before, after, strict=True):
        assert json.loads(new.request_body or "") == json.loads(old.request_body or "")
        assert json.loads(new.response_body or "") == json.loads(old.response_body or "")
        assert (new.status, new.captured, new.source) == (200, True, "import")
    assert (after[0].prompt_tokens, after[0].completion_tokens, after[0].cost_usd) == (5, 2, pytest.approx(1e-05))

    second = tmp_path / "second.jsonl.gz"
    assert export_file(target, "banking", second) == 4
    assert second.read_bytes() == first.read_bytes()


def test_proxy_capture_export_import_keep_the_request_key(tmp_path: Path) -> None:
    raw = b'{"messages":[{"role":"user","content":"Where is my card?"}] ,"model":"example/teacher","temperature":0}'
    upstream_body = json.dumps(_response("card_arrival")).encode()
    source = Store(tmp_path / "a.sqlite")
    http = httpx.AsyncClient()
    with respx.mock(base_url=UPSTREAM) as router:
        router.post("/chat/completions").mock(return_value=httpx.Response(200, content=upstream_body))
        with TestClient(create_proxy_app(source, UPSTREAM, client=http)) as proxy:
            assert proxy.post("/t/banking/v1/chat/completions", content=raw).status_code == 200
    asyncio.run(http.aclose())

    out = tmp_path / "captured.jsonl"
    assert export_file(source, "banking", out) == 1
    target = Store(tmp_path / "b.sqlite")
    import_file(target, "banking", out, "openai")

    [captured] = source.iter_captures("banking")
    [imported] = target.iter_captures("banking")
    assert captured.request_key == imported.request_key == request_key(json.loads(raw))
