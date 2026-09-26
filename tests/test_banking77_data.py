from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import sys
from collections import Counter
from importlib import resources
from pathlib import Path

import httpx
import pytest
import respx

from taskdistill.curate.extract import normalise_text
from taskdistill.demos import banking77
from taskdistill.demos.banking77 import (
    BankingData,
    ChecksumError,
    DatasetError,
    RemoteFile,
    Row,
    check_counts,
    demo_splits,
    fetch_banking77,
    packaged_labels,
    parse_csv,
    profile_splits,
    quick_subset,
    read_parquet,
    to_records,
)

TASK_DIR = resources.files("taskdistill") / "_data" / "tasks" / "banking77"

# A tiny stand-in for the official files, with the quirks of the real data: a leading newline, an
# embedded newline, commas and quotes inside texts, non-ASCII currency signs, and the two labels that are
# not plain snake_case. The label list is deliberately unsorted, as in categories.json.
CATEGORIES = ["card_arrival", "Refund_not_showing_up", "reverted_card_payment?"]
TRAIN_ROWS = [
    ("I am still waiting on my card?", "card_arrival"),
    ("\nWhich ATMs accept this card?", "card_arrival"),
    ("Hi, where is my refund?", "Refund_not_showing_up"),
    ('My "refund" is\nstill missing', "Refund_not_showing_up"),
    ("Why was my £20 payment reverted?", "reverted_card_payment?"),
    ("payment reverted", "reverted_card_payment?"),
]
TEST_ROWS = [
    ("Where is my card?", "card_arrival"),
    ("No refund yet", "Refund_not_showing_up"),
    ("Reverted €5 payment", "reverted_card_payment?"),
]
SORTED_LABELS = ["Refund_not_showing_up", "card_arrival", "reverted_card_payment?"]
# Hugging Face ClassLabel order: case-insensitive sort.
HUB_NAMES = ["card_arrival", "Refund_not_showing_up", "reverted_card_payment?"]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv_bytes(rows: list[tuple[str, str]]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)  # minimal quoting and CRLF record terminators, like the official files
    writer.writerow(["text", "category"])
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _parquet_bytes(rows: list[tuple[str, str]], names: list[str], *, metadata: bool = True) -> bytes:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {"text": [text for text, _ in rows], "label": pa.array([names.index(lb) for _, lb in rows], pa.int64())}
    )
    if metadata:
        features = {
            "text": {"dtype": "string", "_type": "Value"},
            "label": {"names": names, "_type": "ClassLabel"},
        }
        table = table.replace_schema_metadata({b"huggingface": json.dumps({"info": {"features": features}}).encode()})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return bytes(sink.getvalue().to_pybytes())


class FakeSources:
    """Installs tiny fake files behind the real URLs, with their digests and counts pinned in the module."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, hub: bool = False) -> None:
        self.train = _csv_bytes(TRAIN_ROWS)
        self.test = _csv_bytes(TEST_ROWS)
        self.categories = json.dumps(CATEGORIES).encode()
        pinned = {
            "TRAIN_CSV": (banking77.TRAIN_CSV, self.train),
            "TEST_CSV": (banking77.TEST_CSV, self.test),
            "CATEGORIES_JSON": (banking77.CATEGORIES_JSON, self.categories),
        }
        if hub:
            self.hub_train = _parquet_bytes(TRAIN_ROWS, HUB_NAMES)
            self.hub_test = _parquet_bytes(TEST_ROWS, HUB_NAMES)
            pinned["HUB_TRAIN_PARQUET"] = (banking77.HUB_TRAIN_PARQUET, self.hub_train)
            pinned["HUB_TEST_PARQUET"] = (banking77.HUB_TEST_PARQUET, self.hub_test)
        for attr, (remote, payload) in pinned.items():
            monkeypatch.setattr(banking77, attr, RemoteFile(remote.name, remote.url, _sha(payload)))
        monkeypatch.setattr(banking77, "EXPECTED_TRAIN", len(TRAIN_ROWS))
        monkeypatch.setattr(banking77, "EXPECTED_TEST", len(TEST_ROWS))
        monkeypatch.setattr(banking77, "EXPECTED_LABELS", len(CATEGORIES))

    def github_routes(self, router: respx.MockRouter) -> dict[str, respx.Route]:
        return {
            "train": router.get(banking77.TRAIN_CSV.url).mock(return_value=httpx.Response(200, content=self.train)),
            "test": router.get(banking77.TEST_CSV.url).mock(return_value=httpx.Response(200, content=self.test)),
            "categories": router.get(banking77.CATEGORIES_JSON.url).mock(
                return_value=httpx.Response(200, content=self.categories)
            ),
        }

    def hub_routes(self, router: respx.MockRouter) -> dict[str, respx.Route]:
        return {
            "train": router.get(banking77.HUB_TRAIN_PARQUET.url).mock(
                return_value=httpx.Response(200, content=self.hub_train)
            ),
            "test": router.get(banking77.HUB_TEST_PARQUET.url).mock(
                return_value=httpx.Response(200, content=self.hub_test)
            ),
        }


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(banking77, "BACKOFF_S", 0.0)


# --- parsing -----------------------------------------------------------------------------------


def test_parse_csv_keeps_texts_verbatim(tmp_path: Path) -> None:
    raw = (
        'text,category\r\n"\nWhich ATMs accept this card?",atm_support\r\n'
        '"Hi, my card\nis lost",lost_or_stolen_card\r\n'
        '"He said ""hello"" twice",card_arrival\r\n'
        "Is there a £5 fee?,exchange_charge\r\n"
    )
    path = tmp_path / "train.csv"
    path.write_bytes(raw.encode("utf-8"))
    assert parse_csv(path) == [
        ("\nWhich ATMs accept this card?", "atm_support"),
        ("Hi, my card\nis lost", "lost_or_stolen_card"),
        ('He said "hello" twice', "card_arrival"),
        ("Is there a £5 fee?", "exchange_charge"),
    ]


def test_parse_csv_round_trips_the_fake_rows(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_bytes(_csv_bytes(TRAIN_ROWS))
    assert parse_csv(path) == TRAIN_ROWS


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("sentence,label\r\nhello,card_arrival\r\n", "expected the header text,category"),
        ("", "expected the header text,category"),
        ("text,category\r\nhello,card_arrival,extra\r\n", "malformed record ending on line 2"),
        ("text,category\r\nfine,card_arrival\r\nno label\r\n", "malformed record ending on line 3"),
        ("text,category\r\nempty label,\r\n", "malformed record"),
    ],
)
def test_parse_csv_rejects_malformed_files(tmp_path: Path, raw: str, message: str) -> None:
    path = tmp_path / "train.csv"
    path.write_bytes(raw.encode("utf-8"))
    with pytest.raises(DatasetError, match=message):
        parse_csv(path)


# --- download, checksums and counts --------------------------------------------------------------


def test_fetch_downloads_verifies_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch)
    with respx.mock(assert_all_called=True) as router:
        routes = fake.github_routes(router)
        data = fetch_banking77(tmp_path)
    assert data == BankingData(TRAIN_ROWS, TEST_ROWS, SORTED_LABELS, source="github")
    assert {name: route.call_count for name, route in routes.items()} == {"train": 1, "test": 1, "categories": 1}
    assert (tmp_path / "train.csv").read_bytes() == fake.train
    assert (tmp_path / "test.csv").read_bytes() == fake.test
    assert (tmp_path / "categories.json").read_bytes() == fake.categories
    assert not list(tmp_path.glob("*.part"))

    # Second call: everything is served from the verified cache, no request is made.
    with respx.mock(assert_all_called=False) as router:
        again = fetch_banking77(tmp_path)
    assert router.calls.call_count == 0
    assert again == data


def test_fetch_sends_a_user_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch)
    with respx.mock() as router:
        routes = fake.github_routes(router)
        fetch_banking77(tmp_path)
    assert routes["train"].calls.last.request.headers["user-agent"].startswith("taskdistill/")


def test_default_cache_dir_is_under_the_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKDISTILL_HOME", str(tmp_path / "home"))
    fake = FakeSources(monkeypatch)
    with respx.mock() as router:
        fake.github_routes(router)
        fetch_banking77()
    cached = tmp_path / "home" / "_datasets" / "banking77"
    assert sorted(p.name for p in cached.iterdir()) == ["categories.json", "test.csv", "train.csv"]


def test_checksum_mismatch_is_downloaded_once_more(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch)
    with respx.mock() as router:
        routes = fake.github_routes(router)
        routes["train"].side_effect = [
            httpx.Response(200, content=b"<html>captive portal</html>"),
            httpx.Response(200, content=fake.train),
        ]
        data = fetch_banking77(tmp_path)
    assert routes["train"].call_count == 2
    assert data.train == TRAIN_ROWS


def test_checksum_mismatch_twice_fails_without_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch, hub=True)
    with respx.mock(assert_all_called=False) as router:
        routes = fake.github_routes(router)
        hub = fake.hub_routes(router)
        routes["test"].mock(return_value=httpx.Response(200, content=b"tampered"))
        with pytest.raises(ChecksumError) as excinfo:
            fetch_banking77(tmp_path)
    message = str(excinfo.value)
    assert banking77.TEST_CSV.url in message
    assert _sha(b"tampered") in message
    assert banking77.TEST_CSV.sha256 in message
    assert "twice" in message
    assert routes["test"].call_count == 2
    assert hub["train"].call_count == 0
    assert not (tmp_path / "test.csv").exists()


def test_corrupted_cache_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch)
    (tmp_path / "train.csv").write_bytes(_csv_bytes([*TRAIN_ROWS, ("injected", "card_arrival")]))
    (tmp_path / "test.csv").write_bytes(fake.test)
    (tmp_path / "categories.json").write_bytes(fake.categories)
    with respx.mock(assert_all_called=False) as router:
        routes = fake.github_routes(router)
        data = fetch_banking77(tmp_path)
    assert {name: route.call_count for name, route in routes.items()} == {"train": 1, "test": 0, "categories": 0}
    assert (tmp_path / "train.csv").read_bytes() == fake.train
    assert data.train == TRAIN_ROWS


def test_server_errors_are_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch)
    with respx.mock() as router:
        routes = fake.github_routes(router)
        routes["train"].side_effect = [
            httpx.Response(503),
            httpx.Response(429),
            httpx.ConnectError("connection reset"),
            httpx.Response(200, content=fake.train),
        ]
        data = fetch_banking77(tmp_path)
    assert routes["train"].call_count == 4
    assert data.source == "github"


def test_count_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch)
    monkeypatch.setattr(banking77, "EXPECTED_TRAIN", 7)
    with respx.mock() as router, pytest.raises(DatasetError, match=r"train has 6 rows, expected 7"):
        fake.github_routes(router)
        fetch_banking77(tmp_path)


def test_check_counts_reports_every_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(banking77, "EXPECTED_TRAIN", 2)
    monkeypatch.setattr(banking77, "EXPECTED_TEST", 2)
    monkeypatch.setattr(banking77, "EXPECTED_LABELS", 2)
    check_counts(BankingData([("a", "x"), ("b", "y")], [("c", "x"), ("d", "y")], ["x", "y"]))
    with pytest.raises(DatasetError) as excinfo:
        check_counts(BankingData([("a", "x")], [("c", "x"), ("d", "z"), ("e", "x")], ["x", "y", "z"], source="hub"))
    message = str(excinfo.value)
    assert message.startswith("Banking77 (hub) failed its integrity check: ")
    assert "train has 1 rows, expected 2" in message
    assert "test has 3 rows, expected 2" in message
    assert "3 labels, expected 2" in message
    assert "train labels differ from the label list (unknown: [], absent: ['y', 'z'])" in message
    assert "test labels differ from the label list (unknown: [], absent: ['y'])" in message


def test_check_counts_rejects_labels_outside_the_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(banking77, "EXPECTED_TRAIN", 2)
    monkeypatch.setattr(banking77, "EXPECTED_TEST", 2)
    monkeypatch.setattr(banking77, "EXPECTED_LABELS", 2)
    data = BankingData([("a", "x"), ("b", "w")], [("c", "x"), ("d", "y")], ["x", "y"])
    with pytest.raises(
        DatasetError, match=r"train labels differ from the label list \(unknown: \['w'\], absent: \['y'\]\)"
    ):
        check_counts(data)


def test_categories_must_be_a_unique_string_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch)
    duplicated = json.dumps([*CATEGORIES, "card_arrival"]).encode()
    monkeypatch.setattr(
        banking77,
        "CATEGORIES_JSON",
        RemoteFile("categories.json", banking77.CATEGORIES_JSON.url, _sha(duplicated)),
    )
    fake.categories = duplicated
    with respx.mock() as router, pytest.raises(DatasetError, match=r"categories\.json: duplicate labels"):
        fake.github_routes(router)
        fetch_banking77(tmp_path)


# --- Hugging Face Hub fallback -------------------------------------------------------------------


def test_github_404_falls_back_to_hub_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch, hub=True)
    with respx.mock(assert_all_called=False) as router:
        routes = fake.github_routes(router)
        routes["train"].mock(return_value=httpx.Response(404))
        hub = fake.hub_routes(router)
        data = fetch_banking77(tmp_path)
    assert routes["train"].call_count == 1  # a 404 is not retried
    assert {name: route.call_count for name, route in hub.items()} == {"train": 1, "test": 1}
    assert data == BankingData(TRAIN_ROWS, TEST_ROWS, SORTED_LABELS, source="hub")
    assert (tmp_path / "hub" / "train.parquet").read_bytes() == fake.hub_train


def test_network_failure_falls_back_to_hub_after_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch, hub=True)
    with respx.mock(assert_all_called=False) as router:
        routes = fake.github_routes(router)
        routes["train"].side_effect = httpx.ConnectTimeout("timed out")
        fake.hub_routes(router)
        data = fetch_banking77(tmp_path)
    assert routes["train"].call_count == banking77.RETRIES + 1 == 4
    assert data.source == "hub"
    assert data.train == TRAIN_ROWS


def test_redirect_loop_falls_back_to_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch, hub=True)
    with respx.mock(assert_all_called=False) as router:
        routes = fake.github_routes(router)
        routes["train"].mock(return_value=httpx.Response(302, headers={"Location": banking77.TRAIN_CSV.url}))
        hub = fake.hub_routes(router)
        data = fetch_banking77(tmp_path)
    # httpx follows 20 redirects, so each of the 4 attempts makes 21 requests before TooManyRedirects.
    assert routes["train"].call_count == (banking77.RETRIES + 1) * 21 == 84
    assert {name: route.call_count for name, route in hub.items()} == {"train": 1, "test": 1}
    assert data == BankingData(TRAIN_ROWS, TEST_ROWS, SORTED_LABELS, source="hub")


def test_undecodable_body_falls_back_to_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch, hub=True)
    with respx.mock(assert_all_called=False) as router:
        routes = fake.github_routes(router)
        broken_gzip = httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(b"not gzip"))
        routes["train"].mock(return_value=broken_gzip)
        fake.hub_routes(router)
        data = fetch_banking77(tmp_path)
    assert routes["train"].call_count == banking77.RETRIES + 1 == 4
    assert data.source == "hub"
    assert data.train == TRAIN_ROWS


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (httpx.Response(302, headers={"Location": "https://example.com/loop"}), "TooManyRedirects"),
        (httpx.DecodingError("incorrect header check"), "DecodingError: incorrect header check"),
        (httpx.ReadTimeout("read timed out"), "ReadTimeout: read timed out"),
    ],
)
def test_request_errors_end_as_source_unavailable(failure: httpx.Response | Exception, reason: str) -> None:
    url = "https://example.com/loop"
    with respx.mock() as router, httpx.Client(follow_redirects=True) as client:
        route = router.get(url)
        if isinstance(failure, httpx.Response):
            route.mock(return_value=failure)
        else:
            route.mock(side_effect=failure)
        with pytest.raises(banking77.SourceUnavailable, match=f"{url}: {reason}.* after 4 attempts"):
            banking77._download(client, url)


def test_hub_fallback_without_pyarrow_names_the_hub_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch)
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with respx.mock(assert_all_called=False) as router:
        fake.github_routes(router)["train"].mock(return_value=httpx.Response(404))
        hub_route = router.get(url__startswith="https://huggingface.co/")
        with pytest.raises(DatasetError) as excinfo:
            fetch_banking77(tmp_path)
    message = str(excinfo.value)
    assert "HTTP 404" in message
    assert 'install the "hub" extra' in message
    assert "uvx --from 'taskdistill[hub] @ git+https://github.com/B0yko/taskdistill'" in message
    assert hub_route.call_count == 0


def test_both_sources_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSources(monkeypatch, hub=True)
    with respx.mock(assert_all_called=False) as router:
        fake.github_routes(router)["train"].mock(return_value=httpx.Response(404))
        fake.hub_routes(router)["train"].mock(return_value=httpx.Response(404))
        with pytest.raises(DatasetError) as excinfo:
            fetch_banking77(tmp_path)
    message = str(excinfo.value)
    assert f"{banking77.TRAIN_CSV.url}: HTTP 404" in message
    assert f"{banking77.HUB_TRAIN_PARQUET.url}: HTTP 404" in message
    assert "Check the network connection" in message


def test_read_parquet_maps_classlabel_ints_to_names(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    path.write_bytes(_parquet_bytes(TRAIN_ROWS, HUB_NAMES))
    assert read_parquet(path) == TRAIN_ROWS


def test_read_parquet_needs_classlabel_metadata(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    path.write_bytes(_parquet_bytes(TRAIN_ROWS, HUB_NAMES, metadata=False))
    with pytest.raises(DatasetError, match="no ClassLabel names"):
        read_parquet(path)


def test_read_parquet_rejects_out_of_range_labels(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    path.write_bytes(_parquet_bytes(TRAIN_ROWS, HUB_NAMES))
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(path)
    metadata = table.schema.metadata
    names = json.loads(metadata[b"huggingface"])
    names["info"]["features"]["label"]["names"] = HUB_NAMES[:2]
    pq.write_table(table.replace_schema_metadata({b"huggingface": json.dumps(names).encode()}), path)
    with pytest.raises(DatasetError, match="malformed row 4"):
        read_parquet(path)


# --- demo splits ---------------------------------------------------------------------------------


def _synthetic(train_counts: dict[str, int], test_counts: dict[str, int]) -> BankingData:
    """Rows grouped by label in label order, like the official files."""
    train = [(f"{label} train {i}", label) for label, n in train_counts.items() for i in range(n)]
    test = [(f"{label} test {i}", label) for label, n in test_counts.items() for i in range(n)]
    return BankingData(train, test, sorted(set(train_counts) | set(test_counts)))


def _digest(rows: list[Row]) -> str:
    return hashlib.sha256("\n".join(f"{r.split}\t{r.index}\t{r.gold}" for r in rows).encode()).hexdigest()


SMALL_TRAIN = {"e_label": 187, "a_label": 35, "b_label": 20, "c_label": 7, "d_label": 1}
SMALL_TEST = {"e_label": 3, "a_label": 2, "b_label": 2, "c_label": 1, "d_label": 1}


def test_demo_splits_take_ceil_ten_percent_of_each_label() -> None:
    data = _synthetic(SMALL_TRAIN, SMALL_TEST)
    splits = demo_splits(data)
    valid = Counter(row.gold for row in splits["valid"])
    train = Counter(row.gold for row in splits["train"])
    # ceil(187/10)=19, ceil(35/10)=4, ceil(20/10)=2, ceil(7/10)=1, ceil(1/10)=1
    assert valid == {"e_label": 19, "a_label": 4, "b_label": 2, "c_label": 1, "d_label": 1}
    assert train == {"e_label": 168, "a_label": 31, "b_label": 18, "c_label": 6}
    assert (len(splits["train"]), len(splits["valid"]), len(splits["test"])) == (223, 27, 9)


def test_demo_splits_partition_the_official_files() -> None:
    data = _synthetic(SMALL_TRAIN, SMALL_TEST)
    splits = demo_splits(data)
    train_ids = [row.index for row in splits["train"]]
    valid_ids = [row.index for row in splits["valid"]]
    assert sorted(train_ids + valid_ids) == list(range(len(data.train)))
    assert sorted(row.index for row in splits["test"]) == list(range(len(data.test)))
    for split, rows in splits.items():
        source = data.test if split == "test" else data.train
        for row in rows:
            assert row.split == split
            assert (row.text, row.gold) == source[row.index]


def test_demo_splits_are_deterministic_and_ignore_the_global_seed() -> None:
    data = _synthetic(SMALL_TRAIN, SMALL_TEST)
    random.seed(1)
    first = demo_splits(data)
    random.seed(2)
    second = demo_splits(data)
    assert first == second
    # Pinned: the demo recording depends on this exact assignment and order.
    assert sorted(row.index for row in first["valid"] if row.gold == "a_label") == [190, 197, 202, 219]
    assert [row.index for row in first["valid"][:5]] == [157, 122, 12, 242, 104]


def test_demo_splits_are_not_grouped_by_label() -> None:
    splits = demo_splits(_synthetic(SMALL_TRAIN, SMALL_TEST))
    assert [row.index for row in splits["train"]] != sorted(row.index for row in splits["train"])
    assert len({row.gold for row in splits["train"][:20]}) >= 3
    assert len({row.gold for row in splits["test"][:5]}) >= 3


def test_valid_share_is_about_ten_percent_per_label() -> None:
    counts = {f"label_{i:02d}": 30 + i for i in range(77)}
    splits = demo_splits(_synthetic(counts, dict.fromkeys(counts, 40)))
    valid = Counter(row.gold for row in splits["valid"])
    for label, n in counts.items():
        assert valid[label] == -(-n // 10)
        assert 0.1 <= valid[label] / n < 0.1 + 1 / n


def _per_row_valid_ids(data: BankingData) -> set[int]:
    """The plain per-row rule: shuffle each label's rows with one Random(13), take ceil(n / 10)."""
    groups: dict[str, list[int]] = {}
    for index, (_, label) in enumerate(data.train):
        groups.setdefault(label, []).append(index)
    rng = random.Random(13)
    chosen: set[int] = set()
    for label in sorted(groups):
        rng.shuffle(groups[label])
        chosen.update(groups[label][: math.ceil(len(groups[label]) / 10)])
    return chosen


def test_without_equal_texts_the_split_is_the_per_row_rule() -> None:
    counts = {f"label_{i:02d}": 30 + i for i in range(77)}
    data = _synthetic(counts, dict.fromkeys(counts, 40))
    assert {row.index for row in demo_splits(data)["valid"]} == _per_row_valid_ids(data)
    small = _synthetic(SMALL_TRAIN, SMALL_TEST)
    assert {row.index for row in demo_splits(small)["valid"]} == _per_row_valid_ids(small)


# Five texts of one label, each twice, in two forms that curate's normalise_text maps to one input hash.
TWIN_TEXTS = [
    ("Where can I withdraw money from?", "\nWhere can I withdraw money from?"),
    ("I can't seem to be able to use my card", "I can't seem to be able to use my card\n"),
    ("Is  there a fee?", " Is there a fee?"),
    ("ＡＴＭ near me", "ATM near me"),  # NFKC folds the full-width letters
    ("Card\tblocked", "Card blocked"),
]


def _twin_data() -> BankingData:
    # Rows 0-4 hold the first forms and rows 5-9 their twins; rows 10-19 are ten distinct texts.
    twins = [(first, "twins") for first, _ in TWIN_TEXTS] + [(second, "twins") for _, second in TWIN_TEXTS]
    singles = [(f"single text {i}", "singles") for i in range(10)]
    return BankingData(twins + singles, [("a", "twins"), ("b", "singles")], ["singles", "twins"])


def test_equal_texts_never_straddle_train_and_valid() -> None:
    data = _twin_data()
    assert all(normalise_text(first) == normalise_text(second) for first, second in TWIN_TEXTS)
    # The per-row rule puts one row of "twins" in valid and leaves its twin in train.
    (lone,) = _per_row_valid_ids(data) & set(range(10))
    assert (lone + 5) % 10 not in _per_row_valid_ids(data)

    splits = demo_splits(data)
    side = {row.index: row.split for split in ("train", "valid") for row in splits[split]}
    for k in range(5):
        assert side[k] == side[k + 5], TWIN_TEXTS[k]
    # "twins": n = 10, ceil(10 / 10) = 1, but the first shuffled cluster holds two rows, so valid gets both.
    # "singles": n = 10, one row to valid.
    assert Counter(row.gold for row in splits["valid"]) == {"twins": 2, "singles": 1}
    assert Counter(row.gold for row in splits["train"]) == {"twins": 8, "singles": 9}
    twin_valid = sorted(row.index for row in splits["valid"] if row.gold == "twins")
    assert twin_valid[1] == twin_valid[0] + 5
    # Pinned: the demo recording depends on this exact assignment.
    assert sorted(row.index for row in splits["valid"]) == [4, 9, 13]
    assert demo_splits(data) == splits


def test_duplicate_clusters_move_whole_and_valid_stays_near_ten_percent() -> None:
    rng = random.Random(5)
    train: list[tuple[str, str]] = []
    for j in range(20):
        n = 12 + 7 * j
        pool = [f"query {j} {k}" for k in range(max(1, n * 2 // 3))]
        for _ in range(n):
            text = rng.choice(pool)
            train.append((rng.choice([text, f"\n{text}", f"{text} ", text.replace(" ", "  ")]), f"label_{j:02d}"))
    data = BankingData(train, [("t", f"label_{j:02d}") for j in range(20)], [f"label_{j:02d}" for j in range(20)])
    splits = demo_splits(data)
    assert sorted(row.index for split in ("train", "valid") for row in splits[split]) == list(range(len(train)))

    sides: dict[str, set[str]] = {}
    for split in ("train", "valid"):
        for row in splits[split]:
            sides.setdefault(normalise_text(row.text), set()).add(split)
    assert all(len(s) == 1 for s in sides.values())
    assert any("valid" in s for s in sides.values())

    for j in range(20):
        label = f"label_{j:02d}"
        rows = [text for text, gold in train if gold == label]
        target = math.ceil(len(rows) / 10)
        biggest = max(Counter(normalise_text(text) for text in rows).values())
        taken = sum(1 for row in splits["valid"] if row.gold == label)
        assert target <= taken <= target + biggest - 1, label


# --- quick-profile subsets -----------------------------------------------------------------------


def _full_synthetic_splits() -> dict[banking77.Split, list[Row]]:
    # label_i has 30 + i official train rows and 40 test rows, 77 labels.
    counts = {f"label_{i:02d}": 30 + i for i in range(77)}
    return demo_splits(_synthetic(counts, dict.fromkeys(counts, 40)))


def test_quick_sizes_match_the_spec() -> None:
    assert banking77.QUICK_SIZES == {"train": 2000, "valid": 300, "test": 500}


def test_quick_test_subset_is_label_balanced() -> None:
    splits = _full_synthetic_splits()
    subset = quick_subset(splits["test"], "test")
    per_label = Counter(row.gold for row in subset)
    # 500 = 77 * 6 + 38: every label has 40 rows, so the extra row goes to the first 38 labels by name.
    assert per_label == {f"label_{i:02d}": 7 if i < 38 else 6 for i in range(77)}
    assert len(subset) == 500


def test_quick_train_subset_gives_extras_to_the_largest_labels() -> None:
    splits = _full_synthetic_splits()
    subset = quick_subset(splits["train"], "train")
    per_label = Counter(row.gold for row in subset)
    # 2000 = 77 * 25 + 75. The two labels with the fewest train rows (label_00: 30 - 3 = 27,
    # label_01: 31 - 4 = 27) get 25; the other 75 get 26.
    assert per_label == {f"label_{i:02d}": 25 if i < 2 else 26 for i in range(77)}
    assert len(subset) == 2000


def test_quick_valid_subset_caps_small_labels() -> None:
    splits = _full_synthetic_splits()
    subset = quick_subset(splits["valid"], "valid")
    per_label = Counter(row.gold for row in subset)
    # Valid has ceil(n / 10) rows per label: label_00 has 3, label_01..10 have 4, the rest 5 or more.
    # 300 = 77 * 3 + 69. label_00 is full at 3, so the 69 extras go to the 76 open labels with the most
    # rows; label_04..label_10 (4 rows each, last by name) get none.
    expected = {f"label_{i:02d}": 3 if i == 0 or 4 <= i <= 10 else 4 for i in range(77)}
    assert per_label == expected
    assert len(subset) == 300


def test_quick_subsets_are_fixed_subsets_of_the_full_splits() -> None:
    splits = _full_synthetic_splits()
    for split in banking77.SPLITS:
        full = splits[split]
        subset = quick_subset(full, split)
        assert len(subset) == banking77.QUICK_SIZES[split]
        assert len({row.index for row in subset}) == len(subset)
        assert set(subset) <= set(full)
        assert quick_subset(full, split) == subset
        shuffled = list(full)
        random.Random(99).shuffle(shuffled)
        assert quick_subset(shuffled, split) == subset
        assert quick_subset(list(reversed(full)), split) == subset


def test_profile_splits() -> None:
    splits = _full_synthetic_splits()
    assert profile_splits(splits, "full") == splits
    quick = profile_splits(splits, "quick")
    assert {split: len(rows) for split, rows in quick.items()} == {"train": 2000, "valid": 300, "test": 500}
    assert quick["valid"] == quick_subset(splits["valid"], "valid")


@pytest.mark.parametrize(
    ("available", "size", "expected"),
    [
        ({"a": 1, "b": 5, "c": 5}, 7, {"a": 1, "b": 3, "c": 3}),
        ({"a": 1, "b": 5, "c": 5}, 8, {"a": 1, "b": 4, "c": 3}),
        ({"a": 2, "b": 2}, 3, {"a": 2, "b": 1}),
        ({"a": 4, "b": 9}, 13, {"a": 4, "b": 9}),
        ({"a": 4, "b": 9}, 0, {"a": 0, "b": 0}),
        ({"a": 3, "b": 9, "c": 6}, 10, {"a": 3, "b": 4, "c": 3}),
    ],
)
def test_balanced_quota(available: dict[str, int], size: int, expected: dict[str, int]) -> None:
    assert banking77._balanced_quota(available, size) == expected


def test_balanced_quota_rejects_too_large_sizes() -> None:
    with pytest.raises(ValueError, match="cannot take 10 rows from 9"):
        banking77._balanced_quota({"a": 4, "b": 5}, 10)


def test_quick_subset_rejects_bad_input(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [Row("a", "x", "valid", 0), Row("b", "y", "valid", 1), Row("c", "x", "valid", 2)]
    monkeypatch.setitem(banking77.QUICK_SIZES, "valid", 2)
    chosen = quick_subset(rows, "valid")
    assert len(chosen) == 2
    assert {row.gold for row in chosen} == {"x", "y"}
    with pytest.raises(ValueError, match="rows from other splits: valid"):
        quick_subset(rows, "train")
    with pytest.raises(ValueError, match="duplicate row indices"):
        quick_subset([*rows, Row("d", "y", "valid", 1)], "valid")
    with pytest.raises(ValueError, match="fewer than the quick size 2"):
        quick_subset(rows[:1], "valid")
    with pytest.raises(ValueError, match="unknown split"):
        quick_subset(rows, "dev")  # type: ignore[arg-type]


# --- records for capture --import --format inputs --------------------------------------------------


def test_to_records() -> None:
    rows = [Row("Where is my card?", "card_arrival", "test", 3), Row("\nRefund?", "Refund_not_showing_up", "valid", 8)]
    assert to_records(rows) == [
        {"input": "Where is my card?", "gold": "card_arrival", "meta": {"split": "test"}},
        {"input": "\nRefund?", "gold": "Refund_not_showing_up", "meta": {"split": "valid"}},
    ]
    assert json.loads(json.dumps(to_records(rows), ensure_ascii=False)) == to_records(rows)


# --- bundled task files --------------------------------------------------------------------------


def test_labels_file_has_77_unique_sorted_labels() -> None:
    raw = (TASK_DIR / "labels.txt").read_text(encoding="utf-8")
    lines = raw.splitlines()
    assert raw.endswith("\n")
    assert len(lines) == 77
    assert len(set(lines)) == 77
    assert lines == sorted(lines)
    assert all(line and line == line.strip() and " " not in line for line in lines)
    assert packaged_labels() == lines
    # The two labels that are not plain snake_case stay verbatim, as in the CSV category column.
    assert "Refund_not_showing_up" in lines
    assert "reverted_card_payment?" in lines
    assert {"card_arrival", "activate_my_card", "wrong_exchange_rate_for_cash_withdrawal"} <= set(lines)


def test_teacher_prompt_lists_every_label_with_spaces() -> None:
    prompt = (TASK_DIR / "teacher_prompt.md").read_text(encoding="utf-8")
    labels = packaged_labels()
    rendered = [label.replace("_", " ") for label in labels]
    lines = prompt.splitlines()
    start = lines.index(rendered[0])
    assert lines[start : start + 77] == rendered
    assert "_" not in prompt
    assert "Refund not showing up" in lines
    assert "reverted card payment?" in lines
    assert "exactly one" in prompt
    assert "nothing else" in prompt
    assert len(prompt) < 2500


def test_licence_and_citation() -> None:
    assert banking77.LICENCE == "CC BY 4.0"
    assert "casanueva-etal-2020-efficient" in banking77.CITATION
    assert "10.18653/v1/2020.nlp4convai-1.5" in banking77.CITATION
    assert banking77.COMMIT in banking77.SOURCE_URL
    assert all(banking77.COMMIT in f.url for f in (banking77.TRAIN_CSV, banking77.TEST_CSV, banking77.CATEGORIES_JSON))
    assert all(banking77.HUB_REVISION in f.url for f in (banking77.HUB_TRAIN_PARQUET, banking77.HUB_TEST_PARQUET))


# --- the real data (public GitHub and Hugging Face downloads, no paid API) --------------------------

PINNED_SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
}
# Pinned demo split and quick subsets of the real data (split, index, gold, in returned order).
PINNED_SPLIT_DIGESTS = {
    "train": "05379d00d91b513e0cd36efb743830caa4f7344b30da79f429cee17ab15bc239",
    "valid": "1943c562356164ea752699aa9082ba5a092da4cf67bfece52a5bebfb56ba4e3e",
    "test": "8175491c114896e3ba2813829fe52df47b548dae2d9f0ad3c0cd94c2612931cd",
}
PINNED_QUICK_DIGESTS = {
    "train": "30becc9c60515ad60a1d5ff9d066e6328027f0d87b3f39c253cadd6484db166e",
    "valid": "99042ad78a56945367c69e2b542b7637d66b11e3f72a73341467effde1bd73d0",
    "test": "afa4f45e1ec9c9eeeb36d3a108a847f0c2b4a380f67bb63cd045ea05cb532ccf",
}
# train.csv rows whose texts are equal after normalise_text (all within one label), and the official
# test rows whose texts also occur in train.csv.
EQUAL_TRAIN_PAIRS = [(1246, 1290), (1710, 1724), (4594, 4595), (6910, 6965)]
TEST_ROWS_IN_TRAIN = [554, 976, 977, 1432, 1474, 2149, 3070]


@pytest.mark.network
def test_real_pinned_github_data(tmp_path: Path) -> None:
    data = fetch_banking77(tmp_path)
    assert data.source == "github"
    assert (len(data.train), len(data.test), len(data.labels)) == (10_003, 3_080, 77)
    for name, digest in PINNED_SHA256.items():
        assert _sha((tmp_path / name).read_bytes()) == digest
    assert data.labels == packaged_labels()
    assert Counter(label for _, label in data.test) == dict.fromkeys(data.labels, 40)

    splits = demo_splits(data)
    assert {split: len(rows) for split, rows in splits.items()} == {"train": 8965, "valid": 1038, "test": 3080}
    assert {split: _digest(rows) for split, rows in splits.items()} == PINNED_SPLIT_DIGESTS
    quick = profile_splits(splits, "quick")
    assert {split: len(rows) for split, rows in quick.items()} == {"train": 2000, "valid": 300, "test": 500}
    assert {split: _digest(rows) for split, rows in quick.items()} == PINNED_QUICK_DIGESTS
    for split, rows in quick.items():
        per_label = Counter(row.gold for row in rows)
        assert len(per_label) == 77
        assert max(per_label.values()) - min(per_label.values()) <= 1, split

    by_text: dict[str, list[tuple[int, str]]] = {}
    for i, (text, gold) in enumerate(data.train):
        by_text.setdefault(normalise_text(text), []).append((i, gold))
    assert sorted(tuple(i for i, _ in rows) for rows in by_text.values() if len(rows) > 1) == EQUAL_TRAIN_PAIRS
    assert all(len({gold for _, gold in rows}) == 1 for rows in by_text.values())
    side = {row.index: split for split in ("train", "valid") for row in splits[split]}
    for first, second in EQUAL_TRAIN_PAIRS:
        assert side[first] == side[second], (first, second)
    for profile in (splits, quick):
        texts = {split: {normalise_text(row.text) for row in profile[split]} for split in banking77.SPLITS}
        assert not texts["train"] & texts["valid"]
        assert not texts["valid"] & texts["test"]
    test_in_train = sorted(row.index for row in splits["test"] if normalise_text(row.text) in by_text)
    assert test_in_train == TEST_ROWS_IN_TRAIN


@pytest.mark.network
def test_real_hub_fallback_matches_github(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pyarrow")
    github = fetch_banking77(tmp_path / "github")
    missing = banking77.TRAIN_CSV.url.replace("train.csv", "no-such-file.csv")
    monkeypatch.setattr(banking77, "TRAIN_CSV", RemoteFile("train.csv", missing, banking77.TRAIN_CSV.sha256))
    hub = fetch_banking77(tmp_path / "hub-only")
    assert hub.source == "hub"
    assert (hub.train, hub.test, hub.labels) == (github.train, github.test, github.labels)
    assert demo_splits(hub) == demo_splits(github)
