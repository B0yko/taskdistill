"""Banking77 demo data: pinned download, integrity checks, demo splits and quick-profile subsets.

The data comes from the ``banking_data/`` CSVs in PolyAI's ``task-specific-datasets`` repository
(CC BY 4.0), fetched from a pinned commit and checked against SHA-256 digests before use. If GitHub
cannot be reached, the loader falls back to the pinned Parquet copy on the Hugging Face Hub, read with
``pyarrow`` from the ``hub`` extra. ``PolyAI/banking77`` itself is a loading script with no Parquet
conversion, so the fallback uses ``legacy-datasets/banking77``, whose rows equal the CSVs in content
and order. Both sources give the same rows, so the splits below do not depend on the source.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import httpx

from taskdistill import __version__, paths
from taskdistill.curate.extract import normalise_text

log = logging.getLogger(__name__)

Split = Literal["train", "valid", "test"]
SPLITS: tuple[Split, ...] = ("train", "valid", "test")

GITHUB_REPO = "PolyAI-LDN/task-specific-datasets"
COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_URL = f"https://github.com/{GITHUB_REPO}/tree/{COMMIT}/banking_data"
_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{COMMIT}/banking_data"

HUB_REPO = "legacy-datasets/banking77"
HUB_REVISION = "f54121560de48f2852f90be299010d1d6dc612ec"
HUB_URL = f"https://huggingface.co/datasets/{HUB_REPO}/tree/{HUB_REVISION}"
_HUB_BASE = f"https://huggingface.co/datasets/{HUB_REPO}/resolve/{HUB_REVISION}/data"

EXPECTED_TRAIN = 10_003
EXPECTED_TEST = 3_080
EXPECTED_LABELS = 77

LICENCE = "CC BY 4.0"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = (
    "BANKING77 by Casanueva et al. (2020), from PolyAI's task-specific-datasets repository "
    f"({SOURCE_URL}), licensed under {LICENCE} ({LICENCE_URL})."
)
CITATION = """\
@inproceedings{casanueva-etal-2020-efficient,
    title = "Efficient Intent Detection with Dual Sentence Encoders",
    author = "Casanueva, I{\\~n}igo  and
      Tem{\\v{c}}inas, Tadas  and
      Gerz, Daniela  and
      Henderson, Matthew  and
      Vuli{\\'c}, Ivan",
    booktitle = "Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.nlp4convai-1.5/",
    doi = "10.18653/v1/2020.nlp4convai-1.5",
    pages = "38--45"
}
"""

SEED = 13
VALID_PERCENT = 10
QUICK_SIZES: dict[Split, int] = {"train": 2000, "valid": 300, "test": 500}

TIMEOUT = httpx.Timeout(30.0, connect=10.0)
RETRIES = 3
BACKOFF_S = 1.0
_USER_AGENT = f"taskdistill/{__version__} (+https://github.com/B0yko/taskdistill)"
HUB_EXTRA_HINT = (
    'install the "hub" extra, e.g. '
    "uvx --from 'taskdistill[hub] @ git+https://github.com/B0yko/taskdistill' taskdistill demo banking77 "
    "(or: pip install 'taskdistill[hub]')"
)


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """A pinned file: where it is cached (relative to the cache directory), its URL and its SHA-256."""

    name: str
    url: str
    sha256: str


TRAIN_CSV = RemoteFile(
    "train.csv", f"{_RAW_BASE}/train.csv", "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
)
TEST_CSV = RemoteFile(
    "test.csv", f"{_RAW_BASE}/test.csv", "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"
)
CATEGORIES_JSON = RemoteFile(
    "categories.json",
    f"{_RAW_BASE}/categories.json",
    "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
)
HUB_TRAIN_PARQUET = RemoteFile(
    "hub/train.parquet",
    f"{_HUB_BASE}/train-00000-of-00001.parquet",
    "3c648a31689f4ab3acbd4f4f4d120944bb521cf6ba57da77aa87c57e8979be81",
)
HUB_TEST_PARQUET = RemoteFile(
    "hub/test.parquet",
    f"{_HUB_BASE}/test-00000-of-00001.parquet",
    "318da70fb77a0e01bcfaecc97ef6e3645313aab98c429f0f0630c9d48e703ecc",
)


class DatasetError(RuntimeError):
    """Banking77 could not be fetched, verified or parsed; the message says why and what to do."""


class SourceUnavailable(DatasetError):
    """A source could not be reached (network error, timeout or an HTTP error status)."""


class ChecksumError(DatasetError):
    """A downloaded file does not match its pinned SHA-256, even after one re-download."""


@dataclass(frozen=True, slots=True)
class BankingData:
    """The official splits as ``(text, label)`` pairs in file order, and the sorted label list."""

    train: list[tuple[str, str]]
    test: list[tuple[str, str]]
    labels: list[str]
    source: Literal["github", "hub"] = "github"


@dataclass(frozen=True, slots=True)
class Row:
    """One demo example with its gold label and demo split.

    ``index`` is the 0-based data-row position in the official file the row came from: ``train.csv`` for
    the train and valid splits, ``test.csv`` for the test split.
    """

    text: str
    gold: str
    split: Split
    index: int


def packaged_labels() -> list[str]:
    """The 77 canonical labels bundled with the demo task spec, in file order."""
    path = resources.files("taskdistill") / "_data" / "tasks" / "banking77" / "labels.txt"
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def fetch_banking77(cache_dir: Path | str | None = None) -> BankingData:
    """Download (or reuse from ``cache_dir``), verify and parse Banking77.

    Each file is checked against its pinned SHA-256 before use; a mismatching download is fetched once
    more, then the loader fails. The row and label counts (10,003 / 3,080 / 77) are asserted. When GitHub
    is unreachable or answers with an error status, the pinned Hub Parquet copy is used instead.
    """
    directory = Path(cache_dir) if cache_dir is not None else paths.datasets_dir() / "banking77"
    directory.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": _USER_AGENT}
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers=headers) as client:
        try:
            return _load_github(client, directory)
        except SourceUnavailable as exc:
            github_error = str(exc)
        log.warning("Banking77: pinned GitHub source unavailable (%s); using the Hugging Face Hub copy", github_error)
        return _load_hub(client, directory, github_error)


def _load_github(client: httpx.Client, directory: Path) -> BankingData:
    train_path = _fetch_verified(client, TRAIN_CSV, directory)
    test_path = _fetch_verified(client, TEST_CSV, directory)
    categories_path = _fetch_verified(client, CATEGORIES_JSON, directory)
    categories = json.loads(categories_path.read_text(encoding="utf-8"))
    if not isinstance(categories, list) or not all(isinstance(c, str) for c in categories):
        raise DatasetError("categories.json: expected a JSON list of label strings")
    if len(set(categories)) != len(categories):
        raise DatasetError("categories.json: duplicate labels")
    data = BankingData(parse_csv(train_path), parse_csv(test_path), sorted(categories), source="github")
    check_counts(data)
    return data


def _load_hub(client: httpx.Client, directory: Path, github_error: str) -> BankingData:
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise DatasetError(
            f"The pinned GitHub source of Banking77 is unavailable ({github_error}). The Hugging Face Hub "
            f"fallback reads Parquet with pyarrow, which is not installed: {HUB_EXTRA_HINT}."
        ) from exc
    try:
        train_path = _fetch_verified(client, HUB_TRAIN_PARQUET, directory)
        test_path = _fetch_verified(client, HUB_TEST_PARQUET, directory)
    except SourceUnavailable as exc:
        raise DatasetError(
            f"Banking77 could not be downloaded from the pinned GitHub commit ({github_error}) or from the "
            f"Hugging Face Hub copy ({exc}). Check the network connection and retry; the verified files are "
            f"cached in {directory} after the first successful download."
        ) from exc
    train, test = read_parquet(train_path), read_parquet(test_path)
    data = BankingData(train, test, sorted({label for _, label in train}), source="hub")
    check_counts(data)
    return data


def _fetch_verified(client: httpx.Client, file: RemoteFile, directory: Path) -> Path:
    """Return the cached file if its SHA-256 matches, else download it (one re-download on mismatch)."""
    path = directory / file.name
    if path.is_file():
        if _sha256_file(path) == file.sha256:
            return path
        log.warning("Banking77: cached %s fails its SHA-256 check; downloading it again", file.name)
        path.unlink()
    actual = ""
    for _ in range(2):
        payload = _download(client, file.url)
        actual = hashlib.sha256(payload).hexdigest()
        if actual == file.sha256:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".part")
            partial.write_bytes(payload)
            partial.replace(path)
            return path
        log.warning("Banking77: %s has SHA-256 %s, expected %s", file.url, actual, file.sha256)
    raise ChecksumError(
        f"{file.url}: SHA-256 {actual} does not match the pinned {file.sha256}, twice. The response was not "
        "the pinned file (for example a proxy or captive-portal page, or a corrupted transfer); refusing to use it."
    )


def _download(client: httpx.Client, url: str) -> bytes:
    """GET with up to ``RETRIES`` retries on request errors, HTTP 429 and 5xx.

    Request errors are every :class:`httpx.RequestError`: network errors and timeouts, but also redirect
    loops and undecodable bodies, so each failure to fetch ends as :class:`SourceUnavailable`.
    """
    last = ""
    for attempt in range(RETRIES + 1):
        if attempt:
            time.sleep(BACKOFF_S * 2 ** (attempt - 1))
        try:
            response = client.get(url)
        except httpx.RequestError as exc:
            last = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            continue
        if response.status_code == 200:
            return response.content
        last = f"HTTP {response.status_code}"
        if response.status_code != 429 and response.status_code < 500:
            raise SourceUnavailable(f"{url}: {last}")
    raise SourceUnavailable(f"{url}: {last} after {RETRIES + 1} attempts")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv(path: Path) -> list[tuple[str, str]]:
    """Parse a Banking77 CSV (header ``text,category``) into ``(text, label)`` pairs, texts verbatim.

    Uses the ``csv`` module with ``newline=""`` because some quoted texts contain line breaks.
    """
    rows: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != ["text", "category"]:
            raise DatasetError(f"{path.name}: expected the header text,category, found {header}")
        for record in reader:
            if len(record) != 2 or not record[1]:
                raise DatasetError(f"{path.name}: malformed record ending on line {reader.line_num}: {record!r}")
            rows.append((record[0], record[1]))
    return rows


def read_parquet(path: Path) -> list[tuple[str, str]]:
    """Read a Hub Parquet file (``text``, ``label`` as ClassLabel int) into ``(text, label)`` pairs."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetError(f"reading {path.name} needs pyarrow: {HUB_EXTRA_HINT}") from exc
    table = pq.read_table(path, columns=["text", "label"])
    names = _classlabel_names(table.schema.metadata or {}, path.name)
    rows: list[tuple[str, str]] = []
    for text, label in zip(table.column("text").to_pylist(), table.column("label").to_pylist(), strict=True):
        if not isinstance(text, str) or not isinstance(label, int) or not 0 <= label < len(names):
            raise DatasetError(f"{path.name}: malformed row {len(rows)}: text={text!r}, label={label!r}")
        rows.append((text, names[label]))
    return rows


def _classlabel_names(metadata: Mapping[bytes, bytes], name: str) -> list[str]:
    """The ClassLabel names stored in the Hugging Face schema metadata of a Parquet file."""
    try:
        info = json.loads(metadata[b"huggingface"])
        names = info["info"]["features"]["label"]["names"]
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError(f"{name}: no ClassLabel names in the Hugging Face schema metadata") from exc
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise DatasetError(f"{name}: ClassLabel names are not a list of strings")
    return names


def check_counts(data: BankingData) -> None:
    """Fail unless the data has exactly 10,003 train rows, 3,080 test rows and the same 77 labels in both."""
    problems: list[str] = []
    if len(data.train) != EXPECTED_TRAIN:
        problems.append(f"train has {len(data.train):,} rows, expected {EXPECTED_TRAIN:,}")
    if len(data.test) != EXPECTED_TEST:
        problems.append(f"test has {len(data.test):,} rows, expected {EXPECTED_TEST:,}")
    if len(data.labels) != EXPECTED_LABELS:
        problems.append(f"{len(data.labels)} labels, expected {EXPECTED_LABELS}")
    known = set(data.labels)
    for split, rows in (("train", data.train), ("test", data.test)):
        seen = {label for _, label in rows}
        if seen != known:
            extra, missing = sorted(seen - known), sorted(known - seen)
            problems.append(f"{split} labels differ from the label list (unknown: {extra}, absent: {missing})")
    if problems:
        raise DatasetError(f"Banking77 ({data.source}) failed its integrity check: " + "; ".join(problems))


def demo_splits(data: BankingData) -> dict[Split, list[Row]]:
    """Assign every row its demo split. The rule is fixed and never depends on ``--seed``.

    - test: every row of the official ``test.csv``.
    - train/valid: the official ``train.csv`` rows are grouped by gold label, and each label's rows into
      clusters of texts that are equal after :func:`~taskdistill.curate.extract.normalise_text` (the
      text curate hashes to merge records), clusters in order of their first row. One
      ``random.Random(13)`` walks the labels in sorted order and shuffles each label's clusters in turn.
      Whole clusters then go to valid, in shuffled order, until at least ``ceil(n / 10)`` of the label's
      ``n`` rows are there; the remaining clusters go to train. Equal texts therefore never straddle
      train and valid, where curate would merge them into one example with two ``meta.split`` values.
      A label without equal texts gets exactly ``ceil(n / 10)`` valid rows, and a file without any is
      split exactly as by a per-row shuffle. Otherwise a label's valid share overshoots by less than the
      size of the last cluster taken.
    - Each split is then put in ascending ``index`` order and shuffled with a fresh ``random.Random(13)``,
      because the official files are grouped by label and a prefix must not be.

    On the official data this gives 8,965 train, 1,038 valid and 3,080 test rows. Its four pairs of
    equal train texts each stay on one side. Equal texts under different labels (there are none) are
    not clustered. Seven official test texts also occur in ``train.csv``; those cross-file pairs come
    from the official split and are left to curate, which never removes a predefined test row.
    """
    clusters_by_label: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for index, (text, label) in enumerate(data.train):
        clusters_by_label[label].setdefault(normalise_text(text), []).append(index)
    rng = random.Random(SEED)
    valid_ids: set[int] = set()
    for label in sorted(clusters_by_label):
        clusters = list(clusters_by_label[label].values())
        target = -(-sum(len(cluster) for cluster in clusters) * VALID_PERCENT // 100)
        rng.shuffle(clusters)
        taken = 0
        for cluster in clusters:
            if taken >= target:
                break
            valid_ids.update(cluster)
            taken += len(cluster)
    train_rows = [Row(t, g, "train", i) for i, (t, g) in enumerate(data.train) if i not in valid_ids]
    valid_rows = [Row(t, g, "valid", i) for i, (t, g) in enumerate(data.train) if i in valid_ids]
    test_rows = [Row(t, g, "test", i) for i, (t, g) in enumerate(data.test)]
    return {"train": _fixed_order(train_rows), "valid": _fixed_order(valid_rows), "test": _fixed_order(test_rows)}


def quick_subset(rows: Sequence[Row], split: Split) -> list[Row]:
    """The fixed quick-profile subset of one full demo split (``QUICK_SIZES[split]`` rows).

    Labels are balanced as far as possible: every label gets ``n // L`` rows or all it has, and the
    remainder goes one row per label to the labels with the most rows (ties by label name). Within a
    label, rows in ``index`` order are shuffled with one ``random.Random(13)`` that walks the labels in
    sorted order, and the first ones are taken. The result depends only on the set of rows, not their
    order, and is returned in the same fixed shuffled order as :func:`demo_splits`.
    """
    if split not in QUICK_SIZES:
        raise ValueError(f"unknown split {split!r}; expected one of {', '.join(SPLITS)}")
    wrong = sorted({row.split for row in rows if row.split != split})
    if wrong:
        raise ValueError(f"quick_subset({split!r}) got rows from other splits: {', '.join(wrong)}")
    if len({row.index for row in rows}) != len(rows):
        raise ValueError(f"quick_subset({split!r}) got duplicate row indices")
    size = QUICK_SIZES[split]
    if len(rows) < size:
        raise ValueError(f"the {split} split has {len(rows)} rows, fewer than the quick size {size}")
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r.index):
        groups[row.gold].append(row)
    quota = _balanced_quota({label: len(members) for label, members in groups.items()}, size)
    rng = random.Random(SEED)
    chosen: list[Row] = []
    for label in sorted(groups):
        members = groups[label]
        rng.shuffle(members)
        chosen.extend(members[: quota[label]])
    return _fixed_order(chosen)


def profile_splits(splits: Mapping[Split, Sequence[Row]], profile: Literal["quick", "full"]) -> dict[Split, list[Row]]:
    """The demo rows for a profile: the full splits, or their fixed quick subsets."""
    if profile == "full":
        return {split: list(splits[split]) for split in SPLITS}
    return {split: quick_subset(splits[split], split) for split in SPLITS}


def to_records(rows: Iterable[Row]) -> list[dict[str, Any]]:
    """Records for ``capture --import --format inputs``: the input, its gold label and its split."""
    return [{"input": row.text, "gold": row.gold, "meta": {"split": row.split}} for row in rows]


def _balanced_quota(available: Mapping[str, int], size: int) -> dict[str, int]:
    """Split ``size`` across labels as evenly as their ``available`` counts allow (water-filling)."""
    if size > sum(available.values()):
        raise ValueError(f"cannot take {size} rows from {sum(available.values())}")
    quota = dict.fromkeys(available, 0)
    remaining = size
    while remaining:
        open_labels = sorted(label for label in available if quota[label] < available[label])
        share, extra = divmod(remaining, len(open_labels))
        if share == 0:
            by_size = sorted(open_labels, key=lambda label: (-available[label], label))
            for label in by_size[:extra]:
                quota[label] += 1
            break
        for label in open_labels:
            take = min(share, available[label] - quota[label])
            quota[label] += take
            remaining -= take
    return quota


def _fixed_order(rows: list[Row]) -> list[Row]:
    ordered = sorted(rows, key=lambda r: r.index)
    random.Random(SEED).shuffle(ordered)
    return ordered
