"""Duplicate detection: within-split clusters, cross-split matches and an independent leakage check.

A text's dedupe key is its normalised input (NFKC, collapsed whitespace, stripped), case-folded. Two texts are exact
duplicates when their keys are equal, and near-duplicates when the Jaccard similarity of their word 3-shingle sets is at
least the threshold. Texts under 3 words have no 3-shingles and are matched by their exact key only.

MinHash LSH only proposes near-duplicate candidates: texts whose signatures share the key of any band. Every candidate
is verified with the exact Jaccard, so the sketches can cost recall but never precision. The bands are chosen so that a
pair sitting exactly at the threshold is missed with probability at most ``MAX_MISS_RATE``. A threshold so low that
even one-row bands cannot meet that bound is rejected (below about 0.103 with 128 permutations).

Clustering keeps each band bucket grouped by cluster: a candidate cluster that already holds the new text is skipped,
and any other cluster is left at its first verified member, so thousands of mutual near-duplicates cost about one
verification each. One case stays quadratic: many texts that sit just below the threshold of one another, such as a
single template whose varying slot leaves every pair near 0.8. At this recall bound LSH cannot separate 0.8 from 0.9,
so nearly every such pair is verified (13,000 such texts take about a minute).
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from datasketch import MinHash, MinHashLSH

from taskdistill.curate.extract import normalise_text
from taskdistill.teacher.request_key import canonical_json

SHINGLE_SIZE = 3
DEFAULT_THRESHOLD = 0.9
DEFAULT_NUM_PERM = 128
DEFAULT_SEED = 1
#: Upper bound on the probability that LSH fails to propose a pair whose Jaccard equals the threshold.
MAX_MISS_RATE = 1e-6
#: The leakage check builds its own sketches: more permutations, another seed and another hash function.
LEAK_NUM_PERM = 256
LEAK_SEED = 7919

DuplicateKind = Literal["exact", "near"]

_WORD = re.compile(r"\w+")


def dedupe_key(text: str) -> str:
    """Exact-match key: the normalised text, case-folded."""
    return normalise_text(text).casefold()


def words(text: str) -> list[str]:
    """Word tokens (``\\w+`` runs) of the dedupe key."""
    return _WORD.findall(dedupe_key(text))


def shingles(text: str) -> frozenset[str]:
    """Word 3-shingles of ``text`` joined by single spaces; empty for texts under 3 words."""
    tokens = words(text)
    return frozenset(" ".join(tokens[i : i + SHINGLE_SIZE]) for i in range(len(tokens) - SHINGLE_SIZE + 1))


def jaccard(a: AbstractSet[str], b: AbstractSet[str]) -> float:
    """Exact Jaccard similarity of two shingle sets. Two empty sets score 0.0: no shingles, no near-duplicate."""
    common = len(a & b)
    union = len(a) + len(b) - common
    return common / union if union else 0.0


def lsh_bands(threshold: float, num_perm: int, max_miss: float = MAX_MISS_RATE) -> tuple[int, int]:
    """(bands, rows) for MinHash LSH: the widest band whose miss rate at ``threshold`` is at most ``max_miss``.

    A pair with Jaccard ``j`` becomes a candidate with probability ``1 - (1 - j**rows) ** bands``. Wider bands propose
    fewer dissimilar pairs; the recall bound keeps pairs at the threshold. Raises ``ValueError`` when even one-row
    bands miss more often than ``max_miss``.
    """
    if num_perm < 1:
        raise ValueError(f"num_perm must be positive, got {num_perm!r}")
    for rows in range(max(num_perm // 2, 1), 0, -1):
        bands = num_perm // rows
        if (1.0 - threshold**rows) ** bands <= max_miss:
            return bands, rows
    lowest = 1.0 - max_miss ** (1.0 / num_perm)
    raise ValueError(
        f"threshold {threshold!r} is too low for MinHash LSH with num_perm={num_perm}: even one-row bands miss a pair "
        f"at the threshold with probability {(1.0 - threshold) ** num_perm:.3g}, above {max_miss:g}; "
        f"use a threshold of at least {math.ceil(lowest * 1000) / 1000:g}"
    )


def _check_threshold(threshold: float, num_perm: int) -> tuple[int, int]:
    """Validate ``threshold`` for the dedupe path and return its LSH bands."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    return lsh_bands(threshold, num_perm)


def _encoded(shingle_set: AbstractSet[str]) -> list[bytes]:
    return [s.encode("utf-8") for s in shingle_set]


def _sketches(sets: Iterable[AbstractSet[str]], num_perm: int, seed: int) -> Iterator[Any]:
    sketches: Iterator[Any] = MinHash.generator((_encoded(s) for s in sets), num_perm=num_perm, seed=seed)
    return sketches


def _band_keys(sketch: Any, bands: int, rows: int) -> list[bytes]:
    """One bucket key per band: the raw bytes of that band's slice of the signature."""
    hashvalues = sketch.hashvalues
    return [hashvalues[band * rows : (band + 1) * rows].tobytes() for band in range(bands)]


def _unique(items: Iterable[int]) -> Iterator[int]:
    """``items`` without repeats, lazily, so a caller that stops early does not pay for the rest."""
    seen: set[int] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            yield item


class _UnionFind:
    """Disjoint sets over ``0..n-1``; a set's root is its smallest member."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, i: int) -> int:
        parent = self.parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)

    def clusters(self) -> list[list[int]]:
        """Sets with 2+ members, members sorted, ordered by their first member."""
        groups: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            groups.setdefault(self.find(i), []).append(i)
        return sorted((members for members in groups.values() if len(members) > 1), key=lambda members: members[0])


def find_duplicate_clusters(
    texts: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    seed: int = DEFAULT_SEED,
) -> list[list[int]]:
    """Clusters (index lists, 2+ members) of exact and near-duplicate texts, merged transitively.

    Exact-key groups are joined with verified near-duplicate pairs by union-find. Each band bucket groups its members
    by the cluster root they were added under. A new text skips the groups of the cluster it already belongs to and
    joins any other group at its first candidate member with Jaccard at or above the threshold. The skipped pairs are
    already connected, so the clusters equal those of verifying every candidate pair. Members are sorted and clusters
    ordered by their first member.
    """
    bands, rows = _check_threshold(threshold, num_perm)
    components = _UnionFind(len(texts))

    first_by_key: dict[str, int] = {}
    representatives: list[int] = []
    for i, text in enumerate(texts):
        first = first_by_key.setdefault(dedupe_key(text), i)
        if first == i:
            representatives.append(i)
        else:
            components.union(first, i)

    sets = {i: shingles(texts[i]) for i in representatives}
    indexed = [i for i in representatives if sets[i]]
    # Per band: bucket key -> {root when added: members added under it}. ``added`` holds the same groups across bands.
    tables: list[dict[bytes, dict[int, list[int]]]] = [{} for _ in range(bands)]
    added: dict[int, list[int]] = {}
    for i, sketch in zip(indexed, _sketches((sets[i] for i in indexed), num_perm, seed), strict=True):
        keys = _band_keys(sketch, bands, rows)
        buckets = [bucket for table, key in zip(tables, keys, strict=True) if (bucket := table.get(key))]
        roots: set[int] = set()
        for bucket in buckets:
            roots.update(bucket)
        own = components.find(i)
        for root in roots:
            if components.find(root) == own:
                continue
            group = added[root]
            # A group of one is its root alone, and the root shares a bucket with ``i``.
            candidates = group if len(group) == 1 else _unique(m for b in buckets for m in b.get(root, ()))
            for member in candidates:
                if jaccard(sets[member], sets[i]) >= threshold:
                    components.union(member, i)
                    own = components.find(i)
                    break
        added.setdefault(own, []).append(i)
        for table, key in zip(tables, keys, strict=True):
            table.setdefault(key, {}).setdefault(own, []).append(i)

    return components.clusters()


def majority_vote(values: Iterable[Any]) -> tuple[bool, Any]:
    """``(True, winner)`` when one value holds a strict majority of the non-None votes, else ``(False, None)``.

    Values are compared by canonical JSON, so dicts with reordered keys count as one value; the first occurrence of
    the winner is returned as given.
    """
    votes = [v for v in values if v is not None]
    if not votes:
        return False, None
    first: dict[bytes, Any] = {}
    counts: Counter[bytes] = Counter()
    for vote in votes:
        key = canonical_json(vote)
        first.setdefault(key, vote)
        counts[key] += 1
    key, count = counts.most_common(1)[0]
    if 2 * count > len(votes):
        return True, first[key]
    return False, None


def cross_split_duplicates(
    texts: Sequence[str],
    against: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    *,
    num_perm: int = DEFAULT_NUM_PERM,
    seed: int = DEFAULT_SEED,
) -> dict[int, tuple[str, int, float]]:
    """Map each index in ``texts`` that duplicates a text in ``against`` to ``(kind, index in against, jaccard)``.

    An exact match wins and reports the first matching index with Jaccard 1.0; otherwise the near-duplicate with the
    highest Jaccard (then the lowest index) is reported. Finding the highest means verifying every candidate, so
    ``against`` should be deduplicated within itself first, as the curate pipeline does.
    """
    bands, rows = _check_threshold(threshold, num_perm)
    first_by_key: dict[str, int] = {}
    for j, text in enumerate(against):
        first_by_key.setdefault(dedupe_key(text), j)

    found: dict[int, tuple[str, int, float]] = {}
    pending: list[int] = []
    for i, text in enumerate(texts):
        match = first_by_key.get(dedupe_key(text))
        if match is not None:
            found[i] = ("exact", match, 1.0)
        else:
            pending.append(i)

    text_sets = {i: shingles(texts[i]) for i in pending}
    pending = [i for i in pending if text_sets[i]]
    against_sets = [shingles(text) for text in against]
    against_positions = [j for j, s in enumerate(against_sets) if s]
    if pending and against_positions:
        tables: list[dict[bytes, list[int]]] = [{} for _ in range(bands)]
        sketches = _sketches((against_sets[j] for j in against_positions), num_perm, seed)
        for j, sketch in zip(against_positions, sketches, strict=True):
            for table, key in zip(tables, _band_keys(sketch, bands, rows), strict=True):
                table.setdefault(key, []).append(j)
        for i, sketch in zip(pending, _sketches((text_sets[i] for i in pending), num_perm, seed), strict=True):
            candidates: set[int] = set()
            for table, key in zip(tables, _band_keys(sketch, bands, rows), strict=True):
                candidates.update(table.get(key, ()))
            best: tuple[float, int] | None = None
            for j in candidates:
                score = jaccard(text_sets[i], against_sets[j])
                if score >= threshold and (best is None or (-score, j) < (-best[0], best[1])):
                    best = (score, j)
            if best is not None:
                found[i] = ("near", best[1], best[0])
    return dict(sorted(found.items()))


@dataclass(frozen=True)
class LeakagePair:
    """One cross-split duplicate: ``split_a`` precedes ``split_b`` in the order the splits were given."""

    split_a: str
    index_a: int
    split_b: str
    index_b: int
    kind: DuplicateKind
    jaccard: float


@dataclass
class LeakageReport:
    """Result of :func:`leakage_check`; ``ok`` only when no cross-split duplicate was found."""

    pairs: list[LeakagePair]
    sizes: dict[str, int]
    threshold: float = DEFAULT_THRESHOLD
    method: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.pairs

    @property
    def exact(self) -> int:
        return sum(pair.kind == "exact" for pair in self.pairs)

    @property
    def near(self) -> int:
        return sum(pair.kind == "near" for pair in self.pairs)

    def by_split_pair(self) -> dict[str, int]:
        """Pair counts keyed ``"<split_a>/<split_b>"`` for every pair of splits, zeros included."""
        names = list(self.sizes)
        counts = {f"{a}/{b}": 0 for a, b in itertools.combinations(names, 2)}
        for pair in self.pairs:
            name = f"{pair.split_a}/{pair.split_b}"
            counts[name] = counts.get(name, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "threshold": self.threshold,
            "sizes": dict(self.sizes),
            "exact": self.exact,
            "near": self.near,
            "by_split_pair": self.by_split_pair(),
            "method": dict(self.method),
            "pairs": [asdict(pair) for pair in self.pairs],
        }


def _leak_hash(data: bytes) -> int:
    """32-bit BLAKE2b of a shingle, the leakage check's own hash (the dedupe path uses datasketch's SHA-1)."""
    return int.from_bytes(hashlib.blake2b(data, digest_size=4, person=b"taskdistill-leak").digest(), "little")


def _leak_bands(threshold: float) -> tuple[int, int]:
    """(bands, rows) for the leakage check's LSH: the most rows per band that keep the miss rate within bound.

    Computed apart from :func:`lsh_bands`, in log space over every band width that leaves the two bands
    ``MinHashLSH`` needs, so a slip in one is not repeated in the other.
    """
    bound = math.log(MAX_MISS_RATE)
    best: tuple[int, int] | None = None
    for rows in range(1, LEAK_NUM_PERM // 2 + 1):
        bands = LEAK_NUM_PERM // rows
        hit = threshold**rows
        log_miss = -math.inf if hit >= 1.0 else bands * math.log1p(-hit)
        if log_miss <= bound:
            best = (bands, rows)
    if best is None:
        raise ValueError(
            f"threshold {threshold!r} is too low for the leakage check's MinHash LSH with num_perm={LEAK_NUM_PERM}: "
            f"even one-row bands miss a pair at the threshold more often than {MAX_MISS_RATE:g}"
        )
    return best


def leakage_check(splits: Mapping[str, Sequence[str]], threshold: float = DEFAULT_THRESHOLD) -> LeakageReport:
    """Find every exact or near-duplicate pair of texts that sit in different splits.

    Independent of the dedupe path: it shares only the definitions (``dedupe_key``, ``shingles``, ``jaccard`` and
    ``MAX_MISS_RATE``). It builds its own exact-key index, its own band choice and its own MinHash sketches (256
    permutations, another seed, a BLAKE2b hash) held in datasketch's ``MinHashLSH`` rather than the dedupe path's band
    buckets, and verifies every candidate by exact Jaccard. Every pair of splits is checked.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    bands, rows = _leak_bands(threshold)
    names = list(splits)
    items = [(name, index) for name in names for index in range(len(splits[name]))]
    texts = [splits[name][index] for name, index in items]
    keys = [dedupe_key(text) for text in texts]

    found: dict[tuple[int, int], LeakagePair] = {}

    def record(p: int, q: int, kind: DuplicateKind, score: float) -> None:
        (split_a, index_a), (split_b, index_b) = items[p], items[q]
        found[(p, q)] = LeakagePair(split_a, index_a, split_b, index_b, kind, score)

    positions_by_key: dict[str, list[int]] = {}
    for position, key in enumerate(keys):
        positions_by_key.setdefault(key, []).append(position)
    for positions in positions_by_key.values():
        for p, q in itertools.combinations(positions, 2):
            if items[p][0] != items[q][0]:
                record(p, q, "exact", 1.0)

    # One index per split, each text queried against the splits before its own: same-split pairs never come up.
    sets = [shingles(text) for text in texts]
    earlier: list[Any] = []
    start = 0
    for name in names:
        indexed = [p for p in range(start, start + len(splits[name])) if sets[p]]
        start += len(splits[name])
        lsh = MinHashLSH(num_perm=LEAK_NUM_PERM, params=(bands, rows))
        sketches: Iterator[Any] = MinHash.generator(
            ([shingle.encode("utf-8") for shingle in sets[p]] for p in indexed),
            num_perm=LEAK_NUM_PERM,
            seed=LEAK_SEED,
            hashfunc=_leak_hash,
        )
        for q, sketch in zip(indexed, sketches, strict=True):
            for index in earlier:
                for p in index.query(sketch):
                    if keys[p] == keys[q]:
                        continue
                    score = jaccard(sets[p], sets[q])
                    if score >= threshold:
                        record(p, q, "near", score)
            lsh.insert(q, sketch, check_duplication=False)
        earlier.append(lsh)

    return LeakageReport(
        pairs=[found[pq] for pq in sorted(found)],
        sizes={name: len(splits[name]) for name in names},
        threshold=threshold,
        method={"num_perm": LEAK_NUM_PERM, "seed": LEAK_SEED, "bands": bands, "rows": rows, "hash": "blake2b-32"},
    )
