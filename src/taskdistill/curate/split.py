"""Curate stages 6-8: split assignment, dedupe inside each split and cross-split removal.

Split assignment. A row whose predefined split field (``curate.split.predefined``, e.g. ``meta.split``) is set keeps
that split. The other rows take ``val`` and ``test`` fractions of their number. They are ordered by a seeded hash
(SHA-256 of ``curate.split.seed`` and the row's input hash, or its group key), never by a shuffle of the whole
population, so adding rows later moves at most the few rows at a split boundary:

- grouped, when ``group_by`` is set: whole groups go to one split (rows without a group are groups of one). In hash
  order, test takes each group whose midpoint still falls inside its row target, then valid does the same with the
  rest; each gets at least one group when its target is above zero, and train always keeps at least one;
- stratified by label for classification with ``stratify`` (the teacher value, else the gold, else
  ``__unlabelled__``): each label's rows, in hash order, are spread evenly over [0, 1) with a seeded offset, and the
  rows are then taken in that order, so every label reaches each split in proportion;
- otherwise the hash order itself.

Dedupe runs on the PII-scrubbed text the student trains on. A duplicate cluster keeps one example: the teacher
values of its members vote (unlabelled members abstain) and the lowest-index member carrying the strict-majority
value survives. Without a majority the whole cluster is dropped. The survivor takes the strict-majority gold of the
cluster; when the golds conflict without a majority it keeps its own gold. Gold conflicts are counted.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from taskdistill.config import TaskSpec
from taskdistill.curate.dedupe import cross_split_duplicates, dedupe_key, find_duplicate_clusters, majority_vote, words
from taskdistill.curate.merge import SPLITS, Example, canonical_split, meta_get, meta_keys, meta_set
from taskdistill.teacher.request_key import canonical_json

UNLABELLED = "__unlabelled__"

BySplit = dict[str, list[Example]]


def _targets(n: int, val: float, test: float) -> tuple[int, int]:
    n_test = math.floor(test * n + 0.5)
    n_val = min(math.floor(val * n + 0.5), n - n_test)
    return n_test, n_val


def seeded_rank(seed: int, kind: str, key: str) -> int:
    """A 64-bit rank from SHA-256 of the seed and a key: the same key keeps its rank whatever else is present."""
    digest = hashlib.sha256(f"{seed}\x00{kind}\x00{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _row_order(seed: int, rows: Sequence[Example]) -> list[Example]:
    return sorted(rows, key=lambda ex: (seeded_rank(seed, "row", ex.input_hash), ex.input_hash))


def _stratum(ex: Example) -> str:
    value = ex.teacher if ex.teacher is not None else ex.gold
    return UNLABELLED if value is None else str(value)


def _assign_ordered(rows: Sequence[Example], n_test: int, n_val: int) -> None:
    for position, ex in enumerate(rows):
        ex.split = "test" if position < n_test else "valid" if position < n_test + n_val else "train"


def _split_stratified(rows: list[Example], seed: int, n_test: int, n_val: int) -> None:
    strata: dict[str, list[Example]] = {}
    for ex in rows:
        strata.setdefault(_stratum(ex), []).append(ex)
    keyed: list[tuple[float, str, int, Example]] = []
    for label, members in strata.items():
        ordered = _row_order(seed, members)
        offset = seeded_rank(seed, "stratum", label) / 2**64
        size = len(ordered)
        keyed.extend(((i + offset) / size, label, i, ex) for i, ex in enumerate(ordered))
    keyed.sort(key=lambda item: item[:3])
    _assign_ordered([item[3] for item in keyed], n_test, n_val)


def fill_groups(order: Sequence[str], sizes: Mapping[str, int], target: int, keep: int) -> list[str]:
    """The groups one split takes, in ``order``: each group whose midpoint still falls inside ``target`` rows.

    When ``target`` > 0 and no group fits, the smallest group (the first in ``order`` on a tie) is taken, so the
    split is never left empty; at least ``keep`` groups are always left for the splits after it.
    """
    if target <= 0 or len(order) <= keep:
        return []
    chosen: list[str] = []
    count = 0
    for key in order:
        if count + sizes[key] / 2 < target:
            chosen.append(key)
            count += sizes[key]
    if not chosen:
        chosen = [min(order, key=lambda key: sizes[key])]
    return chosen[: len(order) - keep]


def _split_grouped(rows: list[Example], seed: int, n_test: int, n_val: int, group_keys: Sequence[str]) -> int:
    groups: dict[str, list[Example]] = {}
    for ex in rows:
        value = meta_get(ex.meta, group_keys)
        group = ["group", value] if value is not None else ["row", ex.input_hash]
        groups.setdefault(canonical_json(group).decode("utf-8"), []).append(ex)
    order = sorted(groups, key=lambda key: (seeded_rank(seed, "group", key), key))
    sizes = {key: len(members) for key, members in groups.items()}
    keep = 1 if len(rows) - n_test - n_val > 0 else 0
    test = set(fill_groups(order, sizes, n_test, keep))
    rest = [key for key in order if key not in test]
    valid = set(fill_groups(rest, sizes, n_val, keep))
    for key, members in groups.items():
        name = "test" if key in test else "valid" if key in valid else "train"
        for ex in members:
            ex.split = name
    return len(groups)


def assign_splits(spec: TaskSpec, examples: Sequence[Example]) -> dict[str, Any]:
    """Stage 6: set ``example.split`` for every example (predefined first, then the seeded hash-ordered split)."""
    cfg = spec.curate.split
    split_keys = meta_keys(cfg.predefined)
    predefined: Counter[str] = Counter()
    undecided: list[Example] = []
    for ex in examples:
        name = canonical_split(meta_get(ex.meta, split_keys), cfg.predefined or "") if split_keys else None
        if name is None:
            undecided.append(ex)
            continue
        meta_set(ex.meta, split_keys, name)
        ex.split = name
        predefined[name] += 1

    n_test, n_val = _targets(len(undecided), cfg.val, cfg.test)
    groups = None
    if not undecided:
        method = "predefined"
    elif cfg.group_by:
        method = "grouped"
        groups = _split_grouped(undecided, cfg.seed, n_test, n_val, meta_keys(cfg.group_by))
    elif spec.type == "classification" and cfg.stratify:
        method = "stratified"
        _split_stratified(undecided, cfg.seed, n_test, n_val)
    else:
        method = "random"
        _assign_ordered(_row_order(cfg.seed, undecided), n_test, n_val)

    randomly = Counter(ex.split for ex in undecided)
    totals = Counter(ex.split for ex in examples)
    return {
        "method": method,
        "predefined_field": cfg.predefined,
        "group_by": cfg.group_by,
        "seed": cfg.seed,
        "fractions": {"val": cfg.val, "test": cfg.test},
        "targets": {"valid": n_val, "test": n_test},
        "predefined": {name: predefined[name] for name in SPLITS},
        "random": {name: randomly[name] for name in SPLITS},
        "groups": groups,
        "splits": {name: totals[name] for name in SPLITS},
    }


def by_split(examples: Sequence[Example]) -> BySplit:
    """Examples grouped by split, each list in the incoming order (input-hash order in the pipeline)."""
    out: BySplit = {name: [] for name in SPLITS}
    for ex in examples:
        if ex.split not in out:
            raise ValueError(f"example {ex.input_hash[:12]} has no split")
        out[ex.split].append(ex)
    return out


def _value_key(value: Any) -> bytes:
    return canonical_json(value)


def resolve_cluster(members: Sequence[Example]) -> tuple[Example | None, bool]:
    """The example a duplicate cluster keeps (None when its labelled members have no strict majority), and whether
    the members' golds conflict.

    Members are in index order. The survivor is the first member carrying the majority teacher value (the first
    member when none is labelled). Its gold becomes the strict-majority gold of the cluster; when the golds conflict
    without a majority it keeps its own.
    """
    labelled = [ex for ex in members if ex.teacher is not None]
    if labelled:
        ok, winner = majority_vote(ex.teacher for ex in labelled)
        if not ok:
            return None, False
        target = _value_key(winner)
        keep = next(ex for ex in labelled if _value_key(ex.teacher) == target)
    else:
        keep = members[0]
    golds = [ex.gold for ex in members if ex.gold is not None]
    conflict = len({_value_key(gold) for gold in golds}) > 1
    ok, gold = majority_vote(golds)
    if ok:
        keep.gold = gold
    return keep, conflict


def dedupe_split(
    examples: Sequence[Example], threshold: float, *, exact: bool = True
) -> tuple[list[Example], dict[str, Any]]:
    """Exact and near-duplicate removal inside one split, on the scrubbed text.

    With ``exact`` off, texts under 3 words (which only an exact match can catch) are left alone; longer exact
    copies are still near-duplicates at any threshold.
    """
    texts = [ex.text for ex in examples]
    if exact:
        clusters = find_duplicate_clusters(texts, threshold)
    else:
        positions = [i for i, text in enumerate(texts) if len(words(text)) >= 3]
        clusters = [
            [positions[j] for j in c] for c in find_duplicate_clusters([texts[i] for i in positions], threshold)
        ]
    drop: set[int] = set()
    counts: Counter[str] = Counter()
    for cluster in clusters:
        members = [examples[i] for i in cluster]
        keep, gold_conflict = resolve_cluster(members)
        if gold_conflict:
            counts["gold_conflicts"] += 1
        if keep is None:
            counts["conflict_clusters"] += 1
            counts["conflict_examples"] += len(cluster)
            drop.update(cluster)
            continue
        kept_key = dedupe_key(keep.text)
        counts["clusters"] += 1
        for i in cluster:
            if examples[i] is keep:
                continue
            drop.add(i)
            counts["exact" if dedupe_key(examples[i].text) == kept_key else "near"] += 1
    kept = [ex for i, ex in enumerate(examples) if i not in drop]
    return kept, {
        "before": len(examples),
        "after": len(kept),
        "repeated_inputs": sum(ex.repeats for ex in examples),
        "clusters": counts["clusters"],
        "removed_exact": counts["exact"],
        "removed_near": counts["near"],
        "conflict_clusters": counts["conflict_clusters"],
        "conflict_examples": counts["conflict_examples"],
        "gold_conflicts": counts["gold_conflicts"],
    }


def dedupe_within(
    splits: Mapping[str, Sequence[Example]], threshold: float, *, exact: bool = True
) -> tuple[BySplit, dict[str, Any]]:
    """Stage 7: :func:`dedupe_split` for each split."""
    out: BySplit = {}
    per_split: dict[str, Any] = {}
    for name in SPLITS:
        out[name], per_split[name] = dedupe_split(splits.get(name, []), threshold, exact=exact)
    totals = {
        key: sum(stats[key] for stats in per_split.values())
        for key in (
            "repeated_inputs", "removed_exact", "removed_near", "conflict_clusters", "conflict_examples",
            "gold_conflicts",
        )
    }  # fmt: skip
    return out, {"threshold": threshold, "exact": exact, **totals, "splits": per_split}


def remove_cross_split(splits: Mapping[str, Sequence[Example]], threshold: float) -> tuple[BySplit, dict[str, Any]]:
    """Stage 8: drop train rows that duplicate a valid or test row, then valid rows that duplicate a test row.

    Test is never changed.
    """
    train, valid, test = (list(splits.get(name, [])) for name in SPLITS)
    pairs: dict[str, Counter[str]] = {"train/valid": Counter(), "train/test": Counter(), "valid/test": Counter()}

    against = [ex.text for ex in valid] + [ex.text for ex in test]
    found = cross_split_duplicates([ex.text for ex in train], against, threshold)
    for kind, j, _ in found.values():
        pairs["train/valid" if j < len(valid) else "train/test"][kind] += 1
    train = [ex for i, ex in enumerate(train) if i not in found]
    removed_train = len(found)

    found = cross_split_duplicates([ex.text for ex in valid], [ex.text for ex in test], threshold)
    for kind, _, _ in found.values():
        pairs["valid/test"][kind] += 1
    valid = [ex for i, ex in enumerate(valid) if i not in found]
    removed_valid = len(found)

    return {"train": train, "valid": valid, "test": test}, {
        "threshold": threshold,
        "removed": {"train": removed_train, "valid": removed_valid, "test": 0},
        "pairs": {name: {"exact": c["exact"], "near": c["near"]} for name, c in pairs.items()},
    }


def label_distribution(values: Sequence[Any]) -> dict[str, int]:
    """Counts of each value, most common first (ties by value)."""
    counts = Counter(v if isinstance(v, str) else json.dumps(v, sort_keys=True) for v in values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
