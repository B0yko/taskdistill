from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
import sys
import time

import pytest
from datasketch import MinHash, MinHashLSH

from taskdistill.curate import dedupe
from taskdistill.curate.dedupe import (
    LEAK_NUM_PERM,
    MAX_MISS_RATE,
    LeakagePair,
    cross_split_duplicates,
    dedupe_key,
    find_duplicate_clusters,
    jaccard,
    leakage_check,
    lsh_bands,
    majority_vote,
    shingles,
    words,
)


def _tokens(n: int, prefix: str = "t") -> list[str]:
    """``n`` distinct words, so a text of n words has exactly n - 2 distinct 3-shingles."""
    return [f"{prefix}{i}" for i in range(n)]


def _text(tokens: list[str]) -> str:
    return " ".join(tokens)


def _replace(tokens: list[str], position: int, word: str = "changed") -> list[str]:
    out = list(tokens)
    out[position] = word
    return out


# --- keys, words, shingles, jaccard ---------------------------------------------------------------------------------


def test_dedupe_key_folds_case_whitespace_and_width() -> None:
    assert dedupe_key("  Card\tLOST \n today ") == "card lost today"
    assert dedupe_key("ＣＡＲＤ ｌｏｓｔ") == "card lost"  # NFKC maps full-width letters
    assert dedupe_key("Straße") == "strasse"  # casefold, not lower
    assert dedupe_key("") == ""


def test_words_drop_punctuation_and_keep_underscores() -> None:
    assert words("Hello, WORLD! top_up #2") == ["hello", "world", "top_up", "2"]
    assert words("  ...  ") == []


def test_shingles_are_word_trigrams() -> None:
    assert shingles("a b c d") == {"a b c", "b c d"}
    assert shingles("One, two; THREE") == {"one two three"}
    assert shingles("go go go go go") == {"go go go"}  # a set: repeats collapse
    assert shingles("card lost") == frozenset()
    assert shingles("") == frozenset()
    assert len(shingles(_text(_tokens(30)))) == 28


def test_jaccard_hand_computed() -> None:
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == 2 / 4
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), {"a"}) == 0.0
    assert jaccard(set(), set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b", "c", "d", "e"}) == 2 / 5


def test_jaccard_of_one_word_edits_hand_computed() -> None:
    base = _tokens(30)
    # Last word changed: only the last of the 28 shingles differs -> 27 / (28 + 1).
    assert jaccard(shingles(_text(base)), shingles(_text(_replace(base, 29)))) == 27 / 29
    # Middle word changed: 3 shingles differ on each side -> 25 / 31.
    assert jaccard(shingles(_text(base)), shingles(_text(_replace(base, 15)))) == 25 / 31
    # 10 words, last word changed: 7 / 9.
    short = _tokens(10)
    assert jaccard(shingles(_text(short)), shingles(_text(_replace(short, 9)))) == 7 / 9


def test_lsh_bands_bound_the_miss_rate_at_the_threshold() -> None:
    assert lsh_bands(0.9, 128) == (21, 6)
    assert lsh_bands(0.9, 256) == (32, 8)
    for num_perm in (128, 256):
        bands, rows = lsh_bands(0.9, num_perm)
        assert bands * rows <= num_perm
        assert (1 - 0.9**rows) ** bands <= MAX_MISS_RATE
        wider = num_perm // (rows + 1)
        assert (1 - 0.9 ** (rows + 1)) ** wider > MAX_MISS_RATE  # the widest band that meets the bound
    assert lsh_bands(1.0, 128) == (2, 64)
    assert lsh_bands(0.5, 128) == (64, 2)  # 0.75**64 = 1.0e-8; three rows: 0.875**42 = 3.6e-3


def test_lsh_bands_use_one_row_bands_down_to_the_lowest_supported_threshold() -> None:
    # One-row bands miss a pair at threshold t with probability (1 - t) ** num_perm; the bound needs t >= 0.1023 at 128.
    assert lsh_bands(0.11, 128) == (128, 1)
    assert lsh_bands(0.103, 128) == (128, 1)  # 0.897**128 = 9.1e-7
    lowest_miss, below_miss = (1 - 0.103) ** 128, (1 - 0.102) ** 128
    assert lowest_miss <= MAX_MISS_RATE < below_miss
    assert lsh_bands(0.053, 256) == (256, 1)  # 0.947**256 = 8.8e-7


@pytest.mark.parametrize(
    ("threshold", "num_perm", "lowest"),
    [(0.102, 128, "0.103"), (0.05, 128, "0.103"), (0.052, 256, "0.053"), (0.001, 256, "0.053")],
)
def test_lsh_bands_reject_thresholds_below_the_recall_bound(threshold: float, num_perm: int, lowest: str) -> None:
    assert (1 - threshold) ** num_perm > MAX_MISS_RATE  # 0.898**128 = 1.05e-6, 0.95**128 = 1.4e-3
    with pytest.raises(ValueError, match=rf"threshold {threshold} .*num_perm={num_perm}.* at least {lowest}$"):
        lsh_bands(threshold, num_perm)


def test_lsh_bands_reject_a_non_positive_num_perm() -> None:
    with pytest.raises(ValueError, match="num_perm"):
        lsh_bands(0.9, 0)


# --- within-split clusters ------------------------------------------------------------------------------------------


def test_exact_duplicates_differing_in_case_and_whitespace() -> None:
    texts = ["Card lost", "other text entirely here", "card   LOST", "  CARD lost\n", "Other text entirely here"]
    assert find_duplicate_clusters(texts) == [[0, 2, 3], [1, 4]]


def test_near_duplicate_of_a_30_word_text_is_merged() -> None:
    base = _tokens(30)
    last = _text(_replace(base, 29))  # J = 27/29 = 0.931
    first = _text(_replace(base, 0, "other"))  # J = 27/29 = 0.931
    assert find_duplicate_clusters([_text(base), last]) == [[0, 1]]
    assert find_duplicate_clusters([first, _text(base)]) == [[0, 1]]


def test_one_word_edit_below_threshold_is_not_merged() -> None:
    base = _tokens(30)
    assert find_duplicate_clusters([_text(base), _text(_replace(base, 15))]) == []  # J = 25/31 = 0.806
    short = _tokens(10)
    assert find_duplicate_clusters([_text(short), _text(_replace(short, 9))]) == []  # J = 7/9 = 0.778
    assert find_duplicate_clusters([_text(short), _text(_replace(short, 5))]) == []  # J = 5/11 = 0.455


def test_middle_edit_in_a_60_word_text_is_merged() -> None:
    base = _tokens(60)
    edited = _replace(base, 30)
    assert jaccard(shingles(_text(base)), shingles(_text(edited))) == 55 / 61  # 0.9016
    assert find_duplicate_clusters([_text(base), _text(edited)]) == [[0, 1]]


def test_threshold_is_inclusive() -> None:
    twelve = _tokens(12)
    # 12 words vs its first 11: 9 of 10 shingles shared -> exactly 0.9.
    assert jaccard(shingles(_text(twelve)), shingles(_text(twelve[:11]))) == 0.9
    assert find_duplicate_clusters([_text(twelve), _text(twelve[:11])]) == [[0, 1]]
    eleven = _tokens(11)
    # 11 words vs its first 10: 8 / 9 = 0.889.
    assert find_duplicate_clusters([_text(eleven), _text(eleven[:10])]) == []
    assert find_duplicate_clusters([_text(eleven), _text(eleven[:10])], threshold=0.85) == [[0, 1]]


def test_two_word_texts_use_exact_match_only() -> None:
    texts = [
        "Card lost",  # 0
        "card  LOST",  # 1: exact duplicate of 0
        "card found",  # 2: different
        "card lost?",  # 3: punctuation changes the key; under 3 words there is no near match
        "card lost today",  # 4: 3 words, one shingle
        "card lost today please",  # 5: longer text that contains 0
    ]
    assert find_duplicate_clusters(texts) == [[0, 1]]


def test_two_word_text_is_never_a_near_duplicate_of_a_longer_one() -> None:
    assert find_duplicate_clusters(["card lost", "card lost card lost"]) == []
    assert find_duplicate_clusters(["card lost", "card lost card lost"], threshold=0.11) == []  # lowest supported
    assert find_duplicate_clusters(["card lost", "card lost"], threshold=0.11) == [[0, 1]]
    assert find_duplicate_clusters(["card lost", "card found"], threshold=0.11) == []


def test_three_word_texts_differing_in_punctuation_are_near_duplicates() -> None:
    assert find_duplicate_clusters(["Card lost, today", "card lost today!"]) == [[0, 1]]


def test_clusters_merge_exact_groups_and_near_pairs_transitively() -> None:
    base = _tokens(30)
    b = _replace(base, 29)
    c = _replace(b, 0, "other")
    # J(a, b) = J(b, c) = 27/29, J(a, c) = 26/30 = 0.867: still one cluster through b.
    assert jaccard(shingles(_text(base)), shingles(_text(c))) == 26 / 30
    texts = [_text(base), "unrelated words only here", _text(c), _text(b).upper(), _text(b)]
    assert find_duplicate_clusters(texts) == [[0, 2, 3, 4]]


def test_clusters_are_sorted() -> None:
    x, y = _tokens(25, "x"), _tokens(25, "y")
    texts = [_text(y), _text(x), "lonely text with words", _text(_replace(x, 24)), _text(y).upper()]
    assert find_duplicate_clusters(texts) == [[0, 4], [1, 3]]


def test_no_clusters_for_empty_or_unique_input() -> None:
    assert find_duplicate_clusters([]) == []
    assert find_duplicate_clusters(["alpha beta gamma", "delta epsilon zeta"]) == []


def test_empty_texts_are_exact_duplicates_of_each_other() -> None:
    assert find_duplicate_clusters(["", "  ", "text"]) == [[0, 1]]


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        find_duplicate_clusters(["a b c"], threshold=threshold)
    with pytest.raises(ValueError, match="threshold"):
        cross_split_duplicates(["a b c"], ["a b c"], threshold=threshold)
    with pytest.raises(ValueError, match="threshold"):
        leakage_check({"train": ["a b c"]}, threshold=threshold)


def test_thresholds_below_the_recall_bound_are_rejected_before_any_work() -> None:
    # 0.95**128 = 1.4e-3 and 0.95**256 = 2.0e-6: above the bound for both the dedupe path and the leakage check.
    with pytest.raises(ValueError, match=r"threshold 0\.05 .*num_perm=128"):
        find_duplicate_clusters(["card lost"], threshold=0.05)
    with pytest.raises(ValueError, match=r"threshold 0\.05 .*num_perm=128"):
        cross_split_duplicates([], [], threshold=0.05)
    with pytest.raises(ValueError, match=r"threshold 0\.05 .*num_perm=256"):
        leakage_check({}, threshold=0.05)
    # 0.94**256 = 1.3e-7: 256 permutations support 0.06, 128 do not.
    with pytest.raises(ValueError, match="num_perm=128"):
        find_duplicate_clusters(["a b c d"], threshold=0.06)
    assert find_duplicate_clusters(["a b c d", "x y z w"], threshold=0.06, num_perm=256) == []
    report = leakage_check({"train": ["alpha beta gamma delta"], "test": ["epsilon zeta eta theta"]}, threshold=0.06)
    assert report.ok
    assert (report.method["bands"], report.method["rows"]) == (256, 1)


def test_pairs_at_a_low_threshold_are_all_found() -> None:
    # 23 distinct words give 21 shingles; sharing the first 9 words shares 7 of them: J = 7 / (21 + 21 - 7) = 0.2.
    firsts, seconds = [], []
    for k in range(200):
        words_k = _tokens(23, f"a{k}x")
        firsts.append(_text(words_k))
        seconds.append(_text(words_k[:9] + _tokens(14, f"b{k}x")))
    assert jaccard(shingles(firsts[0]), shingles(seconds[0])) == 7 / 35 == 0.2
    assert lsh_bands(0.2, 128) == (128, 1)  # 0.8**128 = 4e-13
    assert cross_split_duplicates(seconds, firsts, threshold=0.2) == {k: ("near", k, 0.2) for k in range(200)}
    assert find_duplicate_clusters(firsts + seconds, threshold=0.2) == [[k, 200 + k] for k in range(200)]
    report = leakage_check({"train": firsts, "test": seconds}, threshold=0.2)
    assert report.pairs == [LeakagePair("train", k, "test", k, "near", 0.2) for k in range(200)]


def _reference_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def _brute_force_clusters(texts: list[str], threshold: float) -> list[list[int]]:
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    keys = [dedupe_key(t) for t in texts]
    sets = [shingles(t) for t in texts]
    for i, j in itertools.combinations(range(len(texts)), 2):
        if keys[i] == keys[j] or _reference_jaccard(sets[i], sets[j]) >= threshold:
            ri, rj = find(i), find(j)
            parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in range(len(texts)):
        groups.setdefault(find(i), []).append(i)
    return sorted((g for g in groups.values() if len(g) > 1), key=lambda g: g[0])


def _edit_variants(rng: random.Random, count: int) -> list[str]:
    """Bases of 20-60 words with random one- or two-word edits, so similarities straddle 0.9."""
    texts: list[str] = []
    for b in range(count):
        base = _tokens(rng.randint(20, 60), f"b{b}w")
        texts.append(_text(base))
        for _ in range(4):
            variant = list(base)
            for _ in range(rng.randint(1, 2)):
                position = rng.choice([0, 1, -2, -1, rng.randrange(len(variant))])
                if rng.random() < 0.5:
                    variant[position] = f"edit{rng.randrange(10**6)}"
                else:
                    del variant[position]
            texts.append(_text(variant))
    rng.shuffle(texts)
    return texts


def test_clusters_match_brute_force() -> None:
    texts = _edit_variants(random.Random(5), 60)
    expected = _brute_force_clusters(texts, 0.9)
    sets = [shingles(t) for t in texts]
    scores = [jaccard(sets[i], sets[j]) for i, j in itertools.combinations(range(len(texts)), 2)]
    assert sum(s >= 0.9 for s in scores) > 100  # plenty of pairs at or above the threshold ...
    assert sum(0.8 <= s < 0.9 for s in scores) > 100  # ... and just below it
    assert find_duplicate_clusters(texts) == expected
    assert find_duplicate_clusters(texts, num_perm=64, seed=3) == expected


def _chain_variants(rng: random.Random, count: int, per_base: int) -> list[str]:
    """Clusters grown by successive one-word edits, so members link to their base and through chains."""
    texts: list[str] = []
    for b in range(count):
        base = _tokens(rng.randint(15, 60), f"c{b}w")
        texts.append(_text(base))
        current = base
        for _ in range(per_base):
            variant = list(rng.choice([base, current]))
            position = rng.randrange(len(variant))
            if rng.random() < 0.5 or len(variant) <= 5:
                variant[position] = f"edit{rng.randrange(10**6)}"
            else:
                del variant[position]
            current = variant
            texts.append(_text(variant))
    rng.shuffle(texts)
    return texts


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_large_chained_clusters_match_brute_force(seed: int) -> None:
    texts = _chain_variants(random.Random(seed), 20, 25)
    for threshold in (0.9, 0.8):
        expected = _brute_force_clusters(texts, threshold)
        assert max(len(cluster) for cluster in expected) >= 8  # multi-member clusters exercise the grouped buckets
        assert find_duplicate_clusters(texts, threshold) == expected


def test_a_text_joins_a_cluster_through_any_candidate_member() -> None:
    base = _tokens(30)
    b = _replace(base, 29)
    c = _replace(b, 0, "other")  # J(b, c) = 27/29, but J(base, c) = 26/30: c reaches the cluster only through b
    for order in itertools.permutations([base, b, c]):
        assert find_duplicate_clusters([_text(tokens) for tokens in order]) == [[0, 1, 2]]


def test_a_text_bridges_two_clusters() -> None:
    base = _tokens(30)
    left = _text(_replace(base, 0, "left"))
    right = _text(_replace(base, 29, "right"))
    # J(left, right) = 26/30 = 0.867; the base is 27/29 from each.
    assert jaccard(shingles(left), shingles(right)) == 26 / 30
    texts = [left, right, right + "?", left.upper()]
    assert find_duplicate_clusters(texts) == [[0, 3], [1, 2]]
    assert find_duplicate_clusters([*texts, _text(base)]) == [[0, 1, 2, 3, 4]]
    assert find_duplicate_clusters([_text(base), *texts]) == [[0, 1, 2, 3, 4]]


def _one_template(n: int, position: int, length: int = 30) -> list[str]:
    """``n`` texts of one template that differ only in the word at ``position``."""
    base = _tokens(length, "w")
    return [_text(_replace(base, position, f"n{k}")) for k in range(n)]


def test_mutual_near_duplicates_cost_about_one_verification_each(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = _one_template(3000, 29)  # every pair: J = 27/29
    calls = 0
    real = dedupe.jaccard

    def counting(a: frozenset[str], b: frozenset[str]) -> float:
        nonlocal calls
        calls += 1
        return real(a, b)

    monkeypatch.setattr(dedupe, "jaccard", counting)
    assert find_duplicate_clusters(texts) == [list(range(3000))]
    assert calls < 2 * 3000  # verifying every candidate pair would take about 4.5 million calls


def test_one_template_just_below_the_threshold_forms_no_cluster() -> None:
    texts = _one_template(300, 15)  # every pair: J = 25/31 = 0.806
    assert find_duplicate_clusters(texts) == []
    assert find_duplicate_clusters(texts, threshold=0.8) == [list(range(300))]


@pytest.mark.timeout(120)
def test_thirteen_thousand_mutual_near_duplicates_cluster_quickly() -> None:
    texts = _one_template(13_000, 29)
    start = time.perf_counter()
    clusters = find_duplicate_clusters(texts)
    elapsed = time.perf_counter() - start
    assert clusters == [list(range(13_000))]
    assert elapsed < 10, f"clustering took {elapsed:.1f}s"


# --- majority vote --------------------------------------------------------------------------------------------------


def test_majority_vote_two_to_one_keeps_the_winner() -> None:
    assert majority_vote(["card_lost", "card_lost", "refund"]) == (True, "card_lost")
    assert majority_vote(["refund", "card_lost", "card_lost"]) == (True, "card_lost")


def test_majority_vote_without_a_strict_majority_drops() -> None:
    assert majority_vote(["card_lost", "refund"]) == (False, None)
    assert majority_vote(["a", "a", "b", "b"]) == (False, None)
    assert majority_vote(["a", "b", "c"]) == (False, None)  # a plurality is not a majority
    assert majority_vote(["a", "a", "b", "c"]) == (False, None)  # 2 of 4 is not more than half


def test_majority_vote_ignores_missing_values() -> None:
    assert majority_vote([]) == (False, None)
    assert majority_vote([None, None]) == (False, None)
    assert majority_vote([None, "a"]) == (True, "a")
    assert majority_vote(["a", None, None, "b", "a"]) == (True, "a")
    assert majority_vote(iter(["x", "x", "y"])) == (True, "x")


def test_majority_vote_compares_dicts_canonically_and_returns_the_original() -> None:
    first = {"total": 12.5, "currency": "EUR"}
    same = {"currency": "EUR", "total": 12.5}
    other = {"currency": "USD", "total": 12.5}
    won, winner = majority_vote([first, other, same])
    assert won is True
    assert winner is first
    assert majority_vote([first, other]) == (False, None)
    assert majority_vote([[1, 2], [1, 2], [2, 1]]) == (True, [1, 2])
    assert majority_vote(["1", 1, 1]) == (True, 1)  # a string and a number are different values


# --- cross-split duplicates -----------------------------------------------------------------------------------------


def test_cross_split_exact_duplicates() -> None:
    texts = ["Card LOST", "a fresh text with enough words", "how do I top up"]
    against = ["something else entirely here", "card lost", "How do I  top up", "card lost"]
    assert cross_split_duplicates(texts, against) == {0: ("exact", 1, 1.0), 2: ("exact", 2, 1.0)}


def test_cross_split_near_duplicates() -> None:
    base = _tokens(30)
    texts = [_text(base), _text(_tokens(10))]
    against = ["unrelated words in this one", _text(_replace(base, 29)), _text(_replace(_tokens(10), 9))]
    assert cross_split_duplicates(texts, against) == {0: ("near", 1, 27 / 29)}


def test_cross_split_prefers_exact_then_highest_jaccard_then_lowest_index() -> None:
    base = _tokens(60)
    middle = _text(_replace(base, 30))  # J = 55/61
    last = _text(_replace(base, 59))  # J = 57/59
    assert cross_split_duplicates([_text(base)], [middle, last]) == {0: ("near", 1, 57 / 59)}
    assert cross_split_duplicates([_text(base)], [last, _text(base).upper()]) == {0: ("exact", 1, 1.0)}
    # Same shingles, different keys: equal Jaccard, so the lower index wins.
    assert cross_split_duplicates([_text(base)], [middle, last + "!", last + "?"]) == {0: ("near", 1, 57 / 59)}


def test_cross_split_short_texts_match_exactly_only() -> None:
    texts = ["card lost", "card found", "top up"]
    against = ["Card Lost", "card found?", "top up failed again"]
    assert cross_split_duplicates(texts, against) == {0: ("exact", 0, 1.0)}
    assert cross_split_duplicates(["card lost today please"], ["card lost"], threshold=0.11) == {}


def test_cross_split_empty_inputs() -> None:
    assert cross_split_duplicates([], ["a b c"]) == {}
    assert cross_split_duplicates(["a b c"], []) == {}


def test_cross_split_result_is_ordered_by_index() -> None:
    base = _tokens(30)
    texts = [_text(base), "x y", "unique words right here", "X  Y"]
    result = cross_split_duplicates(texts, ["x y", _text(_replace(base, 0, "other"))])
    assert list(result) == [0, 1, 3]
    assert result[0] == ("near", 1, 27 / 29)


def test_cross_split_removal_then_leakage_check_reports_zero() -> None:
    rng = random.Random(11)
    texts = _edit_variants(rng, 80)
    train, valid, test = texts[:240], texts[240:320], texts[320:]
    assert not leakage_check({"train": train, "valid": valid, "test": test}).ok

    # Remove from train (against valid and test), then from validation (against test); test is never touched.
    drop_train = cross_split_duplicates(train, valid + test)
    train = [t for i, t in enumerate(train) if i not in drop_train]
    drop_valid = cross_split_duplicates(valid, test)
    valid = [t for i, t in enumerate(valid) if i not in drop_valid]
    assert drop_train and drop_valid

    report = leakage_check({"train": train, "valid": valid, "test": test})
    assert report.ok, report.to_dict()
    assert report.sizes == {"train": len(train), "valid": len(valid), "test": 80}


# --- leakage check --------------------------------------------------------------------------------------------------


def _clean_splits() -> dict[str, list[str]]:
    return {
        "train": [_text(_tokens(25, "a")), "card lost", _text(_tokens(12, "b"))],
        "valid": [_text(_tokens(25, "c")), "card found"],
        "test": [_text(_tokens(25, "d")), "top up", _text(_replace(_tokens(10, "b"), 9))],
    }


def test_leakage_check_is_ok_on_clean_splits() -> None:
    report = leakage_check(_clean_splits())
    assert report.ok
    assert report.pairs == []
    assert report.to_dict() == {
        "ok": True,
        "threshold": 0.9,
        "sizes": {"train": 3, "valid": 2, "test": 3},
        "exact": 0,
        "near": 0,
        "by_split_pair": {"train/valid": 0, "train/test": 0, "valid/test": 0},
        "method": report.method,
        "pairs": [],
    }
    assert report.method["num_perm"] == LEAK_NUM_PERM
    assert report.method["seed"] != dedupe.DEFAULT_SEED


def test_leakage_check_finds_planted_exact_and_near_leaks() -> None:
    splits = _clean_splits()
    splits["test"].append("How do I reset my PIN?")
    splits["train"].append("how do i  reset my pin?")  # exact after normalisation
    splits["valid"].append(_text(_replace(_tokens(25, "a"), 24)))  # near: J = 22/24 with train[0]
    report = leakage_check(splits)
    assert not report.ok
    assert report.pairs == [
        LeakagePair("train", 0, "valid", 2, "near", 22 / 24),
        LeakagePair("train", 3, "test", 3, "exact", 1.0),
    ]
    assert (report.exact, report.near) == (1, 1)
    assert report.by_split_pair() == {"train/valid": 1, "train/test": 1, "valid/test": 0}
    data = json.loads(json.dumps(report.to_dict()))
    assert data["ok"] is False
    assert data["pairs"][1] == {
        "split_a": "train",
        "index_a": 3,
        "split_b": "test",
        "index_b": 3,
        "kind": "exact",
        "jaccard": 1.0,
    }


def test_leakage_check_short_texts_exact_only() -> None:
    report = leakage_check({"train": ["card lost", "top up"], "test": ["Card  LOST", "top up?", "top up failed now"]})
    assert report.pairs == [LeakagePair("train", 0, "test", 0, "exact", 1.0)]


def test_leakage_check_covers_every_pair_of_splits_and_ignores_within_split_duplicates() -> None:
    shared_ab = _text(_tokens(25, "p"))
    shared_bc = _text(_tokens(20, "q"))
    shared_ac = _text(_tokens(20, "r"))
    splits = {
        "a": [shared_ab, shared_ac, "same here", "same here"],
        "b": [shared_bc, _text(_replace(_tokens(25, "p"), 0, "other"))],
        "c": [shared_ac, shared_bc.upper()],
    }
    report = leakage_check(splits)
    assert report.pairs == [
        LeakagePair("a", 0, "b", 1, "near", 22 / 24),
        LeakagePair("a", 1, "c", 0, "exact", 1.0),
        LeakagePair("b", 0, "c", 1, "exact", 1.0),
    ]
    assert report.by_split_pair() == {"a/b": 1, "a/c": 1, "b/c": 1}
    # Split order decides which side is ``split_a``.
    reordered = leakage_check({"c": splits["c"], "a": splits["a"]})
    assert reordered.pairs == [LeakagePair("c", 0, "a", 1, "exact", 1.0)]


def test_leakage_check_threshold_near_boundary() -> None:
    twelve = _tokens(12)
    splits = {"train": [_text(twelve)], "test": [_text(twelve[:11])]}  # J = 0.9 exactly
    assert leakage_check(splits).pairs == [LeakagePair("train", 0, "test", 0, "near", 0.9)]
    assert leakage_check(splits, threshold=0.95).ok


def test_leakage_check_does_not_use_the_dedupe_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("leakage check must not reuse the dedupe path")

    dedupe_path = (
        "find_duplicate_clusters",
        "cross_split_duplicates",
        "lsh_bands",
        "_check_threshold",
        "_sketches",
        "_encoded",
        "_band_keys",
        "_unique",
        "_UnionFind",
    )
    for name in dedupe_path:
        monkeypatch.setattr(dedupe, name, fail)
    splits = _clean_splits()
    splits["valid"].append(_text(_replace(_tokens(25, "a"), 0, "other")))
    splits["test"].append("CARD LOST")
    report = dedupe.leakage_check(splits)
    assert report.pairs == [
        LeakagePair("train", 0, "valid", 2, "near", 22 / 24),
        LeakagePair("train", 1, "test", 3, "exact", 1.0),
    ]
    assert (report.method["bands"], report.method["rows"]) == (32, 8)


@pytest.mark.parametrize("threshold", [0.2, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0])
def test_leakage_bands_agree_with_the_dedupe_path_computation(threshold: float) -> None:
    # Two separate computations of the same rule must agree: the widest band that meets the miss bound.
    bands, rows = dedupe._leak_bands(threshold)
    assert (bands, rows) == lsh_bands(threshold, LEAK_NUM_PERM)
    assert (1 - threshold**rows) ** bands <= MAX_MISS_RATE
    assert bands == LEAK_NUM_PERM // rows


def test_leakage_hash_is_a_32_bit_blake2b() -> None:
    values = {dedupe._leak_hash(shingle.encode()) for shingle in shingles(_text(_tokens(200)))}
    assert len(values) == 198
    assert all(0 <= value < 2**32 for value in values)
    assert dedupe._leak_hash(b"card lost today") == dedupe._leak_hash(b"card lost today")


def test_leakage_check_at_threshold_one_matches_identical_shingle_sets() -> None:
    assert dedupe._leak_bands(1.0) == (2, 128)  # MinHashLSH needs at least two bands
    splits = {"train": ["Card lost, today", _text(_tokens(12))], "test": ["card lost today!", _text(_tokens(12)[:11])]}
    assert leakage_check(splits, threshold=1.0).pairs == [LeakagePair("train", 0, "test", 0, "near", 1.0)]
    assert find_duplicate_clusters([*splits["train"], *splits["test"]], threshold=1.0) == [[0, 2]]


class _LegacyMinHash:
    """Stands in for datasketch before 2.0, whose ``MinHash`` has no ``scheme`` argument."""

    calls = 0

    @classmethod
    def generator(cls, data: object, **kwargs: object) -> object:
        if "scheme" in kwargs:
            raise TypeError("MinHash.__init__() got an unexpected keyword argument 'scheme'")
        cls.calls += 1
        return MinHash.generator(data, **kwargs)


def test_every_path_works_without_the_permutation_scheme_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dedupe, "MinHash", _LegacyMinHash)
    monkeypatch.setattr(_LegacyMinHash, "calls", 0)
    base = _tokens(30)
    near = _text(_replace(base, 29))
    assert find_duplicate_clusters([_text(base), near]) == [[0, 1]]
    assert cross_split_duplicates([near], [_text(base)]) == {0: ("near", 0, 27 / 29)}
    report = leakage_check({"train": [_text(base)], "test": [near]})
    assert report.pairs == [LeakagePair("train", 0, "test", 0, "near", 27 / 29)]
    assert _LegacyMinHash.calls == 5  # sketch runs: one for clustering, two for cross-split, one per leakage split


class _CountingLSH(MinHashLSH):
    """``MinHashLSH`` that counts the candidates its queries return."""

    candidates = 0

    def query(self, minhash: MinHash) -> list[object]:
        found = super().query(minhash)
        type(self).candidates += len(found)
        return list(found)


def test_leakage_check_never_proposes_pairs_within_one_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dedupe, "MinHashLSH", _CountingLSH)
    monkeypatch.setattr(_CountingLSH, "candidates", 0)
    train = _one_template(3000, 29)  # 3000 mutual near-duplicates, all inside train
    test = [_text(_tokens(12, f"q{k}x")) for k in range(50)]
    assert leakage_check({"train": train, "test": test}).ok
    assert _CountingLSH.candidates == 0  # one index over both splits would propose about 4.5 million pairs

    report = leakage_check({"train": train, "test": [*test, train[7].upper()]})
    assert (report.exact, report.near) == (1, 2999)
    assert report.pairs[6:9] == [
        LeakagePair("train", 6, "test", 50, "near", 27 / 29),
        LeakagePair("train", 7, "test", 50, "exact", 1.0),
        LeakagePair("train", 8, "test", 50, "near", 27 / 29),
    ]


def test_leakage_check_matches_brute_force() -> None:
    texts = _edit_variants(random.Random(23), 60)
    splits = {"train": texts[:150], "valid": texts[150:220], "test": texts[220:]}
    expected = []
    items = [(name, i, t) for name, split in splits.items() for i, t in enumerate(split)]
    for (sa, ia, ta), (sb, ib, tb) in itertools.combinations(items, 2):
        if sa == sb:
            continue
        if dedupe_key(ta) == dedupe_key(tb):
            expected.append(LeakagePair(sa, ia, sb, ib, "exact", 1.0))
        elif (score := jaccard(shingles(ta), shingles(tb))) >= 0.9:
            expected.append(LeakagePair(sa, ia, sb, ib, "near", score))
    assert len(expected) > 10
    assert leakage_check(splits).pairs == expected


# --- determinism and performance ------------------------------------------------------------------------------------

_DETERMINISM_SCRIPT = """
import json, random
from taskdistill.curate.dedupe import cross_split_duplicates, find_duplicate_clusters, leakage_check
rng = random.Random(3)
vocab = [f"v{i}" for i in range(300)]
texts = [" ".join(rng.choice(vocab) for _ in range(rng.randint(10, 30))) for _ in range(400)]
texts += [t.rsplit(" ", 1)[0] for t in texts[:100]] + [t.upper() for t in texts[100:150]]
print(json.dumps({
    "clusters": find_duplicate_clusters(texts),
    "cross": cross_split_duplicates(texts[400:], texts[:400]),
    "leak": leakage_check({"train": texts[:300], "test": texts[300:]}).to_dict(),
}, sort_keys=True))
"""


def test_results_are_deterministic_across_processes() -> None:
    outputs = []
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        proc = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_SCRIPT], capture_output=True, text=True, env=env, check=True
        )
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1]
    data = json.loads(outputs[0])
    assert len(data["clusters"]) >= 100
    assert not data["leak"]["ok"]


def test_results_are_deterministic_within_a_process() -> None:
    texts = _edit_variants(random.Random(7), 30)
    splits = {"train": texts[:100], "test": texts[100:]}
    assert find_duplicate_clusters(texts) == find_duplicate_clusters(list(texts))
    assert leakage_check(splits).to_dict() == leakage_check(splits).to_dict()
    assert cross_split_duplicates(texts[:100], texts[100:]) == cross_split_duplicates(texts[:100], texts[100:])


@pytest.mark.timeout(120)
def test_five_thousand_short_texts_cluster_quickly() -> None:
    rng = random.Random(0)
    vocab = [f"v{i}" for i in range(400)] + ["card", "my", "the", "to", "top", "up", "transfer", "pending", "fee"]
    texts = [" ".join(rng.choice(vocab) for _ in range(rng.randint(10, 30))) for _ in range(5000)]
    planted = []
    for k in range(0, 1000, 5):
        texts.append(_text(_replace(texts[k].split(), -1, "planted")))
        planted.append([k, len(texts) - 1])
    for k in range(1000, 1100):
        texts.append(texts[k].upper())
        planted.append([k, len(texts) - 1])
    expected = sorted(
        (pair for pair in planted if jaccard(shingles(texts[pair[0]]), shingles(texts[pair[1]])) >= 0.9),
        key=lambda pair: pair[0],
    )

    start = time.perf_counter()
    clusters = find_duplicate_clusters(texts)
    elapsed = time.perf_counter() - start
    assert clusters == expected
    assert elapsed < 30, f"clustering took {elapsed:.1f}s"

    start = time.perf_counter()
    report = leakage_check({"train": texts[:4000], "valid": texts[4000:5000], "test": texts[5000:]})
    elapsed = time.perf_counter() - start
    assert elapsed < 30, f"leakage check took {elapsed:.1f}s"
    assert len(report.pairs) == len(expected)  # every planted copy sits in test, every original before it
