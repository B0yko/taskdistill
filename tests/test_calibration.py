from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from taskdistill.evaluate.calibration import (
    IsotonicCalibrator,
    auroc,
    bin_index,
    brier,
    calibration_report,
    ece,
    fit_isotonic,
    reliability_bins,
)
from taskdistill.evaluate.plots import plot_loss, plot_reliability, plot_threshold_curve, save_png
from taskdistill.evaluate.splits import EvalRecord, TestSplit, ValidationSplit

CONF = [0.1, 0.45, 0.82, 0.85]
CORRECT = [False, True, True, False]


def test_ece_four_points_by_hand() -> None:
    # 15 bins of width 1/15: 0.1 -> bin 1, 0.45 -> bin 6, 0.82 and 0.85 -> bin 12.
    # ECE = 1/4 * |0 - 0.1| + 1/4 * |1 - 0.45| + 2/4 * |0.5 - 0.835| = 0.025 + 0.1375 + 0.1675 = 0.33
    assert bin_index(np.asarray(CONF), 15).tolist() == [1, 6, 12, 12]
    assert ece(CONF, CORRECT) == pytest.approx(0.33)


def test_ece_two_bins_and_the_closed_last_bin() -> None:
    # bin 0: {0.2, 0.3}, accuracy 0.5, mean 0.25; bin 1: {0.7, 1.0}, accuracy 1, mean 0.85.
    # ECE = 2/4 * 0.25 + 2/4 * 0.15 = 0.2
    assert ece([0.2, 0.3, 0.7, 1.0], [False, True, True, True], n_bins=2) == pytest.approx(0.2)
    assert bin_index(np.asarray([0.0, 1.0]), 15).tolist() == [0, 14]


def test_ece_perfect_and_worst() -> None:
    assert ece([1.0, 1.0, 0.0], [True, True, False]) == 0.0
    assert ece([1.0, 0.0], [False, True]) == 1.0


def test_brier_by_hand() -> None:
    # (0.1^2 + 0.55^2 + 0.18^2 + 0.85^2) / 4 = (0.01 + 0.3025 + 0.0324 + 0.7225) / 4
    assert brier(CONF, CORRECT) == pytest.approx(1.0674 / 4)


def test_auroc_by_hand_and_with_ties() -> None:
    # positives 0.35, 0.8; negatives 0.1, 0.4: pairs ranked correctly 3 of 4.
    assert auroc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == pytest.approx(0.75)
    assert auroc([0.5, 0.5], [False, True]) == pytest.approx(0.5)


def test_auroc_is_none_when_a_class_is_empty() -> None:
    assert auroc([0.2, 0.9, 0.5], [True, True, True]) is None
    assert auroc([0.2, 0.9, 0.5], [False, False, False]) is None
    assert auroc([], []) is None


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ece([1.5], [True])
    with pytest.raises(ValueError, match="NaN"):
        brier([math.nan], [True])
    with pytest.raises(ValueError, match="differ in length"):
        ece([0.5, 0.5], [True])
    with pytest.raises(ValueError, match="empty"):
        ece([], [])
    with pytest.raises(ValueError, match="empty"):
        brier([], [])


def test_reliability_bins() -> None:
    bins = reliability_bins(CONF, CORRECT, n_bins=15)
    assert len(bins) == 15
    assert bins[12] == pytest.approx({"lo": 12 / 15, "hi": 13 / 15, "n": 2, "mean_confidence": 0.835, "accuracy": 0.5})
    assert bins[0] == {"lo": 0.0, "hi": 1 / 15, "n": 0, "mean_confidence": None, "accuracy": None}
    assert sum(b["n"] for b in bins) == 4


def test_calibration_report() -> None:
    report = calibration_report(CONF, CORRECT)
    assert report["n"] == 4 and report["n_correct"] == 2
    assert report["ece"] == pytest.approx(0.33)
    assert report["brier"] == pytest.approx(1.0674 / 4)
    assert report["auroc"] == pytest.approx(0.5)  # positives 0.45, 0.82 vs negatives 0.1, 0.85: 2 of 4
    assert report["accuracy"] == 0.5
    assert report["mean_confidence"] == pytest.approx(0.555)
    assert len(report["bins"]) == 15
    json.dumps(report, allow_nan=False)
    empty = calibration_report([], [])
    assert empty["n"] == 0 and empty["ece"] is None and empty["brier"] is None and empty["auroc"] is None


def _val(conf: list[float | None], split: type[ValidationSplit] | type[TestSplit] = ValidationSplit) -> ValidationSplit:
    records = [EvalRecord(id=str(i), input=f"q{i}", pred="a", confidence=c) for i, c in enumerate(conf)]
    return split(records, "classification")  # type: ignore[return-value]


def test_fit_isotonic_by_hand() -> None:
    # Pool-adjacent-violators on correctness [0, 1, 0, 1] gives [0, 0.5, 0.5, 1].
    calibrator = fit_isotonic(_val([0.1, 0.2, 0.3, 0.4]), [False, True, False, True])
    assert calibrator.n_fit == 4
    assert calibrator.apply([0.25]).tolist() == pytest.approx([0.5])
    assert calibrator.apply([0.15]).tolist() == pytest.approx([0.25])  # linear between 0.1 -> 0 and 0.2 -> 0.5
    assert calibrator.apply([0.0, 0.05, 0.9, 1.0]).tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])  # clipped


def test_fit_isotonic_matches_sklearn_and_round_trips() -> None:
    rng = np.random.default_rng(0)
    conf = rng.random(300)
    correct = rng.random(300) < conf
    split = _val([float(c) for c in conf])
    calibrator = fit_isotonic(split, correct)
    reference = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(conf, correct.astype(float))
    grid = np.linspace(-0.1, 1.1, 101)
    np.testing.assert_allclose(calibrator.apply(grid), reference.predict(grid))
    out = calibrator.apply(grid)
    assert np.all(np.diff(out) >= 0) and out.min() >= 0.0 and out.max() <= 1.0
    restored = IsotonicCalibrator.from_dict(json.loads(json.dumps(calibrator.to_dict())))
    np.testing.assert_allclose(restored.apply(grid), out)


def test_fit_isotonic_accepts_a_correctness_function() -> None:
    split = _val([0.1, 0.2, 0.3, 0.4])
    by_function = fit_isotonic(split, lambda r: r.id in {"1", "3"})
    by_flags = fit_isotonic(split, [False, True, False, True])
    assert by_function == by_flags


def test_fit_isotonic_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="no confidence"):
        fit_isotonic(_val([0.1, None]), [True, False])
    with pytest.raises(ValueError, match="entries"):
        fit_isotonic(_val([0.1, 0.2]), [True])
    with pytest.raises(ValueError, match="empty"):
        fit_isotonic(_val([]), [])


def test_fit_isotonic_refuses_the_test_split() -> None:
    with pytest.raises(TypeError, match="ValidationSplit"):
        fit_isotonic(_val([0.1, 0.2], TestSplit), [True, False])


# --- plots ------------------------------------------------------------------------------------------------------


def _png_chunks(path: Path) -> list[tuple[bytes, bytes]]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, pos = [], 8
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(kind + body) & 0xFFFFFFFF
        chunks.append((kind, body))
        pos += 12 + length
    return chunks


def _text_keywords(path: Path) -> list[bytes]:
    return [body.split(b"\x00", 1)[0] for kind, body in _png_chunks(path) if kind in (b"tEXt", b"iTXt", b"zTXt")]


def test_plots_write_pngs_without_software_chunk(tmp_path: Path) -> None:
    curve = [(0.2, 0.0, 0.9), (0.5, 0.3, 0.96), (0.8, 0.6, 0.99), (math.inf, 1.0, 1.0)]
    paths = [
        plot_threshold_curve(curve, (0.5, 0.3, 0.96), tmp_path / "curve" / "threshold.png", "banking77", target=0.95),
        plot_threshold_curve(curve, (math.inf, 1.0, 1.0), tmp_path / "always.png", "always escalate"),
        plot_reliability(CONF, CORRECT, tmp_path / "reliability.png", calibrated=[0.0, 0.5, 0.5, 0.5]),
        plot_loss([(10, 2.0), (20, 1.5), (30, 1.2)], [(10, 2.1), (30, 1.4)], tmp_path / "loss.png", best_iteration=30),
        plot_loss([], [], tmp_path / "empty_loss.png"),
    ]
    for path in paths:
        assert path.exists()
        kinds = [kind for kind, _ in _png_chunks(path)]
        assert kinds[0] == b"IHDR" and kinds[-1] == b"IEND"
        assert b"Software" not in _text_keywords(path)


def test_save_png_strips_software_from_any_figure(tmp_path: Path) -> None:
    from taskdistill.evaluate.plots import new_figure

    fig = new_figure()
    fig.add_subplot().plot([0, 1], [0, 1])
    path = save_png(fig, tmp_path / "nested" / "plain.png")
    assert b"Software" not in _text_keywords(path)
    # The default matplotlib save would have written it: the check above is meaningful.
    default = tmp_path / "default.png"
    fig.savefig(default, format="png")
    assert b"Software" in _text_keywords(default)
