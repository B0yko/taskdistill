"""Tests for scripts/sync_readme.py: the renderers, --check behaviour and determinism.

Fixtures live under tests/fixtures/sync_readme/: a trimmed copy of the sample reports (banking77/report.json and
invoices/report.json with the large label-distribution table stripped out, plus training.json, spend.json,
demo_timing.json, downloads.json, ablations/lr_schedule.json and the bakeoff files) and a small README.md with the
same three sync blocks as the real one.
"""

from __future__ import annotations

import importlib.util
import shutil
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sync_readme"
REPORTS = FIXTURES / "reports"
README_FIXTURE = FIXTURES / "README.md"


def _load_script(name: str, *, reports: Path, readme: Path, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import a ``scripts/*.py`` file as a fresh module, with the env vars it reads already set.

    Not added to ``sys.path`` and not cached in ``sys.modules``, so each call gets its own module object bound to
    the ``reports``/``readme`` paths given -- the same pattern as ``tests/test_fixes_integration.py``.
    """
    monkeypatch.setenv("SYNC_README_REPORTS", str(reports))
    monkeypatch.setenv("SYNC_README_PATH", str(readme))
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def readme_copy(tmp_path: Path) -> Path:
    """A writable copy of the fixture README, so tests never mutate the checked-in file."""
    target = tmp_path / "README.md"
    shutil.copy(README_FIXTURE, target)
    return target


@pytest.fixture
def sync(monkeypatch: pytest.MonkeyPatch, readme_copy: Path) -> types.ModuleType:
    """``sync_readme`` bound to the full fixture reports directory and a scratch README copy."""
    return _load_script("sync_readme.py", reports=REPORTS, readme=readme_copy, monkeypatch=monkeypatch)


# -- formatting helpers ----------------------------------------------------------------------------------


def test_pct_formats_one_decimal(sync: types.ModuleType) -> None:
    assert sync.pct(0.7580487804878049) == "75.8%"
    assert sync.pct(None) == "—"


def test_pts_signs_and_rounds(sync: types.ModuleType) -> None:
    assert sync.pts(-0.024127812194326692) == "-2.4 pts"
    assert sync.pts(0.003252032520325243) == "+0.3 pts"


def test_usd_three_significant_digits_no_exponent(sync: types.ModuleType) -> None:
    assert sync.usd(0.0059317658536585365) == "$0.00593"
    assert sync.usd(7.833719092634505e-05) == "$0.0000783"
    assert sync.usd(0) == "$0"
    assert sync.usd(None) == "—"


def test_ms_switches_precision_at_100(sync: types.ModuleType) -> None:
    assert sync.ms(594.2275839624926) == "594"
    assert sync.ms(44.49875000864267) == "44.5"
    assert sync.ms(None) == "—"


def test_integer_uses_thousands_separator(sync: types.ModuleType) -> None:
    assert sync.integer(3075) == "3,075"
    assert sync.integer(None) == "—"


def test_hardware_text_matches_report_convention(sync: types.ModuleType) -> None:
    hardware = {"model": "Mac17,4", "cpu": "Apple M5", "memory_gb": 24.0, "os": "macOS 26.6.2"}
    assert sync.hardware_text(hardware) == "Mac17,4, Apple M5, 24 GB, macOS 26.6.2"


# -- teacher-latency (inline block) -----------------------------------------------------------------------


def test_teacher_latency_is_a_single_line_with_the_recorded_numbers(sync: types.ModuleType) -> None:
    body = sync.RENDERERS["teacher-latency"]()
    assert "\n" not in body
    assert "594" in body
    assert "1,242" in body
    assert "2026-09-26" in body


# -- quickstart-timing -------------------------------------------------------------------------------------


def test_quickstart_timing_states_under_five_minutes(sync: types.ModuleType) -> None:
    body = sync.RENDERERS["quickstart-timing"]()
    assert "under five minutes" in body
    assert "Mac17,4" in body
    assert "banking77" in body
    assert "invoices" in body


def test_quickstart_timing_requires_demo_timing_json(
    tmp_path: Path, readme_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_reports = tmp_path / "empty_reports"
    empty_reports.mkdir()
    module = _load_script("sync_readme.py", reports=empty_reports, readme=readme_copy, monkeypatch=monkeypatch)
    with pytest.raises(SystemExit, match=r"demo_timing\.json"):
        module.RENDERERS["quickstart-timing"]()


# -- results: content spot checks -------------------------------------------------------------------------


def test_results_has_every_required_subsection(sync: types.ModuleType) -> None:
    body = sync.RENDERERS["results"]()
    for heading in (
        "## Results",
        "### Banking77 (77 intents)",
        "### Invoices (8-field JSON extraction)",
        "### Cost and latency",
        "### Live bench cross-check",
        "### Training on the M5",
        "### Calibration",
        "### Choosing the base model",
        "### What didn't work",
        "### Spend and downloads",
    ):
        assert heading in body, f"missing section: {heading}"


def test_results_quality_numbers_match_the_report_json(sync: types.ModuleType) -> None:
    body = sync.RENDERERS["results"]()
    # Banking77: teacher accuracy, cascade agreement at the chosen threshold.
    assert "75.8% [74.2, 77.4]" in body
    assert "97.6% [97.0, 98.1]" in body
    # Invoices: the test set is 6 never-seen layouts.
    assert "6 layouts never seen in training" in body
    # Break-even volumes (recorded and list price) for both tasks.
    assert "18,565 requests" in body
    assert "1,858 requests" in body
    assert "1,241 requests" in body
    assert "969 requests" in body


def test_results_references_relative_image_paths_only(sync: types.ModuleType) -> None:
    body = sync.RENDERERS["results"]()
    assert "reports/banking77/threshold_curve.png" in body
    assert "reports/invoices/threshold_curve.png" in body
    assert "reports/banking77/reliability_test.png" in body


def test_results_contains_no_absolute_paths(sync: types.ModuleType) -> None:
    body = sync.RENDERERS["results"]()
    assert str(REPORTS) not in body
    assert str(FIXTURES) not in body


def test_results_what_didnt_work_is_data_driven(sync: types.ModuleType) -> None:
    body = sync.RENDERERS["results"]()
    # The learning-rate ablation shows one seed diverging (seed 14's constant-schedule loss is an outlier).
    assert "one seed (seed 14) diverged" in body
    # The 1.5B student's gain is below the 1-point rule, at just over 2x the p95 latency.
    assert "gained +0.97 pts" in body
    assert "below the 1.00% minimum" in body
    # The gold-label ceiling names the Banking77 label-noise citation.
    assert "Ying and Thomas, 2022" in body
    # Prompt variants: the spec prompt was kept despite a higher-accuracy variant existing.
    assert "spec prompt was kept" in body


def test_results_tolerates_missing_optional_files(
    tmp_path: Path, readme_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the per-task report.json files are required; training/spend/downloads/ablation/bakeoff are optional."""
    partial = tmp_path / "partial_reports"
    (partial / "banking77").mkdir(parents=True)
    (partial / "invoices").mkdir(parents=True)
    shutil.copy(REPORTS / "banking77/report.json", partial / "banking77/report.json")
    shutil.copy(REPORTS / "invoices/report.json", partial / "invoices/report.json")

    module = _load_script("sync_readme.py", reports=partial, readme=readme_copy, monkeypatch=monkeypatch)
    body = module.RENDERERS["results"]()

    assert "Not available yet" in body
    assert "reports/training.json" in body
    assert "reports/spend.json" in body
    assert "reports/downloads.json" in body
    # Bullets that need the ablation/bake-off files are skipped, but the ones derived from the report.json
    # files themselves (already present) still render.
    assert "Learning-rate schedule" not in body
    assert "Teacher prompt variants" not in body
    assert "The bigger Banking77 student" in body
    assert "Ying and Thomas, 2022" in body


def test_results_requires_the_per_task_report_json(
    tmp_path: Path, readme_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_reports = tmp_path / "empty_reports"
    empty_reports.mkdir()
    module = _load_script("sync_readme.py", reports=empty_reports, readme=readme_copy, monkeypatch=monkeypatch)
    with pytest.raises(SystemExit, match=r"banking77/report\.json"):
        module.RENDERERS["results"]()


def test_results_is_deterministic(sync: types.ModuleType) -> None:
    assert sync.RENDERERS["results"]() == sync.RENDERERS["results"]()


# -- render(): block substitution leaves surrounding prose untouched --------------------------------------


def test_render_preserves_prose_outside_sync_blocks(sync: types.ModuleType) -> None:
    original = README_FIXTURE.read_text(encoding="utf-8")
    updated = sync.render(original)
    assert "Some quickstart prose that must survive unchanged by sync_readme.py." in updated
    assert "Prose after the results block that must also survive unchanged." in updated
    assert "<!-- sync:teacher-latency -->" in updated
    assert "<!-- /sync:quickstart-timing -->" in updated
    assert "<!-- /sync:results -->" in updated
    # The heading is emitted by the results renderer itself, so it must appear exactly once.
    assert updated.count("## Results") == 1


def test_render_is_deterministic(sync: types.ModuleType) -> None:
    original = README_FIXTURE.read_text(encoding="utf-8")
    assert sync.render(original) == sync.render(original)


def test_render_raises_for_an_unknown_block_name(sync: types.ModuleType) -> None:
    with pytest.raises(SystemExit, match="no renderer for block 'made-up'"):
        sync.render("<!-- sync:made-up -->x<!-- /sync:made-up -->")


# -- main(): --check exit codes, and a real write --------------------------------------------------------


def test_main_check_fails_on_a_stale_readme(sync: types.ModuleType, readme_copy: Path) -> None:
    assert sync.main(["--check"]) == 1
    text = readme_copy.read_text(encoding="utf-8")
    assert "<!-- sync:teacher-latency -->…<!-- /sync -->" in text  # untouched: --check must not write


def test_main_writes_then_check_passes(sync: types.ModuleType, readme_copy: Path) -> None:
    assert sync.main([]) == 0
    updated = readme_copy.read_text(encoding="utf-8")
    assert "594" in updated
    assert "## Results" in updated
    assert sync.main(["--check"]) == 0


def test_main_second_write_is_a_no_op(sync: types.ModuleType, readme_copy: Path) -> None:
    assert sync.main([]) == 0
    first = readme_copy.read_text(encoding="utf-8")
    assert sync.main([]) == 0
    assert readme_copy.read_text(encoding="utf-8") == first


# -- load()/load_optional(): required vs optional report files --------------------------------------------


def test_load_raises_a_clear_error_for_a_missing_required_file(
    tmp_path: Path, readme_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_reports = tmp_path / "empty_reports"
    empty_reports.mkdir()
    module = _load_script("sync_readme.py", reports=empty_reports, readme=readme_copy, monkeypatch=monkeypatch)
    with pytest.raises(SystemExit, match=r"missing banking77/report\.json"):
        module.load("banking77/report.json")


def test_load_optional_returns_none_for_a_missing_file(
    tmp_path: Path, readme_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_reports = tmp_path / "empty_reports"
    empty_reports.mkdir()
    module = _load_script("sync_readme.py", reports=empty_reports, readme=readme_copy, monkeypatch=monkeypatch)
    assert module.load_optional("training.json") is None


def test_load_optional_reads_an_existing_file(sync: types.ModuleType) -> None:
    data = sync.load_optional("training.json")
    assert data is not None
    assert data["table"][0]["task"] == "banking77"
