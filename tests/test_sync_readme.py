"""Tests for scripts/sync_readme.py: the renderers, --check behaviour and determinism.

Fixtures live under tests/fixtures/sync_readme/: a trimmed copy of the sample reports (banking77/report.json and
invoices/report.json with the large label-distribution table stripped out, plus training.json, spend.json,
demo_timing.json, downloads.json, ablations/lr_schedule.json and the bakeoff files) and a small README.md with the
same three sync blocks as the real one. invoices/report.json carries two zero-shot rows (0.5B and 1.5B) and an
operating point that missed its target on test, matching the shape of the real committed reports.

tests/fixtures/sync_readme/optional/ holds fixtures for reports that are optional and are not part of the shared
reports/ directory above -- reproduction.json and report_mac_studio.json -- so that most tests exercise the
"file does not exist yet" behaviour (the default for these two in the real reports/ directory) and only the tests
that need the "file exists" behaviour build a merged copy with :func:`_reports_plus`.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sync_readme"
REPORTS = FIXTURES / "reports"
OPTIONAL = FIXTURES / "optional"
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


def _reports_plus(tmp_path: Path, *relative: str) -> Path:
    """A writable copy of the full fixture reports directory, with the named fixtures from ``optional/`` (paths
    relative to it, e.g. ``"banking77/report_mac_studio.json"``) copied in on top.

    Keeps ``reproduction.json`` and ``report_mac_studio.json`` out of the shared ``REPORTS`` fixture, so every
    other test using the ``sync`` fixture keeps exercising the "not produced yet" behaviour that is also the
    real repository's current state, while a test that needs the "file exists" behaviour builds this instead.
    """
    target = tmp_path / "reports"
    shutil.copytree(REPORTS, target)
    for rel in relative:
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(OPTIONAL / rel, destination)
    return target


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
    assert "Apple M5, 24 GB" in body
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


def test_results_renders_every_zero_shot_row_smallest_base_first(sync: types.ModuleType) -> None:
    body = sync.RENDERERS["results"]()
    # Banking77 has one zero-shot row (classification): the prompt note names the label list.
    assert "base 0.5B zero-shot (labels in the prompt)" in body
    # Invoices has two (extraction): the note is on the first (smallest) only, naming the JSON Schema.
    assert "base 0.5B zero-shot (schema in the prompt)" in body
    assert "base 1.5B zero-shot" in body
    assert "base 1.5B zero-shot (schema in the prompt)" not in body
    # The provenance paragraph says so too, since invoices has more than the one zero-shot row per task.
    assert "plus one extra zero-shot evaluation (the invoices table also shows the 1.5B zero-shot base)" in body


def test_provenance_paragraph_uses_date_only_and_scoring_counts(sync: types.ModuleType) -> None:
    body = sync.RENDERERS["results"]()
    # The report's "date" is a full ISO timestamp; the paragraph shows only the YYYY-MM-DD prefix.
    assert "on 2026-09-26 on Mac17,4" in body
    assert "2026-09-26T" not in body
    # test_access.count is 8 for Banking77 and 1 for invoices in the fixtures (singular "time").
    assert "scored 8 times (Banking77) and 1 time (invoices)" in body


def test_operating_point_outcome_is_plain_about_a_miss(sync: types.ModuleType) -> None:
    body = sync.RENDERERS["results"]()
    # Banking77 holds the target on test: said plainly, with the numbers that make it true.
    assert "Target held on test: yes — Agreement against the teacher was 97.6% on test" in body
    # Invoices does not (the fixture's operating_point.target_met_on_test is false): never worded as a pass.
    assert "Target held on test: no — Agreement against the teacher met the 97.0% target on validation" in body
    assert "Target held on test: yes — Agreement against the teacher was 90.0%" not in body


def test_what_didnt_work_flags_the_invoices_threshold_that_did_not_transfer(sync: types.ModuleType) -> None:
    body = sync.RENDERERS["results"]()
    # Only invoices' target_met_on_test is false in the fixtures, so only invoices gets this bullet.
    assert "Invoices: the threshold chosen on the validation layouts did not transfer to the unseen test layouts" in (
        body
    )
    assert "Banking77: the threshold chosen on the validation queries did not transfer" not in body


def test_confidence_bullet_never_claims_improvement_the_numbers_contradict(sync: types.ModuleType) -> None:
    """Invoices' alternative confidence definition has a HIGHER test AUROC than the primary (0.846 vs 0.858 in
    the fixture) -- the bullet must say so, never a blanket "did not improve on test"."""
    body = sync.RENDERERS["results"]()
    assert "did not improve" not in body
    # The choice itself was made on validation: AUROC about equal, primary's ECE markedly lower (a ratio, not a
    # fixed "about three times" for every task).
    assert "Banking77 AUROC 0.877 vs 0.874 (about equal), ECE 2.8% vs 9.2%, about 3.3x lower for the primary" in (body)
    assert "Invoices AUROC 0.778 vs 0.778 (about equal), ECE 16.8% vs 81.0%, about 4.8x lower for the primary" in (body)
    # Reported honestly on test, including the surprising direction for invoices.
    assert "Reported honestly, on test:" in body
    assert "Invoices AUROC 0.846 vs 0.858 (higher for the alternative)" in body
    assert "Banking77 AUROC 0.903 vs 0.901 (about equal)" in body


def test_seed_aggregate_row_shows_mean_and_std_only_no_selected_seed_ci(sync: types.ModuleType) -> None:
    """A row aggregating more than one seed must show mean +/- std with no bracketed interval next to it -- that
    interval belonged to one seed's own evaluation, not to the mean, and reads as if it did."""
    body = sync.RENDERERS["results"]()
    assert "75.7% ± 0.5 | 74.6% ± 0.7 | 88.1% ± 0.2" in body  # Banking77 aggregate row: no "[...]" alongside it
    assert "75.7% ± 0.5 [" not in body
    assert "58.0% ± 3.6" in body  # Invoices aggregate row (now 3 seeds in the fixture): same, no bracket either
    assert "58.0% ± 3.6 [" not in body


def test_per_seed_table_renders_for_every_task_with_a_seeds_row(sync: types.ModuleType) -> None:
    """Both fixtures now have a 3-seed student row (teacher labels): each gets its own per-seed table, with the
    validation-selected seed marked and carrying the row's own interval; the other seeds show no interval."""
    body = sync.RENDERERS["results"]()
    assert "Per-seed scores for `student qwen2.5-0.5b (teacher labels, 3 seeds)` (selected run: " in body
    assert body.count("Per-seed scores for `student qwen2.5-0.5b (teacher labels, 3 seeds)`") == 2
    assert "`qwen2.5-0.5b-full-s13` (validation-selected) | 13 | 75.5% [73.8, 77.0]" in body  # Banking77
    assert "| `qwen2.5-0.5b-full-s14` | 14 | 75.3% | 74.1% |" in body  # not selected: no interval
    assert "`qwen2.5-0.5b-quick-s13` (validation-selected) | 13 | 75.0% [61.1, 88.9]" in body  # Invoices
    assert "| `qwen2.5-0.5b-quick-s14` | 14 | 69.4% |" in body  # not selected: no interval


def test_template_cluster_table_formats_auroc_plain_not_percent(sync: types.ModuleType) -> None:
    """The invoices template-cluster bootstrap table must format AUROC like the main quality table does (plain,
    3 decimals), not as a percentage interval."""
    body = sync.RENDERERS["results"]()
    section = body.split("Template-cluster bootstrap")[1].split("Per-template scores")[0]
    assert "[0.600, 0.900]" in section  # the 0.5B zero-shot row's cluster AUROC interval, plain
    assert "[60.0, 90.0]" not in section  # never as if it were a percentage


def test_calibration_line_names_the_selected_run_and_its_reference_basis(sync: types.ModuleType) -> None:
    """The calibration line must describe the report's actual selected run and match the quality table's ECE
    basis -- not just the first teacher-labels student row, which for invoices is a different, unselected run
    at a different base size in the real reports (reproduced here by giving the fixture's only student row a
    different selected seed than seed 13's own point score would suggest if picked blindly)."""
    body = sync.RENDERERS["results"]()
    assert "Banking77 student `qwen2.5-0.5b-full-s13`, ECE on test vs gold: 15.0% raw vs 3.7% after isotonic" in (body)
    assert "Invoices student `qwen2.5-0.5b-quick-s13`, ECE on test vs gold: 12.6% raw vs 14.8% after isotonic" in (body)
    # The named run's ECE must equal the quality table's own ECE cell for that row (not some other row/seed).
    assert "| 12.6% [8.8, 16.7] | — |" in body  # the invoices student row's ECE column, same 12.6%


def test_calibration_line_picks_the_selected_run_not_the_first_student_row(
    tmp_path: Path, readme_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the exact bug reported: when the first teacher-labels student row in quality.rows is
    NOT the report's selected run, the calibration line must still describe the selected one."""
    reports = tmp_path / "reports"
    shutil.copytree(REPORTS, reports)
    invoices_path = reports / "invoices" / "report.json"
    data = json.loads(invoices_path.read_text(encoding="utf-8"))

    first_row = next(r for r in data["quality"]["rows"] if r["system"] == "student" and r["labels"] == "teacher")
    other_row = dict(first_row)
    other_row["run_ids"] = ["other-run-s99"]
    other_row["calibration"] = {**first_row["calibration"], "ece": 0.5, "isotonic_ece": 0.6}
    data["quality"]["rows"].insert(0, other_row)  # the WRONG row now lists first
    data["selected_run"]["run_id"] = first_row["run_ids"][0]  # but this is still the actually selected run
    invoices_path.write_text(json.dumps(data), encoding="utf-8")

    module = _load_script("sync_readme.py", reports=reports, readme=readme_copy, monkeypatch=monkeypatch)
    body = module.RENDERERS["results"]()
    assert "Invoices student `qwen2.5-0.5b-quick-s13`, ECE on test vs gold: 12.6% raw vs 14.8%" in body
    assert "50.0% raw" not in body  # the wrong (first-listed, unselected) row's ECE must never appear


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
    # The learning-rate ablation names the outlier seed's own validation agreement (from runs[]), not an
    # inferred training-loss claim the JSON does not make in those terms.
    assert "with 1 and 0 of 3 seeds diverging (agreement below 10%)" in body
    assert "diverged" not in body
    # The 1.5B student's gain is below the 1-point rule, at just over 2x the p95 latency. The rule's minimum is
    # spelled out without a forced decimal ("1 point", not "1.00 points"); the gain itself keeps its decimals.
    assert "gained +0.97 points" in body
    assert "below the 1 point minimum" in body
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
    # reproduction.json and report_mac_studio.json are optional too: nothing is rendered for either, and the
    # cost/latency and live-bench sections fall back to their current (MacBook Air only) behaviour.
    assert "### Reproducibility on a second machine" not in body
    assert "Measured on" not in body
    assert "eval in-process latency (flagged: not end-to-end through `serve`)" in body
    assert "**Banking77**: not measured yet." in body
    assert "**Invoices**: not measured yet." in body


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


# -- reproducibility on a second machine (optional reports/reproduction.json) -----------------------------


def test_reproducibility_section_absent_without_reproduction_json(sync: types.ModuleType) -> None:
    """The shared fixtures have no reproduction.json (matching the real reports/ directory today): nothing is
    rendered for this optional section."""
    assert "### Reproducibility on a second machine" not in sync.RENDERERS["results"]()


def test_reproducibility_section_renders_when_present(
    tmp_path: Path, readme_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = _reports_plus(tmp_path, "reproduction.json")
    module = _load_script("sync_readme.py", reports=reports, readme=readme_copy, monkeypatch=monkeypatch)
    body = module.RENDERERS["results"]()
    section = body.split("### Reproducibility on a second machine")[1].split("### Calibration")[0]

    assert "Reference: Mac17,4, Apple M5, 24 GB, macOS 26.6.2, 2026-09-27." in section
    assert "Rerun: Mac16,9, Apple M4 Max, 64 GB, macOS 26.6.2, 2026-09-28." in section
    assert "tolerance 2.0 points" in section
    # Per task: the max |difference| and whether every row stayed within tolerance.
    assert "| Banking77 | 0.78 pts | yes |" in section
    assert "| Invoices | 1.53 pts | yes |" in section
    # The compact table is the student/teacher rows at each task's main metric only (accuracy, field micro-F1)
    # -- never a zero-shot row, which the fixture's rows list also carries diffs for.
    assert "| Banking77 | teacher (deepseek/deepseek-v4.1-flash) | Accuracy | 75.8% | 76.0% | +0.21 pts |" in section
    assert "zero-shot" not in section
    # The selected run matched on Banking77 but differed on invoices; only the gain clause of each reason is
    # quoted, not the whole reason (which also names the run and its validation score).
    assert "Banking77: the selected run matched on both machines (`qwen2.5-0.5b-full-s13`)" in section
    assert "large gains 0.90 points, below the 1.00-point minimum" in section
    assert "large gains 3.79 points (>= 1.00) at 1.79x the small p95 (< 3x)" in section
    assert "best validation agreement" not in section
    assert "All rows within tolerance across every task: yes." in section


# -- cost/latency and live bench on a second machine (optional report_mac_studio.json) ----------------------


def test_cost_latency_and_live_bench_keep_current_behaviour_without_mac_studio(sync: types.ModuleType) -> None:
    """The shared fixtures have no report_mac_studio.json for either task: the cost/latency table stays the
    MacBook Air's own, and the live bench stays "not measured yet" (both reports' live_bench is null)."""
    body = sync.RENDERERS["results"]()
    assert "Measured on Mac16,9" not in body
    assert "in-process eval, not through the server" not in body  # only the mac-studio comparison line says this
    assert "**Banking77**: not measured yet." in body
    assert "**Invoices**: not measured yet." in body


def test_cost_latency_and_live_bench_use_mac_studio_when_present(
    tmp_path: Path, readme_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = _reports_plus(tmp_path, "banking77/report_mac_studio.json", "invoices/report_mac_studio.json")
    module = _load_script("sync_readme.py", reports=reports, readme=readme_copy, monkeypatch=monkeypatch)
    body = module.RENDERERS["results"]()

    cost_section = body.split("### Cost and latency")[1].split("### Live bench cross-check")[0]
    assert "Measured on Mac16,9, Apple M4 Max, 64 GB, macOS 26.6.2, 2026-09-28:" in cost_section
    assert "bench against `serve --threshold 0`" in cost_section
    # The MacBook Air's in-process eval latency is still shown, flagged, for comparison.
    assert (
        "For comparison, the MacBook Air's student latency was 44.5 ms p50 / 68.2 ms p95 (in-process eval, "
        "not through the server)." in cost_section
    )
    # Break-even now comes from the mac-studio report, not the MacBook Air one.
    assert "Break-even at 17,420 requests" in cost_section
    assert "18,565 requests" not in cost_section

    bench_section = body.split("### Live bench cross-check")[1].split("### Training on the M5")[0]
    assert "**Banking77** (measured on Mac16,9, Apple M4 Max, 64 GB, macOS 26.6.2, 2026-09-28)" in bench_section
    assert "qwen2.5-0.5b-full-s13" in bench_section  # the run id column
    assert "0.62/0.58/0.55" in bench_section  # load average, read from machine_state, not a top-level field
    assert "24.1%" in bench_section  # cascade escalation rate measured live
    assert "Composed (from the test split) vs measured: p50 12.9 vs 13.1 ms, p95 246 vs 251 ms." in bench_section


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
