"""Command-line interface: capture -> curate -> train -> eval -> serve -> report."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import typer

from taskdistill import __version__, paths

app = typer.Typer(
    name="taskdistill",
    help="Distil a narrow LLM API call into a small local model and serve a calibrated cascade.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
teacher_app = typer.Typer(help="Teacher model tools: bake-off and replay recordings.", no_args_is_help=True)
pricing_app = typer.Typer(help="Model pricing snapshot used by the cost ledger.", no_args_is_help=True)
app.add_typer(teacher_app, name="teacher")
app.add_typer(pricing_app, name="pricing")

MAX_USD_HELP = "Cap on the API dollars this command run may spend (checked before every call)."
YES_HELP = "Confirm live spend when the projected cost exceeds $0.50 (needed for non-interactive runs)."


def _version(value: bool) -> None:
    if value:
        typer.echo(f"taskdistill {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show the version."),
) -> None:
    """taskdistill: capture -> curate -> train -> eval -> serve -> report."""


def _fail(message: str) -> typer.Exit:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def handle_errors[F: Callable[..., Any]](fn: F) -> F:
    """Turn the package's expected errors into a one-line message and exit code 1."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from taskdistill.config import ConfigError
        from taskdistill.teacher.base import TeacherError

        try:
            return fn(*args, **kwargs)
        except (typer.Exit, typer.Abort, click.ClickException):
            raise
        except (ConfigError, TeacherError, FileNotFoundError, ValueError, RuntimeError) as exc:
            raise _fail(str(exc)) from exc
        except KeyboardInterrupt:
            raise _fail("interrupted") from None

    return wrapper  # type: ignore[return-value]


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def _parse_listen(listen: str, default_port: int) -> tuple[str, int]:
    if listen.startswith("["):  # [::1]:8787
        host, _, port = listen[1:].partition("]:")
        return host, int(port or default_port)
    host, sep, port = listen.rpartition(":")
    if not sep:
        return listen, default_port
    return host or "127.0.0.1", int(port)


def parse_duration(text: str) -> float:
    """'24h' -> seconds. Units: s, m, h, d."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", text)
    if not match:
        raise ValueError(f"cannot parse duration {text!r}; use e.g. 30m, 24h or 7d")
    value, unit = float(match.group(1)), match.group(2)
    return value * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def _teacher_mode(live: bool, replay: bool) -> str | None:
    if live and replay:
        raise ValueError("--live and --replay are mutually exclusive")
    return "live" if live else "replay" if replay else None


# ------------------------------------------------------------------------------------------------ init
@app.command()
@handle_errors
def init(
    task: str = typer.Argument(..., help="Task name; the spec is created in ./tasks/<task>/."),
    task_type: str = typer.Option(..., "--type", help="classification | extraction"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing ./tasks/<task>/."),
) -> None:
    """Scaffold tasks/<task>/ (task.yaml, teacher prompt, labels or JSON Schema) from a template."""
    from taskdistill.init import init_task

    directory = init_task(task, task_type, force=force)
    typer.echo(f"created {directory.relative_to(Path.cwd()) if directory.is_relative_to(Path.cwd()) else directory}/")
    for name in sorted(p.name for p in directory.iterdir()):
        typer.echo(f"  {name}")
    typer.echo("next: edit the teacher prompt and labels/schema, then point your client at `taskdistill capture`.")


# --------------------------------------------------------------------------------------------- capture
@app.command()
@handle_errors
def capture(
    task: str = typer.Option(..., "--task", help="Task name (tasks/<task>/task.yaml)."),
    listen: str = typer.Option("127.0.0.1:8787", "--listen", help="host:port for the capture proxy."),
    import_path: Path | None = typer.Option(None, "--import", help="Import a JSONL (or .jsonl.gz) log instead."),
    fmt: str = typer.Option("openai", "--format", help="Import format: openai | pairs | inputs."),
    export_path: Path | None = typer.Option(None, "--export", help="Export captured traffic as openai JSONL."),
    upstream: str | None = typer.Option(
        None, "--upstream", help="Upstream base URL (default: $TASKDISTILL_TEACHER_BASE_URL, else teacher.base_url)."
    ),
) -> None:
    """Capture traffic through an OpenAI-compatible proxy, or import/export existing logs."""
    from taskdistill.store import Store

    store = Store()
    if import_path is not None and export_path is not None:
        raise ValueError("use either --import or --export, not both")
    if import_path is not None:
        from taskdistill.capture.importer import import_file

        counts = import_file(store, task, import_path, fmt)
        typer.echo(f"imported {counts['imported']} of {counts['read']} rows ({fmt}) into task {task}")
        out = _write_json(
            paths.task_home(task) / "last_import.json", {"task": task, "file": import_path.name, **counts}
        )
        typer.echo(f"summary: {paths.relative_to_home(out)}")
        return
    if export_path is not None:
        from taskdistill.capture.export import export_file

        n = export_file(store, task, export_path)
        typer.echo(f"exported {n} captured requests to {export_path}")
        return
    from taskdistill.capture.proxy import run_proxy

    base = upstream or os.environ.get("TASKDISTILL_TEACHER_BASE_URL")
    if not base:
        from taskdistill.config import load_task

        base = load_task(task).teacher.base_url
    host, port = _parse_listen(listen, 8787)
    typer.echo(f"capture proxy for task {task} on http://{host}:{port}/t/{task}/v1 -> {base}")
    typer.echo("point your OpenAI client's base_url there; headers are forwarded, never stored. Ctrl+C to stop.")
    run_proxy(host, port, store=store, upstream_base_url=base, default_task=task)


# ---------------------------------------------------------------------------------------------- curate
@app.command()
@handle_errors
def curate(
    task: str = typer.Option(..., "--task"),
    yes: bool = typer.Option(False, "--yes", help=YES_HELP),
    max_usd: float | None = typer.Option(None, "--max-usd", help=MAX_USD_HELP),
    live: bool = typer.Option(False, "--live", help="Label with the live teacher (needs an API key)."),
    replay: bool = typer.Option(False, "--replay", help="Label from the recorded teacher outputs."),
) -> None:
    """Build train/valid/test data: extract, merge, normalise, scrub, split, dedupe, label, filter, check."""
    from taskdistill.config import load_task
    from taskdistill.curate.pipeline import run_curate
    from taskdistill.store import Store
    from taskdistill.teacher.factory import make_teacher, new_run_id, resolve_mode

    spec = load_task(task)
    teacher = None
    try:
        mode = resolve_mode(spec, _teacher_mode(live, replay))
    except (RuntimeError, ValueError) as exc:
        if live or replay:
            raise
        typer.echo(f"note: no teacher available ({exc}); curate will stop if any example needs labelling")
        mode = None
    if mode is not None:
        teacher = make_teacher(spec, mode=mode, phase="curate-label", run_id=new_run_id("curate"), run_cap=max_usd)
        typer.echo(f"teacher: {mode} ({spec.teacher.model})")
    try:
        result = run_curate(spec, store=Store(), teacher=teacher, yes=yes, max_usd=max_usd)
    finally:
        if teacher is not None:
            asyncio.run(teacher.aclose())
    splits = result.stats.get("splits", {})
    typer.echo("curated: " + ", ".join(f"{name} {count}" for name, count in splits.items()))
    typer.echo(f"stats: {paths.relative_to_home(paths.data_dir(task) / 'curate_stats.json')}")


# ----------------------------------------------------------------------------------------------- train
@app.command()
@handle_errors
def train(
    task: str = typer.Option(..., "--task"),
    base: str | None = typer.Option(None, "--base", help="Base model (default: student.base_model)."),
    profile: str | None = typer.Option(None, "--profile", help="quick | full (default: train.profile)."),
    seed: int | None = typer.Option(None, "--seed", help="Training seed (default: train.seed)."),
    backend: str = typer.Option("mlx", "--backend", help="mlx (Apple Silicon) | torch (transformers + PEFT)."),
    labels: str = typer.Option("teacher", "--labels", help="teacher | gold (gold = label-quality ceiling)."),
) -> None:
    """LoRA fine-tune the student; keeps the checkpoint with the best validation loss."""
    from taskdistill.config import load_task
    from taskdistill.train.runner import run_training

    spec = load_task(task)
    result = run_training(spec, backend=backend, base=base, profile=profile, seed=seed, labels=labels)
    log = result.log
    typer.echo(
        f"run {result.run_id}: {log.get('iterations')} iterations, {log.get('wall_seconds', 0) / 60:.1f} min, "
        f"peak {log.get('peak_memory_gb')} GB, best val loss {log.get('best_val_loss')} at {log.get('best_iteration')}"
    )
    typer.echo(f"log: {paths.relative_to_home(Path(result.run_dir) / 'train_log.json')}")


# ------------------------------------------------------------------------------------------------ eval
@app.command("eval")
@handle_errors
def eval_command(
    task: str = typer.Option(..., "--task"),
    run: str | None = typer.Option(None, "--run", help="Run id (default: selected_run.json)."),
    split: str = typer.Option("test", "--split", help="test | valid"),
    backend: str = typer.Option("mlx", "--backend", help="mlx | torch"),
    select: bool = typer.Option(False, "--select", help="Choose the run on validation first (selected_run.json)."),
    zero_shot: bool = typer.Option(False, "--zero-shot", help="Evaluate the base model zero-shot instead."),
    fast: bool = typer.Option(False, "--fast", help="Skip the alternative confidence scores (quick profile)."),
) -> None:
    """Score the student, the teacher (recorded) and the cascade; pick the threshold on validation."""
    from taskdistill.config import load_task
    from taskdistill.evaluate.runner import run_eval

    if split not in ("test", "valid"):
        raise ValueError("--split must be test or valid")
    spec = load_task(task)
    command = "taskdistill " + " ".join(sys.argv[1:]) if sys.argv else None
    result = run_eval(
        spec, run_id=run, split=split, backend=backend, select=select, zero_shot=zero_shot, fast=fast, command=command
    )
    student = result.get("systems", {}).get("student", {}).get("metrics", {})
    shown = {k: v for k, v in student.items() if isinstance(v, float)}
    typer.echo(
        f"{result.get('run_id')} on {split} (n={result.get('n')}): "
        + ", ".join(f"{k} {v:.4f}" for k, v in shown.items())
    )
    op = (result.get("operating_point") or {}).get(split) or {}
    if op:
        typer.echo(f"cascade: escalation {op.get('escalation_rate', 0):.1%}, target met: {op.get('met')}")


# ----------------------------------------------------------------------------------------------- serve
@app.command()
@handle_errors
def serve(
    task: str = typer.Option(..., "--task"),
    run: str | None = typer.Option(None, "--run", help="Run id (default: selected_run.json)."),
    threshold: str = typer.Option("auto", "--threshold", help="auto (threshold.json) or a float; 0 = never escalate."),
    backend: str = typer.Option("mlx", "--backend", help="mlx | torch"),
    replay: bool = typer.Option(False, "--replay", help="Escalate to the recorded teacher outputs, not the live API."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    max_usd: float | None = typer.Option(None, "--max-usd", help=MAX_USD_HELP),
    yes: bool = typer.Option(False, "--yes", help="Accepted for symmetry; serve has no up-front projection."),
) -> None:
    """Start the OpenAI-compatible cascade server (student first, teacher below the threshold)."""
    from taskdistill.config import load_task
    from taskdistill.serve.runner import run_server

    spec = load_task(task)
    value: str | float = threshold if threshold == "auto" else float(threshold)
    run_server(
        spec,
        run_id=run,
        threshold=value,
        backend=backend,
        replay=replay,
        host=host,
        port=port,
        max_usd=max_usd,
        echo=typer.echo,
        log_level="warning",
    )


# ---------------------------------------------------------------------------------------------- report
@app.command()
@handle_errors
def report(
    task: str = typer.Option(..., "--task"),
    from_serve_log: bool = typer.Option(False, "--from-serve-log", help="Add figures from served traffic."),
    since: str = typer.Option("24h", "--since", help="Window for --from-serve-log, e.g. 30m, 24h, 7d."),
    out: Path | None = typer.Option(None, "--out", help="Output directory (default: reports/<task>)."),
) -> None:
    """Write reports/<task>/report.{md,json}: quality, operating point, cost, latency, break-even."""
    from taskdistill.config import load_task
    from taskdistill.report.builder import build_report

    spec = load_task(task)
    command = "taskdistill " + " ".join(sys.argv[1:]) if sys.argv else None
    since_s = parse_duration(since) if from_serve_log else None
    result = build_report(spec, from_serve_log=from_serve_log, since_s=since_s, out_dir=out, command=command)
    target = out or Path("reports") / task
    typer.echo(f"wrote {target / 'report.md'} and {target / 'report.json'}")
    be = result.get("break_even") or {}
    if be.get("volume") is not None:
        bound = "at least " if be.get("lower_bound") else ""
        typer.echo(f"break-even volume: {bound}{be['volume']:,.0f} requests")
    elif be.get("reason"):
        typer.echo(f"break-even volume: n/a ({be['reason']})")


# ----------------------------------------------------------------------------------------------- bench
@app.command()
@handle_errors
def bench(
    url: str = typer.Option("http://127.0.0.1:8000", "--url", help="Base URL of a running `taskdistill serve`."),
    task: str = typer.Option(..., "--task"),
    n: int = typer.Option(300, "--n", help="Timed requests."),
    warmup: int = typer.Option(20, "--warmup", help="Warm-up requests excluded from the statistics."),
    yes: bool = typer.Option(False, "--yes", help=YES_HELP),
    max_usd: float | None = typer.Option(None, "--max-usd", help=MAX_USD_HELP),
    out: Path | None = typer.Option(None, "--out", help="Write the JSON here too."),
) -> None:
    """Replay test inputs through a running server, one at a time, and measure end-to-end latency."""
    from taskdistill.bench import run_bench
    from taskdistill.config import load_task

    spec = load_task(task)
    result = run_bench(
        url,
        spec,
        n=n,
        warmup=warmup,
        yes=yes,
        max_usd=max_usd,
        token=os.environ.get("TASKDISTILL_SERVER_TOKEN"),
        out=out,
    )
    lat = result.get("latency_ms") or {}
    escalation = result.get("escalation_rate")
    spend = result.get("spend_usd")
    typer.echo(
        f"{result.get('mode')}: p50 {lat.get('p50') or 0:.1f} ms, p95 {lat.get('p95') or 0:.1f} ms, "
        f"escalation {'n/a' if escalation is None else f'{escalation:.1%}'}, "
        f"spend {'not measured' if spend is None else f'${spend:.4f}'}"
    )


# ------------------------------------------------------------------------------------------------ demo
@app.command()
@handle_errors
def demo(
    name: str = typer.Argument(..., help="banking77 | invoices"),
    profile: str = typer.Option("quick", "--profile", help="quick (minutes) | full (the README numbers)."),
    live: bool = typer.Option(False, "--live", help="Call the live teacher instead of the packaged recording."),
    yes: bool = typer.Option(False, "--yes", help=YES_HELP),
    until: str | None = typer.Option(
        None, "--until", help="Stop after a stage: data, capture, curate, train, eval, serve, report."
    ),
    base: str | None = typer.Option(None, "--base", help="Base model (default: the demo's chosen base)."),
    seed: int | None = typer.Option(None, "--seed", help="Training seed (affects training only)."),
    max_usd: float | None = typer.Option(None, "--max-usd", help=MAX_USD_HELP),
    backend: str = typer.Option("mlx", "--backend", help="mlx | torch"),
) -> None:
    """Run the whole pipeline on a bundled demo task, composing the public commands."""
    from taskdistill.demos.runner import run_demo

    run_demo(
        name, profile=profile, live=live, yes=yes, until=until, base=base, seed=seed, max_usd=max_usd, backend=backend
    )


# --------------------------------------------------------------------------------------------- teacher
@teacher_app.command("bakeoff")
@handle_errors
def teacher_bakeoff(
    task: str = typer.Option(..., "--task"),
    models: str = typer.Option(..., "--models", help="Comma-separated slugs, optionally slug@provider-tag."),
    n: int = typer.Option(200, "--n", help="Validation inputs with gold labels to score."),
    prompt_variant: Path | None = typer.Option(None, "--prompt-variant", help="Alternative teacher prompt file."),
    yes: bool = typer.Option(False, "--yes", help=YES_HELP),
    max_usd: float | None = typer.Option(None, "--max-usd", help=MAX_USD_HELP),
) -> None:
    """Score candidate teachers on validation inputs with gold labels; picks the cheapest within 2 points."""
    from taskdistill.config import load_task
    from taskdistill.store import Store
    from taskdistill.teacher.bakeoff import run_bakeoff

    spec = load_task(task)
    candidates = [m.strip() for m in models.split(",") if m.strip()]
    result = run_bakeoff(
        spec, models=candidates, n=n, prompt_variant=prompt_variant, yes=yes, max_usd=max_usd, store=Store()
    )
    chosen = result.get("chosen") or {}
    if not chosen.get("model"):
        raise ValueError("no candidate passed the bake-off checks; see the JSON report for the reasons")
    typer.echo(f"chosen teacher: {chosen['model']}")
    if chosen.get("warning"):
        typer.echo(f"warning: {chosen['warning']}")


@teacher_app.command("record")
@handle_errors
def teacher_record(
    task: str = typer.Option(..., "--task"),
    out: Path = typer.Option(..., "--out", help="Recording path (.jsonl.gz)."),
    demo_name: str | None = typer.Option(None, "--demo", help="Record every request a bundled demo sends."),
) -> None:
    """Write the cached teacher outputs of a task as an offline replay recording."""
    from taskdistill.config import load_task
    from taskdistill.store import Store
    from taskdistill.teacher.record import build_recording, default_keys

    spec = load_task(task)
    if demo_name:
        from taskdistill.demos.runner import demo_requests

        result = build_recording(spec, requests=demo_requests(demo_name, spec), out=out)
    else:
        result = build_recording(spec, keys=default_keys(spec, Store()), out=out)
    typer.echo(f"wrote {out} with {result.get('records')} records")


# --------------------------------------------------------------------------------------- pricing, budget
@pricing_app.command("refresh")
@handle_errors
def pricing_refresh(
    models: str | None = typer.Option(None, "--models", help="Also fetch per-provider prices for these slugs."),
    base_url: str = typer.Option("https://openrouter.ai/api/v1", "--base-url"),
) -> None:
    """Pull the model list (and per-provider endpoints) and save a dated pricing snapshot."""
    from taskdistill.teacher.pricing import refresh

    slugs = [m.strip() for m in models.split(",")] if models else None
    snapshot = refresh(base_url, api_key=os.environ.get("OPENROUTER_API_KEY"), models=slugs)
    path = snapshot.save()
    typer.echo(f"pricing snapshot {snapshot.date}: {len(snapshot.models)} models -> {paths.relative_to_home(path)}")


@app.command()
@handle_errors
def budget(json_path: Path | None = typer.Option(None, "--json", help="Also write the summary as JSON.")) -> None:
    """Print API spend by task and phase from the workspace ledger."""
    from taskdistill.ledger import Ledger

    ledger = Ledger()
    summary = ledger.summary()
    typer.echo(f"total ${summary['total']:.4f} of cap ${summary['cap']:.2f} ({summary['calls']} calls)")
    for key, value in sorted(summary.get("by_task_phase", {}).items()):
        typer.echo(f"  {key}: ${value:.4f}")
    if json_path is not None:
        ledger.export_json(json_path)
        typer.echo(f"wrote {json_path}")


def main() -> None:
    app()
