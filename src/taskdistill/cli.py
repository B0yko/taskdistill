"""Command-line interface."""

from __future__ import annotations

import typer

from taskdistill import __version__

app = typer.Typer(
    name="taskdistill",
    help="Distil a narrow LLM API call into a small local model and serve a calibrated cascade.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"taskdistill {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show the version."),
) -> None:
    """taskdistill: capture -> curate -> train -> eval -> serve -> report."""


def main() -> None:
    app()
