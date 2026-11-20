"""Command-line entry point (`sensorfm ...`). Subcommands are added as tasks land."""

from __future__ import annotations

import typer

from sensorfm import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def hello() -> None:
    """Sanity command so the CLI has more than one subcommand."""
    typer.echo("sensorfm ready")
