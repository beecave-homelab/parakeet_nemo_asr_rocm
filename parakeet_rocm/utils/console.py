"""Shared Rich console helpers for styled CLI output.

This module centralizes the console instance and semantic message helpers
used across CLI-facing code.  It respects ``NO_COLOR`` / ``TERM=dumb``
naturally via Rich's own environment detection and offers a simple quiet
guard so callers do not need to repeat the ``quiet`` check.
"""

from __future__ import annotations

from functools import lru_cache

from rich.console import Console, ConsoleRenderable, RenderableType, RichCast
from rich.markup import escape
from rich.segment import Segment

ConsolePrintItem = ConsoleRenderable | RichCast | RenderableType | Segment

__all__ = [
    "get_console",
    "get_error_console",
    "print_error",
    "print_info",
    "print_status",
    "print_success",
    "print_warning",
]


@lru_cache(maxsize=1)
def get_console() -> Console:
    """Return the shared Rich ``Console`` instance used for CLI stdout output.

    Rich automatically disables color and styling when ``NO_COLOR`` is set
    or ``TERM`` is ``dumb``; callers do not need extra environment checks.

    Returns:
        The shared stdout ``Console`` instance.
    """
    return Console()


@lru_cache(maxsize=1)
def get_error_console() -> Console:
    """Return the shared Rich ``Console`` instance used for CLI stderr output.

    Returns:
        The shared stderr ``Console`` instance.
    """
    return Console(stderr=True)


def _echo(quiet: bool, message: ConsolePrintItem, *, err: bool = False) -> None:
    """Print to the shared console unless quiet mode is enabled.

    Args:
        quiet: When ``True``, the call is a no-op.
        message: Renderable or text forwarded to ``Console.print``.
        err: When ``True``, print to stderr instead of stdout.
    """
    if quiet:
        return
    console = get_error_console() if err else get_console()
    console.print(message)


def print_info(message: str, *, quiet: bool = False) -> None:
    """Print a plain informational message.

    Args:
        message: Text to print.
        quiet: When ``True``, suppress output.
    """
    _echo(quiet, message)


def print_success(message: str, *, quiet: bool = False) -> None:
    """Print a success message in green.

    Args:
        message: Text to print.
        quiet: When ``True``, suppress output.
    """
    _echo(quiet, f"[green]{message}[/green]")


def print_warning(message: str, *, quiet: bool = False) -> None:
    """Print a warning message in yellow.

    Args:
        message: Text to print.
        quiet: When ``True``, suppress output.
    """
    _echo(quiet, f"[yellow]Warning: {message}[/yellow]")


def print_error(message: str, *, quiet: bool = False) -> None:
    """Print an error message in bold red to stderr.

    Args:
        message: Text to print.
        quiet: When ``True``, suppress output.
    """
    _echo(quiet, f"[bold red]Error: {message}[/bold red]", err=True)


def print_status(label: str, message: str, *, quiet: bool = False, err: bool = False) -> None:
    """Print a labeled status line using Rich markup.

    The label is rendered in cyan brackets; the message keeps any markup the
    caller embeds.  This is the preferred replacement for ``typer.echo`` in
    CLI-facing code.

    Args:
        label: Short bracket label, e.g. ``watch`` or ``model``.
        message: Message body; may contain Rich markup.
        quiet: When ``True``, suppress output.
        err: When ``True``, print to stderr instead of stdout.
    """
    if quiet:
        return
    label_text = escape(f"[{label}]")
    console = get_error_console() if err else get_console()
    console.print(f"[cyan bold]{label_text}[/cyan bold] {message}")
