#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
transcribe_and_diff.py

Unified helper to:
  1) Transcribe an input file into three variants
     (default, stabilize, stabilize+vad+demucs).
  2) Run pairwise SRT readability diffs on the three generated outputs.

With no flags it runs BOTH steps. Use --transcribe or --report to select one.

Author: elvee
Date: 11-08-2025
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import typer

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_OUT_DIR: Path = Path("data/test_results")
D_DEFAULT: Path = Path("data/output/default")
D_STABILIZE: Path = Path("data/output/stabilize")
D_SVD: Path = Path("data/output/stabilize_vad_demucs")

LOG_FORMAT: str = "[%(levelname)s] %(message)s"

app = typer.Typer(add_completion=False, no_args_is_help=True)


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Runners:
    """Holds the resolved runners for transcription and diff reporting.

    Attributes:
        transcribe: Command prefix used to run the transcriber.
        diff_report: Command prefix used to run the SRT diff reporter.
    """

    transcribe: Tuple[str, ...]
    diff_report: Tuple[str, ...]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def command_available(cmd: str) -> bool:
    """Return True if a shell command is available on PATH.

    Args:
        cmd: Command name to probe.

    Returns:
        bool: True if the command is found, otherwise False.
    """
    return shutil.which(cmd) is not None


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Ensure a collection of directories exist.

    Args:
        paths: Iterable of directory paths that should exist.

    Returns:
        None
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def resolve_runners() -> Runners:
    """Resolve runners for transcription and reporting, mirroring Bash logic.

    Priority order (transcribe):
        1) `pdm run parakeet-rocm`
        2) `parakeet-rocm`
        3) `python -m parakeet_nemo_asr_rocm.cli`

    Priority order (diff):
        - If using `pdm` for transcribe, prefer
          `pdm run python -m scripts.srt_diff_report`
        - Else if `srt-diff-report` exists, use it
        - Else fallback to `python -m scripts.srt_diff_report`

    Returns:
        Runners: The resolved command prefixes.
    """
    if command_available("pdm"):
        transcribe = ("pdm", "run", "parakeet-rocm")
        diff_report = ("pdm", "run", "python", "-m", "scripts.srt_diff_report")
    elif command_available("parakeet-rocm"):
        transcribe = ("parakeet-rocm",)
        if command_available("srt-diff-report"):
            diff_report = ("srt-diff-report",)
        else:
            diff_report = ("python", "-m", "scripts.srt_diff_report")
    else:
        transcribe = ("python", "-m", "parakeet_nemo_asr_rocm.cli")
        if command_available("srt-diff-report"):
            diff_report = ("srt-diff-report",)
        else:
            diff_report = ("python", "-m", "scripts.srt_diff_report")

    return Runners(transcribe=transcribe, diff_report=diff_report)


def run(cmd: Sequence[str]) -> None:
    """Run a shell command, logging it first and raising on failure.

    Args:
        cmd: Full command list to execute.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the process exits non-zero.
    """
    logging.debug("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_srt(dir_path: Path, stem: str) -> Optional[Path]:
    """Find an SRT file for a given stem within a directory.

    Search order:
      1) Exact match: `<dir>/<stem>.srt`
      2) Newest file matching `<stem>*.srt` by modification time

    Args:
        dir_path: Directory to search within.
        stem: Base filename (without extension) to match.

    Returns:
        Optional[Path]: Path to the found SRT file, or None if not found.
    """
    exact = dir_path / f"{stem}.srt"
    if exact.is_file():
        return exact

    candidates = sorted(
        dir_path.glob(f"{stem}*.srt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# -----------------------------------------------------------------------------
# Core actions
# -----------------------------------------------------------------------------


def transcribe_three(runners: Runners, input_file: Path) -> None:
    """Transcribe an audio file into three variants.

    Variants:
      - default
      - stabilize
      - stabilize + vad + demucs

    Args:
        runners: Resolved runners with the transcribe command prefix.
        input_file: The input audio file path.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If any underlying process fails.
    """
    ensure_dirs([D_DEFAULT, D_STABILIZE, D_SVD])

    base: List[str] = list(runners.transcribe) + [
        "transcribe",
        "--word-timestamps",
        "--output-format",
        "srt",
    ]

    # default
    run(base + ["--output-dir", str(D_DEFAULT), str(input_file)])

    # stabilize
    run(
        base
        + [
            "--output-dir",
            str(D_STABILIZE),
            "--stabilize",
            str(input_file),
        ]
    )

    # stabilize + vad + demucs
    run(
        base
        + [
            "--output-dir",
            str(D_SVD),
            "--stabilize",
            "--vad",
            "--demucs",
            str(input_file),
        ]
    )


def report_diffs(
    runners: Runners,
    stem: str,
    out_dir: Path,
    show_violations: int,
) -> None:
    """Run pairwise SRT readability diffs between the three variants.

    Pairs:
      - default vs stabilize
      - default vs stabilize_vad_demucs
      - stabilize vs stabilize_vad_demucs

    Args:
        runners: Resolved runners with the diff-report command prefix.
        stem: Base filename (without extension) used to find SRT files.
        out_dir: Directory to write Markdown and JSON reports to.
        show_violations: If > 0, limit top-N violations per category.

    Returns:
        None

    Raises:
        FileNotFoundError: If any of the expected SRT files are missing.
        subprocess.CalledProcessError: If any diff process fails.
    """
    ensure_dirs([out_dir])

    srt_default = find_srt(D_DEFAULT, stem)
    srt_stab = find_srt(D_STABILIZE, stem)
    srt_svd = find_srt(D_SVD, stem)

    if not (srt_default and srt_stab and srt_svd):
        msg_lines = [
            f"Missing SRT(s) for '{stem}'. Ensure transcription step ran.",
            f"  default:       {D_DEFAULT / (stem + '.srt')} "
            f"(or newest {stem}*.srt)",
            f"  stabilize:     {D_STABILIZE / (stem + '.srt')} "
            f"(or newest {stem}*.srt)",
            f"  vad+demucs:    {D_SVD / (stem + '.srt')} " f"(or newest {stem}*.srt)",
        ]
        raise FileNotFoundError("\n".join(msg_lines))

    pairs: List[Tuple[Tuple[str, Path], Tuple[str, Path]]] = [
        (("default", srt_default), ("stabilize", srt_stab)),
        (("default", srt_default), ("stabilize_vad_demucs", srt_svd)),
        (("stabilize", srt_stab), ("stabilize_vad_demucs", srt_svd)),
    ]

    for (left_label, left_path), (right_label, right_path) in pairs:
        base_name = f"srt_diff_{left_label}_vs_{right_label}_{stem}"
        md_out = out_dir / f"{base_name}.md"
        json_out = out_dir / f"{base_name}.json"

        cmd_md: List[str] = list(runners.diff_report) + [
            str(left_path),
            str(right_path),
            "--output-format",
            "markdown",
            "-o",
            str(md_out),
        ]
        if show_violations > 0:
            cmd_md += ["--show-violations", str(show_violations)]
        run(cmd_md)

        cmd_json: List[str] = list(runners.diff_report) + [
            str(left_path),
            str(right_path),
            "--output-format",
            "json",
            "-o",
            str(json_out),
        ]
        if show_violations > 0:
            cmd_json += ["--show-violations", str(show_violations)]
        run(cmd_json)


# -----------------------------------------------------------------------------
# CLI (Typer)
# -----------------------------------------------------------------------------


@app.command("run")
def cli(
    audio_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the input audio file.",
    ),
    transcribe: bool = typer.Option(
        False, "--transcribe", help="Run only the transcription step."
    ),
    report: bool = typer.Option(False, "--report", help="Run only the reporting step."),
    show_violations: int = typer.Option(
        0,
        "--show-violations",
        min=0,
        help="If > 0, show top-N violations per category in reports.",
    ),
    out_dir: Path = typer.Option(
        DEFAULT_OUT_DIR,
        "--out-dir",
        help="Directory to write markdown/json reports to.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Transcribe an audio file (3 variants) and/or generate SRT diff reports.

    With no explicit mode flags, this runs both steps in sequence.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)

    if transcribe and report:
        raise typer.BadParameter(
            "Use ONLY one of --transcribe or --report (or neither for both)."
        )

    stem = audio_file.stem
    ensure_dirs([out_dir, D_DEFAULT, D_STABILIZE, D_SVD])

    try:
        runners = resolve_runners()
        logging.info("Transcribe runner: %s", " ".join(runners.transcribe))
        logging.info("Diff runner:       %s", " ".join(runners.diff_report))

        if transcribe:
            transcribe_three(runners, audio_file)
        elif report:
            report_diffs(
                runners=runners,
                stem=stem,
                out_dir=out_dir,
                show_violations=show_violations,
            )
        else:
            transcribe_three(runners, audio_file)
            report_diffs(
                runners=runners,
                stem=stem,
                out_dir=out_dir,
                show_violations=show_violations,
            )

        logging.info("Done.")
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        raise typer.Exit(code=2) from exc
    except subprocess.CalledProcessError as exc:
        logging.error("Command failed with exit code %s", exc.returncode)
        raise typer.Exit(code=exc.returncode or 1) from exc
    # No broad catch-all: allow unexpected exceptions to propagate naturally.


# -----------------------------------------------------------------------------
# Main guard
# -----------------------------------------------------------------------------


def main() -> None:
    """Entrypoint for running via python -m or direct execution."""
    app()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.error("Interrupted by user.")
        sys.exit(130)
