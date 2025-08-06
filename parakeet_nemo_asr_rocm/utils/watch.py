"""Directory / pattern watcher that triggers transcription.

This module centralises the ``--watch`` implementation so that the top-level
`cli.py` remains a thin argument parser.

The primary entry point is :func:`watch_and_transcribe`, which blocks and
continuously monitors for new audio files that match the given *patterns*.

It uses :func:`parakeet_nemo_asr_rocm.utils.file_utils.resolve_input_paths` to
expand wildcards and directories. Already-transcribed files are skipped by
checking whether an output file would be generated (using
:func:`parakeet_nemo_asr_rocm.utils.file_utils.get_unique_filename` with
``overwrite=False``). If the unique filename differs from the intended one, it
assumes a transcription already exists.
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Callable, Iterable, List, Sequence

from parakeet_nemo_asr_rocm.utils.file_utils import (
    AUDIO_EXTENSIONS,
    get_unique_filename,
    resolve_input_paths,
)

__all__ = ["watch_and_transcribe"]


def _default_sig_handler(signum: int, _frame: FrameType | None) -> None:  # noqa: D401
    """Handle ``SIGINT`` (Ctrl-C) gracefully.

    Args:
        signum: Received POSIX signal number (unused).
        _frame: Current stack frame (unused).

    """
    print("\n[watch] Stopping…")
    sys.exit(0)


def _needs_transcription(
    path: Path,
    output_dir: Path,
    output_template: str,
    output_format: str,
) -> bool:  # noqa: D401
    """Check whether *path* still needs to be transcribed.

    Args:
        path: Audio file under consideration.
        output_dir: Directory where output files are written.
        output_template: Filename template provided by the CLI.
        output_format: Desired output extension (``txt``, ``srt``, …).

    Returns:
        ``True`` if no output file exists for *path* yet, ``False`` otherwise.

    """
    target_name = output_template.format(
        parent=path.parent.name,
        filename=path.stem,
        index="",  # handled elsewhere
        date="",  # handled elsewhere
    )
    candidate = output_dir / f"{target_name}.{output_format}"
    # If unique filename differs, file exists ⇒ already transcribed
    return get_unique_filename(candidate, overwrite=False) == candidate


def watch_and_transcribe(
    *,
    patterns: Iterable[str | Path],
    transcribe_fn: Callable[[List[Path]], None],
    poll_interval: float = 2.0,
    output_dir: Path,
    output_format: str,
    output_template: str,
    audio_exts: Sequence[str] | None = None,
    verbose: bool = False,
) -> None:
    """Monitor *patterns* and transcribe newly detected audio files.

    This is a lightweight polling implementation to avoid adding extra
    dependencies such as *watchdog*. A sleep-poll loop every few seconds is
    usually sufficient for batch-style workflows.

    Args:
        patterns: Directory, file, or glob pattern(s) to monitor.
        transcribe_fn: Callback that receives a list of newly detected audio
            files to transcribe.
        poll_interval: Seconds between directory scans.
        output_dir: Directory where transcription outputs are written.
        output_format: Output format (e.g. ``"txt"``, ``"srt"``).
        output_template: Template string used to construct output filenames.
        audio_exts: Allowed audio extensions. ``None`` defaults to
            :data:`parakeet_nemo_asr_rocm.utils.file_utils.AUDIO_EXTENSIONS`.
        verbose: If *True*, prints watcher debug information to *stdout*.

    Returns:
        None. This function blocks indefinitely until interrupted.

    """
    print(
        f"[watch] Monitoring {', '.join(map(str, patterns))} …  (Press Ctrl+C to stop)"
    )

    signal.signal(signal.SIGINT, _default_sig_handler)

    seen: set[Path] = set()

    while True:
        all_matches = resolve_input_paths(
            patterns, audio_exts=audio_exts or AUDIO_EXTENSIONS
        )
        if verbose:
            print(f"[watch] Scan found {len(all_matches)} candidate file(s)")
        new_paths: List[Path] = []
        for p in all_matches:
            if p in seen:
                if verbose:
                    print(f"[watch] ✗ Already processed: {p}")
                continue
            if _needs_transcription(p, output_dir, output_template, output_format):
                new_paths.append(p)
                seen.add(p)
            else:
                if verbose:
                    print(f"[watch] ✗ Output exists, skipping: {p}")
        if new_paths:
            if verbose:
                print(f"[watch] Found {len(new_paths)} new file(s):")
                for file in new_paths:
                    print(f"- {file}")
            transcribe_fn(new_paths)
        else:
            if verbose:
                print("[watch] No new files – waiting…")
        time.sleep(poll_interval)
