"""Directory / pattern watcher that triggers transcription.

This module centralises the ``--watch`` implementation so that the top-level
`cli.py` remains a thin argument parser.

The primary entry point is :func:`watch_and_transcribe`, which blocks and
continuously monitors for new audio files that match the given *patterns*.

It uses :func:`parakeet_rocm.utils.file_utils.resolve_input_paths` to
expand wildcards and directories. Already-transcribed files are skipped by
checking whether an output file would be generated (using
:func:`parakeet_rocm.utils.file_utils.get_unique_filename` with
``overwrite=False``). If the unique filename differs from the intended one, it
assumes a transcription already exists.
"""

from __future__ import annotations

import os
import re
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from types import FrameType

from rich.markup import escape

from parakeet_rocm.models.parakeet import (
    clear_model_cache,
    unload_model_to_cpu,
)
from parakeet_rocm.utils.console import get_console, print_status
from parakeet_rocm.utils.constant import (
    IDLE_CLEAR_TIMEOUT_SEC,
    IDLE_UNLOAD_TIMEOUT_SEC,
    PARAKEET_MODEL_NAME,
)
from parakeet_rocm.utils.file_utils import (
    AUDIO_EXTENSIONS,
    resolve_input_paths,
)

__all__ = ["watch_and_transcribe"]

# Module-level stop event used by the SIGINT handler to cooperatively
# break the poll loop in :func:`watch_and_transcribe`.
_stop_event = threading.Event()


def _default_sig_handler(
    _signum: int,
    _frame: FrameType | None,
    *,
    quiet: bool = False,
) -> None:  # noqa: D401
    """Handle ``SIGINT`` (Ctrl-C) gracefully.

    Sets the module-level stop event so the poll loop in
    :func:`watch_and_transcribe` exits cooperatively.  This avoids
    calling ``sys.exit`` from inside a signal handler, which can
    cause nested ``SystemExit`` during atexit cleanup (e.g. torch
    ``dump_compile_times``).

    Args:
        _signum: Received POSIX signal number (unused).
        _frame: Current stack frame (unused).
        quiet: Whether to suppress the shutdown status message.

    """
    if _stop_event.is_set():
        return
    _stop_event.set()
    if quiet:
        return
    try:
        os.write(sys.stdout.fileno(), "\n[watch] Stopping…\n".encode())
    except (OSError, AttributeError):
        sys.stdout.write("\n[watch] Stopping…\n")
        sys.stdout.flush()


def _needs_transcription(
    path: Path,
    output_dir: Path,
    output_template: str,
    output_format: str,
    watch_base_dirs: Sequence[Path] | None = None,
) -> bool:  # noqa: D401
    """Determine whether an audio file requires a new transcription output.

    Parameters:
        path (Path): Audio file under consideration.
        output_dir (Path): Directory where output files are written.
        output_template (str): Filename template; supports `{parent}` and
            `{filename}` fields.
        output_format (str): Desired output extension (e.g., "txt", "srt").
        watch_base_dirs (Sequence[Path] | None): Optional base directories for
            watch mode. If provided and ``path`` is located beneath one of
            these bases, the output path mirrors the file's subdirectory
            structure under ``output_dir``.

    Returns:
        bool: `True` if no output file exists for `path` yet, `False` otherwise.
    """
    # Mirror subdirectory structure under any matched watch base dir so that
    # detection logic matches the layout used by the transcription pipeline.
    target_dir = output_dir
    if watch_base_dirs:
        for base in watch_base_dirs:
            try:
                rel = path.parent.relative_to(base)
            except Exception:
                continue
            else:
                # If "rel" is not empty (i.e., file is in a subdirectory), mirror it
                if str(rel) != "." and str(rel) != "":
                    target_dir = output_dir / rel
                break

    # Build a regex pattern from the template where {filename} and {parent}
    # are concrete values, while {index} and {date} act as wildcards so that
    # any existing index/date combination is treated as already transcribed.
    escaped_template = re.escape(output_template)
    replacements = {
        "{parent}": re.escape(path.parent.name),
        "{filename}": re.escape(path.stem),
        "{index}": r"\d+",
        "{date}": r"\d{8}",
    }
    for placeholder, replacement in replacements.items():
        escaped_placeholder = re.escape(placeholder)
        escaped_template = escaped_template.replace(escaped_placeholder, replacement)

    pattern = re.compile(rf"^{escaped_template}\." f"{re.escape(output_format)}" r"$")

    if not target_dir.exists():
        return True

    for existing in target_dir.glob(f"*.{output_format}"):
        if pattern.match(existing.name):
            return False

    return True


def watch_and_transcribe(
    *,
    patterns: Iterable[str | Path],
    transcribe_fn: Callable[[list[Path]], None],
    poll_interval: float = 2.0,
    output_dir: Path,
    output_format: str,
    output_template: str,
    watch_base_dirs: Sequence[Path] | None = None,
    audio_exts: Sequence[str] | None = None,
    model_name: str | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """Monitor filesystem patterns and invoke a transcription callback.

    This function polls the given file/directory/glob patterns at a regular
    interval. It determines which matched audio files still require
    transcription based on the configured output directory, template, and
    format, and calls ``transcribe_fn`` with a list of new file paths. When
    idle, it may offload the model to CPU and eventually clear model cache
    after configured idle timeouts.

    Parameters:
        patterns (Iterable[str | Path]): Directory, file, or glob pattern(s)
            to monitor.
        transcribe_fn (Callable[[list[Path]], None]): Callback invoked with a
            list of newly detected audio file paths to transcribe.
        poll_interval (float): Seconds between directory scans.
        output_dir (Path): Directory where transcription outputs are written.
        output_format (str): Output format extension (for example, "txt" or
            "srt").
        output_template (str): Template string used to construct output
            filenames.
        watch_base_dirs (Sequence[Path] | None): Optional base directories
            whose relative subpaths are mirrored under ``output_dir`` when
            computing target output locations.
        audio_exts (Sequence[str] | None): Allowed audio extensions; defaults
            to ``AUDIO_EXTENSIONS`` when ``None``.
        model_name (str | None): Name of the model the watcher should
            offload during idle cleanup. Should match the model selected by
            the CLI (``--model``) so idle cleanup frees the model that is
            actually cached, not the configured default. When ``None``, the
            ``PARAKEET_MODEL_NAME`` default is used.
        verbose: Whether to print watcher debug information.
        quiet: Whether to suppress watcher status output.

    """
    patterns = list(patterns)
    watch_model_name = model_name or PARAKEET_MODEL_NAME
    print_status(
        "watch",
        f"Monitoring {escape(', '.join(map(str, patterns)))} …  (Press Ctrl+C to stop)",
        quiet=quiet,
    )

    original_handler: signal.Handlers | int = signal.SIG_DFL
    try:
        _stop_event.clear()
        if threading.current_thread() is threading.main_thread():
            try:
                original_handler = signal.signal(
                    signal.SIGINT,
                    lambda signum, frame: _default_sig_handler(signum, frame, quiet=quiet),
                )
            except ValueError:  # pragma: no cover
                # In rare environments the signal module may still reject handler changes.
                pass

        seen: set[Path] = set()
        last_activity = time.monotonic()
        unloaded = False  # prevent spamming unload calls/logs while idle
        cleared = False  # whether we already cleared the model cache
        while not _stop_event.is_set():
            all_matches = resolve_input_paths(patterns, audio_exts=audio_exts or AUDIO_EXTENSIONS)
            if verbose:
                print_status(
                    "watch",
                    f"Scan found {len(all_matches)} candidate file(s)",
                    quiet=quiet,
                )
            new_paths: list[Path] = []
            for p in all_matches:
                if p in seen:
                    if verbose:
                        print_status("watch", f"✗ Already processed: {escape(str(p))}", quiet=quiet)
                    continue
                if _needs_transcription(
                    p,
                    output_dir,
                    output_template,
                    output_format,
                    watch_base_dirs=watch_base_dirs,
                ):
                    new_paths.append(p)
                    seen.add(p)
                else:
                    if verbose:
                        print_status(
                            "watch",
                            f"✗ Output exists, skipping: {escape(str(p))}",
                            quiet=quiet,
                        )
            if new_paths:
                if verbose:
                    print_status("watch", f"Found {len(new_paths)} new file(s):", quiet=quiet)
                    for file in new_paths:
                        if not quiet:
                            get_console().print(f"- {escape(str(file))}")
                transcribe_fn(new_paths)
                # Mark activity and reset idle state
                last_activity = time.monotonic()
                if unloaded or cleared:
                    # A new job arrived after idle; the model will be promoted back
                    # to GPU automatically on next get_model(). Reset flag.
                    unloaded = False
                    cleared = False
            else:
                if verbose:
                    print_status("watch", "No new files - waiting…", quiet=quiet)
                # Idle handling: offload model to CPU if idle timeout exceeded
                now = time.monotonic()
                if not unloaded and (now - last_activity) >= IDLE_UNLOAD_TIMEOUT_SEC:
                    try:
                        if verbose:
                            print_status(
                                "watch",
                                f"Idle for >= {IDLE_UNLOAD_TIMEOUT_SEC}s - offloading model to CPU",
                                quiet=quiet,
                            )
                        unload_model_to_cpu(watch_model_name)
                    except Exception:
                        # Never let idle cleanup crash the watcher; retry on
                        # the next idle poll instead of marking it done.
                        print_status(
                            "watch",
                            f"Failed to offload {watch_model_name} to CPU - will retry",
                            quiet=quiet,
                            err=True,
                        )
                    else:
                        unloaded = True
                # If still idle past clear timeout, drop the cache entirely
                if not cleared and (now - last_activity) >= IDLE_CLEAR_TIMEOUT_SEC:
                    if verbose:
                        print_status(
                            "watch",
                            f"Idle for >= {IDLE_CLEAR_TIMEOUT_SEC}s - clearing model cache",
                            quiet=quiet,
                        )
                    # clear_model_cache() reports eviction failure by
                    # returning False instead of raising; a failed clear
                    # leaves ``cleared`` unset and the next idle poll retries.
                    # The except below is a belt-and-braces guard for
                    # unexpected errors (e.g. a future re-raise variant).
                    try:
                        cleared = clear_model_cache()
                    except Exception:
                        # Never let idle cleanup crash the watcher; retry on
                        # the next idle poll instead of marking it done.
                        cleared = False
                    if not cleared:
                        print_status(
                            "watch",
                            "Failed to clear model cache - will retry",
                            quiet=quiet,
                            err=True,
                        )
            time.sleep(poll_interval)
    finally:
        # Restore the previous SIGINT handler so the caller's signal
        # handling is not permanently altered.
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGINT, original_handler)
            except ValueError:  # pragma: no cover
                pass
        _stop_event.clear()
