"""Subtitle refinement utilities.

This module provides a `SubtitleRefiner` that post-processes SRT files to meet
readability guidelines defined in `utils.constant`:

* Minimum cue duration (`MIN_SEGMENT_DURATION_SEC`, default 1.0 s)
* Maximum characters per second (`MAX_CPS`, default 17)
* Mandatory gap between cues in frames (`GAP_FRAMES`, default 2)
* Video frame-rate (`FPS`, default 25)
* Maximum characters per line (`MAX_LINE_CHARS`, default 42)

All values fall back to the defaults above if *utils.constant* does not expose
an attribute (keeps the refiner usable even before constants are added).

No changes are made to the actual words – only timing and line breaks are
altered. This ensures the original ASR text remains intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

# ---------------------------------------------------------------------------
# Constants – import from utils.constant with graceful fallbacks
# ---------------------------------------------------------------------------
from parakeet_nemo_asr_rocm.utils import constant as _c  # pylint: disable=import-error

BOUNDARY_CHARS = getattr(_c, "BOUNDARY_CHARS", ".?!…")
CLAUSE_CHARS = getattr(_c, "CLAUSE_CHARS", ",;:")
SOFT_BOUNDARY_WORDS = getattr(_c, "SOFT_BOUNDARY_WORDS", ("and", "but", "that"))
INTERJECTION_WHITELIST = getattr(
    _c, "INTERJECTION_WHITELIST", ("whoa", "wow", "what", "oh", "hey", "ah")
)

MAX_CPS: int = getattr(_c, "MAX_CPS", 17)
MAX_LINE_CHARS: int = getattr(_c, "MAX_LINE_CHARS", 42)
MIN_DUR: float = getattr(_c, "MIN_SEGMENT_DURATION_SEC", 1.0)
FPS: int = getattr(_c, "FPS", 25)  # video frame-rate (assumed constant)
GAP_FRAMES: int = getattr(_c, "GAP_FRAMES", 2)
GAP_SEC: float = GAP_FRAMES / FPS

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(r"^(\d\d):(\d\d):(\d\d),(\d\d\d)$")


@dataclass
class Cue:
    """Simple container for an SRT cue."""

    index: int
    start: float  # seconds
    end: float  # seconds
    text: str

    # ---------------------------------------------------------------------
    # Formatting helpers
    # ---------------------------------------------------------------------
    def to_srt(self) -> str:
        """Render cue back to SRT block."""
        return (
            f"{self.index}\n{_format_ts(self.start)} --> {_format_ts(self.end)}\n"
            f"{self.text.strip()}\n"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class SubtitleRefiner:
    """Refines SRT cues for readability.

    Workflow:
        refiner = SubtitleRefiner()
        cues = refiner.load_srt(input_path)
        refined = refiner.refine(cues)
        refiner.save_srt(refined, output_path)
    """

    def __init__(
        self,
        max_cps: int = MAX_CPS,
        min_dur: float = MIN_DUR,
        gap_frames: int = GAP_FRAMES,
        fps: int = FPS,
        max_line_chars: int = MAX_LINE_CHARS,
        max_lines_per_block: int = getattr(_c, "MAX_LINES_PER_BLOCK", 2),
    ) -> None:
        self.max_cps = max_cps
        self.min_dur = min_dur
        self.gap = gap_frames / fps
        self.max_line_chars = max_line_chars
        self.max_lines_per_block = max_lines_per_block
        self.max_block_chars = getattr(_c, "MAX_BLOCK_CHARS", max_line_chars * 2)
        self.max_dur = getattr(_c, "MAX_SEGMENT_DURATION_SEC", 6.0)

    # ---------------------------------------------------------------------
    # I/O
    # ---------------------------------------------------------------------
    def load_srt(self, path: Path | str) -> List[Cue]:
        """Load cues from an SRT file."""
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r"\n{2,}", text.strip())
        cues: list[Cue] = []
        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 2:
                continue
            index = int(lines[0].strip())
            start_str, end_str = lines[1].split(" --> ")
            start = _parse_ts(start_str)
            end = _parse_ts(end_str)
            body = "\n".join(lines[2:])
            cues.append(Cue(index=index, start=start, end=end, text=body))
        return cues

    def save_srt(self, cues: Sequence[Cue], path: Path | str) -> None:
        """Write cues back to an SRT file."""
        out_lines = []
        for i, cue in enumerate(cues, start=1):
            cue.index = i  # re-index
            out_lines.append(cue.to_srt())
        Path(path).write_text("\n\n".join(out_lines) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------------
    # Core refinement
    # ---------------------------------------------------------------------
    def refine(self, cues: List[Cue]) -> List[Cue]:
        """Return a refined list of cues."""
        if not cues:
            return []

        cues = self._merge_short_or_fast(cues)
        cues = self._enforce_gaps(cues)
        cues = self._wrap_lines(cues)
        return cues

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _merge_short_or_fast(self, cues: List[Cue]) -> List[Cue]:
        merged: list[Cue] = []
        i = 0
        while i < len(cues):
            current = cues[i]
            while i + 1 < len(cues):
                nxt = cues[i + 1]
                dur = current.end - current.start
                interjection = _is_interjection(current.text)
                cps = len(current.text.replace("\n", " ")) / max(dur, 1e-3)
                gap = nxt.start - current.end

                # Prospective merged values
                prospective_end = nxt.end
                prospective_text = f"{current.text.strip()} \n{nxt.text.strip()}"
                prospective_dur = prospective_end - current.start

                if (
                    (
                        (dur < self.min_dur and not interjection)
                        or cps > self.max_cps
                        or gap < self.gap
                    )
                    and prospective_dur <= self.max_dur
                    and len(prospective_text) <= self.max_block_chars
                    and _is_boundary(prospective_text)  # only merge if boundary reached
                ):
                    # merge
                    current.end = prospective_end
                    current.text = prospective_text
                    i += 1
                else:
                    break
            merged.append(current)
            i += 1
        return merged

    def _enforce_gaps(self, cues: List[Cue]) -> List[Cue]:
        """Ensure a minimum gap between consecutive cues by shifting timings."""
        for prev, curr in zip(cues, cues[1:]):
            required_start = prev.end + self.gap
            if curr.start < required_start:
                shift = required_start - curr.start
                curr.start += shift
                curr.end += shift
        return cues

    def _wrap_lines(self, cues: List[Cue]) -> List[Cue]:
        """Wrap lines so that each line ≤ max_line_chars and ≤ max_lines_per_block."""
        wrapped_cues: list[Cue] = []
        for cue in cues:
            words = cue.text.replace("\n", " ").split()
            new_lines: list[str] = []
            line: list[str] = []
            for word in words:
                prospective = " ".join([*line, word]) if line else word
                if len(prospective) <= self.max_line_chars:
                    line.append(word)
                else:
                    new_lines.append(" ".join(line))
                    line = [word]
            if line:
                new_lines.append(" ".join(line))
            # Smarter split: prefer splitting at punctuation or soft boundary words
            if len(new_lines) > self.max_lines_per_block:
                joined = " ".join(words)
                # try find punctuation near middle
                mid = len(joined) // 2
                punct_idx = max(
                    joined.rfind(ch, 0, mid) for ch in BOUNDARY_CHARS + CLAUSE_CHARS
                )
                if punct_idx == -1:
                    # try soft boundary word index
                    for w in SOFT_BOUNDARY_WORDS:
                        idx = joined.rfind(f" {w} ", 0, mid)
                        if idx > punct_idx:
                            punct_idx = idx + len(w) // 2
                if punct_idx == -1:
                    punct_idx = joined.rfind(" ", 0, mid)
                    if punct_idx == -1:
                        punct_idx = joined.find(" ", mid)
                new_lines = [
                    joined[:punct_idx].strip(),
                    joined[punct_idx + 1 :].strip(),
                ]
            if len(new_lines) > self.max_lines_per_block:
                joined = " ".join(words)
                mid = len(joined) // 2
                split_idx = joined.rfind(" ", 0, mid)
                if split_idx == -1:
                    split_idx = joined.find(" ", mid)
                new_lines = [
                    joined[:split_idx].strip(),
                    joined[split_idx + 1 :].strip(),
                ]
            cue.text = "\n".join(new_lines)
            wrapped_cues.append(cue)
        return wrapped_cues


def _is_interjection(text: str) -> bool:
    """Return True if *text* is a standalone interjection allowed to stay short."""
    pure = re.sub(r"[^A-Za-z]", "", text).lower()
    return pure in INTERJECTION_WHITELIST


def _is_boundary(text: str) -> bool:
    """Return True if *text* ends with a sentence/clause boundary."""
    stripped = text.rstrip()
    if not stripped:
        return False
    last = stripped[-1]
    if last in BOUNDARY_CHARS or last in CLAUSE_CHARS:
        return True
    # Soft boundary word
    last_word = stripped.split()[-1].lower().strip(",;:.")
    return last_word in SOFT_BOUNDARY_WORDS


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _parse_ts(ts: str) -> float:
    """Convert timestamp ``HH:MM:SS,mmm`` to seconds."""
    match = _TIME_RE.match(ts.strip())
    if not match:
        raise ValueError(f"Invalid timestamp '{ts}'")
    hh, mm, ss, ms = map(int, match.groups())
    return hh * 3600 + mm * 60 + ss + ms / 1000.0


def _format_ts(seconds: float) -> str:
    """Convert seconds to ``HH:MM:SS,mmm`` string."""
    ms_total = int(round(seconds * 1000))
    hh, rem = divmod(ms_total, 3600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
