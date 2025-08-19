"""Generate a before/after readability report for SRT refinement.

Usage
-----
    python -m scripts.srt_diff_report original.srt refined.srt [-o report.md]

The script parses both SRT files, computes:
    • cue counts
    • average / min / max duration
    • percentage of cues under `MIN_SEGMENT_DURATION_SEC`
    • percentage of cues over `MAX_SEGMENT_DURATION_SEC`
    • mean characters-per-second (CPS)

It then prints a short Markdown report to STDOUT or writes to the given `-o/--output` path.

Also supports a readability score (0–100) per SRT, JSON output, and
listing top-N worst violations. Thresholds are sourced from
`parakeet_nemo_asr_rocm.utils.constant` to honor project env settings.
"""

from __future__ import annotations

import json
import statistics as stats
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import typer

# Import thresholds from central constants per project policy
from parakeet_nemo_asr_rocm.utils.constant import (
    DISPLAY_BUFFER_SEC,
    MAX_BLOCK_CHARS,
    MAX_BLOCK_CHARS_SOFT,
    MAX_CPS,
    MAX_LINE_CHARS,
    MAX_LINES_PER_BLOCK,
    MAX_SEGMENT_DURATION_SEC,
    MIN_CPS,
    MIN_SEGMENT_DURATION_SEC,
)

_TIME_SPLITTER = " --> "


@dataclass
class Cue:
    """A single subtitle cue.

    Attributes:
        index: 1-based cue index as appears in the SRT.
        start: Cue start time in seconds.
        end: Cue end time in seconds.
        text: Cue text, possibly multi-line with "\n" separators.
    """

    index: int
    start: float  # seconds
    end: float  # seconds
    text: str

    @property
    def duration(self) -> float:
        """Return cue duration in seconds."""
        return self.end - self.start

    @property
    def cps(self) -> float:
        """Return characters-per-second for the cue text."""
        chars = len(self.text.replace("\n", " "))
        return chars / max(self.duration, 1e-3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_timestamp(ts: str) -> float:
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _load_srt(path: Path | str) -> List[Cue]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks = [b for b in text.strip().split("\n\n") if b]
    cues: list[Cue] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue
        idx = int(lines[0])
        start_s, end_s = lines[1].split(_TIME_SPLITTER)
        start, end = _parse_timestamp(start_s), _parse_timestamp(end_s)
        cues.append(Cue(idx, start, end, "\n".join(lines[2:])))
    return cues


def _stats(cues: Sequence[Cue]) -> dict[str, float]:
    durs = [c.duration for c in cues]
    cps_vals = [c.cps for c in cues]
    return {
        "count": len(cues),
        "avg_dur": stats.fmean(durs) if durs else 0,
        "min_dur": min(durs, default=0),
        "max_dur": max(durs, default=0),
        "avg_cps": stats.fmean(cps_vals) if cps_vals else 0,
    }


# ---------------------------------------------------------------------------
# Readability metrics and scoring
# ---------------------------------------------------------------------------


def _line_lengths(text: str) -> List[int]:
    lines = text.splitlines() or [text]
    return [len(ln) for ln in lines]


def _block_chars(text: str) -> int:
    return len(text.replace("\n", " "))


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0 else (1.0 if x >= 1 else x)


def _percentile(xs: Sequence[float], p: float) -> float:
    """Compute the p-th percentile (0..1) using linear interpolation.

    Returns 0.0 for empty input.
    """
    if not xs:
        return 0.0
    if p <= 0:
        return float(min(xs))
    if p >= 1:
        return float(max(xs))
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def _collect_metrics(cues: Sequence[Cue]) -> Dict[str, object]:
    """Collect violation rates (0..1), counts, aggregates, and per-cue details."""
    n = len(cues)
    if n == 0:
        return {
            "counts": {
                "total_cues": 0,
                "overlaps_count": 0,
                "gaps_under_buffer_count": 0,
            },
            "rates": {
                k: 0.0
                for k in (
                    "duration_under",
                    "duration_over",
                    "cps_over",
                    "cps_under",
                    "line_over",
                    "lines_per_block_over",
                    "block_over",
                    "block_over_soft",
                    "block_over_hard",
                    "overlaps",
                    "overlap_severity",
                    "gap_under_buffer",
                )
            },
            "aggregates": {"avg_cps": 0.0, "median_duration": 0.0, "avg_duration": 0.0},
            "per_cue": {
                k: []
                for k in (
                    "duration_under",
                    "duration_over",
                    "cps_over",
                    "cps_under",
                    "line_over",
                    "lines_per_block_over",
                    "block_over",
                    "block_over_soft",
                    "block_over_hard",
                    "overlaps",
                    "overlap_severity",
                    "gap_under_buffer",
                )
            },
        }

    ordered = sorted(cues, key=lambda c: (c.start, c.end))
    duration_under: List[float] = []
    duration_over: List[float] = []
    cps_over: List[float] = []
    cps_under: List[float] = []
    line_over: List[float] = []
    lines_per_block_over: List[float] = []
    block_over: List[float] = []  # legacy alias → hard limit
    block_over_soft: List[float] = []
    block_over_hard: List[float] = []
    overlaps: List[float] = []  # binary
    overlap_severity: List[float] = []
    gap_under_buffer: List[float] = []
    per_cue: Dict[str, List[Tuple[int, float, str]]] = {
        k: []
        for k in (
            "duration_under",
            "duration_over",
            "cps_over",
            "cps_under",
            "line_over",
            "lines_per_block_over",
            "block_over",
            "block_over_soft",
            "block_over_hard",
            "overlaps",
            "overlap_severity",
            "gap_under_buffer",
        )
    }

    prev_end: float | None = None
    for c in ordered:
        dur = c.duration
        cps = c.cps
        under = _clamp01((MIN_SEGMENT_DURATION_SEC - dur) / MIN_SEGMENT_DURATION_SEC)
        over = _clamp01((dur - MAX_SEGMENT_DURATION_SEC) / MAX_SEGMENT_DURATION_SEC)
        duration_under.append(under)
        duration_over.append(over)
        if under > 0:
            per_cue["duration_under"].append(
                (c.index, under, f"{dur:.2f}s < {MIN_SEGMENT_DURATION_SEC:.2f}s")
            )
        if over > 0:
            per_cue["duration_over"].append(
                (c.index, over, f"{dur:.2f}s > {MAX_SEGMENT_DURATION_SEC:.2f}s")
            )

        cps_factor = _clamp01((cps - MAX_CPS) / MAX_CPS)
        cps_over.append(cps_factor)
        if cps_factor > 0:
            per_cue["cps_over"].append(
                (c.index, cps_factor, f"{cps:.2f} > {MAX_CPS:.2f} cps")
            )

        cps_under_factor = _clamp01((MIN_CPS - cps) / max(MIN_CPS, 1e-6))
        cps_under.append(cps_under_factor)
        if cps_under_factor > 0:
            per_cue["cps_under"].append(
                (c.index, cps_under_factor, f"{cps:.2f} < {MIN_CPS:.2f} cps")
            )

        max_line_len = max(_line_lengths(c.text)) if c.text else 0
        line_factor = _clamp01((max_line_len - MAX_LINE_CHARS) / MAX_LINE_CHARS)
        line_over.append(line_factor)
        if line_factor > 0:
            per_cue["line_over"].append(
                (c.index, line_factor, f"line {max_line_len} > {MAX_LINE_CHARS}")
            )

        # Lines per block violation (e.g., > 2 lines)
        line_count = len(c.text.splitlines()) if c.text else 0
        lines_over_factor = _clamp01(
            (line_count - MAX_LINES_PER_BLOCK) / max(MAX_LINES_PER_BLOCK, 1)
        )
        lines_per_block_over.append(lines_over_factor)
        if lines_over_factor > 0:
            per_cue["lines_per_block_over"].append(
                (
                    c.index,
                    lines_over_factor,
                    f"lines {line_count} > {MAX_LINES_PER_BLOCK}",
                )
            )

        blk_chars = _block_chars(c.text)
        block_soft_factor = _clamp01(
            (blk_chars - MAX_BLOCK_CHARS_SOFT) / max(MAX_BLOCK_CHARS_SOFT, 1)
        )
        block_hard_factor = _clamp01(
            (blk_chars - MAX_BLOCK_CHARS) / max(MAX_BLOCK_CHARS, 1)
        )
        block_over_soft.append(block_soft_factor)
        block_over_hard.append(block_hard_factor)
        # Legacy 'block_over' tracks hard limit for backward compatibility
        block_over.append(block_hard_factor)
        if block_soft_factor > 0:
            per_cue["block_over_soft"].append(
                (
                    c.index,
                    block_soft_factor,
                    f"block {blk_chars} > {MAX_BLOCK_CHARS_SOFT}",
                )
            )
        if block_hard_factor > 0:
            per_cue["block_over_hard"].append(
                (c.index, block_hard_factor, f"block {blk_chars} > {MAX_BLOCK_CHARS}")
            )
            per_cue["block_over"].append(
                (c.index, block_hard_factor, f"block {blk_chars} > {MAX_BLOCK_CHARS}")
            )

        if prev_end is not None:
            gap = c.start - prev_end
            if gap < 0:
                # Binary overlap indicator and severity relative to DISPLAY_BUFFER_SEC
                overlaps.append(1.0)
                sev = _clamp01(abs(gap) / max(DISPLAY_BUFFER_SEC, 1e-6))
                overlap_severity.append(sev)
                per_cue["overlaps"].append(
                    (
                        c.index,
                        1.0,
                        f"{c.start:.2f}s < prev_end {prev_end:.2f}s (overlap {gap:.2f}s)",
                    )
                )
                per_cue["overlap_severity"].append(
                    (c.index, sev, f"overlap {abs(gap):.3f}s")
                )
            else:
                overlaps.append(0.0)
                # Check for gaps under display buffer (butt joins)
                if 0 <= gap < DISPLAY_BUFFER_SEC:
                    butt = _clamp01(
                        (DISPLAY_BUFFER_SEC - gap) / max(DISPLAY_BUFFER_SEC, 1e-6)
                    )
                    gap_under_buffer.append(butt)
                    per_cue["gap_under_buffer"].append(
                        (
                            c.index,
                            butt,
                            f"gap {gap:.3f}s < buffer {DISPLAY_BUFFER_SEC:.3f}s",
                        )
                    )
                else:
                    gap_under_buffer.append(0.0)
        prev_end = c.end

    def mean(xs: Iterable[float]) -> float:
        xs = list(xs)
        return (sum(xs) / len(xs)) if xs else 0.0

    rates = {
        "duration_under": mean(duration_under),
        "duration_over": mean(duration_over),
        "cps_over": mean(cps_over),
        "cps_under": mean(cps_under),
        "line_over": mean(line_over),
        "lines_per_block_over": mean(lines_per_block_over),
        "block_over": mean(block_over),
        "block_over_soft": mean(block_over_soft),
        "block_over_hard": mean(block_over_hard),
        "overlaps": mean(overlaps),
        "overlap_severity": mean(overlap_severity),
        "gap_under_buffer": mean(gap_under_buffer),
    }

    cps_list = [c.cps for c in ordered]
    dur_list = [c.duration for c in ordered]
    aggregates = {
        "avg_cps": stats.fmean(cps_list) if cps_list else 0.0,
        "median_duration": stats.median(dur_list) if dur_list else 0.0,
        "avg_duration": stats.fmean(dur_list) if dur_list else 0.0,
    }

    counts = {
        "total_cues": n,
        "overlaps_count": int(sum(overlaps)),
        "gaps_under_buffer_count": sum(1 for x in gap_under_buffer if x > 0),
    }

    percentiles = {
        "duration": {
            "p50": _percentile(dur_list, 0.50),
            "p90": _percentile(dur_list, 0.90),
            "p95": _percentile(dur_list, 0.95),
        },
        "cps": {
            "p50": _percentile(cps_list, 0.50),
            "p90": _percentile(cps_list, 0.90),
            "p95": _percentile(cps_list, 0.95),
        },
    }

    return {
        "counts": counts,
        "rates": rates,
        "aggregates": aggregates,
        "percentiles": percentiles,
        "per_cue": per_cue,
    }


def _score_and_breakdown(
    rates: Dict[str, float], weights: Dict[str, float] | None = None
) -> Tuple[float, Dict[str, Dict[str, float]], Dict[str, float]]:
    """Compute 0–100 score and category breakdown.

    Returns (score, breakdown, normalized_weights).
    """
    default = {
        "duration": 0.35,
        "cps": 0.35,
        "line": 0.15,
        "block": 0.10,
        "hygiene": 0.05,
    }
    w = {**default, **(weights or {})}
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}

    duration_penalty = 0.5 * rates.get("duration_under", 0.0) + 0.5 * rates.get(
        "duration_over", 0.0
    )
    cps_penalty = 0.5 * rates.get("cps_over", 0.0) + 0.5 * rates.get("cps_under", 0.0)
    line_penalty = 0.7 * rates.get("line_over", 0.0) + 0.3 * rates.get(
        "lines_per_block_over", 0.0
    )
    block_penalty = 0.7 * rates.get(
        "block_over_hard", rates.get("block_over", 0.0)
    ) + 0.3 * rates.get("block_over_soft", 0.0)
    hygiene_penalty = 0.7 * rates.get(
        "overlap_severity", rates.get("overlaps", 0.0)
    ) + 0.3 * rates.get("gap_under_buffer", 0.0)

    components = {
        "duration": duration_penalty,
        "cps": cps_penalty,
        "line": line_penalty,
        "block": block_penalty,
        "hygiene": hygiene_penalty,
    }

    weighted_sum = sum(w[k] * v for k, v in components.items())
    score = round(max(0.0, 100.0 * (1.0 - weighted_sum)), 2)

    breakdown = {
        k: {
            "weight": round(w[k], 4),
            "penalty": round(v, 4),
            "contribution": round(w[k] * v, 4),
        }
        for k, v in components.items()
    }
    return score, breakdown, w


def _score(rates: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    """Compute 0–100 readability score from violation rates and weights."""
    score, _breakdown, _w = _score_and_breakdown(rates, weights)
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_report(
    orig: List[Cue],
    refined: List[Cue],
    show_violations: int = 0,
    weights: Dict[str, float] | None = None,
) -> str:
    # Legacy basic stats
    o, r = _stats(orig), _stats(refined)
    delta_count = r["count"] - o["count"]

    def _fmt(val: float | int) -> str:
        return f"{val:.2f}" if isinstance(val, float) else str(val)

    # Readability metrics and scores
    om = _collect_metrics(orig)
    rm = _collect_metrics(refined)
    o_rates = om["rates"]  # type: ignore[arg-type]
    r_rates = rm["rates"]  # type: ignore[arg-type]
    o_score, o_breakdown, w_norm = _score_and_breakdown(o_rates, weights)
    r_score, r_breakdown, _ = _score_and_breakdown(r_rates, weights)
    d_score = r_score - o_score

    lines = ["# SRT Refinement Diff Report", ""]
    lines.append("## Scores")
    lines.append("")
    # Percentiles section
    lines.append("### Percentiles")
    lines.append("")
    lines.append("| Metric | P50 | P90 | P95 |")
    lines.append("| ------ | ---:| ---:| ---:|")
    od = om["percentiles"]["duration"]  # type: ignore[index]
    rd = rm["percentiles"]["duration"]  # type: ignore[index]
    oc = om["percentiles"]["cps"]  # type: ignore[index]
    rc = rm["percentiles"]["cps"]  # type: ignore[index]
    lines.append(
        f"| Duration (s) – Original | {od['p50']:.2f} | {od['p90']:.2f} | {od['p95']:.2f} |"
    )
    lines.append(
        f"| Duration (s) – Refined  | {rd['p50']:.2f} | {rd['p90']:.2f} | {rd['p95']:.2f} |"
    )
    lines.append(
        f"| CPS – Original          | {oc['p50']:.2f} | {oc['p90']:.2f} | {oc['p95']:.2f} |"
    )
    lines.append(
        f"| CPS – Refined           | {rc['p50']:.2f} | {rc['p90']:.2f} | {rc['p95']:.2f} |"
    )
    lines.append("")
    lines.append("| File | Score |")
    lines.append("| ---- | -----:|")
    lines.append(f"| Original | {o_score:.2f} |")
    lines.append(f"| Refined | {r_score:.2f} |")
    lines.append("")
    lines.append(f"**Δ Score:** {d_score:+.2f} (higher is better)")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Original | Refined | Δ |")
    lines.append("| ------ | -------- | ------- | --- |")
    for key, label in [
        ("count", "Cue Count"),
        ("avg_dur", "Avg. Duration (s)"),
        ("min_dur", "Min Duration (s)"),
        ("max_dur", "Max Duration (s)"),
        ("avg_cps", "Avg. CPS"),
    ]:
        delta = r[key] - o[key]
        delta_str = f"{delta:+.2f}" if isinstance(delta, float) else f"{delta:+d}"
        lines.append(f"| {label} | {_fmt(o[key])} | {_fmt(r[key])} | {delta_str} |")
    lines.append("")
    lines.append(
        "**Cue count change:** "
        + ("reduced" if delta_count < 0 else "increased")
        + f" by {abs(delta_count)} cues."
    )
    lines.append("")
    # Environment thresholds section
    lines.append("## Environment (Thresholds)")
    lines.append("")
    lines.append("| Variable | Value |")
    lines.append("| -------- | -----:|")
    lines.append(f"| MIN_CPS | {MIN_CPS} |")
    lines.append(f"| MAX_CPS | {MAX_CPS} |")
    lines.append(f"| MIN_SEGMENT_DURATION_SEC | {MIN_SEGMENT_DURATION_SEC} |")
    lines.append(f"| MAX_SEGMENT_DURATION_SEC | {MAX_SEGMENT_DURATION_SEC} |")
    lines.append(f"| DISPLAY_BUFFER_SEC | {DISPLAY_BUFFER_SEC} |")
    lines.append(f"| MAX_LINE_CHARS | {MAX_LINE_CHARS} |")
    lines.append(f"| MAX_LINES_PER_BLOCK | {MAX_LINES_PER_BLOCK} |")
    lines.append(f"| MAX_BLOCK_CHARS | {MAX_BLOCK_CHARS} |")
    lines.append(f"| MAX_BLOCK_CHARS_SOFT | {MAX_BLOCK_CHARS_SOFT} |")
    lines.append("")
    lines.append("### Violation Rates (%)")
    lines.append("")
    lines.append("| Category | Original | Refined | Δ |")
    lines.append("| -------- | --------:| -------:| ---:|")

    def pct(x: float) -> str:
        return f"{(100.0 * x):.2f}"

    for key, label in [
        ("duration_under", "Short Durations"),
        ("duration_over", "Long Durations"),
        ("cps_over", "High CPS"),
        ("cps_under", "Low CPS"),
        ("line_over", "Line Too Long"),
        ("lines_per_block_over", "Too Many Lines per Block"),
        ("block_over_soft", "Block Over (Soft)"),
        ("block_over", "Block Over (Hard)"),
        ("overlaps", "Overlaps (Binary)"),
        ("overlap_severity", "Overlap Severity"),
        ("gap_under_buffer", "Gap Under Buffer"),
    ]:
        o_r = float(om["rates"][key])  # type: ignore[index]
        r_r = float(rm["rates"][key])  # type: ignore[index]
        d_r = r_r - o_r
        lines.append(f"| {label} | {pct(o_r)} | {pct(r_r)} | {pct(d_r)} |")

    # Penalty breakdown table
    lines.append("")
    lines.append("### Penalty Breakdown (weights · penalties)")
    lines.append("")
    lines.append(
        "| Category | Weight | Original Penalty | Refined Penalty | Δ Contribution |"
    )
    lines.append(
        "| -------- | -----: | ---------------: | --------------: | -------------: |"
    )
    for k, label in [
        ("duration", "Duration"),
        ("cps", "CPS"),
        ("line", "Line"),
        ("block", "Block"),
        ("hygiene", "Hygiene"),
    ]:
        ow = o_breakdown[k]["weight"]
        op = o_breakdown[k]["penalty"]
        rp = r_breakdown[k]["penalty"]
        # contribution delta uses normalized weight
        dcontrib = w_norm[k] * (rp - op)
        lines.append(f"| {label} | {ow:.2f} | {op:.3f} | {rp:.3f} | {dcontrib:+.3f} |")

    # Optional: Top-N violations per category (Original vs Refined)
    if show_violations > 0:

        def topn(
            lst: List[Tuple[int, float, str]], n: int
        ) -> List[Tuple[int, float, str]]:
            return [
                t
                for t in sorted(lst, key=lambda t: (-t[1], t[0]))[: max(0, n)]
                if t[1] > 0
            ]

        lines.append("")
        lines.append(f"### Top {show_violations} Violations (Original)")
        lines.append("")
        lines.append("| Category | Index | Factor | Detail |")
        lines.append("| -------- | ----: | -----: | ------ |")
        for cat in om["per_cue"]:  # type: ignore[union-attr]
            for idx, factor, detail in topn(
                om["per_cue"][cat],  # type: ignore[index]
                show_violations,
            ):
                lines.append(f"| {cat} | {idx} | {factor:.3f} | {detail} |")

        lines.append("")
        lines.append(f"### Top {show_violations} Violations (Refined)")
        lines.append("")
        lines.append("| Category | Index | Factor | Detail |")
        lines.append("| -------- | ----: | -----: | ------ |")
        for cat in rm["per_cue"]:  # type: ignore[union-attr]
            for idx, factor, detail in topn(
                rm["per_cue"][cat],  # type: ignore[index]
                show_violations,
            ):
                lines.append(f"| {cat} | {idx} | {factor:.3f} | {detail} |")
    # Ensure the report ends with a single trailing newline for Markdown hygiene
    return "\n".join(lines) + "\n"


app = typer.Typer(
    add_completion=False,
    help="Generate a before/after readability report for SRT refinement.",
)


@app.command()
def diff(
    original: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to the original SRT file."
    ),
    refined: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to the refined SRT file."
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Optional output file (Markdown or JSON).",
    ),
    output_format: str = typer.Option(
        "markdown",
        "--output-format",
        help="Select output format.",
        metavar="<json|markdown>",
        show_default=True,
    ),
    show_violations: int = typer.Option(
        0, "--show-violations", help="Show top-N worst cues per category."
    ),
    weights_str: str = typer.Option(
        "",
        "--weights",
        help=(
            "Custom score weights as 'duration=0.35,cps=0.35,line=0.15,block=0.1,hygiene=0.05'.\n"
            "Keys: duration,cps,line,block,hygiene. Values will be normalized."
        ),
    ),
    fail_below_score: float | None = typer.Option(
        None,
        "--fail-below-score",
        min=0.0,
        max=100.0,
        help="Exit non-zero if Refined score is below this threshold (0..100).",
    ),
    fail_delta_below: float | None = typer.Option(
        None,
        "--fail-delta-below",
        help="Exit non-zero if (Refined - Original) score delta is below this threshold.",
    ),
) -> None:  # pragma: no cover
    """Compare *original* vs *refined* SRTs and output a report."""
    orig_cues = _load_srt(original)
    refined_cues = _load_srt(refined)

    # Parse weights
    weights: Dict[str, float] | None = None
    if weights_str:
        weights = {}
        allowed = {"duration", "cps", "line", "block", "hygiene"}
        for part in weights_str.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise typer.BadParameter(
                    f"Invalid weights token: '{part}'. Expected key=value."
                )
            k, v = part.split("=", 1)
            k = k.strip()
            if k not in allowed:
                raise typer.BadParameter(
                    f"Unknown weight key: '{k}'. Allowed: {sorted(allowed)}"
                )
            try:
                weights[k] = float(v)
            except ValueError as exc:
                raise typer.BadParameter(
                    f"Invalid weight value for '{k}': '{v}'"
                ) from exc

    want_json_payload = output_format.lower() == "json"
    want_markdown = output_format.lower() == "markdown"

    if want_markdown:
        report = _build_report(
            orig_cues, refined_cues, show_violations=show_violations, weights=weights
        )
        if output:
            output.write_text(report, encoding="utf-8")
            typer.echo(f"Report written to {output.resolve()}")
        else:
            typer.echo(report)

    if want_json_payload:
        om = _collect_metrics(orig_cues)
        rm = _collect_metrics(refined_cues)
        om_rates = om["rates"]  # type: ignore[arg-type]
        rm_rates = rm["rates"]  # type: ignore[arg-type]
        o_score, o_breakdown, w_norm = _score_and_breakdown(om_rates, weights)
        r_score, r_breakdown, _ = _score_and_breakdown(rm_rates, weights)

        def topn(lst: List[Tuple[int, float, str]], n: int) -> List[Dict[str, object]]:
            lst_sorted = sorted(lst, key=lambda t: (-t[1], t[0]))
            return [
                {"index": idx, "factor": round(factor, 4), "detail": detail}
                for idx, factor, detail in lst_sorted[: max(0, n)]
                if factor > 0
            ]

        violations_obj = None
        if show_violations > 0:
            violations_obj = {
                "original": {
                    k: topn(v, show_violations) for k, v in om["per_cue"].items()
                },  # type: ignore[union-attr]
                "refined": {
                    k: topn(v, show_violations) for k, v in rm["per_cue"].items()
                },  # type: ignore[union-attr]
            }

        # Include environment thresholds in JSON as well
        env_info = {
            "MIN_CPS": MIN_CPS,
            "MAX_CPS": MAX_CPS,
            "MIN_SEGMENT_DURATION_SEC": MIN_SEGMENT_DURATION_SEC,
            "MAX_SEGMENT_DURATION_SEC": MAX_SEGMENT_DURATION_SEC,
            "DISPLAY_BUFFER_SEC": DISPLAY_BUFFER_SEC,
            "MAX_LINE_CHARS": MAX_LINE_CHARS,
            "MAX_LINES_PER_BLOCK": MAX_LINES_PER_BLOCK,
            "MAX_BLOCK_CHARS": MAX_BLOCK_CHARS,
            "MAX_BLOCK_CHARS_SOFT": MAX_BLOCK_CHARS_SOFT,
        }

        payload = {
            "schema_version": "1.1",
            "generated_at": datetime.now().isoformat(),
            "inputs": {"original": str(original), "refined": str(refined)},
            "original": {"score": o_score, **om},
            "refined": {"score": r_score, **rm},
            "delta": {
                "score": round(r_score - o_score, 2),
                "count": (len(refined_cues) - len(orig_cues)),
            },
            "env": env_info,
            "score_breakdown": {
                "weights": w_norm,
                "original": o_breakdown,
                "refined": r_breakdown,
            },
        }
        if violations_obj is not None:
            payload["violations"] = violations_obj

        payload_str = json.dumps(payload, ensure_ascii=False)
        # If output ends with .json and --json provided, write pretty JSON to file
        if output and output.suffix.lower() == ".json":
            payload_str = (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        if output:
            output.write_text(payload_str, encoding="utf-8")
            typer.echo(json.dumps({"written": str(output.resolve())}))
        else:
            typer.echo(payload_str)

    # CI/exit code handling
    if (
        weights_str
        or fail_below_score is not None
        or fail_delta_below is not None
        or want_json_payload
    ):
        # Reuse computed scores if available; otherwise recompute quickly
        if "o_score" not in locals():
            om = _collect_metrics(orig_cues)
            rm = _collect_metrics(refined_cues)
            o_score, _ob, _ = _score_and_breakdown(om["rates"], weights)  # type: ignore[arg-type]
            r_score, _rb, _ = _score_and_breakdown(rm["rates"], weights)  # type: ignore[arg-type]
        delta_score = r_score - o_score  # type: ignore[operator]
        should_fail = False
        if fail_below_score is not None and r_score < fail_below_score:  # type: ignore[operator]
            should_fail = True
        if fail_delta_below is not None and delta_score < fail_delta_below:
            should_fail = True
        if should_fail:
            raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
