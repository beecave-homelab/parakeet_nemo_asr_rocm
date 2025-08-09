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
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import typer

# Import thresholds from central constants per project policy
from parakeet_nemo_asr_rocm.utils.constant import (
    MAX_BLOCK_CHARS,
    MAX_BLOCK_CHARS_SOFT,
    MAX_CPS,
    MAX_LINE_CHARS,
    MAX_LINES_PER_BLOCK,
    MAX_SEGMENT_DURATION_SEC,
    MIN_CPS,
    MIN_SEGMENT_DURATION_SEC,
    DISPLAY_BUFFER_SEC,
)

_TIME_SPLITTER = " --> "


@dataclass
class Cue:
    index: int
    start: float  # seconds
    end: float  # seconds
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def cps(self) -> float:
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


def _collect_metrics(cues: Sequence[Cue]) -> Dict[str, object]:
    """Collect violation rates (0..1), counts, aggregates, and per-cue details."""
    n = len(cues)
    if n == 0:
        return {
            "counts": {"total_cues": 0, "overlaps_count": 0},
            "rates": {k: 0.0 for k in ("duration_under","duration_over","cps_over","line_over","block_over","overlaps")},
            "aggregates": {"avg_cps": 0.0, "median_duration": 0.0, "avg_duration": 0.0},
            "per_cue": {k: [] for k in ("duration_under","duration_over","cps_over","line_over","block_over","overlaps")},
        }

    ordered = sorted(cues, key=lambda c: (c.start, c.end))
    duration_under: List[float] = []
    duration_over: List[float] = []
    cps_over: List[float] = []
    line_over: List[float] = []
    block_over: List[float] = []
    overlaps: List[float] = []
    per_cue: Dict[str, List[Tuple[int, float, str]]] = {k: [] for k in ("duration_under","duration_over","cps_over","line_over","block_over","overlaps")}

    prev_end: float | None = None
    for c in ordered:
        dur = c.duration
        cps = c.cps
        under = _clamp01((MIN_SEGMENT_DURATION_SEC - dur) / MIN_SEGMENT_DURATION_SEC)
        over = _clamp01((dur - MAX_SEGMENT_DURATION_SEC) / MAX_SEGMENT_DURATION_SEC)
        duration_under.append(under)
        duration_over.append(over)
        if under > 0:
            per_cue["duration_under"].append((c.index, under, f"{dur:.2f}s < {MIN_SEGMENT_DURATION_SEC:.2f}s"))
        if over > 0:
            per_cue["duration_over"].append((c.index, over, f"{dur:.2f}s > {MAX_SEGMENT_DURATION_SEC:.2f}s"))

        cps_factor = _clamp01((cps - MAX_CPS) / MAX_CPS)
        cps_over.append(cps_factor)
        if cps_factor > 0:
            per_cue["cps_over"].append((c.index, cps_factor, f"{cps:.2f} > {MAX_CPS:.2f} cps"))

        max_line_len = max(_line_lengths(c.text)) if c.text else 0
        line_factor = _clamp01((max_line_len - MAX_LINE_CHARS) / MAX_LINE_CHARS)
        line_over.append(line_factor)
        if line_factor > 0:
            per_cue["line_over"].append((c.index, line_factor, f"line {max_line_len} > {MAX_LINE_CHARS}"))

        blk_chars = _block_chars(c.text)
        block_factor = _clamp01((blk_chars - MAX_BLOCK_CHARS) / MAX_BLOCK_CHARS)
        block_over.append(block_factor)
        if block_factor > 0:
            per_cue["block_over"].append((c.index, block_factor, f"block {blk_chars} > {MAX_BLOCK_CHARS}"))

        if prev_end is not None and c.start < prev_end:
            overlaps.append(1.0)
            per_cue["overlaps"].append((c.index, 1.0, f"{c.start:.2f}s < prev_end {prev_end:.2f}s"))
        else:
            overlaps.append(0.0)
        prev_end = c.end

    def mean(xs: Iterable[float]) -> float:
        xs = list(xs)
        return (sum(xs) / len(xs)) if xs else 0.0

    rates = {
        "duration_under": mean(duration_under),
        "duration_over": mean(duration_over),
        "cps_over": mean(cps_over),
        "line_over": mean(line_over),
        "block_over": mean(block_over),
        "overlaps": mean(overlaps),
    }

    aggregates = {
        "avg_cps": stats.fmean([c.cps for c in ordered]) if ordered else 0.0,
        "median_duration": stats.median([c.duration for c in ordered]) if ordered else 0.0,
        "avg_duration": stats.fmean([c.duration for c in ordered]) if ordered else 0.0,
    }

    counts = {"total_cues": n, "overlaps_count": int(sum(overlaps))}

    return {"counts": counts, "rates": rates, "aggregates": aggregates, "per_cue": per_cue}


def _score(rates: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    """Compute 0–100 readability score from violation rates and weights."""
    default = {"duration": 0.35, "cps": 0.35, "line": 0.15, "block": 0.10, "hygiene": 0.05}
    w = {**default, **(weights or {})}
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}
    duration_penalty = 0.5 * rates.get("duration_under", 0.0) + 0.5 * rates.get("duration_over", 0.0)
    weighted = (
        w["duration"] * duration_penalty
        + w["cps"] * rates.get("cps_over", 0.0)
        + w["line"] * rates.get("line_over", 0.0)
        + w["block"] * rates.get("block_over", 0.0)
        + w["hygiene"] * rates.get("overlaps", 0.0)
    )
    return round(max(0.0, 100.0 * (1.0 - weighted)), 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_report(orig: List[Cue], refined: List[Cue]) -> str:
    # Legacy basic stats
    o, r = _stats(orig), _stats(refined)
    delta_count = r["count"] - o["count"]

    def _fmt(val: float | int) -> str:
        return f"{val:.2f}" if isinstance(val, float) else str(val)

    # Readability metrics and scores
    om = _collect_metrics(orig)
    rm = _collect_metrics(refined)
    o_score = _score(om["rates"])  # type: ignore[arg-type]
    r_score = _score(rm["rates"])  # type: ignore[arg-type]
    d_score = r_score - o_score

    lines = ["# SRT Refinement Diff Report", ""]
    lines.append("## Scores")
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
        ("line_over", "Line Too Long"),
        ("block_over", "Block Too Long"),
        ("overlaps", "Overlaps"),
    ]:
        o_r = float(om["rates"][key])  # type: ignore[index]
        r_r = float(rm["rates"][key])  # type: ignore[index]
        d_r = r_r - o_r
        lines.append(f"| {label} | {pct(o_r)} | {pct(r_r)} | {pct(d_r)} |")
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
        None, "-o", "--output", help="Optional output Markdown file."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON after Markdown."
    ),
    json_only: bool = typer.Option(
        False, "--json-only", help="Emit only JSON (no Markdown)."
    ),
    show_violations: int = typer.Option(
        0, "--show-violations", help="Show top-N worst cues per category."
    ),
) -> None:  # pragma: no cover
    """Compare *original* vs *refined* SRTs and output a report."""
    orig_cues = _load_srt(original)
    refined_cues = _load_srt(refined)

    # Markdown unless JSON-only or user requests JSON file via --output .json
    want_json_payload = json_out or json_only or (output is not None and output.suffix.lower() == ".json")

    if not json_only and not (output and output.suffix.lower() == ".json" and json_out):
        report = _build_report(orig_cues, refined_cues)
        if output:
            output.write_text(report, encoding="utf-8")
            typer.echo(f"Report written to {output.resolve()}")
        else:
            typer.echo(report)

    # JSON payload if requested or output expects JSON
    if want_json_payload:
        om = _collect_metrics(orig_cues)
        rm = _collect_metrics(refined_cues)
        o_score = _score(om["rates"])  # type: ignore[arg-type]
        r_score = _score(rm["rates"])  # type: ignore[arg-type]

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
                "original": {k: topn(v, show_violations) for k, v in om["per_cue"].items()},  # type: ignore[union-attr]
                "refined": {k: topn(v, show_violations) for k, v in rm["per_cue"].items()},   # type: ignore[union-attr]
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
            "original": {"score": o_score, **om},
            "refined": {"score": r_score, **rm},
            "delta": {"score": round(r_score - o_score, 2), "count": (len(refined_cues) - len(orig_cues))},
            "env": env_info,
        }
        if violations_obj is not None:
            payload["violations"] = violations_obj

        payload_str = json.dumps(payload, ensure_ascii=False)
        # If output ends with .json and --json provided, write pretty JSON to file
        if output and output.suffix.lower() == ".json" and (json_out or json_only):
            pretty = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            output.write_text(pretty, encoding="utf-8")
            typer.echo(f"JSON report written to {output.resolve()}")
        else:
            typer.echo(payload_str)


if __name__ == "__main__":  # pragma: no cover
    app()
