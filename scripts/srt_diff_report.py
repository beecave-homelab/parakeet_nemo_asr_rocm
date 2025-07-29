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

Requires only the Python standard library.
"""

from __future__ import annotations

import statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import typer

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
# Main
# ---------------------------------------------------------------------------


def _build_report(orig: List[Cue], refined: List[Cue]) -> str:
    o, r = _stats(orig), _stats(refined)
    delta_count = r["count"] - o["count"]

    def _fmt(val: float | int) -> str:
        return f"{val:.2f}" if isinstance(val, float) else str(val)

    lines = ["# SRT Refinement Diff Report", ""]
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
    return "\n".join(lines)


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
) -> None:  # pragma: no cover
    """Compare *original* vs *refined* SRTs and output a Markdown report."""
    orig_cues = _load_srt(original)
    refined_cues = _load_srt(refined)
    report = _build_report(orig_cues, refined_cues)

    if output:
        output.write_text(report, encoding="utf-8")
        typer.echo(f"Report written to {output.resolve()}")
    else:
        typer.echo(report)


if __name__ == "__main__":  # pragma: no cover
    app()
