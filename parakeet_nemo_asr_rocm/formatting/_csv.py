"""Formatter for CSV (.csv) output containing segment-level timing."""

from __future__ import annotations

import csv
import io

from parakeet_nemo_asr_rocm.timestamps.models import AlignedResult


def to_csv(result: AlignedResult) -> str:  # noqa: D401
    """Convert an ``AlignedResult`` into CSV string (segment-level).

    Columns: start, end, text
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["start", "end", "text"])
    for seg in result.segments:
        writer.writerow([seg.start, seg.end, seg.text.replace("\n", " ")])
    return buffer.getvalue()
