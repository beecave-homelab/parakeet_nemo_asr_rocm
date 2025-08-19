import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import scripts.srt_diff_report as sdr


@pytest.fixture()
def sample_srts(tmp_path: Path):
    # Craft SRTs to exercise various metrics
    # original: short duration, high cps, too many lines, overlap
    orig = (
        "1\n"
        "00:00:00,000 --> 00:00:00,400\n"  # 0.4s -> short
        "This is a very very very very very long line that exceeds forty two characters.\n"
        "And another long line that makes it three lines.\n"
        "Third line here.\n\n"
        "2\n"
        "00:00:00,300 --> 00:00:01,000\n"  # starts before prev ends -> overlap 0.1s
        "Short.\n"
    )
    # refined: remove overlap, but add small butt gap under buffer; still some violations
    ref = (
        "1\n"
        "00:00:00,000 --> 00:00:00,600\n"  # 0.6s
        "This is a very very very very very long line that exceeds forty two characters.\n"
        "Second line.\n\n"
        "2\n"
        "00:00:00,700 --> 00:00:01,300\n"  # gap = 0.1s
        "Okay.\n"
    )
    o = tmp_path / "orig.srt"
    r = tmp_path / "refined.srt"
    o.write_text(orig, encoding="utf-8")
    r.write_text(ref, encoding="utf-8")
    return o, r


def test_collect_metrics_and_breakdown(sample_srts):
    o_path, r_path = sample_srts
    oc = sdr._load_srt(o_path)
    rc = sdr._load_srt(r_path)

    om = sdr._collect_metrics(oc)
    rm = sdr._collect_metrics(rc)

    # New keys exist
    for key in (
        "cps_under",
        "lines_per_block_over",
        "block_over_soft",
        "block_over_hard",
        "overlap_severity",
        "gap_under_buffer",
    ):
        assert key in om["rates"]
        assert key in rm["rates"]

    # Expect some violations to be present in at least one version
    assert om["rates"]["overlaps"] > 0  # original has overlap
    assert rm["rates"]["gap_under_buffer"] >= 0  # refined may have butt gap

    # Score breakdown structure
    o_score, o_breakdown, w = sdr._score_and_breakdown(om["rates"])  # type: ignore[arg-type]
    r_score, r_breakdown, _ = sdr._score_and_breakdown(rm["rates"])  # type: ignore[arg-type]

    for cat in ("duration", "cps", "line", "block", "hygiene"):
        assert cat in o_breakdown
        assert set(o_breakdown[cat].keys()) == {"weight", "penalty", "contribution"}

    # Weights normalized
    assert 0.99 <= sum(w.values()) <= 1.01

    # Score types
    assert isinstance(o_score, float) and isinstance(r_score, float)


def test_cli_json_schema(sample_srts):
    o_path, r_path = sample_srts
    runner = CliRunner()
    # The Typer app exposes a single root command (no subcommand token)
    result = runner.invoke(
        sdr.app,
        [str(o_path), str(r_path), "--output-format", "json", "--show-violations", "3"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    # Schema fields
    assert payload["schema_version"].startswith("1.")
    assert "generated_at" in payload and isinstance(payload["generated_at"], str)
    assert payload["inputs"]["original"].endswith("orig.srt")
    assert payload["inputs"]["refined"].endswith("refined.srt")

    # Score breakdown
    sb = payload["score_breakdown"]
    assert set(sb.keys()) == {"weights", "original", "refined"}
    for cat in ("duration", "cps", "line", "block", "hygiene"):
        assert cat in sb["original"]
        assert cat in sb["refined"]

    # Violations included due to --show-violations
    assert "violations" in payload
    assert "original" in payload["violations"] and "refined" in payload["violations"]


def test_percentiles_present(sample_srts):
    o_path, r_path = sample_srts
    oc = sdr._load_srt(o_path)
    rc = sdr._load_srt(r_path)
    om = sdr._collect_metrics(oc)
    rm = sdr._collect_metrics(rc)

    for m in (om, rm):
        assert "percentiles" in m
        for key in ("duration", "cps"):
            assert key in m["percentiles"]
            p = m["percentiles"][key]
            assert set(p.keys()) == {"p50", "p90", "p95"}
            # Values are floats
            assert all(isinstance(p[k], float) for k in ("p50", "p90", "p95"))


def test_cli_weights_and_breakdown(sample_srts):
    o_path, r_path = sample_srts
    runner = CliRunner()
    # Default weights
    res_def = runner.invoke(
        sdr.app, [str(o_path), str(r_path), "--output-format", "json"]
    )
    assert res_def.exit_code == 0
    payload_def = json.loads(res_def.stdout)
    w_def = payload_def["score_breakdown"]["weights"]
    assert 0.99 <= sum(w_def.values()) <= 1.01

    # Custom weights: emphasize duration only
    res_w = runner.invoke(
        sdr.app,
        [
            str(o_path),
            str(r_path),
            "--output-format",
            "json",
            "--weights",
            "duration=1,cps=0,line=0,block=0,hygiene=0",
        ],
    )
    assert res_w.exit_code == 0
    payload_w = json.loads(res_w.stdout)
    w = payload_w["score_breakdown"]["weights"]
    assert w["duration"] == pytest.approx(1.0, abs=1e-6)
    assert w["cps"] == pytest.approx(0.0, abs=1e-6)
    assert w["line"] == pytest.approx(0.0, abs=1e-6)
    assert w["block"] == pytest.approx(0.0, abs=1e-6)
    assert w["hygiene"] == pytest.approx(0.0, abs=1e-6)


def test_exit_codes(sample_srts):
    o_path, r_path = sample_srts
    runner = CliRunner()

    # This should fail because requiring score >= 100 is unrealistic
    res_fail = runner.invoke(
        sdr.app,
        [
            str(o_path),
            str(r_path),
            "--output-format",
            "json",
            "--fail-below-score",
            "100",
        ],
    )
    assert res_fail.exit_code == 1

    # This should pass with trivial thresholds
    res_ok = runner.invoke(
        sdr.app,
        [
            str(o_path),
            str(r_path),
            "--output-format",
            "json",
            "--fail-below-score",
            "0",
            "--fail-delta-below",
            "-100",
        ],
    )
    assert res_ok.exit_code == 0
