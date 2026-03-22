"""Tests for preflight calibration reports."""

from __future__ import annotations

from pathlib import Path

from chesslab.validation import load_support_catalog, run_preflight_calibration


def test_preflight_on_real_fixture_returns_report() -> None:
    catalog = load_support_catalog()
    profile = catalog.get_profile("chesscom_green_white")
    assert profile is not None

    images = sorted(Path("tests/integration/fixtures/images").glob("*.png"))
    report = run_preflight_calibration(images, profile)

    assert report.setup_id == "chesscom_green_white"
    assert report.metrics.frame_count == len(images)
    assert report.metrics.detection_rate >= 0.9
    assert report.passed is False
    assert any("Ready frame rate" in blocker for blocker in report.blockers)
