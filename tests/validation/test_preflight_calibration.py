"""Tests for preflight calibration reports."""

from __future__ import annotations

from pathlib import Path

from chesslab.validation import load_support_catalog, run_preflight_calibration
from chesslab.validation.real_dataset import load_real_dataset_manifest


def test_preflight_on_real_fixture_returns_report() -> None:
    catalog = load_support_catalog()
    profile = catalog.get_profile("chesscom_green_white")
    assert profile is not None

    manifest = load_real_dataset_manifest(
        Path("tests/integration/fixtures/real_setups_manifest.json")
    )
    images = [sample.image_path for sample in manifest["chesscom_green_white"]]
    report = run_preflight_calibration(images, profile)

    assert report.setup_id == "chesscom_green_white"
    assert report.metrics.frame_count == len(images)
    assert report.metrics.detection_rate >= 0.9
    assert report.metrics.theme_match_rate >= 0.85
    assert report.metrics.ready_rate >= 0.60
    assert report.metrics.uncertain_rate <= 0.30
    assert report.passed is True
