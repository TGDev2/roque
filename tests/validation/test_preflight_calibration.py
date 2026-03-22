"""Tests for preflight calibration reports."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from chesslab.validation import load_support_catalog, run_preflight_calibration
from chesslab.validation.calibration import run_preflight_calibration_on_frames
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
    assert report.corrective_actions == ()


def test_preflight_on_frames_matches_image_path_contract() -> None:
    catalog = load_support_catalog()
    profile = catalog.get_profile("chesscom_green_white")
    assert profile is not None

    manifest = load_real_dataset_manifest(
        Path("tests/integration/fixtures/real_setups_manifest.json")
    )
    images = [sample.image_path for sample in manifest["chesscom_green_white"]]
    frames = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in images]

    assert all(frame is not None for frame in frames)

    path_report = run_preflight_calibration(images, profile)
    frame_report = run_preflight_calibration_on_frames(frames, profile)

    assert frame_report.setup_id == path_report.setup_id
    assert frame_report.passed == path_report.passed
    assert frame_report.blockers == path_report.blockers
    assert frame_report.warnings == path_report.warnings
    assert frame_report.corrective_actions == path_report.corrective_actions
    assert frame_report.metrics.frame_count == path_report.metrics.frame_count
    assert frame_report.metrics.status_counts == path_report.metrics.status_counts
    assert frame_report.metrics.detection_rate == pytest.approx(
        path_report.metrics.detection_rate
    )
    assert frame_report.metrics.theme_match_rate == pytest.approx(
        path_report.metrics.theme_match_rate
    )
    assert frame_report.metrics.ready_rate == pytest.approx(
        path_report.metrics.ready_rate
    )
    assert frame_report.metrics.uncertain_rate == pytest.approx(
        path_report.metrics.uncertain_rate
    )
    assert frame_report.metrics.invalid_rate == pytest.approx(
        path_report.metrics.invalid_rate
    )
