"""Tests for real-dataset validation by setup."""

from __future__ import annotations

from pathlib import Path

from chesslab.validation import load_support_catalog, validate_real_dataset


def test_validate_real_dataset_returns_metrics() -> None:
    catalog = load_support_catalog()
    manifest = Path("tests/integration/fixtures/real_setups_manifest.json")

    results = validate_real_dataset(
        manifest_path=manifest,
        catalog=catalog,
        setup_ids=["chesscom_green_white"],
    )

    assert len(results) == 1
    result = results[0]
    assert result.setup_id == "chesscom_green_white"
    assert result.metrics.sample_count == 4
    assert result.metrics.detection_rate >= 0.9
    assert result.metrics.mean_iou >= 0.9
    assert result.metrics.theme_match_rate >= 0.85
    assert result.metrics.ready_rate >= 0.60
    assert result.metrics.uncertain_rate <= 0.30
    assert result.passed is True


def test_black_candidate_real_dataset_surfaces_evidence_backed_failures() -> None:
    catalog = load_support_catalog()
    manifest = Path("tests/integration/fixtures/real_setups_manifest.json")

    results = validate_real_dataset(
        manifest_path=manifest,
        catalog=catalog,
        setup_ids=["chesscom_green_black"],
    )

    assert len(results) == 1
    result = results[0]
    assert result.setup_id == "chesscom_green_black"
    assert result.support_level == "candidate"
    assert result.metrics.sample_count >= 1
    assert result.passed is False
    assert (
        "No dataset thresholds configured for this setup in support catalog."
        not in result.failures
    )
    assert any("Ready frame rate too low" in failure for failure in result.failures)
    assert any("Invalid frame rate too high" in failure for failure in result.failures)
