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
    assert result.metrics.sample_count == 8
    assert result.metrics.detection_rate >= 0.9
    assert result.metrics.mean_iou >= 0.9
    assert result.passed is False
