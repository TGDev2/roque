"""E2E measurement contract on real annotated datasets by setup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chesslab.validation import load_support_catalog, validate_real_dataset
from chesslab.validation.catalog import SupportLevel

DEFAULT_MANIFEST = Path("tests/integration/fixtures/real_setups_manifest.json")
DATASET_MANIFEST = Path(
    os.environ.get("CHESSLAB_REAL_SETUPS_MANIFEST", DEFAULT_MANIFEST)
)


def test_real_setups_manifest_measures_e2e_metrics() -> None:
    if not DATASET_MANIFEST.exists():
        pytest.skip(
            "Real setup manifest missing. "
            "Set CHESSLAB_REAL_SETUPS_MANIFEST to run this contract."
        )

    catalog = load_support_catalog()
    results = validate_real_dataset(
        manifest_path=DATASET_MANIFEST,
        catalog=catalog,
    )

    assert results, "Real setup manifest is empty."
    profiles_by_id = {profile.id: profile for profile in catalog.profiles}
    for result in results:
        profile = profiles_by_id[result.setup_id]
        assert result.metrics.sample_count > 0
        assert result.metrics.p95_total_latency_ms > 0.0
        assert result.metrics.p95_detection_latency_ms > 0.0
        if profile.has_real_dataset:
            assert (
                "No dataset thresholds configured for this setup in support catalog."
                not in result.failures
            )

    official_ids = {
        profile.id
        for profile in catalog.profiles
        if profile.support_level is SupportLevel.OFFICIAL
    }
    for result in results:
        if result.setup_id in official_ids:
            assert result.passed, (
                f"Official setup '{result.setup_id}' must pass e2e dataset contract; "
                f"failures={result.failures}"
            )
