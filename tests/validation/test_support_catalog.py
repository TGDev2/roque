"""Tests for the setup support catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from chesslab.validation.catalog import SupportCatalogError, load_support_catalog


def test_default_catalog_loads_profiles() -> None:
    catalog = load_support_catalog()

    assert catalog.version == 1
    assert len(catalog.profiles) >= 1
    assert catalog.get_profile("chesscom_green_white") is not None


def test_official_profile_has_thresholds() -> None:
    catalog = load_support_catalog()
    profile = catalog.get_profile("chesscom_green_white")

    assert profile is not None
    assert profile.is_official is True
    assert profile.preflight_thresholds.min_detection_rate >= 0.9
    assert profile.dataset_thresholds is not None


def test_invalid_catalog_raises(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{\"version\": 1, \"setups\": []}", encoding="utf-8")

    with pytest.raises(SupportCatalogError):
        load_support_catalog(broken)
