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
    assert profile.is_recommended is True
    assert profile.preflight_thresholds.min_detection_rate >= 0.9
    assert profile.dataset_thresholds is not None


def test_black_official_profile_is_backed_by_real_dataset() -> None:
    catalog = load_support_catalog()
    profile = catalog.get_profile("chesscom_green_black")

    assert profile is not None
    assert profile.is_official is True
    assert profile.is_recommended is True
    assert profile.uses_real_templates is True
    assert profile.has_real_dataset is True
    assert profile.dataset_thresholds is not None


def test_brown_candidate_profile_is_backed_by_real_templates_and_dataset() -> None:
    catalog = load_support_catalog()
    profile = catalog.get_profile("chesscom_brown_white")

    assert profile is not None
    assert profile.support_level.value == "candidate"
    assert profile.uses_real_templates is True
    assert profile.has_real_dataset is True
    assert profile.dataset_thresholds is not None
    assert profile.preflight_thresholds.min_detection_rate >= 0.9


def test_live_profiles_expose_only_real_template_official_or_candidate_setups() -> None:
    catalog = load_support_catalog()

    profiles = catalog.live_profiles()

    assert {profile.id for profile in profiles} == {
        "chesscom_brown_white",
        "chesscom_green_white",
        "chesscom_green_black",
    }
    assert all(profile.uses_real_templates for profile in profiles)


def test_invalid_catalog_raises(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{\"version\": 1, \"setups\": []}", encoding="utf-8")

    with pytest.raises(SupportCatalogError):
        load_support_catalog(broken)
