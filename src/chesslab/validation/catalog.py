"""Support perimeter catalog for real-world live configurations."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_CATALOG_PATH = Path(__file__).with_name("supported_setups.json")


class SupportCatalogError(RuntimeError):
    """Raised when the support catalog is invalid."""


class SupportLevel(str, enum.Enum):
    """Support maturity level for a setup profile."""

    OFFICIAL = "official"
    CANDIDATE = "candidate"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class PreflightThresholds:
    """Thresholds used by pre-session calibration."""

    min_detection_rate: float
    min_theme_match_rate: float
    min_ready_rate: float
    max_uncertain_rate: float
    max_p95_total_latency_ms: float


@dataclass(frozen=True)
class DatasetThresholds:
    """Thresholds used by real-dataset validation."""

    min_detection_rate: float
    min_mean_iou: float
    min_theme_match_rate: float
    min_ready_rate: float
    max_uncertain_rate: float
    max_invalid_rate: float
    max_p95_total_latency_ms: float
    max_p95_detection_latency_ms: float
    max_p95_split_latency_ms: float
    max_p95_recognition_latency_ms: float


@dataclass(frozen=True)
class SetupSupportProfile:
    """Single setup definition from the support matrix."""

    id: str
    platform: str
    site_theme: str
    perspective: str
    template_theme: str
    support_level: SupportLevel
    template_quality: str
    notes: str
    preflight_thresholds: PreflightThresholds
    dataset_thresholds: DatasetThresholds | None = None
    dataset_manifest: str | None = None

    @property
    def expected_theme_id(self) -> str:
        return f"{self.platform}/{self.site_theme}"

    @property
    def is_official(self) -> bool:
        return self.support_level is SupportLevel.OFFICIAL

    @property
    def uses_real_templates(self) -> bool:
        return self.template_quality == "real"

    @property
    def has_real_dataset(self) -> bool:
        return self.dataset_manifest is not None

    @property
    def is_recommended(self) -> bool:
        return self.is_official and self.uses_real_templates


@dataclass(frozen=True)
class SupportCatalog:
    """Versioned setup support matrix."""

    version: int
    profiles: tuple[SetupSupportProfile, ...]

    def get_profile(self, profile_id: str) -> SetupSupportProfile | None:
        for profile in self.profiles:
            if profile.id == profile_id:
                return profile
        return None

    def official_profiles(self) -> tuple[SetupSupportProfile, ...]:
        return tuple(profile for profile in self.profiles if profile.is_official)

    def calibration_profiles(self) -> tuple[SetupSupportProfile, ...]:
        return tuple(
            profile
            for profile in self.profiles
            if profile.support_level in {
                SupportLevel.OFFICIAL,
                SupportLevel.CANDIDATE,
                SupportLevel.EXPERIMENTAL,
            }
        )

    def live_profiles(self) -> tuple[SetupSupportProfile, ...]:
        """Return the narrow user-facing Observe Live perimeter."""
        return tuple(
            profile
            for profile in self.profiles
            if profile.uses_real_templates
            and profile.support_level in {
                SupportLevel.OFFICIAL,
                SupportLevel.CANDIDATE,
            }
        )


def load_support_catalog(path: Path | None = None) -> SupportCatalog:
    """Load and validate the support matrix JSON."""
    catalog_path = path or _DEFAULT_CATALOG_PATH
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupportCatalogError(
            f"Unable to read support catalog at {catalog_path}."
        ) from exc

    version = _read_int(payload, "version", ctx="catalog root")
    raw_setups = payload.get("setups")
    if not isinstance(raw_setups, list) or not raw_setups:
        raise SupportCatalogError(
            f"Support catalog {catalog_path} must contain a non-empty 'setups' list."
        )

    profiles: list[SetupSupportProfile] = []
    seen_ids: set[str] = set()

    for idx, raw_profile in enumerate(raw_setups):
        if not isinstance(raw_profile, dict):
            raise SupportCatalogError(
                f"Invalid profile at index {idx} in {catalog_path}."
            )
        profile = _parse_profile(raw_profile, idx=idx, path=catalog_path)
        if profile.id in seen_ids:
            raise SupportCatalogError(
                f"Duplicate setup id '{profile.id}' in {catalog_path}."
            )
        seen_ids.add(profile.id)
        profiles.append(profile)

    return SupportCatalog(version=version, profiles=tuple(profiles))


def _parse_profile(
    raw: dict[str, object],
    *,
    idx: int,
    path: Path,
) -> SetupSupportProfile:
    ctx = f"profile[{idx}] in {path}"
    profile_id = _read_str(raw, "id", ctx=ctx)
    support_level_raw = _read_str(raw, "support_level", ctx=ctx)
    try:
        support_level = SupportLevel(support_level_raw)
    except ValueError as exc:
        allowed = ", ".join(level.value for level in SupportLevel)
        raise SupportCatalogError(
            f"Invalid support level '{support_level_raw}' for setup '{profile_id}'. "
            f"Allowed values: {allowed}."
        ) from exc

    dataset_thresholds_raw = raw.get("dataset_thresholds")
    dataset_thresholds = (
        None
        if dataset_thresholds_raw is None
        else _parse_dataset_thresholds(dataset_thresholds_raw, ctx=ctx)
    )

    dataset_manifest_raw = raw.get("dataset_manifest")
    dataset_manifest = None
    if dataset_manifest_raw is not None:
        if not isinstance(dataset_manifest_raw, str) or not dataset_manifest_raw.strip():
            raise SupportCatalogError(
                f"Invalid dataset_manifest in {ctx}; expected a non-empty string."
            )
        dataset_manifest = dataset_manifest_raw.strip()

    return SetupSupportProfile(
        id=profile_id,
        platform=_read_str(raw, "platform", ctx=ctx),
        site_theme=_read_str(raw, "site_theme", ctx=ctx),
        perspective=_read_perspective(raw, ctx=ctx),
        template_theme=_read_str(raw, "template_theme", ctx=ctx),
        support_level=support_level,
        template_quality=_read_str(raw, "template_quality", ctx=ctx),
        notes=_read_str(raw, "notes", ctx=ctx),
        preflight_thresholds=_parse_preflight_thresholds(
            raw.get("preflight_thresholds"),
            ctx=ctx,
        ),
        dataset_thresholds=dataset_thresholds,
        dataset_manifest=dataset_manifest,
    )


def _parse_preflight_thresholds(
    value: object,
    *,
    ctx: str,
) -> PreflightThresholds:
    if not isinstance(value, dict):
        raise SupportCatalogError(
            f"Missing or invalid preflight_thresholds in {ctx}."
        )
    return PreflightThresholds(
        min_detection_rate=_read_ratio(value, "min_detection_rate", ctx=ctx),
        min_theme_match_rate=_read_ratio(value, "min_theme_match_rate", ctx=ctx),
        min_ready_rate=_read_ratio(value, "min_ready_rate", ctx=ctx),
        max_uncertain_rate=_read_ratio(value, "max_uncertain_rate", ctx=ctx),
        max_p95_total_latency_ms=_read_non_negative_float(
            value,
            "max_p95_total_latency_ms",
            ctx=ctx,
        ),
    )


def _parse_dataset_thresholds(
    value: object,
    *,
    ctx: str,
) -> DatasetThresholds:
    if not isinstance(value, dict):
        raise SupportCatalogError(
            f"Invalid dataset_thresholds in {ctx}; expected an object."
        )
    return DatasetThresholds(
        min_detection_rate=_read_ratio(value, "min_detection_rate", ctx=ctx),
        min_mean_iou=_read_ratio(value, "min_mean_iou", ctx=ctx),
        min_theme_match_rate=_read_ratio(value, "min_theme_match_rate", ctx=ctx),
        min_ready_rate=_read_ratio(value, "min_ready_rate", ctx=ctx),
        max_uncertain_rate=_read_ratio(value, "max_uncertain_rate", ctx=ctx),
        max_invalid_rate=_read_ratio(value, "max_invalid_rate", ctx=ctx),
        max_p95_total_latency_ms=_read_non_negative_float(
            value,
            "max_p95_total_latency_ms",
            ctx=ctx,
        ),
        max_p95_detection_latency_ms=_read_non_negative_float(
            value,
            "max_p95_detection_latency_ms",
            ctx=ctx,
        ),
        max_p95_split_latency_ms=_read_non_negative_float(
            value,
            "max_p95_split_latency_ms",
            ctx=ctx,
        ),
        max_p95_recognition_latency_ms=_read_non_negative_float(
            value,
            "max_p95_recognition_latency_ms",
            ctx=ctx,
        ),
    )


def _read_int(data: dict[str, object], key: str, *, ctx: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise SupportCatalogError(
            f"Invalid '{key}' in {ctx}; expected integer."
        )
    return value


def _read_str(data: dict[str, object], key: str, *, ctx: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SupportCatalogError(
            f"Invalid '{key}' in {ctx}; expected non-empty string."
        )
    return value.strip()


def _read_perspective(data: dict[str, object], *, ctx: str) -> str:
    value = _read_str(data, "perspective", ctx=ctx).lower()
    if value not in {"white", "black"}:
        raise SupportCatalogError(
            f"Invalid perspective '{value}' in {ctx}; expected 'white' or 'black'."
        )
    return value


def _read_ratio(data: dict[str, object], key: str, *, ctx: str) -> float:
    value = _read_non_negative_float(data, key, ctx=ctx)
    if value > 1.0:
        raise SupportCatalogError(
            f"Invalid '{key}' in {ctx}; expected value in [0, 1], got {value}."
        )
    return value


def _read_non_negative_float(
    data: dict[str, object],
    key: str,
    *,
    ctx: str,
) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)):
        raise SupportCatalogError(
            f"Invalid '{key}' in {ctx}; expected number."
        )
    numeric = float(value)
    if numeric < 0:
        raise SupportCatalogError(
            f"Invalid '{key}' in {ctx}; expected >= 0, got {numeric}."
        )
    return numeric
