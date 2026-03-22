"""Setup-driven preflight helpers for the Observe Live journey."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from numpy.typing import NDArray

from chesslab.capture.screen_capture import (
    MIN_CAPTURE_INTERVAL_MS,
    CaptureConfig,
    ScreenCapture,
)
from chesslab.gui.live_observer import LiveObserverConfig
from chesslab.validation import (
    CalibrationReport,
    SetupSupportProfile,
    SupportCatalog,
    SupportLevel,
    load_support_catalog,
    run_preflight_calibration_on_frames,
)


_SUPPORT_PRIORITY: dict[SupportLevel, int] = {
    SupportLevel.OFFICIAL: 0,
    SupportLevel.CANDIDATE: 1,
    SupportLevel.EXPERIMENTAL: 2,
    SupportLevel.UNSUPPORTED: 3,
}


@dataclass(frozen=True)
class LiveSetupOption:
    """Display-friendly view of a declared live setup."""

    profile_id: str
    label: str
    support_label: str
    platform: str
    site_theme: str
    perspective: str
    expected_theme: str
    template_theme: str
    template_quality: str
    notes: str
    recommended: bool

    @classmethod
    def from_profile(cls, profile: SetupSupportProfile) -> LiveSetupOption:
        support_label = profile.support_level.value.upper()
        perspective = profile.perspective.title()
        return cls(
            profile_id=profile.id,
            label=(
                f"{support_label}  {profile.platform} / {profile.site_theme} / "
                f"{perspective}"
            ),
            support_label=support_label,
            platform=profile.platform,
            site_theme=profile.site_theme,
            perspective=perspective,
            expected_theme=profile.expected_theme_id,
            template_theme=profile.template_theme,
            template_quality=profile.template_quality,
            notes=profile.notes,
            recommended=profile.support_level is SupportLevel.OFFICIAL,
        )


class LiveSetupAssistant:
    """Coordinates support-catalog setup selection and live preflight checks."""

    def __init__(
        self,
        *,
        catalog: SupportCatalog | None = None,
        capture_factory: Callable[[CaptureConfig], ScreenCapture] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._catalog = catalog or load_support_catalog()
        self._capture_factory = capture_factory or ScreenCapture
        self._sleep = sleeper

    def list_options(self) -> tuple[LiveSetupOption, ...]:
        """Return calibration-capable setup options in UX-friendly order."""
        profiles = sorted(self._catalog.calibration_profiles(), key=_profile_sort_key)
        return tuple(LiveSetupOption.from_profile(profile) for profile in profiles)

    def get_profile(self, profile_id: str) -> SetupSupportProfile | None:
        """Return a declared setup profile by id."""
        return self._catalog.get_profile(profile_id)

    def require_profile(self, profile_id: str) -> SetupSupportProfile:
        """Return a declared setup profile or raise a helpful error."""
        profile = self.get_profile(profile_id)
        if profile is None:
            raise ValueError(f"Unknown setup profile '{profile_id}'.")
        return profile

    def build_observer_config(
        self,
        monitor_index: int,
        profile: SetupSupportProfile,
    ) -> LiveObserverConfig:
        """Convert a support-catalog profile into the live observer config."""
        return LiveObserverConfig(
            monitor_index=monitor_index,
            theme=profile.template_theme,
            white_perspective=profile.perspective == "white",
            enable_game_tracking=True,
            setup_id=profile.id,
            support_level=profile.support_level.value,
        )

    def capture_preflight_frames(
        self,
        monitor_index: int,
        *,
        frame_count: int = 6,
        interval_ms: int = 150,
    ) -> tuple[NDArray, ...]:
        """Capture a short burst of frames from the selected monitor."""
        if frame_count < 1:
            raise ValueError("frame_count must be >= 1.")
        effective_interval_ms = max(interval_ms, MIN_CAPTURE_INTERVAL_MS)
        capture = self._capture_factory(
            CaptureConfig(
                interval_ms=effective_interval_ms,
                monitor_index=monitor_index,
            )
        )

        frames: list[NDArray] = []
        for idx in range(frame_count):
            frames.append(capture.capture_once())
            if idx + 1 < frame_count:
                self._sleep(effective_interval_ms / 1000.0)
        return tuple(frames)

    def run_preflight(
        self,
        monitor_index: int,
        profile_id: str,
        *,
        frame_count: int = 6,
        interval_ms: int = 150,
        confidence_threshold: float = 0.60,
    ) -> CalibrationReport:
        """Capture frames and run the declared setup preflight."""
        profile = self.require_profile(profile_id)
        frames = self.capture_preflight_frames(
            monitor_index,
            frame_count=frame_count,
            interval_ms=interval_ms,
        )
        return run_preflight_calibration_on_frames(
            frames,
            profile,
            confidence_threshold=confidence_threshold,
        )


def _profile_sort_key(profile: SetupSupportProfile) -> tuple[int, str, str, str]:
    return (
        _SUPPORT_PRIORITY.get(profile.support_level, 99),
        profile.platform,
        profile.site_theme,
        profile.perspective,
    )
