"""Tests for the Observe Live setup assistant."""

from __future__ import annotations

import numpy as np
import pytest

from chesslab.capture.screen_capture import MIN_CAPTURE_INTERVAL_MS, CaptureConfig
from chesslab.gui.live_setup_assistant import LiveSetupAssistant
from chesslab.validation import CalibrationMetrics, CalibrationReport, load_support_catalog


class _FakeCapture:
    def __init__(self, config: CaptureConfig, frames: list[np.ndarray]) -> None:
        self.config = config
        self._frames = frames
        self._idx = 0

    def capture_once(self) -> np.ndarray:
        frame = self._frames[min(self._idx, len(self._frames) - 1)]
        self._idx += 1
        return frame


def _make_report() -> CalibrationReport:
    return CalibrationReport(
        setup_id="chesscom_green_white",
        support_level="official",
        template_theme="chess.com-green",
        expected_theme="chess.com/green",
        passed=True,
        blockers=(),
        warnings=(),
        corrective_actions=(),
        metrics=CalibrationMetrics(
            frame_count=2,
            detection_rate=1.0,
            theme_match_rate=1.0,
            ready_rate=1.0,
            uncertain_rate=0.0,
            invalid_rate=0.0,
            p95_total_latency_ms=100.0,
            p95_detection_latency_ms=20.0,
            p95_split_latency_ms=10.0,
            p95_recognition_latency_ms=10.0,
            status_counts={"ready": 2},
        ),
    )


def test_list_options_prioritizes_official_profiles_and_hides_synthetic_setups() -> None:
    assistant = LiveSetupAssistant(catalog=load_support_catalog())

    options = assistant.list_options()

    assert options
    assert options[0].profile_id == "chesscom_green_white"
    assert options[0].support_label == "OFFICIAL"
    assert options[0].recommended is True
    assert [option.profile_id for option in options] == [
        "chesscom_green_white",
        "chesscom_green_black",
    ]
    assert all(option.template_quality == "real" for option in options)


def test_build_observer_config_uses_declared_profile() -> None:
    assistant = LiveSetupAssistant(catalog=load_support_catalog())
    profile = assistant.require_profile("chesscom_green_black")

    config = assistant.build_observer_config(2, profile)

    assert config.monitor_index == 2
    assert config.theme == "chess.com-green"
    assert config.white_perspective is False
    assert config.enable_game_tracking is True
    assert config.setup_id == "chesscom_green_black"
    assert config.support_level == "candidate"


def test_capture_preflight_frames_uses_monitor_and_min_interval() -> None:
    frames = [
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.ones((8, 8, 4), dtype=np.uint8),
        np.full((8, 8, 4), 2, dtype=np.uint8),
    ]
    created_configs: list[CaptureConfig] = []
    sleep_calls: list[float] = []

    def factory(config: CaptureConfig) -> _FakeCapture:
        created_configs.append(config)
        return _FakeCapture(config, frames)

    assistant = LiveSetupAssistant(
        catalog=load_support_catalog(),
        capture_factory=factory,
        sleeper=sleep_calls.append,
    )

    captured = assistant.capture_preflight_frames(
        2,
        frame_count=3,
        interval_ms=10,
    )

    assert len(captured) == 3
    assert created_configs[0].monitor_index == 2
    assert created_configs[0].interval_ms == MIN_CAPTURE_INTERVAL_MS
    assert sleep_calls == [MIN_CAPTURE_INTERVAL_MS / 1000.0] * 2


def test_run_preflight_uses_captured_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = load_support_catalog()
    frames = [np.zeros((8, 8, 4), dtype=np.uint8)] * 2
    expected_report = _make_report()
    created_configs: list[CaptureConfig] = []

    def factory(config: CaptureConfig) -> _FakeCapture:
        created_configs.append(config)
        return _FakeCapture(config, frames)

    seen: dict[str, object] = {}

    def fake_calibration(
        received_frames: tuple[np.ndarray, ...],
        profile: object,
        *,
        confidence_threshold: float = 0.60,
    ) -> CalibrationReport:
        seen["frames"] = received_frames
        seen["profile"] = profile
        seen["confidence_threshold"] = confidence_threshold
        return expected_report

    monkeypatch.setattr(
        "chesslab.gui.live_setup_assistant.run_preflight_calibration_on_frames",
        fake_calibration,
    )

    assistant = LiveSetupAssistant(
        catalog=catalog,
        capture_factory=factory,
        sleeper=lambda _seconds: None,
    )
    report = assistant.run_preflight(
        1,
        "chesscom_green_white",
        frame_count=2,
        interval_ms=75,
    )

    assert report is expected_report
    assert created_configs[0].monitor_index == 1
    assert len(seen["frames"]) == 2
    assert getattr(seen["profile"], "id") == "chesscom_green_white"
    assert seen["confidence_threshold"] == 0.60
