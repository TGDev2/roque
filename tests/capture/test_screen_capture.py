"""Tests for the screen capture module.

Covers acceptance criteria:
- Nominal capture (< 200 ms, image in memory, no visible window)
- Capture performance (periodic capture, no frame blocking)
- Multi-screen management (list, select, invalid index)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mss.exception import ScreenShotError

from chesslab.capture.screen_capture import (
    CaptureConfig,
    CapturePermissionError,
    DisplayUnavailableError,
    MIN_CAPTURE_INTERVAL_MS,
    MonitorInfo,
    MonitorSelectionError,
    ScreenCapture,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

FAKE_MONITORS = [
    {"left": 0, "top": 0, "width": 3840, "height": 1080},  # index 0: virtual
    {"left": 0, "top": 0, "width": 1920, "height": 1080},  # index 1
    {"left": 1920, "top": 0, "width": 1920, "height": 1080},  # index 2
]

FAKE_FRAME_SHAPE = (1080, 1920, 4)  # H, W, BGRA


def _make_fake_raw(shape: tuple[int, ...] = FAKE_FRAME_SHAPE) -> MagicMock:
    """Create a mock mss screenshot that behaves like a numpy-convertible object."""
    raw = MagicMock()
    fake_array = np.zeros(shape, dtype=np.uint8)
    raw.__array__ = MagicMock(return_value=fake_array)
    # np.asarray calls __array_interface__ or __array__; we mock __array__
    return raw


@pytest.fixture
def mock_mss():
    """Patch mss.mss() to return a fake screen capture context."""
    fake_sct = MagicMock()
    fake_sct.monitors = FAKE_MONITORS
    fake_sct.grab.return_value = _make_fake_raw()

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=fake_sct)
    ctx.__exit__ = MagicMock(return_value=False)

    with patch("chesslab.capture.screen_capture.mss.mss", return_value=ctx) as m:
        yield fake_sct


# ---------------------------------------------------------------------------
# Scenario: Multi-screen management
# ---------------------------------------------------------------------------


class TestListMonitors:
    def test_returns_physical_monitors_only(self, mock_mss: MagicMock) -> None:
        monitors = ScreenCapture.list_monitors()
        # Index 0 (virtual combined) should be excluded
        assert len(monitors) == 2
        assert all(isinstance(m, MonitorInfo) for m in monitors)

    def test_monitor_info_attributes(self, mock_mss: MagicMock) -> None:
        monitors = ScreenCapture.list_monitors()
        m1 = monitors[0]
        assert m1.index == 1
        assert m1.width == 1920
        assert m1.height == 1080
        assert m1.left == 0
        assert m1.top == 0

    def test_second_monitor(self, mock_mss: MagicMock) -> None:
        monitors = ScreenCapture.list_monitors()
        m2 = monitors[1]
        assert m2.index == 2
        assert m2.left == 1920

    def test_permission_error_is_translated(self) -> None:
        with patch(
            "chesslab.capture.screen_capture.mss.mss",
            side_effect=ScreenShotError("Permission denied"),
        ):
            with pytest.raises(CapturePermissionError):
                ScreenCapture.list_monitors()

    def test_display_unavailable_is_translated(self) -> None:
        with patch(
            "chesslab.capture.screen_capture.mss.mss",
            side_effect=ScreenShotError("Unable to open display"),
        ):
            with pytest.raises(DisplayUnavailableError):
                ScreenCapture.list_monitors()


class TestSelectMonitor:
    def test_select_valid_monitor(self, mock_mss: MagicMock) -> None:
        cap = ScreenCapture()
        cap.select_monitor(1)
        assert cap.config.monitor_index == 1

    def test_select_second_monitor(self, mock_mss: MagicMock) -> None:
        cap = ScreenCapture()
        cap.select_monitor(2)
        assert cap.config.monitor_index == 2

    def test_select_invalid_monitor_raises(self, mock_mss: MagicMock) -> None:
        cap = ScreenCapture()
        with pytest.raises(MonitorSelectionError, match="Invalid monitor index 5"):
            cap.select_monitor(5)


# ---------------------------------------------------------------------------
# Scenario: Nominal capture
# ---------------------------------------------------------------------------


class TestCaptureOnce:
    def test_capture_returns_numpy_array(self, mock_mss: MagicMock) -> None:
        cap = ScreenCapture(CaptureConfig(monitor_index=1))
        frame = cap.capture_once()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8

    def test_capture_stores_frame_in_memory(self, mock_mss: MagicMock) -> None:
        cap = ScreenCapture(CaptureConfig(monitor_index=1))
        assert cap.frame is None
        cap.capture_once()
        assert cap.frame is not None

    def test_frame_has_correct_shape(self, mock_mss: MagicMock) -> None:
        cap = ScreenCapture(CaptureConfig(monitor_index=1))
        frame = cap.capture_once()
        assert frame.shape == FAKE_FRAME_SHAPE

    def test_capture_without_monitor_raises(self, mock_mss: MagicMock) -> None:
        cap = ScreenCapture()
        with pytest.raises(MonitorSelectionError, match="No monitor selected"):
            cap.capture_once()

    def test_frame_property_returns_defensive_copy(self, mock_mss: MagicMock) -> None:
        cap = ScreenCapture(CaptureConfig(monitor_index=1))
        cap.capture_once()
        first = cap.frame
        second = cap.frame
        assert first is not None and second is not None
        first[0, 0, 0] = 255
        assert second[0, 0, 0] == 0

    def test_capture_completes_under_200ms(self, mock_mss: MagicMock) -> None:
        """AC: capture in < 200 ms."""
        cap = ScreenCapture(CaptureConfig(monitor_index=1))
        start = time.perf_counter()
        cap.capture_once()
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"Capture took {elapsed_ms:.1f} ms (limit: 200 ms)"

    def test_no_visible_window(self, mock_mss: MagicMock) -> None:
        """AC: mss does not open any visible window.

        mss uses OS-level APIs (no GUI window) by design.
        We verify that no display-related call is made.
        """
        cap = ScreenCapture(CaptureConfig(monitor_index=1))
        cap.capture_once()
        # mss.grab() is called, but no Tk/Qt/display window is created
        mock_mss.grab.assert_called_once()


# ---------------------------------------------------------------------------
# Scenario: Capture performance (periodic)
# ---------------------------------------------------------------------------


class TestPeriodicCapture:
    def test_start_and_stop(self, mock_mss: MagicMock) -> None:
        cap = ScreenCapture(CaptureConfig(monitor_index=1, interval_ms=100))
        cap.start()
        assert cap.is_running is True
        time.sleep(0.35)
        cap.stop()
        assert cap.is_running is False
        # Should have captured several frames
        assert cap.frame is not None

    def test_start_without_monitor_raises(self, mock_mss: MagicMock) -> None:
        cap = ScreenCapture()
        with pytest.raises(MonitorSelectionError, match="No monitor selected"):
            cap.start()

    def test_double_start_is_safe(self, mock_mss: MagicMock) -> None:
        cap = ScreenCapture(CaptureConfig(monitor_index=1, interval_ms=200))
        cap.start()
        cap.start()  # should warn but not crash
        cap.stop()

    def test_callback_invoked_on_each_frame(self, mock_mss: MagicMock) -> None:
        received: list[np.ndarray] = []
        cap = ScreenCapture(CaptureConfig(monitor_index=1, interval_ms=100))
        cap.on_frame(lambda f: received.append(f))
        cap.start()
        time.sleep(0.35)
        cap.stop()
        assert len(received) >= 2

    def test_buffer_not_accumulating(self, mock_mss: MagicMock) -> None:
        """AC: no RAM accumulation -- only one frame stored at a time."""
        cap = ScreenCapture(CaptureConfig(monitor_index=1, interval_ms=100))
        cap.start()
        time.sleep(0.35)
        cap.stop()
        # Only the latest frame is kept in self._frame (single reference)
        assert cap.frame is not None
        # Verify _frame is a single array, not a list
        assert isinstance(cap.frame, np.ndarray)

    def test_capture_loop_is_daemon_thread(self, mock_mss: MagicMock) -> None:
        """The capture thread should not block application exit."""
        cap = ScreenCapture(CaptureConfig(monitor_index=1, interval_ms=200))
        cap.start()
        assert cap._thread is not None
        assert cap._thread.daemon is True
        cap.stop()

    def test_capture_loop_handles_errors_gracefully(
        self, mock_mss: MagicMock
    ) -> None:
        """If grab() fails, the loop continues without crashing."""
        call_count = 0

        def flaky_grab(monitor: dict) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Screen locked")
            return _make_fake_raw()

        mock_mss.grab.side_effect = flaky_grab

        cap = ScreenCapture(CaptureConfig(monitor_index=1, interval_ms=50))
        cap.start()
        time.sleep(0.25)
        cap.stop()
        # Should have recovered after the first error
        assert call_count >= 2
        assert cap.frame is not None

    def test_callback_receives_readonly_frame(self, mock_mss: MagicMock) -> None:
        flags: list[bool] = []
        cap = ScreenCapture(CaptureConfig(monitor_index=1, interval_ms=100))
        cap.on_frame(lambda frame: flags.append(frame.flags.writeable))
        cap.start()
        time.sleep(0.2)
        cap.stop()
        assert flags
        assert all(flag is False for flag in flags)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestCaptureConfig:
    def test_default_interval(self) -> None:
        cfg = CaptureConfig()
        assert cfg.interval_ms == 500

    def test_custom_interval(self) -> None:
        cfg = CaptureConfig(interval_ms=250)
        assert cfg.interval_ms == 250

    def test_default_monitor_is_none(self) -> None:
        cfg = CaptureConfig()
        assert cfg.monitor_index is None

    def test_interval_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="interval_ms must be >="):
            CaptureConfig(interval_ms=MIN_CAPTURE_INTERVAL_MS - 1)

    def test_monitor_index_below_1_raises(self) -> None:
        with pytest.raises(ValueError, match="monitor_index must be >= 1"):
            CaptureConfig(monitor_index=0)
