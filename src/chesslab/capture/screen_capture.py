"""Screen capture module using mss for low-overhead, periodic screen grabbing."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import mss
import numpy as np
from mss.exception import ScreenShotError
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


MIN_CAPTURE_INTERVAL_MS = 50


class ScreenCaptureError(RuntimeError):
    """Base class for screen capture runtime failures."""


class CapturePermissionError(ScreenCaptureError):
    """Raised when OS-level screen recording permissions are missing."""


class DisplayUnavailableError(ScreenCaptureError):
    """Raised when no graphical display is available."""


class MonitorSelectionError(ScreenCaptureError):
    """Raised when monitor selection is invalid or missing."""


@dataclass
class CaptureConfig:
    """Configuration for screen capture."""

    interval_ms: int = 500
    """Capture interval in milliseconds (default: 500)."""

    monitor_index: int | None = None
    """Monitor index to capture (1-based). None means not yet selected."""

    def __post_init__(self) -> None:
        if self.interval_ms < MIN_CAPTURE_INTERVAL_MS:
            raise ValueError(
                f"interval_ms must be >= {MIN_CAPTURE_INTERVAL_MS}, "
                f"got {self.interval_ms}."
            )
        if self.monitor_index is not None and self.monitor_index < 1:
            raise ValueError(
                f"monitor_index must be >= 1 or None, got {self.monitor_index}."
            )


@dataclass
class MonitorInfo:
    """Description of a detected monitor."""

    index: int
    left: int
    top: int
    width: int
    height: int

    def __str__(self) -> str:
        return (
            f"Monitor {self.index}: {self.width}x{self.height} "
            f"at ({self.left}, {self.top})"
        )


class ScreenCapture:
    """Captures the screen periodically using mss.

    Designed for low CPU overhead and no visible capture window.
    Stores a single frame in memory (no accumulation).
    """

    def __init__(self, config: CaptureConfig | None = None) -> None:
        self._config = config or CaptureConfig()
        self._frame: NDArray[np.uint8] | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._on_frame: Callable[[NDArray[np.uint8]], None] | None = None
        self._on_stopped: Callable[[ScreenCaptureError | None], None] | None = None
        self._stop_notified = False

    @property
    def config(self) -> CaptureConfig:
        return self._config

    @property
    def frame(self) -> NDArray[np.uint8] | None:
        """Return a copy of the latest frame (or None if no capture yet)."""
        return self.get_frame(copy=True)

    def get_frame(self, *, copy: bool = True) -> NDArray[np.uint8] | None:
        """Return the latest captured frame.

        Args:
            copy: When True (default), returns a defensive copy to protect the
                internal buffer from external mutation.
        """
        with self._lock:
            if self._frame is None:
                return None
            if copy:
                return self._frame.copy()
            return self._frame

    @staticmethod
    def _translate_screenshot_error(exc: ScreenShotError) -> ScreenCaptureError:
        message = str(exc).lower()
        if "permission" in message or "not authorized" in message or "denied" in message:
            return CapturePermissionError(
                "Screen capture permission denied by the OS."
            )
        if "display" in message or "unable to open" in message:
            return DisplayUnavailableError(
                "No graphical display is available for screen capture."
            )
        return ScreenCaptureError(f"Screen capture failed: {exc}")

    @staticmethod
    def check_capture_access() -> None:
        """Check that the process can enumerate monitors and capture the screen."""
        try:
            with mss.mss() as sct:
                _ = sct.monitors
        except ScreenShotError as exc:
            raise ScreenCapture._translate_screenshot_error(exc) from exc

    @staticmethod
    def list_monitors() -> list[MonitorInfo]:
        """Detect and return all available monitors.

        Returns a list of MonitorInfo (1-based index, matching mss convention).
        Index 0 in mss is the virtual "all monitors" screen; we skip it.
        """
        try:
            with mss.mss() as sct:
                monitors: list[MonitorInfo] = []
                for i, mon in enumerate(sct.monitors):
                    if i == 0:
                        continue  # skip virtual combined monitor
                    monitors.append(
                        MonitorInfo(
                            index=i,
                            left=mon["left"],
                            top=mon["top"],
                            width=mon["width"],
                            height=mon["height"],
                        )
                    )
                return monitors
        except ScreenShotError as exc:
            raise ScreenCapture._translate_screenshot_error(exc) from exc

    def select_monitor(self, index: int) -> None:
        """Select which monitor to capture (1-based index)."""
        monitors = self.list_monitors()
        if not monitors:
            raise MonitorSelectionError("No physical monitor detected.")

        valid_indices = {m.index for m in monitors}
        if index not in valid_indices:
            raise MonitorSelectionError(
                f"Invalid monitor index {index}. "
                f"Available: {sorted(valid_indices)}"
            )
        self._config.monitor_index = index
        logger.info("Monitor %d selected for capture.", index)

    def _require_monitor_selected(self) -> int:
        if self._config.monitor_index is None:
            raise MonitorSelectionError(
                "No monitor selected. Call select_monitor() first "
                "or use list_monitors() to discover available screens."
            )
        return self._config.monitor_index

    def capture_once(self) -> NDArray[np.uint8]:
        """Capture a single frame from the selected monitor.

        Returns the captured image as a numpy array (H, W, 4) in BGRA format.

        Raises:
            MonitorSelectionError: If no monitor has been selected.
        """
        monitor_index = self._require_monitor_selected()

        try:
            with mss.mss() as sct:
                monitor = sct.monitors[monitor_index]
                raw = sct.grab(monitor)
                frame = np.asarray(raw, dtype=np.uint8).copy()
        except ScreenShotError as exc:
            raise self._translate_screenshot_error(exc) from exc
        except IndexError as exc:
            raise MonitorSelectionError(
                f"Monitor index {monitor_index} no longer exists."
            ) from exc

        with self._lock:
            frame.setflags(write=False)
            self._frame = frame

        logger.debug(
            "Frame captured: %dx%d from monitor %d.",
            frame.shape[1],
            frame.shape[0],
            monitor_index,
        )
        return frame

    def on_frame(self, callback: Callable[[NDArray[np.uint8]], None]) -> None:
        """Register a callback invoked after each capture (e.g. detection module)."""
        self._on_frame = callback

    def on_stopped(
        self,
        callback: Callable[[ScreenCaptureError | None], None],
    ) -> None:
        """Register a callback invoked when capture stops.

        The callback receives ``None`` for a normal stop, or a translated
        :class:`ScreenCaptureError` when capture aborts due to a runtime issue.
        """
        self._on_stopped = callback

    def start(self) -> None:
        """Start periodic capture in a background thread."""
        self._require_monitor_selected()
        self.check_capture_access()
        if self._running:
            logger.warning("Capture already running.")
            return

        self._running = True
        self._stop_notified = False
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="screen-capture"
        )
        self._thread.start()
        logger.info(
            "Capture started on monitor %d every %d ms.",
            self._config.monitor_index,
            self._config.interval_ms,
        )

    def stop(self) -> None:
        """Stop the periodic capture loop."""
        was_running = self._running
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if was_running:
            self._notify_stopped(None)
        logger.info("Capture stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    def _capture_loop(self) -> None:
        """Internal loop running in background thread."""
        interval_s = self._config.interval_ms / 1000.0

        monitor_index = self._require_monitor_selected()

        try:
            with mss.mss() as sct:
                while self._running:
                    start = time.perf_counter()
                    try:
                        monitor = sct.monitors[monitor_index]
                        raw = sct.grab(monitor)
                        frame = np.asarray(raw, dtype=np.uint8).copy()
                        frame.setflags(write=False)

                        with self._lock:
                            self._frame = frame

                        if self._on_frame is not None:
                            try:
                                self._on_frame(frame)
                            except Exception:
                                logger.exception("Frame callback raised an error.")

                    except ScreenShotError as exc:
                        translated = self._translate_screenshot_error(exc)
                        logger.error("%s", translated)
                        self._running = False
                        self._notify_stopped(translated)
                        break
                    except IndexError:
                        err = MonitorSelectionError(
                            f"Monitor index {monitor_index} no longer exists."
                        )
                        logger.error(
                            "Monitor index %d no longer exists. Stopping capture.",
                            monitor_index,
                        )
                        self._running = False
                        self._notify_stopped(err)
                        break
                    except Exception:
                        logger.exception("Error during screen capture.")

                    elapsed = time.perf_counter() - start
                    sleep_time = max(0.0, interval_s - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except ScreenShotError as exc:
            translated = self._translate_screenshot_error(exc)
            logger.error("%s", translated)
            self._running = False
            self._notify_stopped(translated)

    def _notify_stopped(self, error: ScreenCaptureError | None) -> None:
        if self._stop_notified:
            return
        self._stop_notified = True
        if self._on_stopped is None:
            return
        try:
            self._on_stopped(error)
        except Exception:
            logger.exception("Capture stop callback raised an error.")
