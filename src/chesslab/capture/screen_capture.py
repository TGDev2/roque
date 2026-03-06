"""Screen capture module using mss for low-overhead, periodic screen grabbing."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import mss
import mss.tools
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class CaptureConfig:
    """Configuration for screen capture."""

    interval_ms: int = 500
    """Capture interval in milliseconds (default: 500)."""

    monitor_index: int | None = None
    """Monitor index to capture (1-based). None means not yet selected."""


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

    @property
    def config(self) -> CaptureConfig:
        return self._config

    @property
    def frame(self) -> NDArray[np.uint8] | None:
        """Return the latest captured frame (or None if no capture yet)."""
        with self._lock:
            return self._frame

    @staticmethod
    def list_monitors() -> list[MonitorInfo]:
        """Detect and return all available monitors.

        Returns a list of MonitorInfo (1-based index, matching mss convention).
        Index 0 in mss is the virtual "all monitors" screen; we skip it.
        """
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

    def select_monitor(self, index: int) -> None:
        """Select which monitor to capture (1-based index)."""
        monitors = self.list_monitors()
        valid_indices = {m.index for m in monitors}
        if index not in valid_indices:
            raise ValueError(
                f"Invalid monitor index {index}. "
                f"Available: {sorted(valid_indices)}"
            )
        self._config.monitor_index = index
        logger.info("Monitor %d selected for capture.", index)

    def capture_once(self) -> NDArray[np.uint8]:
        """Capture a single frame from the selected monitor.

        Returns the captured image as a numpy array (H, W, 4) in BGRA format.

        Raises:
            RuntimeError: If no monitor has been selected.
        """
        if self._config.monitor_index is None:
            raise RuntimeError(
                "No monitor selected. Call select_monitor() first "
                "or use list_monitors() to discover available screens."
            )

        with mss.mss() as sct:
            monitor = sct.monitors[self._config.monitor_index]
            raw = sct.grab(monitor)
            frame = np.asarray(raw, dtype=np.uint8).copy()

        with self._lock:
            self._frame = frame

        logger.debug(
            "Frame captured: %dx%d from monitor %d.",
            frame.shape[1],
            frame.shape[0],
            self._config.monitor_index,
        )
        return frame

    def on_frame(self, callback: Callable[[NDArray[np.uint8]], None]) -> None:
        """Register a callback invoked after each capture (e.g. detection module)."""
        self._on_frame = callback

    def start(self) -> None:
        """Start periodic capture in a background thread."""
        if self._config.monitor_index is None:
            raise RuntimeError(
                "No monitor selected. Call select_monitor() before start()."
            )
        if self._running:
            logger.warning("Capture already running.")
            return

        self._running = True
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
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Capture stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    def _capture_loop(self) -> None:
        """Internal loop running in background thread."""
        interval_s = self._config.interval_ms / 1000.0

        with mss.mss() as sct:
            while self._running:
                start = time.perf_counter()
                try:
                    monitor = sct.monitors[self._config.monitor_index]  # type: ignore[index]
                    raw = sct.grab(monitor)
                    frame = np.asarray(raw, dtype=np.uint8).copy()

                    with self._lock:
                        self._frame = frame

                    if self._on_frame is not None:
                        self._on_frame(frame)

                except Exception:
                    logger.exception("Error during screen capture.")

                elapsed = time.perf_counter() - start
                sleep_time = max(0.0, interval_s - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
