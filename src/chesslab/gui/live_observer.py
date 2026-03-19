"""Live observation mode — bridges the CV pipeline into the GUI.

Orchestrates screen capture, board detection, piece recognition, and FEN
generation in a background thread.  Position changes are detected via the
monitoring module and enqueued as structured events for the GUI's main-thread
poll loop to consume.

Usage::

    obs_queue: queue.Queue[ObservationEvent] = queue.Queue()
    config = LiveObserverConfig(monitor_index=1, theme="chess.com/green")
    observer = LiveObserver(config, obs_queue)
    observer.start()
    # ... poll obs_queue from the main thread ...
    observer.stop()
"""

from __future__ import annotations

import enum
import logging
import queue
import threading
from dataclasses import dataclass

import chess
from numpy.typing import NDArray

from chesslab.capture.screen_capture import CaptureConfig, ScreenCapture
from chesslab.live.live_position_processor import (
    LiveFrameStatus,
    LivePositionProcessor,
    LiveProcessorConfig,
)
from chesslab.monitoring.position_watcher import (
    PositionChangeEvent,
    PositionWatcher,
    PositionWatcherConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LiveObserverConfig:
    """Configuration for the live observation mode.

    Parameters
    ----------
    monitor_index:
        Screen monitor to capture (1-based, matching mss convention).
    theme:
        Chess board theme for piece recognition (e.g. ``"chess.com/green"``).
    white_perspective:
        ``True`` if the observed board has white at the bottom.
    capture_interval_ms:
        Milliseconds between screen captures.
    debounce_ms:
        Debounce delay before firing a position change event.
    """

    monitor_index: int = 1
    theme: str = "chess.com/green"
    white_perspective: bool = True
    capture_interval_ms: int = 500
    debounce_ms: int = 300


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class ObservationEventType(enum.Enum):
    """Type of event emitted by the live observer."""

    MOVE_DETECTED = "move_detected"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass(frozen=True)
class ObservationEvent:
    """Structured event passed from the observer to the GUI main thread.

    Attributes
    ----------
    type:
        The kind of event.
    fen:
        Full FEN string when available (``MOVE_DETECTED``).
    piece_placement:
        First FEN field (piece placement only) for move deduction.
    message:
        Human-readable description of the event.
    """

    type: ObservationEventType
    fen: str | None = None
    piece_placement: str | None = None
    message: str = ""


# ---------------------------------------------------------------------------
# Move deduction
# ---------------------------------------------------------------------------


def deduce_move_from_fen(
    board: chess.Board,
    detected_placement: str,
) -> chess.Move | None:
    """Find the legal move that produces *detected_placement*.

    Iterates all legal moves on *board*, pushes each on a copy, and
    compares the resulting piece-placement field (first FEN token) with
    *detected_placement*.  Returns the matching :class:`chess.Move` or
    ``None`` if no legal move produces the detected position.

    Only the piece-placement portion of the FEN is compared because the
    CV pipeline cannot reliably determine active colour, castling rights,
    or en-passant state.

    Parameters
    ----------
    board:
        The current board state (from the coach).
    detected_placement:
        The piece-placement field extracted from the detected FEN
        (e.g. ``"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"``).
    """
    for move in board.legal_moves:
        board.push(move)
        placement = board.fen().split(" ", 1)[0]
        board.pop()
        if placement == detected_placement:
            return move
    return None


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


class LiveObserver:
    """Orchestrates the CV pipeline for live observation mode.

    Owns :class:`ScreenCapture`, :class:`LivePositionProcessor`, and
    :class:`PositionWatcher`.  All communication with the GUI happens
    through a :class:`queue.Queue` of :class:`ObservationEvent` objects.
    """

    def __init__(
        self,
        config: LiveObserverConfig,
        event_queue: queue.Queue[ObservationEvent],
        *,
        screen_capture: ScreenCapture | None = None,
        processor: LivePositionProcessor | None = None,
        watcher: PositionWatcher | None = None,
    ) -> None:
        self._config = config
        self._event_queue = event_queue

        self._capture = screen_capture or ScreenCapture(
            CaptureConfig(
                interval_ms=config.capture_interval_ms,
                monitor_index=config.monitor_index,
            ),
        )
        self._processor = processor or LivePositionProcessor(
            LiveProcessorConfig(
                theme=config.theme,
                white_perspective=config.white_perspective,
            ),
        )
        self._watcher = watcher or PositionWatcher(
            PositionWatcherConfig(debounce_ms=config.debounce_ms),
        )

        self._running = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """``True`` if the observer is actively capturing."""
        return self._running

    def start(self) -> None:
        """Start screen capture and position monitoring.

        Raises
        ------
        chesslab.capture.screen_capture.ScreenCaptureError
            If monitor selection or capture access fails.
        """
        with self._lock:
            if self._running:
                logger.warning("LiveObserver already running.")
                return

            self._watcher.on_change(self._on_position_change)
            self._capture.on_frame(self._on_frame)
            self._capture.start()
            self._running = True

        logger.info(
            "LiveObserver started: monitor=%d theme=%s",
            self._config.monitor_index,
            self._config.theme,
        )

    def stop(self) -> None:
        """Stop capture and monitoring.  Safe to call multiple times."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        self._capture.stop()
        self._watcher.stop()
        logger.info("LiveObserver stopped.")

    # -- Internal callbacks --------------------------------------------------

    def _on_frame(self, frame: NDArray) -> None:
        """Called by ScreenCapture for each captured frame (background thread)."""
        if not self._running:
            return

        result = self._processor.process_frame(frame)

        if result.status is LiveFrameStatus.READY:
            if result.changed and result.fen is not None:
                self._watcher.update(result.fen)
        else:
            self._event_queue.put(ObservationEvent(
                type=ObservationEventType.ERROR,
                message=result.message,
            ))

    def _on_position_change(
        self,
        event: PositionChangeEvent,
        cancellation: threading.Event,
    ) -> None:
        """Called by PositionWatcher after debounce (timer thread)."""
        if cancellation.is_set() or not self._running:
            return

        placement = event.current_fen.split(" ", 1)[0]

        self._event_queue.put(ObservationEvent(
            type=ObservationEventType.MOVE_DETECTED,
            fen=event.current_fen,
            piece_placement=placement,
            message=f"Position change detected ({event.elapsed_ms:.0f} ms)",
        ))
