"""Tests for the LiveObserver lifecycle and event pipeline.

All tests are headless — no mss, no tkinter, no Stockfish required.
Move deduction tests have been moved to tests/live/test_move_deduction.py.
"""

from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock

import chess

from chesslab.capture.screen_capture import CapturePermissionError
from chesslab.gui.live_observer import (
    LiveObserver,
    LiveObserverConfig,
    ObservationEvent,
    ObservationEventType,
)
from chesslab.live.live_position_processor import LiveFrameResult, LiveFrameStatus
from chesslab.monitoring.position_watcher import PositionChangeEvent


# ---------------------------------------------------------------------------
# LiveObserver lifecycle
# ---------------------------------------------------------------------------


class TestLiveObserverLifecycle:
    """Tests for observer start/stop behaviour using mocks."""

    def _make_observer(
        self,
    ) -> tuple[LiveObserver, queue.Queue[ObservationEvent], MagicMock, MagicMock, MagicMock]:
        eq: queue.Queue[ObservationEvent] = queue.Queue()
        mock_capture = MagicMock()
        mock_capture.is_running = False
        mock_processor = MagicMock()
        mock_watcher = MagicMock()

        config = LiveObserverConfig(monitor_index=1, theme="chess.com/green")
        observer = LiveObserver(
            config,
            eq,
            screen_capture=mock_capture,
            processor=mock_processor,
            watcher=mock_watcher,
        )
        return observer, eq, mock_capture, mock_processor, mock_watcher

    def test_start_calls_capture_start(self) -> None:
        observer, _, mock_capture, _, _ = self._make_observer()

        observer.start()

        mock_capture.on_frame.assert_called_once()
        mock_capture.on_stopped.assert_called_once()
        mock_capture.start.assert_called_once()
        assert observer.is_running

    def test_stop_calls_capture_and_watcher_stop(self) -> None:
        observer, _, mock_capture, _, mock_watcher = self._make_observer()
        observer.start()

        observer.stop()

        mock_capture.stop.assert_called_once()
        mock_watcher.stop.assert_called_once()
        assert not observer.is_running

    def test_double_stop_is_safe(self) -> None:
        observer, _, mock_capture, _, mock_watcher = self._make_observer()
        observer.start()

        observer.stop()
        observer.stop()

        # Only called once despite double stop.
        mock_capture.stop.assert_called_once()

    def test_double_start_is_safe(self) -> None:
        observer, _, mock_capture, _, _ = self._make_observer()

        observer.start()
        observer.start()

        mock_capture.start.assert_called_once()

    def test_is_running_reflects_state(self) -> None:
        observer, _, _, _, _ = self._make_observer()

        assert not observer.is_running
        observer.start()
        assert observer.is_running
        observer.stop()
        assert not observer.is_running

    def test_unexpected_capture_stop_enqueues_stopped_event(self) -> None:
        observer, eq, mock_capture, _, _ = self._make_observer()
        observer.start()

        stop_callback = mock_capture.on_stopped.call_args.args[0]
        stop_callback(CapturePermissionError("permission denied"))

        event = eq.get_nowait()
        assert event.type is ObservationEventType.STOPPED
        assert "permission" in event.message.lower()
        assert not observer.is_running


# ---------------------------------------------------------------------------
# Frame processing → event queue
# ---------------------------------------------------------------------------


class TestLiveObserverFrameProcessing:
    """Tests for the frame callback pipeline using injected mocks."""

    def _make_started_observer(
        self,
    ) -> tuple[LiveObserver, queue.Queue[ObservationEvent], MagicMock, MagicMock, MagicMock]:
        eq: queue.Queue[ObservationEvent] = queue.Queue()
        mock_capture = MagicMock()
        mock_processor = MagicMock()
        mock_watcher = MagicMock()

        config = LiveObserverConfig(monitor_index=1)
        observer = LiveObserver(
            config,
            eq,
            screen_capture=mock_capture,
            processor=mock_processor,
            watcher=mock_watcher,
        )
        observer.start()
        return observer, eq, mock_capture, mock_processor, mock_watcher

    def _make_frame_result(
        self,
        status: LiveFrameStatus = LiveFrameStatus.READY,
        changed: bool = True,
        fen: str | None = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        message: str = "OK",
    ) -> LiveFrameResult:
        return LiveFrameResult(
            status=status,
            message=message,
            changed=changed,
            total_elapsed_ms=50.0,
            detection_elapsed_ms=20.0,
            split_elapsed_ms=10.0,
            recognition_elapsed_ms=10.0,
            state_elapsed_ms=5.0,
            fen_elapsed_ms=5.0,
            board=None,
            selected_theme="chess.com/green",
            recognition_status="ready",
            uncertain_squares=(),
            fen=fen,
            piece_placement=fen.split(" ", 1)[0] if fen else None,
        )

    def test_ready_changed_feeds_watcher(self) -> None:
        observer, eq, _, mock_processor, mock_watcher = self._make_started_observer()
        result = self._make_frame_result(status=LiveFrameStatus.READY, changed=True)
        mock_processor.process_frame.return_value = result

        import numpy as np
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        observer._on_frame(frame)

        mock_watcher.update.assert_called_once_with(result.fen)
        observer.stop()

    def test_ready_unchanged_does_not_feed_watcher(self) -> None:
        observer, eq, _, mock_processor, mock_watcher = self._make_started_observer()
        result = self._make_frame_result(status=LiveFrameStatus.READY, changed=False)
        mock_processor.process_frame.return_value = result

        import numpy as np
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        observer._on_frame(frame)

        mock_watcher.update.assert_not_called()
        observer.stop()

    def test_board_not_found_enqueues_error(self) -> None:
        observer, eq, _, mock_processor, _ = self._make_started_observer()
        result = self._make_frame_result(
            status=LiveFrameStatus.BOARD_NOT_FOUND,
            changed=False,
            fen=None,
            message="No chessboard detected in frame.",
        )
        mock_processor.process_frame.return_value = result

        import numpy as np
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        observer._on_frame(frame)

        event = eq.get_nowait()
        assert event.type is ObservationEventType.ERROR
        assert "No chessboard" in event.message
        observer.stop()

    def test_position_uncertain_enqueues_error(self) -> None:
        observer, eq, _, mock_processor, _ = self._make_started_observer()
        result = self._make_frame_result(
            status=LiveFrameStatus.POSITION_UNCERTAIN,
            changed=False,
            fen=None,
            message="Low confidence squares.",
        )
        mock_processor.process_frame.return_value = result

        import numpy as np
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        observer._on_frame(frame)

        event = eq.get_nowait()
        assert event.type is ObservationEventType.ERROR
        assert "confidence" in event.message.lower()
        observer.stop()

    def test_theme_unrecognized_enqueues_error(self) -> None:
        observer, eq, _, mock_processor, _ = self._make_started_observer()
        result = self._make_frame_result(
            status=LiveFrameStatus.THEME_UNRECOGNIZED,
            changed=False,
            fen=None,
            message="Theme not recognized.",
        )
        mock_processor.process_frame.return_value = result

        import numpy as np
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        observer._on_frame(frame)

        event = eq.get_nowait()
        assert event.type is ObservationEventType.ERROR
        observer.stop()


# ---------------------------------------------------------------------------
# Position change → MOVE_DETECTED event
# ---------------------------------------------------------------------------


class TestPositionChangeEvent:
    """Tests that PositionWatcher change events produce MOVE_DETECTED."""

    def test_position_change_enqueues_move_detected(self) -> None:
        eq: queue.Queue[ObservationEvent] = queue.Queue()
        config = LiveObserverConfig(monitor_index=1)
        mock_capture = MagicMock()
        mock_processor = MagicMock()
        mock_watcher = MagicMock()

        observer = LiveObserver(
            config,
            eq,
            screen_capture=mock_capture,
            processor=mock_processor,
            watcher=mock_watcher,
        )
        observer.start()

        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        change_event = PositionChangeEvent(
            previous_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            current_fen=fen,
            elapsed_ms=320.0,
        )
        cancel = threading.Event()

        observer._on_position_change(change_event, cancel)

        obs_event = eq.get_nowait()
        assert obs_event.type is ObservationEventType.MOVE_DETECTED
        assert obs_event.fen == fen
        assert obs_event.piece_placement == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
        observer.stop()

    def test_cancelled_change_is_ignored(self) -> None:
        eq: queue.Queue[ObservationEvent] = queue.Queue()
        config = LiveObserverConfig(monitor_index=1)
        observer = LiveObserver(
            config,
            eq,
            screen_capture=MagicMock(),
            processor=MagicMock(),
            watcher=MagicMock(),
        )
        observer.start()

        cancel = threading.Event()
        cancel.set()  # Pre-cancelled.

        change_event = PositionChangeEvent(
            previous_fen="",
            current_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            elapsed_ms=300.0,
        )
        observer._on_position_change(change_event, cancel)

        assert eq.empty()
        observer.stop()

    def test_stopped_observer_ignores_change(self) -> None:
        eq: queue.Queue[ObservationEvent] = queue.Queue()
        config = LiveObserverConfig(monitor_index=1)
        observer = LiveObserver(
            config,
            eq,
            screen_capture=MagicMock(),
            processor=MagicMock(),
            watcher=MagicMock(),
        )
        # Not started — _running is False.

        change_event = PositionChangeEvent(
            previous_fen="",
            current_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            elapsed_ms=300.0,
        )
        observer._on_position_change(change_event, threading.Event())

        assert eq.empty()


# ---------------------------------------------------------------------------
# Position change with tracking enabled
# ---------------------------------------------------------------------------


class TestTrackedPositionChangeEvent:
    """Ensure tracking emits explicit sync/new-game/resync events."""

    def _make_started_observer(self) -> tuple[LiveObserver, queue.Queue[ObservationEvent]]:
        eq: queue.Queue[ObservationEvent] = queue.Queue()
        observer = LiveObserver(
            LiveObserverConfig(monitor_index=1, enable_game_tracking=True),
            eq,
            screen_capture=MagicMock(),
            processor=MagicMock(),
            watcher=MagicMock(),
        )
        observer.start()
        return observer, eq

    @staticmethod
    def _change_event(fen: str) -> PositionChangeEvent:
        return PositionChangeEvent(previous_fen="", current_fen=fen, elapsed_ms=300.0)

    def test_first_frame_enqueues_tracking_sync(self) -> None:
        observer, eq = self._make_started_observer()

        start_fen = chess.Board().fen()
        observer._on_position_change(self._change_event(start_fen), threading.Event())

        event = eq.get_nowait()
        assert event.type is ObservationEventType.TRACKING_SYNC
        assert event.tracking is not None
        assert event.fen == start_fen
        observer.stop()

    def test_sequence_enqueues_move_detected_from_tracker(self) -> None:
        observer, eq = self._make_started_observer()
        board = chess.Board()

        observer._on_position_change(self._change_event(board.fen()), threading.Event())
        eq.get_nowait()  # TRACKING_SYNC

        board.push_san("e4")
        observer._on_position_change(self._change_event(board.fen()), threading.Event())

        event = eq.get_nowait()
        assert event.type is ObservationEventType.MOVE_DETECTED
        assert event.tracking is not None
        assert event.tracking.move_san == "e4"
        assert event.fen == board.fen()
        observer.stop()

    def test_persistent_discontinuity_enqueues_resync(self) -> None:
        observer, eq = self._make_started_observer()
        board = chess.Board()

        observer._on_position_change(self._change_event(board.fen()), threading.Event())
        eq.get_nowait()  # TRACKING_SYNC

        impossible = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"
        observer._on_position_change(self._change_event(impossible), threading.Event())
        assert eq.empty()  # Pending confirmation only.

        observer._on_position_change(self._change_event(impossible), threading.Event())
        event = eq.get_nowait()
        assert event.type is ObservationEventType.RESYNC
        assert event.tracking is not None
        observer.stop()

    def test_new_game_enqueues_new_game_event(self) -> None:
        observer, eq = self._make_started_observer()
        board = chess.Board()

        observer._on_position_change(self._change_event(board.fen()), threading.Event())
        eq.get_nowait()  # TRACKING_SYNC

        board.push_san("e4")
        observer._on_position_change(self._change_event(board.fen()), threading.Event())
        eq.get_nowait()  # MOVE_DETECTED

        start_fen = chess.Board().fen()
        observer._on_position_change(self._change_event(start_fen), threading.Event())
        assert eq.empty()  # Pending confirmation.

        observer._on_position_change(self._change_event(start_fen), threading.Event())
        event = eq.get_nowait()
        assert event.type is ObservationEventType.NEW_GAME
        assert event.tracking is not None
        observer.stop()
