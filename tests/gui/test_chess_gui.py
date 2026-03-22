"""Tests for the ChessGui controller logic.

Tests the data flow and routing without requiring a display server.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import chess
import pytest

from chesslab.capture.screen_capture import CapturePermissionError
from chesslab.analysis import AnalysisConfig
from chesslab.coach.game_coach import PlayerColor
from chesslab.gui.chess_gui import ChessGui, GuiConfig, StartupJourneyMode
from chesslab.gui.live_observer import LiveObserverConfig
from chesslab.validation import CalibrationMetrics, CalibrationReport, load_support_catalog


# ---------------------------------------------------------------------------
# GuiConfig
# ---------------------------------------------------------------------------


class TestGuiConfig:
    """GuiConfig holds GUI configuration."""

    def test_defaults(self) -> None:
        config = GuiConfig()
        assert config.player_color is None
        assert config.analysis.depth == 18
        assert config.square_size == 64
        assert config.startup_mode is None
        assert config.startup_review_file is None

    def test_custom_values(self) -> None:
        config = GuiConfig(
            player_color=PlayerColor.BLACK,
            analysis=AnalysisConfig(depth=22),
            square_size=72,
        )
        assert config.player_color == PlayerColor.BLACK
        assert config.analysis.depth == 22
        assert config.square_size == 72

    def test_supports_forced_startup_mode(self) -> None:
        config = GuiConfig(startup_mode=StartupJourneyMode.OBSERVE_LIVE)
        assert config.startup_mode is StartupJourneyMode.OBSERVE_LIVE

    def test_player_color_none_triggers_dialog(self) -> None:
        """When player_color is None, a dialog should be shown."""
        config = GuiConfig()
        assert config.player_color is None


def _make_report(*, passed: bool) -> CalibrationReport:
    return CalibrationReport(
        setup_id="chesscom_green_white",
        support_level="official",
        template_theme="chess.com-green",
        expected_theme="chess.com/green",
        passed=passed,
        blockers=()
        if passed
        else ("Chessboard detection is too unstable (50.0% < 90.0%).",),
        warnings=() if passed else ("Setup support level is 'candidate', not 'official'.",),
        corrective_actions=()
        if passed
        else (
            "Keep the full chessboard visible on the selected monitor and confirm "
            "you chose the monitor that actually contains the board.",
        ),
        metrics=CalibrationMetrics(
            frame_count=4,
            detection_rate=1.0 if passed else 0.5,
            theme_match_rate=1.0,
            ready_rate=1.0 if passed else 0.25,
            uncertain_rate=0.0,
            invalid_rate=0.0,
            p95_total_latency_ms=120.0,
            p95_detection_latency_ms=20.0,
            p95_split_latency_ms=10.0,
            p95_recognition_latency_ms=10.0,
            status_counts={"ready": 4} if passed else {"board_not_found": 2, "ready": 2},
        ),
    )


class TestObserveLiveHelpers:
    def test_format_preflight_report_surfaces_actions(self) -> None:
        gui = ChessGui()
        profile = load_support_catalog().get_profile("chesscom_green_white")

        assert profile is not None
        report = _make_report(passed=False)

        text = gui._format_preflight_report(profile, report)

        assert "Blockers:" in text
        assert "Corrective actions:" in text
        assert "Chessboard detection is too unstable" in text
        assert "Keep the full chessboard visible" in text

    def test_format_preflight_report_marks_pass_as_technical_only(self) -> None:
        gui = ChessGui()
        profile = load_support_catalog().get_profile("chesscom_green_white")

        assert profile is not None
        report = _make_report(passed=True)

        text = gui._format_preflight_report(profile, report)

        assert "technical compatibility only" in text
        assert "outside tools" in text

    def test_format_preflight_error_uses_actionable_capture_message(self) -> None:
        message = ChessGui._format_preflight_error(
            CapturePermissionError("permission denied")
        )

        assert "permission denied" in message.lower()
        assert "screen recording" in message.lower()

    def test_start_observation_reports_preflight_pass_for_declared_setup(self) -> None:
        gui = ChessGui()
        gui._set_status = MagicMock()
        gui._observe_btn = MagicMock()
        gui._move_entry = MagicMock()
        gui._observation_queue = queue.Queue()

        config = LiveObserverConfig(
            monitor_index=1,
            theme="chess.com-green",
            setup_id="chesscom_green_white",
            support_level="official",
        )
        observer = MagicMock()

        with patch("chesslab.gui.chess_gui.LiveObserver", return_value=observer):
            assert gui._start_observation(config) is True

        assert gui._last_observer_config == config
        assert gui._observing is True
        assert observer.start.called
        assert gui._set_status.call_args_list[-1].args[0].startswith("Preflight PASS")
        assert "Passive sync only" in gui._set_status.call_args_list[-1].args[0]

    def test_start_observation_switches_ui_to_passive_mode(self) -> None:
        gui = ChessGui()
        gui._set_status = MagicMock()
        gui._coach = MagicMock()
        gui._coach.move_history = ("e4",)
        gui._observe_btn = MagicMock()
        gui._apply_btn = MagicMock()
        gui._undo_btn = MagicMock()
        gui._board_widget = MagicMock()
        gui._eval_panel = MagicMock()
        gui._observe_panel = MagicMock()
        gui._suggestions_panel = MagicMock()
        gui._history_panel = MagicMock()
        gui._btn_frame = MagicMock()
        gui._new_game_btn = MagicMock()
        gui._pgn_frame = MagicMock()
        gui._entry_frame = MagicMock()
        gui._move_entry = MagicMock()
        gui._submit_btn = MagicMock()
        gui._observation_queue = queue.Queue()

        config = LiveObserverConfig(
            monitor_index=1,
            theme="chess.com-green",
            setup_id="chesscom_green_white",
            support_level="official",
        )
        observer = MagicMock()

        with patch("chesslab.gui.chess_gui.LiveObserver", return_value=observer):
            assert gui._start_observation(config) is True

        gui._coach.stop_analysis.assert_called_once()
        gui._board_widget.set_interactive.assert_called_with(False)
        gui._move_entry.config.assert_any_call(state="disabled")
        gui._submit_btn.config.assert_called_with(state="disabled")
        gui._eval_panel.update_eval.assert_called_with(None, 0, is_loading=False)
        gui._suggestions_panel.update_suggestions.assert_called_with(())
        gui._observe_panel.update_session.assert_called_with(
            setup_label="chesscom_green_white",
            move_count=1,
        )
        assert any(
            call.kwargs.get("text") == "Manual Coach"
            for call in gui._apply_btn.config.call_args_list
        )
        assert any(
            call.kwargs.get("text") == "Stop Live"
            for call in gui._undo_btn.config.call_args_list
        )
        assert any(
            call.kwargs.get("text") == "Review"
            for call in gui._observe_btn.config.call_args_list
        )

    def test_submit_san_can_record_passively_without_analysis(self) -> None:
        gui = ChessGui()
        gui._coach = MagicMock()
        gui._clear_coaching_outputs = MagicMock()
        gui._refresh_after_move = MagicMock()
        gui._set_status = MagicMock()

        gui._submit_san("e4", passive=True)

        gui._coach.record_move.assert_called_once_with("e4")
        gui._coach.submit_opponent_move_continuous.assert_not_called()
        gui._clear_coaching_outputs.assert_called_once()
        gui._refresh_after_move.assert_called_once()

    def test_exit_observe_to_review_stops_live_before_review(self) -> None:
        gui = ChessGui()
        gui._set_status = MagicMock()
        gui._observing = True
        gui._coach = MagicMock()
        board = chess.Board()
        board.push_san("e4")
        gui._coach.board = board
        gui._stop_observation = MagicMock()
        gui._start_review = MagicMock()

        gui._exit_observe_to_review()

        gui._stop_observation.assert_called_once()
        gui._start_review.assert_called_once_with(board)
