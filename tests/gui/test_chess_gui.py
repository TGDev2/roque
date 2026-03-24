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
from chesslab.review.game_reviewer import (
    GamePhase,
    GameReview,
    LessonTheme,
    MoveClassification,
    PhaseSummary,
    ReviewLesson,
    ReviewedMove,
)
from chesslab.validation import CalibrationMetrics, CalibrationReport, load_support_catalog
from chesslab.analysis import Evaluation


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
        assert config.progression_sample_size == 20

    def test_custom_values(self) -> None:
        config = GuiConfig(
            player_color=PlayerColor.BLACK,
            analysis=AnalysisConfig(depth=22),
            square_size=72,
            progression_sample_size=30,
        )
        assert config.player_color == PlayerColor.BLACK
        assert config.analysis.depth == 22
        assert config.square_size == 72
        assert config.progression_sample_size == 30

    def test_supports_forced_startup_mode(self) -> None:
        config = GuiConfig(startup_mode=StartupJourneyMode.OBSERVE_LIVE)
        assert config.startup_mode is StartupJourneyMode.OBSERVE_LIVE

    def test_player_color_none_triggers_dialog(self) -> None:
        """When player_color is None, a dialog should be shown."""
        config = GuiConfig()
        assert config.player_color is None

    def test_supports_progression_startup_mode(self) -> None:
        config = GuiConfig(startup_mode=StartupJourneyMode.PROGRESSION)
        assert config.startup_mode is StartupJourneyMode.PROGRESSION


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


def _make_review_with_lesson() -> GameReview:
    move = ReviewedMove(
        ply=0,
        san="e4",
        uci="e2e4",
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        classification=MoveClassification.MISTAKE,
        eval_before=Evaluation(centipawns=150, mate_in=None),
        eval_after=Evaluation(centipawns=-20, mate_in=None),
        best_move_san="d4",
        best_move_uci="d2d4",
        best_eval=Evaluation(centipawns=150, mate_in=None),
        cp_loss=130,
        best_pv_san=("d4", "d5"),
        phase=GamePhase.OPENING,
        is_white=True,
        is_forced=False,
    )

    return GameReview(
        moves=(move,),
        phase_summaries=(
            PhaseSummary(
                phase=GamePhase.OPENING,
                move_count=1,
                avg_cp_loss=130.0,
                classification_counts={MoveClassification.MISTAKE: 1},
            ),
        ),
        white_classification_counts={MoveClassification.MISTAKE: 1},
        black_classification_counts={},
        white_avg_cp_loss=130.0,
        black_avg_cp_loss=0.0,
        critical_moments=(0,),
        lessons=(
            ReviewLesson(
                priority=1,
                theme=LessonTheme.TURNING_POINT,
                title="Turning point: 1. e4",
                finding="1. e4 cost about 130cp for White.",
                takeaway="Compare forcing alternatives before committing.",
                focus_ply=0,
                related_plies=(0,),
                phase=GamePhase.OPENING,
                is_white=True,
            ),
        ),
        total_elapsed_ms=250.0,
    )


def _make_checkmate_board() -> chess.Board:
    board = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        board.push_san(san)
    return board


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

    def test_review_complete_wires_lesson_navigation(self) -> None:
        gui = ChessGui()
        gui._close_progress_dialog = MagicMock()
        gui._save_review_summary = MagicMock(return_value=True)
        gui._play_mode_widgets = [MagicMock()]
        gui._entry_frame = MagicMock()
        gui._review_panel = MagicMock()
        gui._review_nav = MagicMock()
        gui._summary_panel = MagicMock()
        gui._history_panel = MagicMock()
        gui._eval_panel = MagicMock()
        gui._set_status = MagicMock()
        gui._review_navigate_to = MagicMock()

        review = _make_review_with_lesson()

        gui._on_review_complete(review)

        callback = gui._summary_panel.update_summary.call_args.kwargs["on_lesson_click"]
        callback(0)

        gui._review_navigate_to.assert_called_with(1)

    def test_apply_startup_mode_progression_opens_dashboard(self) -> None:
        gui = ChessGui()
        gui._root = MagicMock()
        gui._set_status = MagicMock()
        gui._show_progression_dialog = MagicMock()

        gui._apply_startup_mode(StartupJourneyMode.PROGRESSION)

        gui._show_progression_dialog.assert_called_once()


class TestManualCoachReviewOffer:
    def test_submit_san_offers_review_when_player_move_finishes_game(self) -> None:
        gui = ChessGui()
        gui._coach = MagicMock()
        gui._coach.is_player_turn = True
        gui._coach.is_game_over = True
        gui._refresh_after_move = MagicMock()
        gui._handle_finished_manual_game = MagicMock()

        gui._submit_san("Qxf7#")

        gui._coach.submit_player_move.assert_called_once_with("Qxf7#")
        gui._handle_finished_manual_game.assert_called_once()
        gui._coach.submit_opponent_move_continuous.assert_not_called()

    def test_submit_san_skips_loading_state_when_opponent_move_finishes_game(self) -> None:
        gui = ChessGui()
        gui._coach = MagicMock()
        gui._coach.is_player_turn = False
        gui._coach.is_game_over = True
        gui._refresh_after_move = MagicMock()
        gui._handle_finished_manual_game = MagicMock()
        gui._eval_panel = MagicMock()

        handle = MagicMock()
        gui._coach.submit_opponent_move_continuous.return_value = handle

        gui._submit_san("Qh4#")

        gui._coach.stop_analysis.assert_called_once()
        gui._coach.submit_opponent_move_continuous.assert_called_once()
        gui._handle_finished_manual_game.assert_called_once()
        gui._eval_panel.update_eval.assert_not_called()

    def test_handle_finished_manual_game_auto_saves_and_offers_review(self) -> None:
        gui = ChessGui()
        gui._coach = MagicMock()
        gui._coach.board = _make_checkmate_board()
        gui._clear_coaching_outputs = MagicMock()
        gui._auto_save = MagicMock()
        gui._propose_manual_review = MagicMock()

        gui._handle_finished_manual_game()

        gui._clear_coaching_outputs.assert_called_once()
        gui._auto_save.assert_called_once()
        proposed_board = gui._propose_manual_review.call_args.args[0]
        assert proposed_board.result() == "0-1"
        assert len(proposed_board.move_stack) == 4

    def test_new_game_closes_post_game_review_prompt(self) -> None:
        gui = ChessGui()
        gui._auto_save = MagicMock()
        gui._coach = MagicMock()
        gui._board_widget = MagicMock()
        gui._eval_panel = MagicMock()
        gui._suggestions_panel = MagicMock()
        gui._history_panel = MagicMock()
        gui._update_turn_status = MagicMock()
        gui._root = MagicMock()
        gui._show_color_dialog = MagicMock(
            return_value=(PlayerColor.WHITE, AnalysisConfig.DEFAULT_ENGINE_ELO)
        )

        new_coach = MagicMock()
        new_coach.current_fen = chess.Board().fen()
        gui._create_coach = MagicMock(return_value=new_coach)

        dialog = MagicMock()
        dialog.winfo_exists.return_value = True
        gui._post_game_review_dialog = dialog

        gui._on_new_game()

        dialog.destroy.assert_called_once()

    def test_return_to_launcher_closes_post_game_review_prompt(self) -> None:
        gui = ChessGui()
        gui._exit_review_mode = MagicMock()
        gui._root = MagicMock()
        gui._show_startup_dialog = MagicMock(
            return_value=StartupJourneyMode.MANUAL_COACH
        )
        gui._apply_startup_mode = MagicMock()

        dialog = MagicMock()
        dialog.winfo_exists.return_value = True
        gui._post_game_review_dialog = dialog

        gui._return_to_launcher()

        dialog.destroy.assert_called_once()
        gui._show_startup_dialog.assert_called_once_with(gui._root)
        gui._apply_startup_mode.assert_called_once_with(
            StartupJourneyMode.MANUAL_COACH
        )
