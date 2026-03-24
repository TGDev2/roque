"""Tests for the ChessGui controller logic.

Tests the data flow and routing without requiring a display server.
"""

from __future__ import annotations

import queue
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import pytest

from chesslab.capture.screen_capture import CapturePermissionError
from chesslab.analysis import AnalysisConfig, ContinuousUpdate
from chesslab.coach.game_coach import ContinuousAdvice, PlayerColor
from chesslab.gui.chess_gui import ChessGui, GuiConfig, StartupJourneyMode
from chesslab.gui.live_observer import LiveObserverConfig
from chesslab.persistence.remote_import import RemoteGameSource
from chesslab.persistence.review_history import TrainingPositionSummary
from chesslab.review.batch_review import (
    BatchReviewItemResult,
    BatchReviewItemStatus,
    BatchReviewResult,
    BatchReviewTarget,
)
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
from chesslab.training import TrainingExercise, TrainingFeedbackLabel, TrainingMoveFeedback


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


def _make_review_move(
    classification: MoveClassification,
    *,
    best_move_uci: str = "d2d4",
    played_uci: str = "e2e4",
) -> ReviewedMove:
    return ReviewedMove(
        ply=0,
        san="e4",
        uci=played_uci,
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        classification=classification,
        eval_before=Evaluation(centipawns=150, mate_in=None),
        eval_after=Evaluation(centipawns=-20, mate_in=None),
        best_move_san="d4" if best_move_uci else "",
        best_move_uci=best_move_uci,
        best_eval=Evaluation(centipawns=150, mate_in=None),
        cp_loss=130,
        best_pv_san=("d4", "d5") if best_move_uci else (),
        phase=GamePhase.OPENING,
        is_white=True,
        is_forced=classification is MoveClassification.FORCED,
    )


def _make_single_move_review(move: ReviewedMove) -> GameReview:
    return GameReview(
        moves=(move,),
        phase_summaries=(),
        white_classification_counts={move.classification: 1},
        black_classification_counts={},
        white_avg_cp_loss=float(move.cp_loss),
        black_avg_cp_loss=0.0,
        critical_moments=(0,),
        lessons=(),
        total_elapsed_ms=100.0,
    )


def _make_training_exercise() -> TrainingExercise:
    return TrainingExercise(
        game_id="game-1",
        reviewed_at="2026-03-24T12:00:00+00:00",
        source_path=None,
        position=TrainingPositionSummary(
            fen=chess.Board().fen(),
            ply=0,
            played_san="e4",
            played_uci="e2e4",
            best_move_san="d4",
            best_move_uci="d2d4",
            best_pv_san=("d4", "d5"),
            classification=MoveClassification.MISTAKE,
            phase=GamePhase.MIDDLEGAME,
            cp_loss=130,
            lesson_themes=(LessonTheme.TURNING_POINT,),
        ),
        relevance_score=250,
        focus_reason="Recurring theme: Turning point",
        focus_theme=LessonTheme.TURNING_POINT,
        focus_phase=GamePhase.MIDDLEGAME,
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

    def test_apply_startup_mode_import_lichess_opens_dialog(self) -> None:
        gui = ChessGui()
        gui._root = MagicMock()
        gui._set_status = MagicMock()
        gui._open_remote_import_dialog = MagicMock()

        gui._apply_startup_mode(StartupJourneyMode.IMPORT_LICHESS)

        gui._open_remote_import_dialog.assert_called_once_with(
            RemoteGameSource.LICHESS,
        )

    def test_apply_startup_mode_import_chess_com_opens_dialog(self) -> None:
        gui = ChessGui()
        gui._root = MagicMock()
        gui._set_status = MagicMock()
        gui._open_remote_import_dialog = MagicMock()

        gui._apply_startup_mode(StartupJourneyMode.IMPORT_CHESS_COM)

        gui._open_remote_import_dialog.assert_called_once_with(
            RemoteGameSource.CHESS_COM,
        )


class TestBoardMoveHints:
    def test_apply_advice_shows_manual_coach_hint(self) -> None:
        gui = ChessGui()
        gui._board_widget = MagicMock()
        gui._eval_panel = MagicMock()
        gui._suggestions_panel = MagicMock()

        advice = ContinuousAdvice(
            opponent_move_san="e4",
            current_fen=chess.Board().fen(),
            update=ContinuousUpdate(
                fen=chess.Board().fen(),
                variations=(),
                evaluation=Evaluation(centipawns=35, mate_in=None),
                best_move_uci="e2e4",
                best_move_san="e4",
                depth_reached=14,
                elapsed_ms=120.0,
            ),
            move_number=1,
        )

        gui._apply_advice(advice)

        assert gui._best_move_uci == "e2e4"
        gui._board_widget.set_move_hint.assert_called_once_with(
            "e2",
            "e4",
            color="#3b82f6",
        )

    def test_apply_advice_clears_hint_when_no_best_move_is_available(self) -> None:
        gui = ChessGui()
        gui._board_widget = MagicMock()
        gui._eval_panel = MagicMock()
        gui._suggestions_panel = MagicMock()
        gui._best_move_uci = "e2e4"

        advice = ContinuousAdvice(
            opponent_move_san="e4",
            current_fen=chess.Board().fen(),
            update=ContinuousUpdate(
                fen=chess.Board().fen(),
                variations=(),
                evaluation=Evaluation(centipawns=0, mate_in=None),
                best_move_uci="",
                best_move_san="",
                depth_reached=14,
                elapsed_ms=120.0,
            ),
            move_number=1,
        )

        gui._apply_advice(advice)

        assert gui._best_move_uci is None
        gui._board_widget.clear_highlights.assert_called_once()

    @pytest.mark.parametrize(
        ("classification", "colour"),
        [
            (MoveClassification.INACCURACY, "#ffee58"),
            (MoveClassification.MISTAKE, "#ffb74d"),
            (MoveClassification.BLUNDER, "#ef5350"),
        ],
    )
    def test_review_navigation_shows_hint_for_pedagogical_errors(
        self,
        classification: MoveClassification,
        colour: str,
    ) -> None:
        gui = ChessGui()
        gui._in_review_mode = True
        gui._board_widget = MagicMock()
        gui._eval_panel = MagicMock()
        gui._review_panel = MagicMock()
        gui._review_nav = MagicMock()
        gui._history_panel = MagicMock()
        gui._set_status = MagicMock()

        move = _make_review_move(classification)
        gui._game_review = _make_single_move_review(move)
        gui._review_positions = [move.fen_before, move.fen_after]

        gui._review_navigate_to(1)

        gui._board_widget.set_move_hint.assert_called_once_with(
            "d2",
            "d4",
            color=colour,
        )

    @pytest.mark.parametrize(
        "classification",
        [
            MoveClassification.EXCELLENT,
            MoveClassification.GOOD,
            MoveClassification.FORCED,
        ],
    )
    def test_review_navigation_hides_hint_for_non_error_moves(
        self,
        classification: MoveClassification,
    ) -> None:
        gui = ChessGui()
        gui._in_review_mode = True
        gui._board_widget = MagicMock()
        gui._eval_panel = MagicMock()
        gui._review_panel = MagicMock()
        gui._review_nav = MagicMock()
        gui._history_panel = MagicMock()
        gui._set_status = MagicMock()

        move = _make_review_move(classification)
        gui._game_review = _make_single_move_review(move)
        gui._review_positions = [move.fen_before, move.fen_after]

        gui._review_navigate_to(1)

        gui._board_widget.clear_highlights.assert_called_once()
        gui._board_widget.set_move_hint.assert_not_called()


class TestTrainingMode:
    def test_board_move_routes_to_training_when_mode_is_active(self) -> None:
        gui = ChessGui()
        gui._in_training_mode = True
        gui._on_training_board_move = MagicMock()

        gui._on_board_move("e2", "e4")

        gui._on_training_board_move.assert_called_once_with("e2", "e4")

    def test_start_training_from_review_uses_current_review_as_focus(self) -> None:
        gui = ChessGui()
        review = _make_review_with_lesson()
        gui._game_review = review
        gui._start_training_session = MagicMock()

        gui._start_training_from_review()

        gui._start_training_session.assert_called_once_with(
            sample_size=gui._progression_sample_size,
            focus_review=review,
        )

    def test_start_training_session_surfaces_unavailable_message(self) -> None:
        gui = ChessGui()
        gui._review_history = MagicMock()
        gui._review_history.list_recent_reviews.return_value = []
        gui._training_service = MagicMock()
        gui._training_service.build_session.return_value = ()
        gui._show_training_unavailable = MagicMock()

        gui._start_training_session(sample_size=5, focus_review=None)

        gui._show_training_unavailable.assert_called_once()

    def test_training_feedback_updates_panel_and_status(self) -> None:
        gui = ChessGui()
        gui._in_training_mode = True
        gui._training_panel = MagicMock()
        gui._training_session = (_make_training_exercise(),)
        gui._set_status = MagicMock()

        feedback = TrainingMoveFeedback(
            label=TrainingFeedbackLabel.GOOD,
            played_san="Nf3",
            played_uci="g1f3",
            cp_loss=5,
            move_rank=3,
            best_move_san="e4",
            best_move_uci="e2e4",
            best_line_san=("e4", "e5"),
            top_move_sans=("e4", "d4", "Nf3"),
            message="Bon coup.",
        )

        gui._on_training_feedback(feedback)

        gui._training_panel.show_feedback.assert_called_once_with(
            feedback,
            is_last=True,
        )
        gui._set_status.assert_called_once_with("Training — good move")

    def test_exit_training_restores_history_without_unpacked_button_anchor(self) -> None:
        gui = ChessGui()
        gui._in_training_mode = True
        gui._training_origin_review = False
        gui._training_mode_widgets = [MagicMock()]
        gui._btn_frame = MagicMock()
        gui._btn_frame.winfo_manager.return_value = ""
        gui._history_panel = MagicMock()
        gui._history_panel.winfo_manager.return_value = ""
        gui._set_observe_mode_ui = MagicMock()
        gui._set_status = MagicMock()

        gui._exit_training_mode()

        gui._history_panel.pack.assert_called_once_with(
            fill="both",
            expand=True,
            pady=(0, 6),
        )

    def test_show_history_panel_keeps_button_frame_order_when_anchor_is_packed(self) -> None:
        gui = ChessGui()
        gui._btn_frame = MagicMock()
        gui._btn_frame.winfo_manager.return_value = "pack"
        gui._history_panel = MagicMock()
        gui._history_panel.winfo_manager.return_value = ""

        gui._show_history_panel()

        gui._history_panel.pack.assert_called_once_with(
            before=gui._btn_frame,
            fill="both",
            expand=True,
            pady=(0, 6),
        )


class TestBatchReviewHelpers:
    def test_build_batch_review_completion_text_reports_ready_activation(self) -> None:
        gui = ChessGui()
        gui._progress_tracker = MagicMock()
        gui._progress_tracker.build_snapshot.return_value = MagicMock(
            has_enough_games=True,
            games_remaining=0,
        )
        gui._review_history = MagicMock()
        gui._review_history.list_recent_reviews.return_value = ["summary"]
        gui._training_service = MagicMock()
        gui._training_service.build_session.return_value = ("exercise",)

        message, progression_ready, training_ready = (
            gui._build_batch_review_completion_text(
            BatchReviewResult(items=(
                BatchReviewItemResult(
                    target=BatchReviewTarget(file_path=Path("/tmp/game.pgn"), label="Game"),
                    status=BatchReviewItemStatus.COMPLETED,
                    message="done",
                ),
            )),
            )
        )

        assert progression_ready is True
        assert training_ready is True
        assert "Progression and targeted training are ready" in message

    def test_on_batch_review_complete_enables_activation_buttons(self) -> None:
        gui = ChessGui()
        gui._refresh_batch_review_progress = MagicMock()
        gui._set_status = MagicMock()
        gui._batch_review_current_label = MagicMock()
        gui._batch_review_progression_btn = MagicMock()
        gui._batch_review_training_btn = MagicMock()
        gui._batch_review_close_btn = MagicMock()
        gui._build_batch_review_completion_text = MagicMock(
            return_value=("Batch ready", True, True)
        )

        result = BatchReviewResult(items=(
            BatchReviewItemResult(
                target=BatchReviewTarget(file_path=Path("/tmp/game.pgn"), label="Game"),
                status=BatchReviewItemStatus.COMPLETED,
                message="done",
            ),
        ))

        gui._on_batch_review_complete(result)

        gui._batch_review_progression_btn.config.assert_called_once_with(
            state="normal",
        )
        gui._batch_review_training_btn.config.assert_called_once_with(
            state="normal",
        )
        gui._batch_review_close_btn.config.assert_called_once()
        gui._set_status.assert_called_once()


class TestManualCoachReviewOffer:
    def test_submit_san_offers_review_when_player_move_finishes_game(self) -> None:
        gui = ChessGui()
        gui._coach = MagicMock()
        gui._coach.is_player_turn = True
        gui._coach.is_game_over = True
        gui._clear_coaching_outputs = MagicMock()
        gui._refresh_after_move = MagicMock()
        gui._handle_finished_manual_game = MagicMock()

        gui._submit_san("Qxf7#")

        gui._coach.submit_player_move.assert_called_once_with("Qxf7#")
        gui._clear_coaching_outputs.assert_called_once()
        gui._handle_finished_manual_game.assert_called_once()
        gui._coach.submit_opponent_move_continuous.assert_not_called()

    def test_submit_san_clears_stale_coaching_outputs_after_player_move(self) -> None:
        gui = ChessGui()
        gui._coach = MagicMock()
        gui._coach.is_player_turn = True
        gui._coach.is_game_over = False
        gui._refresh_after_move = MagicMock()
        gui._clear_coaching_outputs = MagicMock()

        gui._submit_san("Nf3")

        gui._coach.submit_player_move.assert_called_once_with("Nf3")
        gui._clear_coaching_outputs.assert_called_once()
        gui._refresh_after_move.assert_called_once()

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
