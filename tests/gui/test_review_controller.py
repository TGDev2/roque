"""Tests for the review mode controller."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import chess

from chesslab.analysis import Evaluation
from chesslab.gui.controllers import AppMode
from chesslab.gui.controllers.review import ReviewController
from chesslab.persistence.game_storage import GameRecord
from chesslab.persistence.training_history import TrainingAttemptOutcome
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

from tests.gui.controller_test_utils import DummyHost


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


class TestReviewController:
    def _make_controller(self) -> tuple[DummyHost, ReviewController]:
        host = DummyHost()
        controller = ReviewController(host)
        controller._eval_panel = MagicMock()
        controller._history_panel = MagicMock()
        controller._review_panel = MagicMock()
        controller._review_nav = MagicMock()
        controller._summary_panel = MagicMock()
        return host, controller

    def test_review_complete_wires_lesson_navigation(self) -> None:
        host, controller = self._make_controller()
        controller._close_progress_dialog = MagicMock()
        controller._save_review_summary = MagicMock(return_value=True)
        controller._review_navigate_to = MagicMock()

        review = _make_review_with_lesson()
        controller._on_review_complete(review)

        callback = controller._summary_panel.update_summary.call_args.kwargs["on_lesson_click"]
        callback(0)

        controller._review_navigate_to.assert_called_with(1)
        host.set_status.assert_called()

    def test_review_navigation_shows_hint_for_pedagogical_errors(self) -> None:
        host, controller = self._make_controller()
        move = _make_review_move(MoveClassification.MISTAKE)
        controller._game_review = _make_single_move_review(move)
        controller._review_positions = [move.fen_before, move.fen_after]

        controller._review_navigate_to(1)

        host.widgets.board_widget.set_move_hint.assert_called_once_with(
            "d2",
            "d4",
            color="#ffb74d",
        )

    def test_review_navigation_hides_hint_for_non_error_moves(self) -> None:
        host, controller = self._make_controller()
        move = _make_review_move(MoveClassification.GOOD)
        controller._game_review = _make_single_move_review(move)
        controller._review_positions = [move.fen_before, move.fen_after]

        controller._review_navigate_to(1)

        host.widgets.board_widget.clear_highlights.assert_called_once()

    def test_build_batch_review_completion_text_reports_ready_activation(self) -> None:
        host, controller = self._make_controller()
        host.progression_sample_size = 20
        host.services.progress_tracker.build_snapshot.return_value = MagicMock(
            has_enough_games=True,
            games_remaining=0,
        )
        host.services.review_history.list_recent_reviews.return_value = ["summary"]
        host.services.training_history.list_recent_attempts.return_value = ["attempt"]
        host.services.training_service.build_session.return_value = ("exercise",)

        message, progression_ready, training_ready = controller._build_batch_review_completion_text(
            BatchReviewResult(items=(
                BatchReviewItemResult(
                    target=BatchReviewTarget(file_path=Path("/tmp/game.pgn"), label="Game"),
                    status=BatchReviewItemStatus.COMPLETED,
                    message="done",
                ),
            )),
        )

        assert progression_ready is True
        assert training_ready is True
        assert "Progression and targeted training are ready" in message

    def test_on_batch_review_complete_enables_activation_buttons(self) -> None:
        host, controller = self._make_controller()
        controller._refresh_batch_review_progress = MagicMock()
        controller._batch_review_current_label = MagicMock()
        controller._batch_review_progression_btn = MagicMock()
        controller._batch_review_training_btn = MagicMock()
        controller._batch_review_close_btn = MagicMock()
        controller._build_batch_review_completion_text = MagicMock(
            return_value=("Batch ready", True, True)
        )

        result = BatchReviewResult(items=(
            BatchReviewItemResult(
                target=BatchReviewTarget(file_path=Path("/tmp/game.pgn"), label="Game"),
                status=BatchReviewItemStatus.COMPLETED,
                message="done",
            ),
        ))

        controller._on_batch_review_complete(result)

        controller._batch_review_progression_btn.config.assert_called_once_with(state="normal")
        controller._batch_review_training_btn.config.assert_called_once_with(state="normal")
        controller._batch_review_close_btn.config.assert_called_once()
        host.set_status.assert_called_once()

    def test_exit_review_loads_record_back_into_play_mode(self) -> None:
        host, controller = self._make_controller()
        controller._return_mode = AppMode.PLAY
        controller._play_return_record = GameRecord(
            headers={},
            moves=(),
            board=chess.Board(),
            result="*",
            file_path=Path("/tmp/game.pgn"),
        )
        controller._game_review = _make_review_with_lesson()

        controller._exit_review_mode()

        mode, request = host.last_switch
        assert mode is AppMode.PLAY
        assert request.record.file_path == Path("/tmp/game.pgn")
