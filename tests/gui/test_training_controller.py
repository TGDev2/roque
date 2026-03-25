"""Tests for the training mode controller."""

from __future__ import annotations

from unittest.mock import MagicMock

import chess

from chesslab.gui.controllers import AppMode, TrainingModeRequest
from chesslab.gui.controllers.training import TrainingController
from chesslab.persistence.review_history import TrainingPositionSummary
from chesslab.persistence.training_history import TrainingAttemptOutcome
from chesslab.review.game_reviewer import GamePhase, LessonTheme, MoveClassification
from chesslab.training import TrainingExercise, TrainingFeedbackLabel, TrainingMoveFeedback

from tests.gui.controller_test_utils import DummyHost


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


class TestTrainingController:
    def _make_controller(self) -> tuple[DummyHost, TrainingController]:
        host = DummyHost()
        controller = TrainingController(host)
        controller._panel = MagicMock()
        return host, controller

    def test_start_training_session_surfaces_unavailable_message(self) -> None:
        host, controller = self._make_controller()
        host.services.review_history.list_recent_reviews.return_value = []
        host.services.training_history.list_recent_attempts.return_value = []
        host.services.training_service.build_session.return_value = ()
        controller._show_training_unavailable = MagicMock()

        controller._build_session(
            TrainingModeRequest(sample_size=5, focus_review=None, return_mode=AppMode.REVIEW)
        )

        controller._show_training_unavailable.assert_called_once()

    def test_start_training_session_passes_persisted_attempts_to_builder(self) -> None:
        host, controller = self._make_controller()
        host.services.review_history.list_recent_reviews.return_value = ["summary"]
        host.services.training_history.list_recent_attempts.return_value = ["attempt"]
        host.services.training_service.build_session.return_value = ("exercise",)

        controller._build_session(
            TrainingModeRequest(sample_size=5, focus_review=None, return_mode=AppMode.REVIEW)
        )

        host.services.training_service.build_session.assert_called_once_with(
            ["summary"],
            sample_size=5,
            focus_review=None,
            training_attempts=["attempt"],
        )

    def test_training_feedback_updates_panel_and_status(self) -> None:
        host, controller = self._make_controller()
        controller._session = (_make_training_exercise(),)

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

        controller._on_training_feedback(feedback)

        controller._panel.show_feedback.assert_called_once_with(feedback, is_last=True)
        attempt = host.services.training_history.record_attempt.call_args.args[0]
        assert attempt.outcome is TrainingAttemptOutcome.SUCCESS
        assert attempt.weakness_id == controller._session[0].position.weakness_id
        host.set_status.assert_called_once_with("Training — good move")

    def test_exit_training_returns_to_requested_mode(self) -> None:
        host, controller = self._make_controller()
        controller._return_mode = AppMode.PROGRESSION
        controller._session = (_make_training_exercise(),)

        controller._exit_training_mode()

        assert host.last_switch[0] is AppMode.PROGRESSION
