"""Tests for targeted weakness-based training selection and feedback."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from chesslab.analysis import AnalysisResult, Evaluation, VariationInfo
from chesslab.persistence.review_history import ReviewSummary, TrainingPositionSummary
from chesslab.review.game_reviewer import (
    GamePhase,
    GameReview,
    LessonTheme,
    MoveClassification,
    ReviewLesson,
)
from chesslab.training import TrainingFeedbackLabel, WeaknessTrainingService


def _training_position(
    *,
    fen: str,
    played_uci: str,
    best_move_uci: str,
    best_move_san: str,
    classification: MoveClassification,
    phase: GamePhase,
    cp_loss: int,
    lesson_themes: tuple[LessonTheme, ...] = (),
) -> TrainingPositionSummary:
    return TrainingPositionSummary(
        fen=fen,
        ply=0,
        played_san="played",
        played_uci=played_uci,
        best_move_san=best_move_san,
        best_move_uci=best_move_uci,
        best_pv_san=(best_move_san, "reply"),
        classification=classification,
        phase=phase,
        cp_loss=cp_loss,
        lesson_themes=lesson_themes,
    )


def _summary(
    *,
    game_id: str,
    reviewed_at: str,
    positions: tuple[TrainingPositionSummary, ...],
    error_phase_counts: dict[GamePhase, int] | None = None,
    lesson_theme_counts: dict[LessonTheme, int] | None = None,
) -> ReviewSummary:
    return ReviewSummary(
        game_id=game_id,
        reviewed_at=reviewed_at,
        move_count=12,
        avg_cp_loss=65.0,
        classification_counts={cls: 0 for cls in (
            MoveClassification.EXCELLENT,
            MoveClassification.GOOD,
            MoveClassification.INACCURACY,
            MoveClassification.MISTAKE,
            MoveClassification.BLUNDER,
        )},
        error_phase_counts={
            phase: (error_phase_counts or {}).get(phase, 0)
            for phase in GamePhase
        },
        lesson_theme_counts={
            theme: (lesson_theme_counts or {}).get(theme, 0)
            for theme in LessonTheme
        },
        training_positions=positions,
    )


def _focus_review() -> GameReview:
    return GameReview(
        moves=(),
        phase_summaries=(),
        white_classification_counts={},
        black_classification_counts={},
        white_avg_cp_loss=0.0,
        black_avg_cp_loss=0.0,
        critical_moments=(),
        lessons=(
            ReviewLesson(
                priority=1,
                theme=LessonTheme.TURNING_POINT,
                title="Turning point",
                finding="A middlegame swing.",
                takeaway="Look for forcing moves first.",
                focus_ply=10,
                related_plies=(10,),
                phase=GamePhase.MIDDLEGAME,
                is_white=True,
            ),
        ),
        total_elapsed_ms=100.0,
    )


class TestWeaknessTrainingSelection:
    def test_build_session_prioritises_theme_and_phase_from_focus_review(self) -> None:
        starting_fen = chess.Board().fen()
        service = WeaknessTrainingService()

        summaries = [
            _summary(
                game_id="middlegame",
                reviewed_at="2026-03-24T10:00:00+00:00",
                positions=(
                    _training_position(
                        fen=starting_fen,
                        played_uci="e2e4",
                        best_move_uci="d2d4",
                        best_move_san="d4",
                        classification=MoveClassification.MISTAKE,
                        phase=GamePhase.MIDDLEGAME,
                        cp_loss=180,
                        lesson_themes=(LessonTheme.TURNING_POINT,),
                    ),
                ),
                error_phase_counts={GamePhase.MIDDLEGAME: 3},
                lesson_theme_counts={LessonTheme.TURNING_POINT: 2},
            ),
            _summary(
                game_id="opening",
                reviewed_at="2026-03-24T11:00:00+00:00",
                positions=(
                    _training_position(
                        fen=starting_fen,
                        played_uci="g1f3",
                        best_move_uci="d2d4",
                        best_move_san="d4",
                        classification=MoveClassification.BLUNDER,
                        phase=GamePhase.OPENING,
                        cp_loss=240,
                    ),
                ),
                error_phase_counts={GamePhase.OPENING: 1},
            ),
        ]

        session = service.build_session(
            summaries,
            sample_size=2,
            focus_review=_focus_review(),
            session_size=2,
        )

        assert len(session) == 2
        assert session[0].game_id == "middlegame"
        assert session[0].focus_theme is LessonTheme.TURNING_POINT
        assert session[0].focus_phase is GamePhase.MIDDLEGAME
        assert "Turning point" in session[0].focus_reason


@dataclass
class _FakeAnalyser:
    config: object
    results: dict[str, AnalysisResult]
    started: bool = False
    stopped: bool = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def analyse(self, fen: str) -> AnalysisResult:
        return self.results[fen]


class _FakeAnalyserFactory:
    def __init__(self, results: dict[str, AnalysisResult]) -> None:
        self._results = results

    def __call__(self, config: object) -> _FakeAnalyser:
        return _FakeAnalyser(config, self._results)


def _variation(
    *,
    rank: int,
    move_uci: str,
    move_san: str,
    evaluation_cp: int,
) -> VariationInfo:
    return VariationInfo(
        rank=rank,
        move_uci=move_uci,
        move_san=move_san,
        evaluation=Evaluation(centipawns=evaluation_cp, mate_in=None),
        pv_uci=(move_uci,),
        pv_san=(move_san,),
        depth=16,
    )


def _result(
    *,
    fen: str,
    best_move_uci: str,
    best_move_san: str,
    evaluation_cp: int,
    variations: tuple[VariationInfo, ...],
) -> AnalysisResult:
    return AnalysisResult(
        fen=fen,
        best_move_uci=best_move_uci,
        best_move_san=best_move_san,
        evaluation=Evaluation(centipawns=evaluation_cp, mate_in=None),
        variations=variations,
        depth_reached=16,
        is_partial=False,
        elapsed_ms=125.0,
    )


def _after_fen(fen: str, move_uci: str) -> str:
    board = chess.Board(fen)
    board.push(chess.Move.from_uci(move_uci))
    return board.fen()


class TestWeaknessTrainingFeedback:
    def test_marks_top_three_move_as_good(self) -> None:
        fen = chess.Board().fen()
        move_uci = "g1f3"
        after_fen = _after_fen(fen, move_uci)
        results = {
            fen: _result(
                fen=fen,
                best_move_uci="e2e4",
                best_move_san="e4",
                evaluation_cp=40,
                variations=(
                    _variation(rank=1, move_uci="e2e4", move_san="e4", evaluation_cp=50),
                    _variation(rank=2, move_uci="d2d4", move_san="d4", evaluation_cp=45),
                    _variation(rank=3, move_uci="g1f3", move_san="Nf3", evaluation_cp=40),
                ),
            ),
            after_fen: _result(
                fen=after_fen,
                best_move_uci="d7d5",
                best_move_san="d5",
                evaluation_cp=-35,
                variations=(),
            ),
        }
        service = WeaknessTrainingService(
            analyser_factory=_FakeAnalyserFactory(results),
        )

        feedback = service.evaluate_move(fen, move_uci)

        assert feedback.label is TrainingFeedbackLabel.GOOD
        assert feedback.move_rank == 3
        assert feedback.cp_loss == 5

    def test_marks_non_top_move_as_better_available_when_loss_is_small(self) -> None:
        fen = chess.Board().fen()
        move_uci = "c2c4"
        after_fen = _after_fen(fen, move_uci)
        results = {
            fen: _result(
                fen=fen,
                best_move_uci="e2e4",
                best_move_san="e4",
                evaluation_cp=60,
                variations=(
                    _variation(rank=1, move_uci="e2e4", move_san="e4", evaluation_cp=60),
                    _variation(rank=2, move_uci="d2d4", move_san="d4", evaluation_cp=55),
                    _variation(rank=3, move_uci="g1f3", move_san="Nf3", evaluation_cp=50),
                ),
            ),
            after_fen: _result(
                fen=after_fen,
                best_move_uci="d7d5",
                best_move_san="d5",
                evaluation_cp=-10,
                variations=(),
            ),
        }
        service = WeaknessTrainingService(
            analyser_factory=_FakeAnalyserFactory(results),
        )

        feedback = service.evaluate_move(fen, move_uci)

        assert feedback.label is TrainingFeedbackLabel.BETTER_AVAILABLE
        assert feedback.move_rank is None
        assert feedback.cp_loss == 50
        assert feedback.best_move_san == "e4"

    def test_marks_large_drop_as_significant_error(self) -> None:
        fen = chess.Board().fen()
        move_uci = "a2a3"
        after_fen = _after_fen(fen, move_uci)
        results = {
            fen: _result(
                fen=fen,
                best_move_uci="e2e4",
                best_move_san="e4",
                evaluation_cp=250,
                variations=(
                    _variation(rank=1, move_uci="e2e4", move_san="e4", evaluation_cp=250),
                    _variation(rank=2, move_uci="d2d4", move_san="d4", evaluation_cp=220),
                    _variation(rank=3, move_uci="g1f3", move_san="Nf3", evaluation_cp=200),
                ),
            ),
            after_fen: _result(
                fen=after_fen,
                best_move_uci="d7d5",
                best_move_san="d5",
                evaluation_cp=50,
                variations=(),
            ),
        }
        service = WeaknessTrainingService(
            analyser_factory=_FakeAnalyserFactory(results),
        )

        feedback = service.evaluate_move(fen, move_uci)

        assert feedback.label is TrainingFeedbackLabel.SIGNIFICANT_ERROR
        assert feedback.cp_loss == 300
        assert "e4" in feedback.message
