"""Tests for review module data models and enums."""

from __future__ import annotations

from chesslab.review.game_reviewer import (
    GamePhase,
    GameReview,
    MoveClassification,
    PhaseSummary,
    ReviewConfig,
    ReviewedMove,
)
from chesslab.analysis import Evaluation


class TestMoveClassification:
    """MoveClassification enum values match expected labels."""

    def test_values(self) -> None:
        assert MoveClassification.EXCELLENT.value == "excellent"
        assert MoveClassification.GOOD.value == "good"
        assert MoveClassification.INACCURACY.value == "inaccuracy"
        assert MoveClassification.MISTAKE.value == "mistake"
        assert MoveClassification.BLUNDER.value == "blunder"
        assert MoveClassification.FORCED.value == "forced"

    def test_all_members(self) -> None:
        assert len(MoveClassification) == 6


class TestGamePhase:
    def test_values(self) -> None:
        assert GamePhase.OPENING.value == "opening"
        assert GamePhase.MIDDLEGAME.value == "middlegame"
        assert GamePhase.ENDGAME.value == "endgame"


class TestReviewConfig:
    def test_defaults(self) -> None:
        cfg = ReviewConfig()
        assert cfg.depth == 16
        assert cfg.time_limit == 1.5
        assert cfg.multipv == 1
        assert cfg.excellent_threshold == 10
        assert cfg.good_threshold == 30
        assert cfg.inaccuracy_threshold == 80
        assert cfg.mistake_threshold == 200
        assert cfg.opening_end_ply == 30
        assert cfg.endgame_material_threshold == 13

    def test_custom_thresholds(self) -> None:
        cfg = ReviewConfig(
            excellent_threshold=5,
            good_threshold=20,
            inaccuracy_threshold=60,
            mistake_threshold=150,
        )
        assert cfg.excellent_threshold == 5
        assert cfg.mistake_threshold == 150


class TestReviewedMove:
    def test_frozen(self) -> None:
        move = ReviewedMove(
            ply=0,
            san="e4",
            uci="e2e4",
            fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            classification=MoveClassification.EXCELLENT,
            eval_before=Evaluation(centipawns=20, mate_in=None),
            eval_after=Evaluation(centipawns=-20, mate_in=None),
            best_move_san="e4",
            best_move_uci="e2e4",
            best_eval=Evaluation(centipawns=20, mate_in=None),
            cp_loss=0,
            best_pv_san=("e4", "e5"),
            phase=GamePhase.OPENING,
            is_white=True,
            is_forced=False,
        )
        assert move.san == "e4"
        assert move.classification == MoveClassification.EXCELLENT


class TestGameReview:
    def test_empty_review(self) -> None:
        review = GameReview(
            moves=(),
            phase_summaries=(),
            white_classification_counts={cls: 0 for cls in MoveClassification},
            black_classification_counts={cls: 0 for cls in MoveClassification},
            white_avg_cp_loss=0.0,
            black_avg_cp_loss=0.0,
            critical_moments=(),
            total_elapsed_ms=0.0,
        )
        assert len(review.moves) == 0
        assert review.white_avg_cp_loss == 0.0
        assert len(review.critical_moments) == 0


class TestPhaseSummary:
    def test_frozen(self) -> None:
        ps = PhaseSummary(
            phase=GamePhase.OPENING,
            move_count=10,
            avg_cp_loss=15.0,
            classification_counts={MoveClassification.EXCELLENT: 5, MoveClassification.GOOD: 5},
        )
        assert ps.phase == GamePhase.OPENING
        assert ps.move_count == 10
