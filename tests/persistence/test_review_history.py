"""Tests for persisted review summaries used by the progression dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import chess

from chesslab.analysis import Evaluation
from chesslab.persistence.review_history import ReviewHistoryStorage
from chesslab.review.game_reviewer import (
    GamePhase,
    GameReview,
    LessonTheme,
    MoveClassification,
    ReviewLesson,
    ReviewedMove,
)


def _make_review(
    *,
    classifications: tuple[MoveClassification, ...] = (
        MoveClassification.GOOD,
        MoveClassification.BLUNDER,
    ),
    cp_losses: tuple[int, ...] = (10, 120),
    phases: tuple[GamePhase, ...] = (
        GamePhase.OPENING,
        GamePhase.ENDGAME,
    ),
    themes: tuple[LessonTheme, ...] = (
        LessonTheme.TURNING_POINT,
        LessonTheme.PHASE_PATTERN,
    ),
) -> GameReview:
    board = chess.Board()
    san_moves = ("e4", "e5", "Nf3", "Nc6")
    moves: list[ReviewedMove] = []

    for idx, classification in enumerate(classifications):
        fen_before = board.fen()
        move = board.parse_san(san_moves[idx])
        san = board.san(move)
        board.push(move)
        fen_after = board.fen()
        moves.append(ReviewedMove(
            ply=idx,
            san=san,
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=fen_after,
            classification=classification,
            eval_before=Evaluation(centipawns=30, mate_in=None),
            eval_after=Evaluation(centipawns=10, mate_in=None),
            best_move_san=san,
            best_move_uci=move.uci(),
            best_eval=Evaluation(centipawns=30, mate_in=None),
            cp_loss=cp_losses[idx],
            best_pv_san=(san,),
            phase=phases[idx],
            is_white=idx % 2 == 0,
            is_forced=False,
        ))

    lessons = tuple(
        ReviewLesson(
            priority=index + 1,
            theme=theme,
            title=f"Lesson {index + 1}",
            finding="Recurring pattern",
            takeaway="Study this pattern.",
            focus_ply=min(index, len(moves) - 1),
            related_plies=(min(index, len(moves) - 1),),
            phase=phases[min(index, len(phases) - 1)],
            is_white=True,
        )
        for index, theme in enumerate(themes)
    )

    white_counts = {cls: 0 for cls in MoveClassification}
    black_counts = {cls: 0 for cls in MoveClassification}
    for move in moves:
        if move.is_white:
            white_counts[move.classification] += 1
        else:
            black_counts[move.classification] += 1

    return GameReview(
        moves=tuple(moves),
        phase_summaries=(),
        white_classification_counts=white_counts,
        black_classification_counts=black_counts,
        white_avg_cp_loss=float(cp_losses[0]),
        black_avg_cp_loss=float(cp_losses[1]) if len(cp_losses) > 1 else 0.0,
        critical_moments=tuple(
            index for index, move in enumerate(moves)
            if move.classification in {
                MoveClassification.INACCURACY,
                MoveClassification.MISTAKE,
                MoveClassification.BLUNDER,
            }
        ),
        lessons=lessons,
        total_elapsed_ms=250.0,
    )


class TestReviewHistoryStorage:
    def test_record_review_persists_summary(self, tmp_path: Path) -> None:
        history_file = tmp_path / "reviews.json"
        storage = ReviewHistoryStorage(history_file=history_file)
        review = _make_review()

        storage.record_review(
            review,
            reviewed_at=datetime(2026, 3, 20, tzinfo=timezone.utc),
            source_path=tmp_path / "game-1.pgn",
        )

        saved = storage.list_recent_reviews(limit=None)

        assert history_file.exists()
        assert len(saved) == 1
        assert saved[0].move_count == 2
        assert saved[0].avg_cp_loss == 65.0
        assert saved[0].classification_counts[MoveClassification.GOOD] == 1
        assert saved[0].classification_counts[MoveClassification.BLUNDER] == 1
        assert saved[0].error_phase_counts[GamePhase.OPENING] == 0
        assert saved[0].error_phase_counts[GamePhase.ENDGAME] == 1
        assert saved[0].lesson_theme_counts[LessonTheme.TURNING_POINT] == 1
        assert saved[0].lesson_theme_counts[LessonTheme.PHASE_PATTERN] == 1

    def test_record_review_upserts_identical_game(self, tmp_path: Path) -> None:
        storage = ReviewHistoryStorage(history_file=tmp_path / "reviews.json")
        review = _make_review()
        source_path = tmp_path / "game-1.pgn"

        storage.record_review(
            review,
            reviewed_at=datetime(2026, 3, 20, tzinfo=timezone.utc),
            source_path=source_path,
        )
        storage.record_review(
            review,
            reviewed_at=datetime(2026, 3, 21, tzinfo=timezone.utc),
            source_path=source_path,
        )

        saved = storage.list_recent_reviews(limit=None)

        assert len(saved) == 1
        assert saved[0].reviewed_at == "2026-03-21T00:00:00+00:00"
