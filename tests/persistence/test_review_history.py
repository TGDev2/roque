"""Tests for persisted review summaries used by the progression dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import chess

from chesslab.analysis import Evaluation
from chesslab.persistence.review_history import ReviewHistoryStorage
from chesslab.review.game_reviewer import (
    ErrorMotif,
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
    motifs: tuple[ErrorMotif, ...] = (
        ErrorMotif.UNKNOWN,
        ErrorMotif.MATERIAL_LOSS,
    ),
    best_move_ucis: tuple[str, ...] = (
        "d2d4",
        "d7d5",
    ),
) -> GameReview:
    board = chess.Board()
    san_moves = ("e4", "e5", "Nf3", "Nc6")
    moves: list[ReviewedMove] = []

    for idx, classification in enumerate(classifications):
        fen_before = board.fen()
        before = board.copy()
        move = board.parse_san(san_moves[idx])
        san = board.san(move)
        best_move = chess.Move.from_uci(best_move_ucis[idx])
        best_move_san = before.san(best_move)
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
            best_move_san=best_move_san,
            best_move_uci=best_move.uci(),
            best_eval=Evaluation(centipawns=30, mate_in=None),
            cp_loss=cp_losses[idx],
            best_pv_san=(best_move_san, san),
            phase=phases[idx],
            is_white=idx % 2 == 0,
            is_forced=False,
            error_motif=motifs[idx],
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
        assert saved[0].error_motif_counts[ErrorMotif.MATERIAL_LOSS] == 1
        assert len(saved[0].training_positions) == 2
        assert saved[0].training_positions[0].best_move_uci == "d7d5"
        assert saved[0].training_positions[0].classification is MoveClassification.BLUNDER
        assert saved[0].training_positions[0].error_motif is ErrorMotif.MATERIAL_LOSS
        assert saved[0].training_positions[0].lesson_themes == (
            LessonTheme.PHASE_PATTERN,
        )

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

    def test_reads_schema_v1_history_with_default_motif_counts(self, tmp_path: Path) -> None:
        history_file = tmp_path / "reviews.json"
        payload = {
            "schema_version": 1,
            "reviews": [
                {
                    "game_id": "legacy",
                    "reviewed_at": "2026-03-20T00:00:00+00:00",
                    "move_count": 1,
                    "avg_cp_loss": 45.0,
                    "classification_counts": {
                        "excellent": 0,
                        "good": 0,
                        "inaccuracy": 1,
                        "mistake": 0,
                        "blunder": 0,
                    },
                    "error_phase_counts": {
                        "opening": 1,
                        "middlegame": 0,
                        "endgame": 0,
                    },
                    "lesson_theme_counts": {
                        "phase_pattern": 0,
                        "turning_point": 1,
                        "advantage_management": 0,
                    },
                    "source_path": None,
                    "training_positions": [
                        {
                            "fen": chess.Board().fen(),
                            "ply": 0,
                            "played_san": "e4",
                            "played_uci": "e2e4",
                            "best_move_san": "d4",
                            "best_move_uci": "d2d4",
                            "best_pv_san": ["d4", "d5"],
                            "classification": "mistake",
                            "phase": "opening",
                            "cp_loss": 90,
                            "lesson_themes": ["turning_point"],
                        },
                    ],
                },
            ],
        }
        history_file.write_text(json.dumps(payload))

        storage = ReviewHistoryStorage(history_file=history_file)
        saved = storage.list_recent_reviews(limit=None)

        assert len(saved) == 1
        assert saved[0].error_motif_counts[ErrorMotif.UNKNOWN] == 0
        assert saved[0].training_positions[0].error_motif is ErrorMotif.UNKNOWN
