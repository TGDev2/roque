"""Tests for batch post-game review orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import chess

from chesslab.analysis import Evaluation
from chesslab.persistence.game_storage import GameStorage
from chesslab.persistence.review_history import ReviewHistoryStorage
from chesslab.review.batch_review import (
    BatchReviewItemStatus,
    BatchReviewService,
    BatchReviewTarget,
)
from chesslab.review.game_reviewer import (
    GamePhase,
    GameReview,
    MoveClassification,
    ReviewCancelledError,
    ReviewedMove,
)

PGN_A = """\
[Event "Batch A"]
[Site "ChessLab"]
[Date "2026.03.24"]
[White "Alpha"]
[Black "Beta"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"""

PGN_B = """\
[Event "Batch B"]
[Site "ChessLab"]
[Date "2026.03.24"]
[White "Gamma"]
[Black "Delta"]
[Result "0-1"]

1. d4 d5 2. c4 e6 0-1
"""


def _write_pgn(tmp_path: Path, name: str, pgn: str) -> Path:
    file_path = tmp_path / name
    file_path.write_text(pgn, encoding="utf-8")
    return file_path


def _make_review(board: chess.Board) -> GameReview:
    root = board.copy()
    while root.move_stack:
        root.pop()

    replay = root
    moves: list[ReviewedMove] = []
    for index, move in enumerate(board.move_stack):
        fen_before = replay.fen()
        san = replay.san(move)
        replay.push(move)
        moves.append(ReviewedMove(
            ply=index,
            san=san,
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=replay.fen(),
            classification=(
                MoveClassification.MISTAKE
                if index == 0
                else MoveClassification.GOOD
            ),
            eval_before=Evaluation(centipawns=30, mate_in=None),
            eval_after=Evaluation(centipawns=10, mate_in=None),
            best_move_san=san,
            best_move_uci=move.uci(),
            best_eval=Evaluation(centipawns=30, mate_in=None),
            cp_loss=120 if index == 0 else 10,
            best_pv_san=(san,),
            phase=GamePhase.OPENING,
            is_white=index % 2 == 0,
            is_forced=False,
        ))

    return GameReview(
        moves=tuple(moves),
        phase_summaries=(),
        white_classification_counts={MoveClassification.MISTAKE: 1},
        black_classification_counts={MoveClassification.GOOD: max(0, len(moves) - 1)},
        white_avg_cp_loss=120.0,
        black_avg_cp_loss=10.0,
        critical_moments=(0,) if moves else (),
        lessons=(),
        total_elapsed_ms=150.0,
    )


class TestBatchReviewService:
    def test_run_persists_new_reviews_and_skips_reviewed_or_duplicate_targets(
        self,
        tmp_path: Path,
    ) -> None:
        first_file = _write_pgn(tmp_path, "first.pgn", PGN_A)
        second_file = _write_pgn(tmp_path, "second.pgn", PGN_B)

        history = ReviewHistoryStorage(history_file=tmp_path / "reviews.json")
        existing_review = _make_review(GameStorage.load_game(first_file).board)
        history.record_review(existing_review, source_path=first_file)

        service = BatchReviewService(
            storage=GameStorage(games_dir=tmp_path),
            review_history=history,
        )

        second_review = _make_review(GameStorage.load_game(second_file).board)
        targets = [
            BatchReviewTarget(file_path=first_file, label="First"),
            BatchReviewTarget(file_path=first_file, label="First duplicate"),
            BatchReviewTarget(file_path=second_file, label="Second"),
        ]

        with patch(
            "chesslab.review.batch_review.GameReviewer.review",
            return_value=second_review,
        ) as mocked_review:
            result = service.run(targets)

        assert mocked_review.call_count == 1
        assert result.completed_count == 1
        assert result.skipped_reviewed_count == 1
        assert result.skipped_duplicate_count == 1
        assert result.failed_count == 0
        assert result.cancelled is False
        assert result.items[0].status is BatchReviewItemStatus.SKIPPED_ALREADY_REVIEWED
        assert result.items[1].status is BatchReviewItemStatus.SKIPPED_DUPLICATE_SELECTION
        assert result.items[2].status is BatchReviewItemStatus.COMPLETED
        assert len(history.list_recent_reviews(limit=None)) == 2

    def test_run_marks_cancellation_and_stops_processing(
        self,
        tmp_path: Path,
    ) -> None:
        first_file = _write_pgn(tmp_path, "first.pgn", PGN_A)
        second_file = _write_pgn(tmp_path, "second.pgn", PGN_B)

        history = ReviewHistoryStorage(history_file=tmp_path / "reviews.json")
        service = BatchReviewService(
            storage=GameStorage(games_dir=tmp_path),
            review_history=history,
        )

        with patch(
            "chesslab.review.batch_review.GameReviewer.review",
            side_effect=ReviewCancelledError("stop"),
        ):
            result = service.run([
                BatchReviewTarget(file_path=first_file, label="First"),
                BatchReviewTarget(file_path=second_file, label="Second"),
            ])

        assert result.cancelled is True
        assert result.cancelled_count == 1
        assert result.processed_count == 1
        assert result.items[0].status is BatchReviewItemStatus.CANCELLED
        assert history.list_recent_reviews(limit=None) == []
