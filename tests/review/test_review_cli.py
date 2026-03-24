"""Tests for python -m chesslab.review CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import pytest

from chesslab.persistence.game_storage import GameRecord, GameStorage
from chesslab.review.__main__ import _format_review, main
from chesslab.review.game_reviewer import (
    GamePhase,
    GameReview,
    LessonTheme,
    MoveClassification,
    PhaseSummary,
    ReviewLesson,
    ReviewedMove,
)
from chesslab.analysis import Evaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_review() -> tuple[GameReview, dict[str, str]]:
    """Create a minimal GameReview for formatting tests."""
    move = ReviewedMove(
        ply=0,
        san="e4",
        uci="e2e4",
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        classification=MoveClassification.GOOD,
        eval_before=Evaluation(centipawns=20, mate_in=None),
        eval_after=Evaluation(centipawns=30, mate_in=None),
        best_move_san="e4",
        best_move_uci="e2e4",
        best_eval=Evaluation(centipawns=30, mate_in=None),
        cp_loss=0,
        best_pv_san=("e4",),
        phase=GamePhase.OPENING,
        is_white=True,
        is_forced=False,
    )

    review = GameReview(
        moves=(move,),
        phase_summaries=(
            PhaseSummary(
                phase=GamePhase.OPENING,
                move_count=1,
                avg_cp_loss=0.0,
                classification_counts={MoveClassification.GOOD: 1},
            ),
        ),
        white_classification_counts={MoveClassification.GOOD: 1},
        black_classification_counts={},
        white_avg_cp_loss=0.0,
        black_avg_cp_loss=0.0,
        critical_moments=(),
        lessons=(
            ReviewLesson(
                priority=1,
                theme=LessonTheme.TURNING_POINT,
                title="Turning point: 1. e4",
                finding="1. e4 created the sharpest swing in the game.",
                takeaway="Compare forcing alternatives before committing.",
                focus_ply=0,
                related_plies=(0,),
                phase=GamePhase.OPENING,
                is_white=True,
            ),
        ),
        total_elapsed_ms=1234.0,
    )

    headers = {"White": "Alice", "Black": "Bob", "Result": "1-0"}
    return review, headers


def _make_board(*san_moves: str) -> chess.Board:
    board = chess.Board()
    for san in san_moves:
        board.push_san(san)
    return board


def _make_record(
    *san_moves: str,
    headers: dict[str, str] | None = None,
) -> GameRecord:
    board = _make_board(*san_moves)
    h = headers or {"White": "Alice", "Black": "Bob", "Result": "*"}
    return GameRecord(
        headers=h,
        moves=san_moves,
        board=board,
        result=h.get("Result", "*"),
        file_path=None,
    )


# ---------------------------------------------------------------------------
# Tests: _format_review
# ---------------------------------------------------------------------------


class TestFormatReview:
    """Tests for the output formatter."""

    def test_contains_player_names(self) -> None:
        review, headers = _make_dummy_review()
        output = _format_review(review, headers)
        assert "Alice" in output
        assert "Bob" in output

    def test_contains_move(self) -> None:
        review, headers = _make_dummy_review()
        output = _format_review(review, headers)
        assert "e4" in output

    def test_contains_classification(self) -> None:
        review, headers = _make_dummy_review()
        output = _format_review(review, headers)
        assert "ok" in output  # GOOD -> "ok"

    def test_contains_lessons_section(self) -> None:
        review, headers = _make_dummy_review()
        output = _format_review(review, headers)
        assert "Lessons:" in output
        assert "Fact:" in output
        assert "Lesson:" in output


# ---------------------------------------------------------------------------
# Tests: main()
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the CLI entry point."""

    def test_missing_file_exits(self, tmp_path: Path) -> None:
        nonexistent = str(tmp_path / "nope.pgn")
        with patch("sys.argv", ["prog", nonexistent]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_no_moves_exits(self, tmp_path: Path) -> None:
        """A PGN with headers only (no moves) should exit with error."""
        pgn_file = tmp_path / "empty.pgn"
        pgn_file.write_text('[Event "Empty"]\n[Result "*"]\n\n*\n')

        with patch("sys.argv", ["prog", str(pgn_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("chesslab.review.__main__.ReviewHistoryStorage")
    @patch("chesslab.review.__main__.GameReviewer")
    def test_success_prints_output(
        self,
        mock_reviewer_cls: MagicMock,
        mock_history_cls: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """With a mocked reviewer, verify CLI produces formatted output."""
        # Write a valid PGN.
        pgn_file = tmp_path / "game.pgn"
        pgn_file.write_text(
            '[White "Alice"]\n[Black "Bob"]\n[Result "1-0"]\n\n'
            "1. e4 e5 2. Nf3 Nc6 1-0\n"
        )

        # Mock the reviewer to return a dummy review.
        review, _ = _make_dummy_review()
        mock_instance = MagicMock()
        mock_instance.review.return_value = review
        mock_reviewer_cls.return_value = mock_instance

        with patch("sys.argv", ["prog", str(pgn_file)]):
            main()

        captured = capsys.readouterr()
        assert "Alice" in captured.out
        assert "Bob" in captured.out
        mock_history_cls.return_value.record_review.assert_called_once()
