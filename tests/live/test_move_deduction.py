"""Tests for the move deduction module.

Moved from tests/gui/test_live_observer.py and extended with
``is_starting_position`` tests.
"""

from __future__ import annotations

import chess

from chesslab.live.move_deduction import (
    _STANDARD_START_PLACEMENT,
    deduce_move_from_fen,
    is_starting_position,
)


# ---------------------------------------------------------------------------
# deduce_move_from_fen
# ---------------------------------------------------------------------------


class TestDeduceMoveFromFen:
    """Unit tests for the pure move deduction function."""

    def test_e2e4_from_start(self) -> None:
        board = chess.Board()
        board_copy = board.copy()
        board_copy.push_san("e4")
        placement = board_copy.fen().split(" ", 1)[0]

        move = deduce_move_from_fen(board, placement)

        assert move is not None
        assert move == chess.Move.from_uci("e2e4")

    def test_nf3_from_start(self) -> None:
        board = chess.Board()
        board_copy = board.copy()
        board_copy.push_san("Nf3")
        placement = board_copy.fen().split(" ", 1)[0]

        move = deduce_move_from_fen(board, placement)

        assert move is not None
        assert move == chess.Move.from_uci("g1f3")

    def test_same_position_returns_none(self) -> None:
        board = chess.Board()
        placement = board.fen().split(" ", 1)[0]

        move = deduce_move_from_fen(board, placement)

        assert move is None

    def test_impossible_placement_returns_none(self) -> None:
        board = chess.Board()
        placement = "8/8/8/8/8/8/8/8"

        move = deduce_move_from_fen(board, placement)

        assert move is None

    def test_castling_kingside_white(self) -> None:
        board = chess.Board(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        )
        board_copy = board.copy()
        board_copy.push_san("O-O")
        placement = board_copy.fen().split(" ", 1)[0]

        move = deduce_move_from_fen(board, placement)

        assert move is not None
        assert move == chess.Move.from_uci("e1g1")

    def test_castling_queenside_black(self) -> None:
        board = chess.Board(
            "r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR b KQkq - 6 4",
        )
        board_copy = board.copy()
        board_copy.push_san("O-O-O")
        placement = board_copy.fen().split(" ", 1)[0]

        move = deduce_move_from_fen(board, placement)

        assert move is not None
        assert move == chess.Move.from_uci("e8c8")

    def test_promotion_to_queen(self) -> None:
        board = chess.Board("8/P3k3/8/8/8/8/8/4K3 w - - 0 1")
        board_copy = board.copy()
        board_copy.push_san("a8=Q")
        placement = board_copy.fen().split(" ", 1)[0]

        move = deduce_move_from_fen(board, placement)

        assert move is not None
        assert move.promotion == chess.QUEEN
        assert move.from_square == chess.A7
        assert move.to_square == chess.A8

    def test_promotion_to_knight(self) -> None:
        board = chess.Board("8/P3k3/8/8/8/8/8/4K3 w - - 0 1")
        board_copy = board.copy()
        board_copy.push_san("a8=N")
        placement = board_copy.fen().split(" ", 1)[0]

        move = deduce_move_from_fen(board, placement)

        assert move is not None
        assert move.promotion == chess.KNIGHT

    def test_en_passant(self) -> None:
        board = chess.Board(
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        )
        board_copy = board.copy()
        board_copy.push_san("exd6")
        placement = board_copy.fen().split(" ", 1)[0]

        move = deduce_move_from_fen(board, placement)

        assert move is not None
        assert move == chess.Move.from_uci("e5d6")

    def test_capture(self) -> None:
        board = chess.Board()
        for san in ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"):
            board.push_san(san)

        board_copy = board.copy()
        board_copy.push_san("Qxf7#")
        placement = board_copy.fen().split(" ", 1)[0]

        move = deduce_move_from_fen(board, placement)

        assert move is not None
        assert board.san(move) == "Qxf7#"


# ---------------------------------------------------------------------------
# is_starting_position
# ---------------------------------------------------------------------------


class TestIsStartingPosition:
    """Tests for the starting-position detection helper."""

    def test_standard_start_returns_true(self) -> None:
        assert is_starting_position(_STANDARD_START_PLACEMENT)

    def test_after_e4_returns_false(self) -> None:
        board = chess.Board()
        board.push_san("e4")
        placement = board.fen().split(" ", 1)[0]
        assert not is_starting_position(placement)

    def test_empty_board_returns_false(self) -> None:
        assert not is_starting_position("8/8/8/8/8/8/8/8")

    def test_board_default_fen_placement(self) -> None:
        board = chess.Board()
        placement = board.fen().split(" ", 1)[0]
        assert is_starting_position(placement)
