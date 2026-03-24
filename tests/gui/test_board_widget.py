"""Tests for the board widget pure helper functions.

These tests validate coordinate mapping, move-hint geometry, FEN parsing, and flip logic
without requiring a display server.
"""

from __future__ import annotations

import pytest

from chesslab.gui.board_widget import (
    coords_to_square,
    move_hint_line_coords,
    parse_fen_placement,
    square_center,
    square_to_coords,
)

SQUARE_SIZE = 64


# ---------------------------------------------------------------------------
# square_to_coords
# ---------------------------------------------------------------------------


class TestSquareToCoords:
    """Pixel mapping from algebraic square to canvas position."""

    def test_a1_not_flipped(self) -> None:
        x, y = square_to_coords("a1", SQUARE_SIZE, flipped=False)
        assert x == 0
        assert y == 7 * SQUARE_SIZE

    def test_h8_not_flipped(self) -> None:
        x, y = square_to_coords("h8", SQUARE_SIZE, flipped=False)
        assert x == 7 * SQUARE_SIZE
        assert y == 0

    def test_e4_not_flipped(self) -> None:
        x, y = square_to_coords("e4", SQUARE_SIZE, flipped=False)
        assert x == 4 * SQUARE_SIZE  # file e = index 4
        assert y == 4 * SQUARE_SIZE  # rank 4 → row 4 (from top)

    def test_a1_flipped(self) -> None:
        x, y = square_to_coords("a1", SQUARE_SIZE, flipped=True)
        assert x == 7 * SQUARE_SIZE
        assert y == 0

    def test_h8_flipped(self) -> None:
        x, y = square_to_coords("h8", SQUARE_SIZE, flipped=True)
        assert x == 0
        assert y == 7 * SQUARE_SIZE

    def test_all_corners(self) -> None:
        """All four corners map to distinct positions."""
        corners = ["a1", "a8", "h1", "h8"]
        positions = {square_to_coords(sq, SQUARE_SIZE, False) for sq in corners}
        assert len(positions) == 4


# ---------------------------------------------------------------------------
# coords_to_square
# ---------------------------------------------------------------------------


class TestCoordsToSquare:
    """Pixel click to algebraic square mapping."""

    def test_top_left_not_flipped(self) -> None:
        sq = coords_to_square(0, 0, SQUARE_SIZE, flipped=False)
        assert sq == "a8"

    def test_bottom_right_not_flipped(self) -> None:
        sq = coords_to_square(
            7 * SQUARE_SIZE + 10, 7 * SQUARE_SIZE + 10,
            SQUARE_SIZE, flipped=False,
        )
        assert sq == "h1"

    def test_top_left_flipped(self) -> None:
        sq = coords_to_square(0, 0, SQUARE_SIZE, flipped=True)
        assert sq == "h1"

    def test_roundtrip(self) -> None:
        """square_to_coords → coords_to_square round-trips correctly."""
        for sq_name in ["a1", "d4", "h8", "e2", "c7"]:
            x, y = square_to_coords(sq_name, SQUARE_SIZE, flipped=False)
            result = coords_to_square(x + 1, y + 1, SQUARE_SIZE, flipped=False)
            assert result == sq_name, f"Failed for {sq_name}"

    def test_roundtrip_flipped(self) -> None:
        """Round-trip works when board is flipped."""
        for sq_name in ["a1", "d4", "h8", "e2", "c7"]:
            x, y = square_to_coords(sq_name, SQUARE_SIZE, flipped=True)
            result = coords_to_square(x + 1, y + 1, SQUARE_SIZE, flipped=True)
            assert result == sq_name, f"Failed for {sq_name} (flipped)"

    def test_clamped_to_bounds(self) -> None:
        """Out-of-bounds clicks are clamped to valid squares."""
        sq = coords_to_square(-100, -100, SQUARE_SIZE, flipped=False)
        assert sq == "a8"

        sq = coords_to_square(9999, 9999, SQUARE_SIZE, flipped=False)
        assert sq == "h1"


# ---------------------------------------------------------------------------
# move hint geometry
# ---------------------------------------------------------------------------


class TestMoveHintGeometry:
    """Helpers for the pedagogical move hint overlay."""

    def test_square_center_uses_square_middle(self) -> None:
        center_x, center_y = square_center("e4", SQUARE_SIZE, flipped=False)

        assert center_x == pytest.approx(4 * SQUARE_SIZE + SQUARE_SIZE / 2)
        assert center_y == pytest.approx(4 * SQUARE_SIZE + SQUARE_SIZE / 2)

    def test_move_hint_line_is_trimmed_inside_both_squares(self) -> None:
        start_x, start_y, end_x, end_y = move_hint_line_coords(
            "e2",
            "e4",
            SQUARE_SIZE,
            flipped=False,
        )
        source_center_x, source_center_y = square_center(
            "e2",
            SQUARE_SIZE,
            flipped=False,
        )
        target_center_x, target_center_y = square_center(
            "e4",
            SQUARE_SIZE,
            flipped=False,
        )

        assert start_x == pytest.approx(source_center_x)
        assert end_x == pytest.approx(target_center_x)
        assert start_y < source_center_y
        assert end_y > target_center_y

    def test_move_hint_line_reorients_when_board_is_flipped(self) -> None:
        _start_x, start_y, _end_x, end_y = move_hint_line_coords(
            "e2",
            "e4",
            SQUARE_SIZE,
            flipped=True,
        )

        assert start_y < end_y


# ---------------------------------------------------------------------------
# parse_fen_placement
# ---------------------------------------------------------------------------


class TestParseFenPlacement:
    """FEN piece-placement string parsing."""

    def test_starting_position(self) -> None:
        placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        pieces = parse_fen_placement(placement)

        assert pieces["e1"] == "K"
        assert pieces["d8"] == "q"
        assert pieces["a2"] == "P"
        assert pieces["h7"] == "p"
        assert "e4" not in pieces  # empty square

    def test_counts_64_or_fewer_pieces(self) -> None:
        placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        pieces = parse_fen_placement(placement)
        assert len(pieces) == 32  # 16 white + 16 black pieces

    def test_after_e4(self) -> None:
        placement = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
        pieces = parse_fen_placement(placement)
        assert pieces["e4"] == "P"
        assert "e2" not in pieces

    def test_empty_board(self) -> None:
        placement = "8/8/8/8/8/8/8/8"
        pieces = parse_fen_placement(placement)
        assert len(pieces) == 0

    def test_endgame_position(self) -> None:
        placement = "8/8/8/8/8/kq6/8/K7"
        pieces = parse_fen_placement(placement)
        assert pieces["a1"] == "K"
        assert pieces["a3"] == "k"
        assert pieces["b3"] == "q"
        assert len(pieces) == 3
