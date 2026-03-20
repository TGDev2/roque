"""Move deduction from detected piece placements.

Provides logic to identify which legal move on a ``chess.Board`` produces
a given piece-placement string (the first FEN field).  This is the bridge
between the stateless CV pipeline (which outputs FEN) and stateful game
tracking (which needs actual moves).
"""

from __future__ import annotations

import chess

_STANDARD_START_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def deduce_move_from_fen(
    board: chess.Board,
    detected_placement: str,
) -> chess.Move | None:
    """Find the legal move that produces *detected_placement*.

    Iterates all legal moves on *board*, pushes each on a copy, and
    compares the resulting piece-placement field (first FEN token) with
    *detected_placement*.  Returns the matching :class:`chess.Move` or
    ``None`` if no legal move produces the detected position.

    Only the piece-placement portion of the FEN is compared because the
    CV pipeline cannot reliably determine active colour, castling rights,
    or en-passant state.

    Parameters
    ----------
    board:
        The current board state.
    detected_placement:
        The piece-placement field extracted from the detected FEN
        (e.g. ``"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"``).
    """
    for move in board.legal_moves:
        board.push(move)
        placement = board.fen().split(" ", 1)[0]
        board.pop()
        if placement == detected_placement:
            return move
    return None


def is_starting_position(placement: str) -> bool:
    """Return ``True`` if *placement* is the standard chess starting position."""
    return placement == _STANDARD_START_PLACEMENT
