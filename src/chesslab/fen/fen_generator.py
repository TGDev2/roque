"""FEN generation module for converting recognised positions to standard notation.

Converts a set of 64 square classifications produced by the piece recogniser
into a valid FEN string, validates the position for legality, and confirms
the result against the python-chess library.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import chess

from chesslab.recognition.piece_recognizer import PieceType, SquareClassification

logger = logging.getLogger(__name__)

_FILES = "abcdefgh"
_RANKS = range(8, 0, -1)  # 8 → 1

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class FenError(RuntimeError):
    """Base class for FEN generation failures."""


class InvalidPositionError(FenError):
    """Raised when the detected position is not a legal chess position."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FenConfig:
    """Non-visual FEN fields that cannot be inferred from a screenshot.

    Provides sensible defaults (white to move, full castling rights,
    no en-passant, move 1) that can be overridden by the caller.
    """

    active_color: str = "w"
    castling: str = "KQkq"
    en_passant: str = "-"
    halfmove_clock: int = 0
    fullmove_number: int = 1

    def __post_init__(self) -> None:
        if self.active_color not in ("w", "b"):
            raise ValueError(
                f"active_color must be 'w' or 'b', got '{self.active_color}'."
            )
        valid_castling = {"-"} | {
            "".join(combo)
            for length in range(1, 5)
            for combo in _iter_ordered_subsets("KQkq", length)
        }
        if self.castling not in valid_castling:
            raise ValueError(
                f"castling must be a valid FEN castling field, got '{self.castling}'."
            )
        if self.en_passant != "-" and (
            len(self.en_passant) != 2
            or self.en_passant[0] not in _FILES
            or self.en_passant[1] not in ("3", "6")
        ):
            raise ValueError(
                f"en_passant must be '-' or a valid square (e.g. 'e3'), "
                f"got '{self.en_passant}'."
            )
        if self.halfmove_clock < 0:
            raise ValueError(
                f"halfmove_clock must be >= 0, got {self.halfmove_clock}."
            )
        if self.fullmove_number < 1:
            raise ValueError(
                f"fullmove_number must be >= 1, got {self.fullmove_number}."
            )


@dataclass(frozen=True)
class FenResult:
    """Outcome of a successful FEN generation."""

    fen: str
    piece_placement: str
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class FenGenerator:
    """Converts square classifications to a validated FEN string."""

    def generate(
        self,
        classifications: dict[str, SquareClassification],
        config: FenConfig | None = None,
    ) -> FenResult:
        """Build and validate a FEN string from 64 square classifications.

        Parameters
        ----------
        classifications:
            Mapping of algebraic square names (``a1`` … ``h8``) to their
            classification results.  Must contain exactly 64 entries.
        config:
            Optional overrides for the non-visual FEN fields.

        Returns
        -------
        FenResult
            The validated FEN string, piece placement, and elapsed time.

        Raises
        ------
        ValueError
            If *classifications* does not contain exactly 64 squares.
        InvalidPositionError
            If the position fails legality validation.
        """
        start = time.perf_counter()
        config = config or FenConfig()

        if len(classifications) != 64:
            raise ValueError(
                f"Expected 64 square classifications, got {len(classifications)}."
            )

        placement = self._build_placement(classifications)
        self._validate_position(classifications)

        fen = (
            f"{placement} {config.active_color} {config.castling} "
            f"{config.en_passant} {config.halfmove_clock} "
            f"{config.fullmove_number}"
        )

        self._validate_with_chess(fen)

        elapsed = (time.perf_counter() - start) * 1000.0
        logger.info("FEN generated in %.1f ms: %s", elapsed, fen)

        return FenResult(fen=fen, piece_placement=placement, elapsed_ms=elapsed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_placement(
        classifications: dict[str, SquareClassification],
    ) -> str:
        """Build the piece-placement field (rank 8 → rank 1)."""
        rank_strings: list[str] = []
        for rank in _RANKS:
            empty_count = 0
            rank_str = ""
            for file_char in _FILES:
                square_name = f"{file_char}{rank}"
                piece = classifications[square_name].piece
                if piece is PieceType.EMPTY:
                    empty_count += 1
                else:
                    if empty_count:
                        rank_str += str(empty_count)
                        empty_count = 0
                    rank_str += piece.value
            if empty_count:
                rank_str += str(empty_count)
            rank_strings.append(rank_str)
        return "/".join(rank_strings)

    @staticmethod
    def _validate_position(
        classifications: dict[str, SquareClassification],
    ) -> None:
        """Check basic legality constraints on the detected position."""
        pieces = [c.piece for c in classifications.values()]

        white_kings = pieces.count(PieceType.WHITE_KING)
        black_kings = pieces.count(PieceType.BLACK_KING)
        if white_kings != 1:
            raise InvalidPositionError(
                f"Position must have exactly 1 white king, found {white_kings}."
            )
        if black_kings != 1:
            raise InvalidPositionError(
                f"Position must have exactly 1 black king, found {black_kings}."
            )

        white_count = sum(
            1 for p in pieces if p.value.isupper() and p is not PieceType.EMPTY
        )
        black_count = sum(1 for p in pieces if p.value.islower())
        if white_count > 16:
            raise InvalidPositionError(
                f"White cannot have more than 16 pieces, found {white_count}."
            )
        if black_count > 16:
            raise InvalidPositionError(
                f"Black cannot have more than 16 pieces, found {black_count}."
            )

        for file_char in _FILES:
            for edge_rank in (1, 8):
                square_name = f"{file_char}{edge_rank}"
                piece = classifications[square_name].piece
                if piece in (PieceType.WHITE_PAWN, PieceType.BLACK_PAWN):
                    raise InvalidPositionError(
                        f"Pawn on {square_name} is illegal (rank {edge_rank})."
                    )

    @staticmethod
    def _validate_with_chess(fen: str) -> None:
        """Final validation via the python-chess library.

        Castling rights, en-passant validity, and opposite-check status
        are ignored because they cannot be determined from a screenshot.
        """
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise InvalidPositionError(
                f"python-chess rejected the FEN: {exc}"
            ) from exc

        status = board.status() & _CHECKED_STATUS_MASK
        if status:
            issues = ", ".join(
                name
                for name, mask in _CHECKED_FLAGS.items()
                if status & mask
            )
            raise InvalidPositionError(
                f"python-chess position validation failed: {issues}."
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Flags that represent truly illegal piece configurations.
# Castling, en-passant, and opposite-check flags are excluded because
# those aspects cannot be inferred from a board screenshot.
_CHECKED_FLAGS: dict[str, int] = {
    "no_white_king": chess.STATUS_NO_WHITE_KING,
    "no_black_king": chess.STATUS_NO_BLACK_KING,
    "too_many_kings": chess.STATUS_TOO_MANY_KINGS,
    "too_many_white_pieces": chess.STATUS_TOO_MANY_WHITE_PIECES,
    "too_many_black_pieces": chess.STATUS_TOO_MANY_BLACK_PIECES,
    "pawns_on_backrank": chess.STATUS_PAWNS_ON_BACKRANK,
    "too_many_white_pawns": chess.STATUS_TOO_MANY_WHITE_PAWNS,
    "too_many_black_pawns": chess.STATUS_TOO_MANY_BLACK_PAWNS,
}

_CHECKED_STATUS_MASK: int = 0
for _flag in _CHECKED_FLAGS.values():
    _CHECKED_STATUS_MASK |= _flag


def _iter_ordered_subsets(s: str, length: int):
    """Yield all ordered subsets of *s* with the given *length*."""
    if length == 0:
        yield ""
        return
    for i, ch in enumerate(s):
        for rest in _iter_ordered_subsets(s[i + 1 :], length - 1):
            yield ch + rest
