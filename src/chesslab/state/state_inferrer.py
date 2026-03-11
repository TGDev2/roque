"""Game state inference module for detecting active color and castling rights.

Infers non-visual FEN fields from piece positions detected by the recognition
module.  Castling rights are determined heuristically by checking whether kings
and rooks occupy their starting squares.  Active color can be supplied as an
explicit override; when absent the module defaults to white and flags the
result as uncertain so that the caller can prompt the user.

Limitations
-----------
* Castling heuristic over-grants rights when a king or rook has moved and
  returned to its starting square (inherent to single-screenshot analysis).
* Active color visual detection is not implemented in MVP — relies on caller
  override.
* En passant is not detectable from a single screenshot and always defaults
  to ``"-"``.
* Assumes standard chess (not Chess960 / Fischer Random).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from chesslab.fen.fen_generator import FenConfig
from chesslab.recognition.piece_recognizer import PieceType, SquareClassification

logger = logging.getLogger(__name__)

_FILES = "abcdefgh"

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class StateInferenceError(RuntimeError):
    """Base class for state inference failures."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StateInferenceConfig:
    """Configuration for game state inference.

    Parameters
    ----------
    active_color_override:
        Explicit active color (``"w"`` or ``"b"``).  When *None* the module
        defaults to ``"w"`` and marks the result as uncertain.
    en_passant:
        En-passant target square in FEN notation, or ``"-"``.  Defaults to
        ``"-"`` (not detectable from a single screenshot).
    halfmove_clock:
        Number of half-moves since the last capture or pawn advance.
    fullmove_number:
        Current full-move number (starts at 1).
    """

    active_color_override: str | None = None
    en_passant: str = "-"
    halfmove_clock: int = 0
    fullmove_number: int = 1

    def __post_init__(self) -> None:
        if self.active_color_override is not None and self.active_color_override not in (
            "w",
            "b",
        ):
            raise ValueError(
                f"active_color_override must be 'w', 'b', or None, "
                f"got '{self.active_color_override}'."
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
class CastlingRights:
    """Structured representation of castling availability.

    Each boolean indicates whether the corresponding castling right is
    still available based on the heuristic analysis of piece positions.
    """

    white_kingside: bool
    white_queenside: bool
    black_kingside: bool
    black_queenside: bool

    @property
    def fen(self) -> str:
        """Return the FEN castling field (e.g. ``'KQkq'``, ``'Kq'``, ``'-'``)."""
        result = ""
        if self.white_kingside:
            result += "K"
        if self.white_queenside:
            result += "Q"
        if self.black_kingside:
            result += "k"
        if self.black_queenside:
            result += "q"
        return result or "-"


@dataclass(frozen=True)
class StateInferenceResult:
    """Outcome of game state inference.

    Attributes
    ----------
    fen_config:
        A :class:`~chesslab.fen.fen_generator.FenConfig` ready to be passed
        to :meth:`~chesslab.fen.fen_generator.FenGenerator.generate`.
    castling_rights:
        Structured castling availability for programmatic inspection.
    active_color_uncertain:
        ``True`` when no explicit active color override was provided and
        the caller should prompt the user.
    elapsed_ms:
        Wall-clock time spent on inference in milliseconds.
    """

    fen_config: FenConfig
    castling_rights: CastlingRights
    active_color_uncertain: bool
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Inferrer
# ---------------------------------------------------------------------------


class StateInferrer:
    """Infers game state from piece positions and optional caller overrides.

    The primary output is a :class:`~chesslab.fen.fen_generator.FenConfig`
    that can be fed directly into
    :meth:`~chesslab.fen.fen_generator.FenGenerator.generate`.
    """

    def __init__(self, config: StateInferenceConfig | None = None) -> None:
        self._config = config or StateInferenceConfig()

    def infer(
        self,
        classifications: dict[str, SquareClassification],
    ) -> StateInferenceResult:
        """Infer game state from 64 square classifications.

        Parameters
        ----------
        classifications:
            Mapping of algebraic square names (``a1`` … ``h8``) to their
            classification results.  Must contain exactly 64 entries.

        Returns
        -------
        StateInferenceResult
            Contains a ready-to-use ``FenConfig``, castling details, and
            an uncertainty flag for the active color.

        Raises
        ------
        ValueError
            If *classifications* does not contain exactly 64 squares.
        """
        start = time.perf_counter()

        if len(classifications) != 64:
            raise ValueError(
                f"Expected 64 square classifications, got {len(classifications)}."
            )

        active_color, uncertain = self._resolve_active_color()
        castling = self._infer_castling(classifications)

        fen_config = FenConfig(
            active_color=active_color,
            castling=castling.fen,
            en_passant=self._config.en_passant,
            halfmove_clock=self._config.halfmove_clock,
            fullmove_number=self._config.fullmove_number,
        )

        elapsed = (time.perf_counter() - start) * 1000.0
        logger.info(
            "State inferred in %.1f ms: color=%s (uncertain=%s), castling=%s",
            elapsed,
            active_color,
            uncertain,
            castling.fen,
        )

        return StateInferenceResult(
            fen_config=fen_config,
            castling_rights=castling,
            active_color_uncertain=uncertain,
            elapsed_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_active_color(self) -> tuple[str, bool]:
        """Determine active color and whether it is uncertain.

        Returns
        -------
        tuple[str, bool]
            ``(color, uncertain)`` where *color* is ``"w"`` or ``"b"``
            and *uncertain* is ``True`` when no override was provided.
        """
        if self._config.active_color_override is not None:
            return self._config.active_color_override, False
        return "w", True

    @staticmethod
    def _infer_castling(
        classifications: dict[str, SquareClassification],
    ) -> CastlingRights:
        """Infer castling rights from king and rook positions.

        The heuristic grants a castling right only when both the king and
        the corresponding rook occupy their standard starting squares.
        """
        white_king_home = classifications["e1"].piece is PieceType.WHITE_KING
        black_king_home = classifications["e8"].piece is PieceType.BLACK_KING

        white_kingside = (
            white_king_home
            and classifications["h1"].piece is PieceType.WHITE_ROOK
        )
        white_queenside = (
            white_king_home
            and classifications["a1"].piece is PieceType.WHITE_ROOK
        )
        black_kingside = (
            black_king_home
            and classifications["h8"].piece is PieceType.BLACK_ROOK
        )
        black_queenside = (
            black_king_home
            and classifications["a8"].piece is PieceType.BLACK_ROOK
        )

        return CastlingRights(
            white_kingside=white_kingside,
            white_queenside=white_queenside,
            black_kingside=black_kingside,
            black_queenside=black_queenside,
        )
