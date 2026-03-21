"""Stateful game tracking for the live observation pipeline.

Maintains a :class:`chess.Board` that is updated move-by-move as new
piece placements are detected by the CV pipeline.  Handles game
continuity, new-game detection, and position resets.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass

import chess

from chesslab.live.move_deduction import deduce_move_from_fen, is_starting_position

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & result types
# ---------------------------------------------------------------------------


class GameTrackingEventType(enum.Enum):
    """Type of event produced by :class:`LiveGameTracker`."""

    GAME_STARTED = "game_started"
    MOVE = "move"
    NEW_GAME_DETECTED = "new_game"
    POSITION_RESET = "position_reset"
    NO_CHANGE = "no_change"


@dataclass(frozen=True)
class GameTrackingEvent:
    """Result of a single :meth:`LiveGameTracker.update` call."""

    type: GameTrackingEventType
    fen: str
    move: chess.Move | None = None
    move_san: str | None = None
    move_uci: str | None = None
    move_number: int = 1
    active_color: str = "w"
    is_new_game: bool = False
    elapsed_ms: float = 0.0
    message: str = ""


@dataclass
class GameTrackerConfig:
    """Configuration for :class:`LiveGameTracker`.

    Parameters
    ----------
    initial_fen:
        Override the starting position.  If ``None``, the first detected
        placement is used to initialise the board.
    require_confirmation:
        When ``True`` (default), a position discontinuity (no legal move
        found) must be confirmed across multiple consecutive frames before
        the tracker acts on it.  This absorbs transient OCR errors.
    confirmation_frames:
        Number of consecutive discontinuity frames required before a
        reset/new-game action is triggered when *require_confirmation*
        is enabled.
    """

    initial_fen: str | None = None
    require_confirmation: bool = True
    confirmation_frames: int = 2

    def __post_init__(self) -> None:
        if self.confirmation_frames < 1:
            raise ValueError(
                f"confirmation_frames must be >= 1, got {self.confirmation_frames}."
            )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class LiveGameTracker:
    """Tracks a chess game by consuming piece-placement strings.

    Call :meth:`update` each time the CV pipeline detects a (changed)
    position.  The tracker maintains a :class:`chess.Board`, deduces the
    move played, and returns a :class:`GameTrackingEvent` describing what
    happened.
    """

    def __init__(self, config: GameTrackerConfig | None = None) -> None:
        self._config = config or GameTrackerConfig()
        self._board: chess.Board | None = None
        self._pending_placement: str | None = None
        self._pending_count: int = 0

        if self._config.initial_fen is not None:
            self._board = chess.Board(self._config.initial_fen)

    # -- Public API ----------------------------------------------------------

    def update(self, piece_placement: str) -> GameTrackingEvent:
        """Feed a new piece placement and return a tracking event.

        Parameters
        ----------
        piece_placement:
            The first FEN field (piece positions only), e.g.
            ``"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"``.
        """
        t0 = time.perf_counter()

        # First frame — initialise board.
        if self._board is None:
            return self._init_board(piece_placement, t0)

        # Same position — no change.
        current_placement = self._board.fen().split(" ", 1)[0]
        if piece_placement == current_placement:
            self._clear_pending_state()
            return GameTrackingEvent(
                type=GameTrackingEventType.NO_CHANGE,
                fen=self._board.fen(),
                move_number=self._board.fullmove_number,
                active_color="w" if self._board.turn == chess.WHITE else "b",
                elapsed_ms=_elapsed(t0),
                message="Position unchanged.",
            )

        # Try to find a legal move.
        move = deduce_move_from_fen(self._board, piece_placement)
        if move is not None:
            self._clear_pending_state()
            return self._apply_move(move, t0)

        # Initial sync ambiguity: first non-start frame may have unknown side-to-move.
        recovered = self._try_recover_initial_turn(piece_placement, t0)
        if recovered is not None:
            return recovered

        # No legal move — handle discontinuity.
        return self._handle_discontinuity(piece_placement, t0)

    @property
    def board(self) -> chess.Board | None:
        """Copy of the internal board, or ``None`` if not yet initialised."""
        return self._board.copy() if self._board is not None else None

    @property
    def move_stack(self) -> tuple[chess.Move, ...]:
        """Accumulated moves since the last reset / game start."""
        if self._board is None:
            return ()
        return tuple(self._board.move_stack)

    @property
    def is_tracking(self) -> bool:
        """``True`` once the first position has been established."""
        return self._board is not None

    def reset(self) -> None:
        """Reset to uninitialised state."""
        self._board = None
        self._clear_pending_state()

    # -- Internal helpers ----------------------------------------------------

    def _init_board(
        self, placement: str, t0: float,
    ) -> GameTrackingEvent:
        """Set up the board from the first detected placement."""
        if is_starting_position(placement):
            self._board = chess.Board()
        else:
            self._board = self._board_from_placement(placement, active_color="w")

        logger.info("Game tracking started: %s", self._board.fen())
        return GameTrackingEvent(
            type=GameTrackingEventType.GAME_STARTED,
            fen=self._board.fen(),
            move_number=self._board.fullmove_number,
            active_color="w" if self._board.turn == chess.WHITE else "b",
            is_new_game=is_starting_position(placement),
            elapsed_ms=_elapsed(t0),
            message="Game tracking started.",
        )

    def _apply_move(
        self, move: chess.Move, t0: float,
    ) -> GameTrackingEvent:
        """Push *move* onto the board and return a MOVE event."""
        san = self._board.san(move)  # type: ignore[union-attr]
        uci = move.uci()
        self._board.push(move)  # type: ignore[union-attr]

        logger.debug(
            "Move %d: %s (%s)",
            self._board.fullmove_number,  # type: ignore[union-attr]
            san,
            uci,
        )
        return GameTrackingEvent(
            type=GameTrackingEventType.MOVE,
            fen=self._board.fen(),  # type: ignore[union-attr]
            move=move,
            move_san=san,
            move_uci=uci,
            move_number=self._board.fullmove_number,  # type: ignore[union-attr]
            active_color="w" if self._board.turn == chess.WHITE else "b",  # type: ignore[union-attr]
            elapsed_ms=_elapsed(t0),
            message=f"Move: {san}",
        )

    def _handle_discontinuity(
        self, placement: str, t0: float,
    ) -> GameTrackingEvent:
        """Handle a position that no legal move can produce."""
        if self._config.require_confirmation:
            if self._pending_placement == placement:
                self._pending_count += 1
            else:
                self._pending_placement = placement
                self._pending_count = 1

            if self._pending_count < self._required_confirmations:
                logger.debug(
                    "Pending discontinuity confirmation (%d/%d).",
                    self._pending_count,
                    self._required_confirmations,
                )
                return GameTrackingEvent(
                    type=GameTrackingEventType.NO_CHANGE,
                    fen=self._board.fen(),  # type: ignore[union-attr]
                    move_number=self._board.fullmove_number,  # type: ignore[union-attr]
                    active_color="w" if self._board.turn == chess.WHITE else "b",  # type: ignore[union-attr]
                    elapsed_ms=_elapsed(t0),
                    message=(
                        "Awaiting discontinuity confirmation "
                        f"({self._pending_count}/{self._required_confirmations})."
                    ),
                )
            # Confirmed.

        self._clear_pending_state()

        if is_starting_position(placement):
            self._board = chess.Board()
            logger.info("New game detected.")
            return GameTrackingEvent(
                type=GameTrackingEventType.NEW_GAME_DETECTED,
                fen=self._board.fen(),
                move_number=1,
                active_color="w",
                is_new_game=True,
                elapsed_ms=_elapsed(t0),
                message="New game detected — board reset.",
            )

        # Non-starting discontinuity.
        previous_turn = "w" if self._board.turn == chess.WHITE else "b"  # type: ignore[union-attr]
        self._board = self._board_from_placement(placement, active_color=previous_turn)
        logger.info("Position reset to: %s", self._board.fen())
        return GameTrackingEvent(
            type=GameTrackingEventType.POSITION_RESET,
            fen=self._board.fen(),
            move_number=1,
            active_color=previous_turn,
            elapsed_ms=_elapsed(t0),
            message="Position discontinuity — board reset.",
        )

    def _try_recover_initial_turn(
        self,
        placement: str,
        t0: float,
    ) -> GameTrackingEvent | None:
        """Recover from unknown active colour right after an initial sync."""
        if self._board is None or self._board.move_stack:
            return None

        candidate = self._board.copy(stack=False)
        candidate.turn = not candidate.turn

        move = deduce_move_from_fen(candidate, placement)
        if move is None:
            return None

        self._board = candidate
        self._clear_pending_state()
        logger.info(
            "Recovered live continuity by flipping active color during initial sync."
        )
        return self._apply_move(move, t0)

    def _board_from_placement(
        self,
        placement: str,
        *,
        active_color: str,
    ) -> chess.Board:
        """Create a board from placement with best-effort castling inference."""
        castling = self._infer_castling_rights(placement)
        fen = f"{placement} {active_color} {castling} - 0 1"
        return chess.Board(fen)

    @staticmethod
    def _infer_castling_rights(placement: str) -> str:
        """Infer likely castling availability from piece homes."""
        board = chess.Board(f"{placement} w - - 0 1")
        rights = ""

        if board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
            if board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
                rights += "K"
            if board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
                rights += "Q"

        if board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
            if board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
                rights += "k"
            if board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
                rights += "q"

        return rights or "-"

    @property
    def _required_confirmations(self) -> int:
        if not self._config.require_confirmation:
            return 1
        return self._config.confirmation_frames

    def _clear_pending_state(self) -> None:
        self._pending_placement = None
        self._pending_count = 0


def _elapsed(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0
