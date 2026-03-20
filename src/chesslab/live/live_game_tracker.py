"""Stateful game tracking for the live observation pipeline.

Maintains a :class:`chess.Board` that is updated move-by-move as new
piece placements are detected by the CV pipeline.  Handles game
continuity, new-game detection, and position resets.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field

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
        found) must be confirmed by a second identical frame before the
        tracker acts on it.  This absorbs transient OCR errors.
    """

    initial_fen: str | None = None
    require_confirmation: bool = True


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
            self._pending_placement = None
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
            self._pending_placement = None
            return self._apply_move(move, t0)

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
        self._pending_placement = None

    # -- Internal helpers ----------------------------------------------------

    def _init_board(
        self, placement: str, t0: float,
    ) -> GameTrackingEvent:
        """Set up the board from the first detected placement."""
        if is_starting_position(placement):
            self._board = chess.Board()
        else:
            # Best-effort: assume white to move, full castling, no en passant.
            fen = f"{placement} w KQkq - 0 1"
            self._board = chess.Board(fen)

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
            if self._pending_placement is None or self._pending_placement != placement:
                # First occurrence or different from previous pending — wait.
                self._pending_placement = placement
                logger.debug("Pending confirmation for discontinuity.")
                return GameTrackingEvent(
                    type=GameTrackingEventType.NO_CHANGE,
                    fen=self._board.fen(),  # type: ignore[union-attr]
                    move_number=self._board.fullmove_number,  # type: ignore[union-attr]
                    active_color="w" if self._board.turn == chess.WHITE else "b",  # type: ignore[union-attr]
                    elapsed_ms=_elapsed(t0),
                    message="Awaiting confirmation of position change.",
                )
            # Second identical frame — confirmed.

        self._pending_placement = None

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
        fen = f"{placement} w KQkq - 0 1"
        self._board = chess.Board(fen)
        logger.info("Position reset to: %s", self._board.fen())
        return GameTrackingEvent(
            type=GameTrackingEventType.POSITION_RESET,
            fen=self._board.fen(),
            move_number=1,
            active_color="w",
            elapsed_ms=_elapsed(t0),
            message="Position discontinuity — board reset.",
        )


def _elapsed(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0
