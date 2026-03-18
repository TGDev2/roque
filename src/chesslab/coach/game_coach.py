"""Interactive game coaching via manual move input and Stockfish analysis.

Provides an alternative to screen capture-based position recognition by
allowing the user to manually input opponent moves in Standard Algebraic
Notation (SAN).  After each opponent move the coach queries Stockfish for
the best responses.

Usage::

    config = CoachConfig(
        player_color=PlayerColor.BLACK,
        analysis=AnalysisConfig(depth=20),
    )
    with GameCoach(config) as coach:
        advice = coach.submit_opponent_move("e4")
        for v in advice.analysis.variations:
            print(f"  {v.rank}. {v.move_san} ({v.evaluation})")
        coach.submit_player_move("e5")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import chess

from typing import Callable

from chesslab.analysis import (
    AnalysisConfig,
    AnalysisResult,
    ContinuousAnalysisHandle,
    ContinuousUpdate,
    PositionAnalyser,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class CoachError(RuntimeError):
    """Base class for coach module failures."""


class IllegalMoveError(CoachError):
    """Raised when a submitted move is not legal in the current position."""


class GameOverError(CoachError):
    """Raised when a move is submitted after the game has ended."""


class NotYourTurnError(CoachError):
    """Raised when a move is submitted for the wrong side."""


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class PlayerColor(Enum):
    """The color the human player controls."""

    WHITE = "white"
    BLACK = "black"


@dataclass
class CoachConfig:
    """Configuration for the interactive game coach.

    Parameters
    ----------
    player_color:
        The color the human player controls.  Default ``WHITE``.
    analysis:
        Stockfish analysis configuration (depth, time_limit, multipv, etc.).
    """

    player_color: PlayerColor = PlayerColor.WHITE
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


@dataclass(frozen=True)
class MoveAdvice:
    """Analysis result returned after registering an opponent move.

    Attributes
    ----------
    opponent_move_san:
        The opponent move that triggered this analysis (normalised SAN).
    current_fen:
        Board FEN after the opponent move — the player is now to move.
    analysis:
        Full Stockfish analysis with top variations for the player.
    move_number:
        Current full-move number.
    """

    opponent_move_san: str
    current_fen: str
    analysis: AnalysisResult
    move_number: int


@dataclass(frozen=True)
class ContinuousAdvice:
    """Incremental advice update emitted during continuous analysis.

    Attributes
    ----------
    opponent_move_san:
        The opponent move that triggered this analysis (normalised SAN).
    current_fen:
        Board FEN after the opponent move.
    update:
        The latest continuous analysis update.
    move_number:
        Current full-move number.
    """

    opponent_move_san: str
    current_fen: str
    update: ContinuousUpdate
    move_number: int


AdviceUpdateCallback = Callable[[ContinuousAdvice], None]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class GameCoach:
    """Interactive game coach with manual move input and Stockfish analysis.

    Maintains a chess board, accepts moves in Standard Algebraic Notation,
    and provides Stockfish analysis after each opponent move.  Turn order is
    enforced: opponent moves and player moves must alternate correctly based
    on the configured player color.

    Usage::

        with GameCoach(CoachConfig(player_color=PlayerColor.BLACK)) as coach:
            advice = coach.submit_opponent_move("e4")
            print(advice.analysis.best_move_san)
            coach.submit_player_move("e5")
    """

    def __init__(self, config: CoachConfig | None = None) -> None:
        self._config = config or CoachConfig()
        self._board = chess.Board()
        self._analyser = PositionAnalyser(self._config.analysis)
        self._opponent_chess_color = (
            chess.BLACK
            if self._config.player_color == PlayerColor.WHITE
            else chess.WHITE
        )
        self._current_handle: ContinuousAnalysisHandle | None = None

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the Stockfish engine."""
        self._analyser.start()
        logger.info(
            "Game coach started: player=%s depth=%d",
            self._config.player_color.value,
            self._config.analysis.depth,
        )

    def stop(self) -> None:
        """Stop any running analysis and the Stockfish engine."""
        self.stop_analysis()
        self._analyser.stop()
        logger.info("Game coach stopped.")

    @property
    def is_running(self) -> bool:
        """``True`` if the underlying Stockfish engine is running."""
        return self._analyser.is_running

    def __enter__(self) -> GameCoach:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- Move submission -----------------------------------------------------

    def submit_opponent_move(self, move_san: str) -> MoveAdvice:
        """Register an opponent move and return Stockfish analysis.

        Pushes the opponent's move onto the board and analyses the resulting
        position to suggest the best responses for the player.

        Parameters
        ----------
        move_san:
            The opponent's move in Standard Algebraic Notation (e.g.
            ``"e4"``, ``"Nf3"``, ``"O-O"``).

        Returns
        -------
        MoveAdvice
            The analysis result with top suggested responses.

        Raises
        ------
        GameOverError
            If the game is already over.
        NotYourTurnError
            If it is currently the player's turn, not the opponent's.
        IllegalMoveError
            If *move_san* is not a legal move in the current position.
        """
        self._check_game_not_over()

        if self._board.turn != self._opponent_chess_color:
            raise NotYourTurnError(
                "It is not the opponent's turn. "
                f"Expected {self._config.player_color.value} (player) move.",
            )

        move = self._parse_san(move_san)
        normalised_san = self._board.san(move)
        self._board.push(move)

        current_fen = self._board.fen()
        logger.info("Opponent played: %s (FEN: %s)", normalised_san, current_fen)

        result = self._analyser.analyse(current_fen)

        return MoveAdvice(
            opponent_move_san=normalised_san,
            current_fen=current_fen,
            analysis=result,
            move_number=self._board.fullmove_number,
        )

    def submit_player_move(self, move_san: str) -> None:
        """Register the player's chosen move to advance the game.

        Parameters
        ----------
        move_san:
            The player's move in Standard Algebraic Notation.

        Raises
        ------
        GameOverError
            If the game is already over.
        NotYourTurnError
            If it is currently the opponent's turn, not the player's.
        IllegalMoveError
            If *move_san* is not a legal move in the current position.
        """
        self._check_game_not_over()

        if self._board.turn == self._opponent_chess_color:
            raise NotYourTurnError(
                "It is not the player's turn. Expected opponent move.",
            )

        move = self._parse_san(move_san)
        normalised_san = self._board.san(move)
        self._board.push(move)

        logger.info("Player played: %s (FEN: %s)", normalised_san, self._board.fen())

    def submit_opponent_move_continuous(
        self,
        move_san: str,
        callback: AdviceUpdateCallback,
    ) -> ContinuousAnalysisHandle:
        """Register an opponent move and start continuous Stockfish analysis.

        Like :meth:`submit_opponent_move` but starts a streaming analysis
        that calls *callback* with a :class:`ContinuousAdvice` each time the
        search depth increases.

        Parameters
        ----------
        move_san:
            The opponent's move in Standard Algebraic Notation.
        callback:
            Called from an analysis thread with each incremental update.

        Returns
        -------
        ContinuousAnalysisHandle
            A handle to stop or wait for the analysis.

        Raises
        ------
        GameOverError
            If the game is already over.
        NotYourTurnError
            If it is currently the player's turn, not the opponent's.
        IllegalMoveError
            If *move_san* is not a legal move in the current position.
        """
        self._check_game_not_over()

        if self._board.turn != self._opponent_chess_color:
            raise NotYourTurnError(
                "It is not the opponent's turn. "
                f"Expected {self._config.player_color.value} (player) move.",
            )

        move = self._parse_san(move_san)
        normalised_san = self._board.san(move)
        self._board.push(move)

        current_fen = self._board.fen()
        move_number = self._board.fullmove_number
        logger.info("Opponent played: %s (FEN: %s)", normalised_san, current_fen)

        # Stop any previous continuous analysis.
        self.stop_analysis()

        def _wrapped_callback(update: ContinuousUpdate) -> None:
            callback(ContinuousAdvice(
                opponent_move_san=normalised_san,
                current_fen=current_fen,
                update=update,
                move_number=move_number,
            ))

        handle = self._analyser.analyse_continuous(current_fen, _wrapped_callback)
        self._current_handle = handle
        return handle

    def undo_move(self) -> str | None:
        """Undo the last move on the board.

        Stops any running analysis and pops the most recent move.

        Returns
        -------
        str or None
            The SAN of the undone move, or ``None`` if no moves to undo.
        """
        self.stop_analysis()

        if not self._board.move_stack:
            return None

        # Recover SAN before popping.
        move = self._board.peek()
        self._board.pop()
        # Replay to get SAN of the undone move.
        san = self._board.san(move)
        logger.info("Undo move: %s (FEN: %s)", san, self._board.fen())
        return san

    def stop_analysis(self) -> None:
        """Stop any running continuous analysis."""
        if self._current_handle is not None and self._current_handle.is_running:
            self._current_handle.stop()
            self._current_handle = None

    # -- Properties ----------------------------------------------------------

    @property
    def current_fen(self) -> str:
        """The current board position as a FEN string."""
        return self._board.fen()

    @property
    def move_number(self) -> int:
        """The current full-move number."""
        return self._board.fullmove_number

    @property
    def is_game_over(self) -> bool:
        """``True`` if the game has ended (checkmate, stalemate, etc.)."""
        return self._board.is_game_over()

    @property
    def is_player_turn(self) -> bool:
        """``True`` if it is currently the player's turn to move."""
        return self._board.turn != self._opponent_chess_color

    @property
    def move_history(self) -> tuple[str, ...]:
        """SAN moves played so far, in order."""
        replay = chess.Board()
        san_list: list[str] = []
        for move in self._board.move_stack:
            san_list.append(replay.san(move))
            replay.push(move)
        return tuple(san_list)

    # -- Private helpers -----------------------------------------------------

    def _check_game_not_over(self) -> None:
        """Raise if the game has ended."""
        if self._board.is_game_over():
            outcome = self._board.outcome()
            reason = outcome.termination.name if outcome else "unknown"
            raise GameOverError(f"Game is over: {reason}.")

    def _parse_san(self, move_san: str) -> chess.Move:
        """Parse a SAN string, raising IllegalMoveError on failure."""
        try:
            return self._board.parse_san(move_san)
        except (
            chess.InvalidMoveError,
            chess.IllegalMoveError,
            chess.AmbiguousMoveError,
        ) as exc:
            raise IllegalMoveError(
                f"Illegal move '{move_san}': {exc}",
            ) from exc
