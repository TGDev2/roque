"""Interactive game coaching with manual move input."""

from chesslab.coach.game_coach import (
    CoachConfig,
    CoachError,
    GameCoach,
    GameOverError,
    IllegalMoveError,
    MoveAdvice,
    NotYourTurnError,
    PlayerColor,
)

__all__ = [
    "CoachConfig",
    "CoachError",
    "GameCoach",
    "GameOverError",
    "IllegalMoveError",
    "MoveAdvice",
    "NotYourTurnError",
    "PlayerColor",
]
