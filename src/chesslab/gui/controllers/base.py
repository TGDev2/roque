"""Shared controller contracts and request types for ChessLab GUI modes."""

from __future__ import annotations

import enum
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chess

from chesslab.analysis import AnalysisConfig
from chesslab.coach.game_coach import GameCoach, PlayerColor
from chesslab.gui.board_widget import BoardWidget
from chesslab.gui.live_observer import LiveObserverConfig
from chesslab.persistence.game_storage import GameRecord, GameStorage
from chesslab.persistence.remote_import import RemoteGameSource, RemoteImportService
from chesslab.persistence.review_history import ReviewHistoryStorage
from chesslab.persistence.training_history import TrainingHistoryStorage
from chesslab.progression import ProgressTracker
from chesslab.review.batch_review import BatchReviewService
from chesslab.review.game_reviewer import GameReview, ReviewConfig
from chesslab.training import WeaknessTrainingService


class AppMode(enum.Enum):
    PLAY = "play"
    REVIEW = "review"
    TRAINING = "training"
    PROGRESSION = "progression"


@dataclass(frozen=True)
class SharedWidgets:
    root: tk.Tk
    board_host: tk.Frame
    board_widget: BoardWidget
    side_host: tk.Frame
    status_host: tk.Frame
    status_label: tk.Label
    status_icon: tk.Label


@dataclass(frozen=True)
class GuiServices:
    analysis_config: AnalysisConfig
    review_config: ReviewConfig
    storage: GameStorage
    remote_import: RemoteImportService
    review_history: ReviewHistoryStorage
    training_history: TrainingHistoryStorage
    progress_tracker: ProgressTracker
    batch_review: BatchReviewService
    training_service: WeaknessTrainingService


@dataclass(frozen=True)
class PlayModeRequest:
    record: GameRecord | None = None
    status_message: str | None = None
    start_observe_config: LiveObserverConfig | None = None


@dataclass(frozen=True)
class ReviewModeRequest:
    action: str
    board: chess.Board | None = None
    file_path: Path | None = None
    record: GameRecord | None = None
    remote_source: RemoteGameSource | None = None
    return_mode: AppMode = AppMode.PLAY
    from_observation: bool = False
    source_path: Path | None = None


@dataclass(frozen=True)
class TrainingModeRequest:
    sample_size: int
    focus_review: GameReview | None
    return_mode: AppMode
    opening_filter_key: str | None = None


@dataclass(frozen=True)
class ProgressionModeRequest:
    sample_size: int | None = None
    focus_review: GameReview | None = None
    return_mode: AppMode = AppMode.PLAY


class ModeHost(Protocol):
    @property
    def widgets(self) -> SharedWidgets:
        """Return the shared shell widgets."""

    @property
    def services(self) -> GuiServices:
        """Return long-lived service objects."""

    @property
    def current_mode(self) -> AppMode | None:
        """Return the currently active top-level mode."""

    @property
    def engine_elo(self) -> int | None:
        """Return the active engine Elo setting."""

    @engine_elo.setter
    def engine_elo(self, value: int | None) -> None:
        """Update the active engine Elo setting."""

    @property
    def progression_sample_size(self) -> int:
        """Return the shared progression sample window."""

    @progression_sample_size.setter
    def progression_sample_size(self, value: int) -> None:
        """Persist the shared progression sample window."""

    def create_coach(
        self,
        player_color: PlayerColor,
        engine_elo: int | None,
    ) -> GameCoach:
        """Create and start a new GameCoach instance."""

    def show_color_dialog(
        self,
        *,
        default_engine_elo: int | None = None,
    ) -> tuple[PlayerColor, int]:
        """Ask the user for a new-game colour and engine strength."""

    def switch_mode(self, mode: AppMode, request: object | None = None) -> None:
        """Activate *mode*, optionally with a typed request payload."""

    def set_status(
        self,
        text: str,
        *,
        error: bool = False,
        waiting: bool = False,
        game_over: bool = False,
    ) -> None:
        """Update the shared status bar."""

    def set_board_visible(self, visible: bool) -> None:
        """Show or hide the shared board host."""

    def relaunch_startup(self) -> None:
        """Show the startup launcher again."""


class ModeController(Protocol):
    mode: AppMode

    def activate(self, request: object | None = None) -> None:
        """Enter this mode, optionally using *request*."""

    def deactivate(self) -> None:
        """Leave this mode without destroying long-lived state."""

    def poll(self) -> None:
        """Drain any mode-specific queues on the Tk main thread."""

    def on_board_move(self, from_sq: str, to_sq: str) -> None:
        """Handle a shared-board click, if this mode uses the board."""

    def on_key_left(self) -> None:
        """Handle left navigation, if supported."""

    def on_key_right(self) -> None:
        """Handle right navigation, if supported."""

    def on_key_home(self) -> None:
        """Handle home navigation, if supported."""

    def on_key_end(self) -> None:
        """Handle end navigation, if supported."""

    def shutdown(self) -> None:
        """Release mode-specific resources before the shell exits."""


class ControllerBase:
    """Default no-op implementation shared by concrete mode controllers."""

    mode: AppMode

    def __init__(self, host: ModeHost) -> None:
        self._host = host

    @property
    def widgets(self) -> SharedWidgets:
        return self._host.widgets

    @property
    def services(self) -> GuiServices:
        return self._host.services

    def activate(self, request: object | None = None) -> None:
        del request

    def deactivate(self) -> None:
        return

    def poll(self) -> None:
        return

    def on_board_move(self, from_sq: str, to_sq: str) -> None:
        del from_sq, to_sq

    def on_key_left(self) -> None:
        return

    def on_key_right(self) -> None:
        return

    def on_key_home(self) -> None:
        return

    def on_key_end(self) -> None:
        return

    def shutdown(self) -> None:
        return
