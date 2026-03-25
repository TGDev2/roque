"""Tkinter shell for ChessLab GUI mode controllers."""

from __future__ import annotations

import logging
import tkinter as tk
from dataclasses import dataclass, field, replace
from pathlib import Path

from chesslab.analysis import AnalysisConfig
from chesslab.coach.game_coach import CoachConfig, GameCoach, PlayerColor
from chesslab.gui import theme
from chesslab.gui.board_widget import BoardWidget
from chesslab.gui.controllers import (
    AppMode,
    GuiServices,
    ModeController,
    PlayController,
    PlayModeRequest,
    ProgressionController,
    ProgressionModeRequest,
    ReviewController,
    ReviewModeRequest,
    SharedWidgets,
    TrainingController,
)
from chesslab.gui.dialog_utils import create_scrollable_dialog, finalize_scrollable_dialog
from chesslab.gui.journeys import StartupJourneyMode, StartupJourneyOrchestrator
from chesslab.gui.live_observer import LiveObserverConfig
from chesslab.persistence.game_storage import GameStorage
from chesslab.persistence.remote_import import RemoteGameSource, RemoteImportService
from chesslab.persistence.review_history import ReviewHistoryStorage
from chesslab.persistence.training_history import TrainingHistoryStorage
from chesslab.product_usage import PRODUCT_BOUNDARY, PRODUCT_SCOPE
from chesslab.progression import ProgressTracker
from chesslab.review.batch_review import BatchReviewService
from chesslab.review.game_reviewer import ReviewConfig
from chesslab.training import WeaknessTrainingService

logger = logging.getLogger(__name__)

_POLL_INTERVAL_MS = 100


@dataclass
class GuiConfig:
    """Configuration for the ChessLab GUI shell."""

    player_color: PlayerColor | None = None
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    square_size: int = 64
    startup_mode: StartupJourneyMode | None = None
    startup_review_file: Path | None = None
    progression_sample_size: int = 20


class ChessGui:
    """Tkinter shell that orchestrates GUI mode controllers."""

    def __init__(self, config: GuiConfig | None = None) -> None:
        self._config = config or GuiConfig()
        self._engine_elo: int | None = self._config.analysis.engine_elo
        self._progression_sample_size = max(3, self._config.progression_sample_size)
        self._root: tk.Tk | None = None
        self._widgets: SharedWidgets | None = None

        review_config = ReviewConfig(
            stockfish_path=self._config.analysis.stockfish_path,
            depth=self._config.analysis.depth,
            time_limit=self._config.analysis.time_limit,
        )
        storage = GameStorage()
        remote_import = RemoteImportService(storage=storage)
        review_history = ReviewHistoryStorage()
        training_history = TrainingHistoryStorage()
        progress_tracker = ProgressTracker(review_history, training_history)
        self._services = GuiServices(
            analysis_config=self._config.analysis,
            review_config=review_config,
            storage=storage,
            remote_import=remote_import,
            review_history=review_history,
            training_history=training_history,
            progress_tracker=progress_tracker,
            batch_review=BatchReviewService(
                storage=storage,
                review_history=review_history,
                review_config=review_config,
            ),
            training_service=WeaknessTrainingService(review_config),
        )

        self._controllers: dict[AppMode, ModeController] = {}
        self._active_mode: AppMode | None = None
        self._startup_journeys = StartupJourneyOrchestrator()

    @property
    def widgets(self) -> SharedWidgets:
        assert self._widgets is not None
        return self._widgets

    @property
    def services(self) -> GuiServices:
        return self._services

    @property
    def current_mode(self) -> AppMode | None:
        return self._active_mode

    @property
    def engine_elo(self) -> int | None:
        return self._engine_elo

    @engine_elo.setter
    def engine_elo(self, value: int | None) -> None:
        self._engine_elo = value

    @property
    def progression_sample_size(self) -> int:
        return self._progression_sample_size

    @progression_sample_size.setter
    def progression_sample_size(self, value: int) -> None:
        self._progression_sample_size = max(3, min(int(value), 100))

    @property
    def startup_review_file(self) -> Path | None:
        return self._config.startup_review_file

    def run(self) -> None:
        root = tk.Tk()
        root.title("ChessLab")
        root.configure(bg=theme.BG)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root = root

        startup_mode = self._config.startup_mode
        if startup_mode is None:
            startup_mode = self._show_startup_dialog(root)

        player_color = self._resolve_initial_player_color(startup_mode, root)
        self._build_shell(root, player_color)
        self._create_controllers(player_color)
        self.switch_mode(AppMode.PLAY)

        root.after(_POLL_INTERVAL_MS, self._poll_active_controller)
        root.after(80, lambda: self._apply_startup_mode(startup_mode))

        logger.info(
            "ChessLab GUI started: player=%s elo=%s",
            player_color.value,
            self._engine_elo if self._engine_elo is not None else "max",
        )
        root.mainloop()

    def create_coach(
        self,
        player_color: PlayerColor,
        engine_elo: int | None,
    ) -> GameCoach:
        analysis_config = replace(self._config.analysis, engine_elo=engine_elo)
        coach = GameCoach(
            CoachConfig(
                player_color=player_color,
                analysis=analysis_config,
            )
        )
        coach.start()
        return coach

    def show_color_dialog(
        self,
        *,
        default_engine_elo: int | None = None,
    ) -> tuple[PlayerColor, int]:
        assert self._root is not None

        initial_elo = (
            default_engine_elo
            if default_engine_elo is not None
            else AnalysisConfig.DEFAULT_ENGINE_ELO
        )
        initial_elo = AnalysisConfig.snap_engine_elo(initial_elo)
        result: list[tuple[PlayerColor, int]] = [(PlayerColor.WHITE, initial_elo)]
        elo_var = tk.IntVar(value=initial_elo)

        dialog = tk.Toplevel(self._root)
        dialog.title("ChessLab")
        dialog.configure(bg=theme.BG)
        dialog.transient(self._root)
        dialog.grab_set()
        dialog.resizable(False, False)

        container = tk.Frame(dialog, bg=theme.BG)
        container.pack(fill="both", expand=True, padx=20, pady=18)

        tk.Label(
            container,
            text="New Game",
            bg=theme.BG,
            fg=theme.FG,
            font=("sans-serif", 16, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        elo_card = tk.Frame(container, bg=theme.CARD_BORDER)
        elo_card.pack(fill="x", pady=(0, 12))
        elo_inner = tk.Frame(elo_card, bg=theme.BG_SECONDARY)
        elo_inner.pack(fill="both", expand=True, padx=1, pady=1)

        elo_header = tk.Frame(elo_inner, bg=theme.BG_SECONDARY)
        elo_header.pack(fill="x", padx=12, pady=(10, 6))

        tk.Label(
            elo_header,
            text="Engine Strength",
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("sans-serif", 11, "bold"),
        ).pack(side="left")

        elo_value_frame = tk.Frame(elo_header, bg=theme.BG_SECONDARY)
        elo_value_frame.pack(side="right")

        elo_value = tk.Label(
            elo_value_frame,
            text=str(initial_elo),
            bg=theme.BG_SECONDARY,
            fg=theme.ACCENT,
            font=("monospace", 14, "bold"),
        )
        elo_value.pack(side="left")

        elo_suffix = tk.Label(
            elo_value_frame,
            text="",
            bg=theme.BG_SECONDARY,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
        )
        elo_suffix.pack(side="left", padx=(6, 0))

        def refresh_elo_label() -> None:
            value = AnalysisConfig.snap_engine_elo(elo_var.get())
            elo_var.set(value)
            elo_value.config(text=str(value))
            elo_suffix.config(
                text="(default)" if value == AnalysisConfig.DEFAULT_ENGINE_ELO else ""
            )

        scale = tk.Scale(
            elo_inner,
            from_=AnalysisConfig.MIN_ENGINE_ELO,
            to=AnalysisConfig.MAX_ENGINE_ELO,
            orient="horizontal",
            resolution=50,
            variable=elo_var,
            showvalue=False,
            bg=theme.BG_SECONDARY,
            fg=theme.FG_DIM,
            troughcolor=theme.BTN_BG,
            highlightthickness=0,
            bd=0,
            command=lambda _value: refresh_elo_label(),
        )
        scale.pack(fill="x", padx=12)

        range_frame = tk.Frame(elo_inner, bg=theme.BG_SECONDARY)
        range_frame.pack(fill="x", padx=12, pady=(0, 10))
        tk.Label(
            range_frame,
            text=str(AnalysisConfig.MIN_ENGINE_ELO),
            bg=theme.BG_SECONDARY,
            fg=theme.FG_MUTED,
            font=("monospace", 8),
        ).pack(side="left")
        tk.Label(
            range_frame,
            text=str(AnalysisConfig.MAX_ENGINE_ELO),
            bg=theme.BG_SECONDARY,
            fg=theme.FG_MUTED,
            font=("monospace", 8),
        ).pack(side="right")

        tk.Label(
            container,
            text="Play as",
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
        ).pack(anchor="w", pady=(0, 6))

        btn_frame = tk.Frame(container, bg=theme.BG)
        btn_frame.pack(fill="x")

        def choose(color: PlayerColor) -> None:
            result[0] = (color, AnalysisConfig.snap_engine_elo(elo_var.get()))
            dialog.destroy()

        theme.styled_button(
            btn_frame,
            text="White",
            command=lambda: choose(PlayerColor.WHITE),
            bg=theme.BTN_WHITE_BG,
            fg=theme.BTN_WHITE_FG,
            hover_bg=theme.BTN_WHITE_HOVER,
            active_bg="#f5ede1",
            font=("sans-serif", 11, "bold"),
            padx=14,
            pady=8,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        theme.styled_button(
            btn_frame,
            text="Black",
            command=lambda: choose(PlayerColor.BLACK),
            bg=theme.BTN_BLACK_BG,
            fg=theme.BTN_BLACK_FG,
            hover_bg=theme.BTN_BLACK_HOVER,
            active_bg="#4a4a5c",
            font=("sans-serif", 11, "bold"),
            padx=14,
            pady=8,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        refresh_elo_label()
        dialog.update_idletasks()
        x = self._root.winfo_x() + (self._root.winfo_width() - dialog.winfo_width()) // 2
        y = self._root.winfo_y() + (self._root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
        self._root.wait_window(dialog)
        return result[0]

    def switch_mode(self, mode: AppMode, request: object | None = None) -> None:
        controller = self._controllers[mode]
        if self._active_mode == mode:
            controller.activate(request)
            return

        if self._active_mode is not None:
            self._controllers[self._active_mode].deactivate()

        self._active_mode = mode
        controller.activate(request)

    def set_status(
        self,
        text: str,
        *,
        error: bool = False,
        waiting: bool = False,
        game_over: bool = False,
    ) -> None:
        if self._widgets is None:
            return

        fg = (
            theme.ERROR
            if error
            else theme.FG_MUTED
            if waiting
            else theme.ACCENT
            if game_over
            else theme.FG_DIM
        )
        self._widgets.status_label.config(text=text, fg=fg)

        if error:
            self._widgets.status_icon.config(text="●", fg=theme.ERROR)
        elif waiting:
            self._widgets.status_icon.config(text="○", fg=theme.FG_MUTED)
        elif game_over:
            self._widgets.status_icon.config(text="★", fg=theme.ACCENT)
        else:
            self._widgets.status_icon.config(text="●", fg=theme.SUCCESS)

    def set_board_visible(self, visible: bool) -> None:
        if self._widgets is None:
            return

        manager = str(self._widgets.board_host.winfo_manager())
        if visible:
            if manager != "pack":
                self._widgets.board_host.pack(
                    side="left",
                    padx=(0, 8),
                    before=self._widgets.side_host,
                )
        elif manager == "pack":
            self._widgets.board_host.pack_forget()

    def relaunch_startup(self) -> None:
        assert self._root is not None
        mode = self._show_startup_dialog(self._root)
        self._apply_startup_mode(mode)

    def set_startup_status(self, text: str, *, error: bool = False) -> None:
        self.set_status(text, error=error)

    def open_observe_live_startup(self) -> LiveObserverConfig | None:
        return self._play_controller().open_observe_live_startup()

    def start_observe_live(self, config: LiveObserverConfig) -> bool:
        return self._play_controller().start_observation(config)

    def open_progression_startup(self) -> None:
        self.switch_mode(
            AppMode.PROGRESSION,
            ProgressionModeRequest(
                sample_size=self._progression_sample_size,
                return_mode=AppMode.PLAY,
            ),
        )

    def open_review_recent_startup(self) -> None:
        self.switch_mode(
            AppMode.REVIEW,
            ReviewModeRequest(
                action="open_recent",
                return_mode=AppMode.PLAY,
            ),
        )

    def open_review_paste_startup(self) -> None:
        self.switch_mode(
            AppMode.REVIEW,
            ReviewModeRequest(
                action="open_paste",
                return_mode=AppMode.PLAY,
            ),
        )

    def open_review_file_startup(self, file_path: Path) -> None:
        self.switch_mode(
            AppMode.REVIEW,
            ReviewModeRequest(
                action="review_file",
                file_path=file_path,
                return_mode=AppMode.PLAY,
                source_path=file_path,
            ),
        )

    def open_import_startup(self, source: RemoteGameSource) -> None:
        self.switch_mode(
            AppMode.REVIEW,
            ReviewModeRequest(
                action="open_import",
                remote_source=source,
                return_mode=AppMode.PLAY,
            ),
        )

    def _resolve_initial_player_color(
        self,
        startup_mode: StartupJourneyMode,
        root: tk.Tk,
    ) -> PlayerColor:
        player_color = self._config.player_color
        if player_color is not None:
            return player_color

        if startup_mode in {
            StartupJourneyMode.PROGRESSION,
            StartupJourneyMode.REVIEW_RECENT,
            StartupJourneyMode.REVIEW_PASTE,
            StartupJourneyMode.REVIEW_FILE,
            StartupJourneyMode.IMPORT_LICHESS,
            StartupJourneyMode.IMPORT_CHESS_COM,
        }:
            if self._engine_elo is None:
                self._engine_elo = AnalysisConfig.DEFAULT_ENGINE_ELO
            return PlayerColor.WHITE

        player_color, self._engine_elo = self.show_color_dialog(
            default_engine_elo=self._engine_elo,
        )
        return player_color

    def _build_shell(self, root: tk.Tk, player_color: PlayerColor) -> None:
        flipped = player_color == PlayerColor.BLACK

        main_frame = tk.Frame(root, bg=theme.BG)
        main_frame.pack(fill="both", expand=True, padx=6, pady=(6, 0))

        board_outer = tk.Frame(main_frame, bg=theme.BOARD_SHADOW_BG, bd=0)
        board_outer.pack(side="left", padx=(0, 8))

        board_frame = tk.Frame(board_outer, bg=theme.BG)
        board_frame.pack(padx=2, pady=2)

        board_widget = BoardWidget(
            board_frame,
            square_size=self._config.square_size,
            flipped=flipped,
            on_move=self._on_board_move,
        )
        board_widget.pack()

        side_host = tk.Frame(main_frame, bg=theme.BG)
        side_host.pack(side="left", fill="both", expand=True)

        status_bar = tk.Frame(root, bg=theme.STATUS_BG, height=40)
        status_bar.pack(fill="x", padx=0, pady=0)
        status_bar.pack_propagate(False)

        inner_status = tk.Frame(status_bar, bg=theme.STATUS_BG)
        inner_status.pack(fill="both", expand=True, padx=8)

        status_host = tk.Frame(inner_status, bg=theme.STATUS_BG)
        status_host.pack(side="left", fill="y", pady=4)

        status_icon = tk.Label(
            inner_status,
            text="●",
            bg=theme.STATUS_BG,
            fg=theme.SUCCESS,
            font=("sans-serif", 10),
        )
        status_icon.pack(side="left", padx=(16, 4))

        status_label = tk.Label(
            inner_status,
            text="",
            bg=theme.STATUS_BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 10),
            anchor="w",
        )
        status_label.pack(side="left", fill="x", expand=True)

        self._widgets = SharedWidgets(
            root=root,
            board_host=board_outer,
            board_widget=board_widget,
            side_host=side_host,
            status_host=status_host,
            status_label=status_label,
            status_icon=status_icon,
        )

        root.bind("<Left>", self._on_key_left)
        root.bind("<Right>", self._on_key_right)
        root.bind("<Home>", self._on_key_home)
        root.bind("<End>", self._on_key_end)

    def _create_controllers(self, player_color: PlayerColor) -> None:
        self._controllers = {
            AppMode.PLAY: PlayController(self, initial_player_color=player_color),
            AppMode.REVIEW: ReviewController(self),
            AppMode.TRAINING: TrainingController(self),
            AppMode.PROGRESSION: ProgressionController(self),
        }

    def _play_controller(self) -> PlayController:
        return self._controllers[AppMode.PLAY]  # type: ignore[return-value]

    def _poll_active_controller(self) -> None:
        if self._active_mode is not None:
            self._controllers[self._active_mode].poll()
        if self._root is not None:
            self._root.after(_POLL_INTERVAL_MS, self._poll_active_controller)

    def _on_board_move(self, from_sq: str, to_sq: str) -> None:
        if self._active_mode is None:
            return
        self._controllers[self._active_mode].on_board_move(from_sq, to_sq)

    def _on_key_left(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        del event
        if self._active_mode is not None:
            self._controllers[self._active_mode].on_key_left()

    def _on_key_right(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        del event
        if self._active_mode is not None:
            self._controllers[self._active_mode].on_key_right()

    def _on_key_home(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        del event
        if self._active_mode is not None:
            self._controllers[self._active_mode].on_key_home()

    def _on_key_end(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        del event
        if self._active_mode is not None:
            self._controllers[self._active_mode].on_key_end()

    def _show_startup_dialog(self, parent: tk.Tk) -> StartupJourneyMode:
        result: list[StartupJourneyMode] = [StartupJourneyMode.MANUAL_COACH]

        layout = create_scrollable_dialog(
            parent,
            title="ChessLab",
            padx=20,
            pady=18,
        )
        dialog = layout.dialog
        container = layout.container

        tk.Label(
            container,
            text="Choose your journey",
            bg=theme.BG,
            fg=theme.FG,
            font=("sans-serif", 16, "bold"),
        ).pack(anchor="w")

        tk.Label(
            container,
            text=(
                "Start from one place and switch modes in the same app without "
                "changing the product boundary."
            ),
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=430,
        ).pack(anchor="w", pady=(4, 12))

        scope_card = tk.Frame(container, bg=theme.CARD_BORDER)
        scope_card.pack(fill="x", pady=(0, 12))
        scope_inner = tk.Frame(scope_card, bg=theme.BG_SECONDARY)
        scope_inner.pack(fill="both", expand=True, padx=1, pady=1)

        tk.Label(
            scope_inner,
            text="Product Scope",
            bg=theme.BG_SECONDARY,
            fg=theme.ACCENT,
            font=("sans-serif", 9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(
            scope_inner,
            text=PRODUCT_SCOPE,
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("sans-serif", 9, "bold"),
            justify="left",
            wraplength=410,
            anchor="w",
        ).pack(fill="x", padx=10)

        tk.Label(
            scope_inner,
            text=PRODUCT_BOUNDARY,
            bg=theme.BG_SECONDARY,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=410,
            anchor="w",
        ).pack(fill="x", padx=10, pady=(4, 8))

        def choose(mode: StartupJourneyMode) -> None:
            result[0] = mode
            dialog.destroy()

        for journey in self._startup_journeys.launcher_entries():
            card = tk.Frame(container, bg=theme.CARD_BORDER)
            card.pack(fill="x", pady=(0, 8))
            inner = tk.Frame(card, bg=theme.BG_SECONDARY)
            inner.pack(fill="both", expand=True, padx=1, pady=1)

            row = tk.Frame(inner, bg=theme.BG_SECONDARY)
            row.pack(fill="x", padx=10, pady=(8, 3))

            tk.Label(
                row,
                text=journey.label,
                bg=theme.BG_SECONDARY,
                fg=theme.FG,
                font=("sans-serif", 11, "bold"),
                anchor="w",
            ).pack(side="left")

            theme.styled_button(
                row,
                text="Start",
                command=lambda m=journey.mode: choose(m),
                bg=theme.BTN_ACCENT_BG,
                fg=theme.BTN_ACCENT_FG,
                hover_bg=theme.BTN_ACCENT_HOVER,
                active_bg="#1e88e5",
                font=("sans-serif", 9, "bold"),
                padx=10,
                pady=4,
            ).pack(side="right")

            tk.Label(
                inner,
                text=journey.description,
                bg=theme.BG_SECONDARY,
                fg=theme.FG_DIM,
                font=("sans-serif", 9),
                justify="left",
                wraplength=390,
                anchor="w",
            ).pack(fill="x", padx=10, pady=(0, 8))

        footer = tk.Frame(container, bg=theme.BG)
        footer.pack(fill="x", pady=(2, 0))
        theme.styled_button(
            footer,
            text="Default to Manual Coach",
            command=dialog.destroy,
            font=("sans-serif", 9),
            padx=10,
            pady=5,
        ).pack(side="right")

        finalize_scrollable_dialog(parent, layout)
        parent.wait_window(dialog)
        return result[0]

    def _apply_startup_mode(self, mode: StartupJourneyMode) -> None:
        if self._root is None:
            return
        self._startup_journeys.launch(self, mode)

    def _on_close(self) -> None:
        for controller in self._controllers.values():
            controller.shutdown()
        if self._root is not None:
            self._root.destroy()
            self._root = None
        logger.info("ChessLab GUI closed.")
