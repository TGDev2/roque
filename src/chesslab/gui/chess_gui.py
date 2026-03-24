"""Main GUI application wiring board, panels, and GameCoach.

Provides an interactive chess interface where the user can play moves via
board clicks or SAN text entry, with continuous Stockfish analysis displayed
in real time.

Usage::

    from chesslab.gui import ChessGui, GuiConfig

    gui = ChessGui(GuiConfig(square_size=72))
    gui.run()

Or simply::

    python -m chesslab.gui
"""

from __future__ import annotations

import logging
import enum
import queue
import threading
import tkinter as tk
import tkinter.ttk as ttk
from dataclasses import dataclass, field, replace
from pathlib import Path
from tkinter import filedialog

import chess

from chesslab.persistence.game_storage import GameRecord, GameStorage, build_headers
from chesslab.persistence.review_history import ReviewHistoryStorage

from chesslab.analysis import AnalysisConfig, ContinuousAnalysisHandle
from chesslab.coach.game_coach import (
    AdviceUpdateCallback,
    CoachConfig,
    ContinuousAdvice,
    GameCoach,
    GameOverError,
    IllegalMoveError,
    NotYourTurnError,
    PlayerColor,
)
from chesslab.capture.screen_capture import ScreenCapture
from chesslab.capture.screen_capture import (
    CapturePermissionError,
    DisplayUnavailableError,
    MonitorSelectionError,
    ScreenCaptureError,
)
from chesslab.progression import ProgressTracker
from chesslab.gui.board_widget import BoardWidget
from chesslab.gui.live_setup_assistant import LiveSetupAssistant
from chesslab.gui.live_observer import (
    LiveObserver,
    LiveObserverConfig,
    ObservationEvent,
    ObservationEventType,
)
from chesslab.live.live_position_processor import LiveFrameStatus
from chesslab.live.move_deduction import deduce_move_from_fen
from chesslab.gui.panels import (
    EvalPanel,
    GameSummaryPanel,
    MoveHistoryPanel,
    ObserveLivePanel,
    ProgressionPanel,
    ReviewNavigationPanel,
    ReviewPanel,
    SuggestionsPanel,
)
from chesslab.review.game_reviewer import (
    GameReview,
    GameReviewer,
    MoveClassification,
    ReviewCancelledError,
    ReviewConfig,
)
from chesslab.product_usage import (
    OBSERVE_LIVE_ACKNOWLEDGEMENT,
    OBSERVE_LIVE_DIALOG_INTRO,
    OBSERVE_LIVE_GUARDRAIL_BODY,
    OBSERVE_LIVE_GUARDRAIL_TITLE,
    PRODUCT_BOUNDARY,
    PRODUCT_SCOPE,
    journey_usage,
)
from chesslab.validation import CalibrationReport, SetupSupportProfile, SupportCatalogError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette — modern dark theme
# ---------------------------------------------------------------------------

_BG = "#16161e"
_BG_SECONDARY = "#1e1e2e"
_CARD_BORDER = "#2a2a3c"
_FG = "#e0e0e8"
_FG_DIM = "#7a7a8c"
_FG_MUTED = "#555566"
_ACCENT = "#64b5f6"
_ACCENT_HOVER = "#90caf9"
_ACCENT_DARK = "#3d7ab8"
_ERROR = "#ef5350"
_SUCCESS = "#66bb6a"
_COACH_MOVE_HINT = "#3b82f6"
_REVIEW_MOVE_HINTS: dict[MoveClassification, str] = {
    MoveClassification.INACCURACY: "#ffee58",
    MoveClassification.MISTAKE: "#ffb74d",
    MoveClassification.BLUNDER: "#ef5350",
}

# Button colours.
_BTN_BG = "#2a2a3c"
_BTN_FG = "#e0e0e8"
_BTN_HOVER = "#3a3a4c"
_BTN_ACTIVE = "#4a4a5c"
_BTN_ACCENT_BG = "#2962ff"
_BTN_ACCENT_FG = "#ffffff"
_BTN_ACCENT_HOVER = "#448aff"
_BTN_WHITE_BG = "#e8e0d4"
_BTN_WHITE_FG = "#1a1a1a"
_BTN_WHITE_HOVER = "#f0e8dc"
_BTN_BLACK_BG = "#2a2a3c"
_BTN_BLACK_FG = "#e0e0e8"
_BTN_BLACK_HOVER = "#3a3a4c"

# Entry field.
_ENTRY_BG = "#1a1a2e"
_ENTRY_FG = "#e0e0e8"
_ENTRY_BORDER = "#3a3a4c"
_ENTRY_FOCUS_BORDER = "#64b5f6"

# Status bar.
_STATUS_BG = "#12121a"

_POLL_INTERVAL_MS = 100


# ---------------------------------------------------------------------------
# Styled button helper
# ---------------------------------------------------------------------------


def _styled_button(
    parent: tk.Widget,
    text: str,
    command: object,
    *,
    bg: str = _BTN_BG,
    fg: str = _BTN_FG,
    hover_bg: str = _BTN_HOVER,
    active_bg: str = _BTN_ACTIVE,
    font: tuple[str, ...] = ("sans-serif", 10),
    padx: int = 12,
    pady: int = 6,
    cursor: str = "hand2",
    **kwargs: object,
) -> tk.Button:
    """Create a flat-styled button with hover effect."""
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        font=font,
        bg=bg,
        fg=fg,
        activebackground=active_bg,
        activeforeground=fg,
        relief="flat",
        bd=0,
        padx=padx,
        pady=pady,
        cursor=cursor,
        highlightthickness=0,
        **kwargs,  # type: ignore[arg-type]
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg))
    return btn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class StartupJourneyMode(enum.Enum):
    """Initial product flow selected at startup."""

    MANUAL_COACH = "manual_coach"
    OBSERVE_LIVE = "observe_live"
    PROGRESSION = "progression"
    REVIEW_RECENT = "review_recent"
    REVIEW_PASTE = "review_paste"
    REVIEW_FILE = "review_file"


@dataclass
class GuiConfig:
    """Configuration for the chess GUI.

    Parameters
    ----------
    player_color:
        The color the human player controls.  If ``None``, a dialog is
        shown at launch.
    analysis:
        Stockfish analysis configuration.
    square_size:
        Board square size in pixels.
    startup_mode:
        Optional startup flow to bypass the guided launcher.  ``None``
        shows the startup launcher dialog.
    startup_review_file:
        Optional PGN file to review immediately when ``startup_mode`` is
        :attr:`StartupJourneyMode.REVIEW_FILE`.
    progression_sample_size:
        Default sample window for the progression dashboard.
    """

    player_color: PlayerColor | None = None
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    square_size: int = 64
    startup_mode: StartupJourneyMode | None = None
    startup_review_file: Path | None = None
    progression_sample_size: int = 20


# ---------------------------------------------------------------------------
# Main GUI class
# ---------------------------------------------------------------------------


class ChessGui:
    """Main GUI application with interactive board and Stockfish analysis.

    Call :meth:`run` to start the tkinter event loop (blocking).
    """

    def __init__(self, config: GuiConfig | None = None) -> None:
        self._config = config or GuiConfig()
        self._engine_elo: int | None = self._config.analysis.engine_elo
        self._root: tk.Tk | None = None
        self._coach: GameCoach | None = None
        self._current_handle: ContinuousAnalysisHandle | None = None
        self._update_queue: queue.Queue[ContinuousAdvice] = queue.Queue()

        # Current best move from analysis (UCI string, e.g. "e2e4").
        self._best_move_uci: str | None = None

        # Live observation state.
        self._observer: LiveObserver | None = None
        self._observation_queue: queue.Queue[ObservationEvent] = queue.Queue()
        self._observing: bool = False
        self._last_observer_config: LiveObserverConfig | None = None
        self._live_setup_assistant: LiveSetupAssistant | None = None

        # Persistence.
        self._storage = GameStorage()
        self._review_history = ReviewHistoryStorage()
        self._progress_tracker = ProgressTracker(self._review_history)
        self._game_saved: bool = False
        self._current_game_file: Path | None = None
        self._progression_sample_size = max(3, self._config.progression_sample_size)

        # Review mode state.
        self._from_observation: bool = False
        self._in_review_mode: bool = False
        self._game_review: GameReview | None = None
        self._review_ply: int = 0
        self._review_positions: list[str] = []
        self._review_cancel: threading.Event | None = None
        self._review_queue: queue.Queue[int | GameReview | Exception] = queue.Queue()
        self._progress_dialog: tk.Toplevel | None = None
        self._progress_bar: ttk.Progressbar | None = None
        self._post_game_review_dialog: tk.Toplevel | None = None

        # Widgets (populated by _build_window).
        self._observe_btn: tk.Button | None = None
        self._apply_btn: tk.Button | None = None
        self._undo_btn: tk.Button | None = None
        self._board_widget: BoardWidget | None = None
        self._eval_panel: EvalPanel | None = None
        self._observe_panel: ObserveLivePanel | None = None
        self._suggestions_panel: SuggestionsPanel | None = None
        self._history_panel: MoveHistoryPanel | None = None
        self._move_entry: tk.Entry | None = None
        self._status_label: tk.Label | None = None
        self._status_icon: tk.Label | None = None

        # Review mode widgets.
        self._review_panel: ReviewPanel | None = None
        self._review_nav: ReviewNavigationPanel | None = None
        self._summary_panel: GameSummaryPanel | None = None
        self._progression_panel: ProgressionPanel | None = None

        # Play mode widget groups (for hiding/showing during review).
        self._play_mode_widgets: list[tk.Widget] = []
        self._review_mode_widgets: list[tk.Widget] = []

        # Frame refs for layout management.
        self._btn_frame: tk.Frame | None = None
        self._new_game_btn: tk.Button | None = None
        self._pgn_frame: tk.Frame | None = None
        self._progression_btn: tk.Button | None = None
        self._entry_frame: tk.Frame | None = None
        self._submit_btn: tk.Button | None = None
        self._progression_dialog: tk.Toplevel | None = None
        self._progression_limit_var: tk.IntVar | None = None

    def run(self) -> None:
        """Build the window and enter the tkinter mainloop (blocking)."""
        root = tk.Tk()
        root.title("ChessLab")
        root.configure(bg=_BG)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root = root

        startup_mode = self._config.startup_mode
        if startup_mode is None:
            startup_mode = self._show_startup_dialog(root)

        # Determine player color.
        player_color = self._config.player_color
        if player_color is None:
            if startup_mode in {
                StartupJourneyMode.PROGRESSION,
                StartupJourneyMode.REVIEW_RECENT,
                StartupJourneyMode.REVIEW_PASTE,
                StartupJourneyMode.REVIEW_FILE,
            }:
                player_color = PlayerColor.WHITE
                if self._engine_elo is None:
                    self._engine_elo = AnalysisConfig.DEFAULT_ENGINE_ELO
            else:
                player_color, self._engine_elo = self._show_color_dialog(
                    root,
                    default_engine_elo=self._engine_elo,
                )

        # Create and start coach.
        self._coach = self._create_coach(player_color, self._engine_elo)

        self._build_window(root, player_color)

        # Start polling.
        root.after(_POLL_INTERVAL_MS, self._poll_queue)

        # Update status for initial turn.
        self._update_turn_status()

        # Optional guided startup action.
        root.after(80, lambda: self._apply_startup_mode(startup_mode))

        logger.info(
            "ChessLab GUI started: player=%s elo=%s",
            player_color.value,
            self._engine_elo if self._engine_elo is not None else "max",
        )
        root.mainloop()

    def _create_coach(
        self,
        player_color: PlayerColor,
        engine_elo: int | None,
    ) -> GameCoach:
        """Create and start a GameCoach with the selected setup."""
        analysis_config = replace(self._config.analysis, engine_elo=engine_elo)
        coach = GameCoach(CoachConfig(
            player_color=player_color,
            analysis=analysis_config,
        ))
        coach.start()
        return coach

    # -- Startup launcher ----------------------------------------------------

    def _show_startup_dialog(self, parent: tk.Tk) -> StartupJourneyMode:
        """Show a startup launcher for the main product journeys."""
        result: list[StartupJourneyMode] = [StartupJourneyMode.MANUAL_COACH]

        dialog = tk.Toplevel(parent)
        dialog.title("ChessLab")
        dialog.configure(bg=_BG)
        dialog.transient(parent)
        dialog.grab_set()
        dialog.resizable(False, False)

        container = tk.Frame(dialog, bg=_BG)
        container.pack(fill="both", expand=True, padx=20, pady=18)

        tk.Label(
            container,
            text="Choose your journey",
            bg=_BG,
            fg=_FG,
            font=("sans-serif", 16, "bold"),
        ).pack(anchor="w")

        tk.Label(
            container,
            text=(
                "Start from one place and switch modes in the same app without "
                "changing the product boundary."
            ),
            bg=_BG,
            fg=_FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=430,
        ).pack(anchor="w", pady=(4, 12))

        scope_card = tk.Frame(container, bg=_CARD_BORDER)
        scope_card.pack(fill="x", pady=(0, 12))
        scope_inner = tk.Frame(scope_card, bg=_BG_SECONDARY)
        scope_inner.pack(fill="both", expand=True, padx=1, pady=1)

        tk.Label(
            scope_inner,
            text="Product Scope",
            bg=_BG_SECONDARY,
            fg=_ACCENT,
            font=("sans-serif", 9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(
            scope_inner,
            text=PRODUCT_SCOPE,
            bg=_BG_SECONDARY,
            fg=_FG,
            font=("sans-serif", 9, "bold"),
            justify="left",
            wraplength=410,
            anchor="w",
        ).pack(fill="x", padx=10)

        tk.Label(
            scope_inner,
            text=PRODUCT_BOUNDARY,
            bg=_BG_SECONDARY,
            fg=_FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=410,
            anchor="w",
        ).pack(fill="x", padx=10, pady=(4, 8))

        def choose(mode: StartupJourneyMode) -> None:
            result[0] = mode
            dialog.destroy()

        manual_journey = journey_usage("manual_coach")
        observe_journey = journey_usage("observe_live")
        review_journey = journey_usage("review")
        actions = [
            (
                manual_journey.label,
                manual_journey.launcher_summary,
                StartupJourneyMode.MANUAL_COACH,
            ),
            (
                observe_journey.label,
                observe_journey.launcher_summary,
                StartupJourneyMode.OBSERVE_LIVE,
            ),
            (
                "Progression",
                (
                    "Open your learning profile built from recent reviews: move "
                    "quality mix, centipawn-loss trend, recurring lesson themes, "
                    "and where your errors happen most often."
                ),
                StartupJourneyMode.PROGRESSION,
            ),
            (
                "Review Recent",
                review_journey.launcher_summary,
                StartupJourneyMode.REVIEW_RECENT,
            ),
            (
                "Paste PGN",
                review_journey.launcher_summary,
                StartupJourneyMode.REVIEW_PASTE,
            ),
        ]

        for label, description, mode in actions:
            card = tk.Frame(container, bg=_CARD_BORDER)
            card.pack(fill="x", pady=(0, 8))
            inner = tk.Frame(card, bg=_BG_SECONDARY)
            inner.pack(fill="both", expand=True, padx=1, pady=1)

            row = tk.Frame(inner, bg=_BG_SECONDARY)
            row.pack(fill="x", padx=10, pady=(8, 3))

            tk.Label(
                row,
                text=label,
                bg=_BG_SECONDARY,
                fg=_FG,
                font=("sans-serif", 11, "bold"),
                anchor="w",
            ).pack(side="left")

            _styled_button(
                row,
                text="Start",
                command=lambda m=mode: choose(m),
                bg=_BTN_ACCENT_BG,
                fg=_BTN_ACCENT_FG,
                hover_bg=_BTN_ACCENT_HOVER,
                active_bg="#1e88e5",
                font=("sans-serif", 9, "bold"),
                padx=10,
                pady=4,
            ).pack(side="right")

            tk.Label(
                inner,
                text=description,
                bg=_BG_SECONDARY,
                fg=_FG_DIM,
                font=("sans-serif", 9),
                justify="left",
                wraplength=390,
                anchor="w",
            ).pack(fill="x", padx=10, pady=(0, 8))

        footer = tk.Frame(container, bg=_BG)
        footer.pack(fill="x", pady=(2, 0))
        _styled_button(
            footer,
            text="Default to Manual Coach",
            command=dialog.destroy,
            font=("sans-serif", 9),
            padx=10,
            pady=5,
        ).pack(side="right")

        dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

        parent.wait_window(dialog)
        return result[0]

    def _apply_startup_mode(self, mode: StartupJourneyMode) -> None:
        """Execute optional startup actions chosen from the launcher."""
        if self._root is None:
            return

        if mode is StartupJourneyMode.OBSERVE_LIVE:
            config = self._show_observe_dialog(self._root)
            if config is not None:
                self._start_observation(config)
            return

        if mode is StartupJourneyMode.PROGRESSION:
            self._set_status("Progression — review your recent learning trends.")
            self._show_progression_dialog()
            return

        if mode is StartupJourneyMode.REVIEW_RECENT:
            self._set_status("Review — select a saved game for post-game analysis.")
            self._on_recent_games()
            return

        if mode is StartupJourneyMode.REVIEW_PASTE:
            self._set_status("Review — paste a PGN to launch post-game analysis.")
            self._on_paste_pgn()
            return

        if mode is StartupJourneyMode.REVIEW_FILE:
            if self._config.startup_review_file is None:
                self._set_status("No review file was provided.", error=True)
                return
            self._set_status(
                f"Review — loading post-game analysis from "
                f"{self._config.startup_review_file.name}"
            )
            self._review_from_file(self._config.startup_review_file)

    # -- Color selection dialog ----------------------------------------------

    def _show_color_dialog(
        self,
        parent: tk.Tk,
        *,
        default_engine_elo: int | None = None,
    ) -> tuple[PlayerColor, int]:
        """Show a dialog to choose player color and Stockfish Elo."""
        initial_elo = (
            default_engine_elo
            if default_engine_elo is not None
            else AnalysisConfig.DEFAULT_ENGINE_ELO
        )
        initial_elo = AnalysisConfig.snap_engine_elo(initial_elo)
        result: list[tuple[PlayerColor, int]] = [(PlayerColor.WHITE, initial_elo)]
        elo_var = tk.IntVar(value=initial_elo)

        dialog = tk.Toplevel(parent)
        dialog.title("ChessLab")
        dialog.configure(bg=_BG)
        dialog.transient(parent)
        dialog.grab_set()
        dialog.resizable(False, False)

        # Main container with padding.
        container = tk.Frame(dialog, bg=_BG)
        container.pack(fill="both", expand=True, padx=24, pady=20)

        # Title.
        tk.Label(
            container, text="New Game", bg=_BG, fg=_FG,
            font=("sans-serif", 18, "bold"),
        ).pack(pady=(0, 16))

        # Elo section — card style.
        elo_card = tk.Frame(container, bg=_CARD_BORDER)
        elo_card.pack(fill="x", pady=(0, 16))
        elo_inner = tk.Frame(elo_card, bg=_BG_SECONDARY)
        elo_inner.pack(fill="both", expand=True, padx=1, pady=1)

        elo_header = tk.Frame(elo_inner, bg=_BG_SECONDARY)
        elo_header.pack(fill="x", padx=12, pady=(10, 0))

        tk.Label(
            elo_header, text="Engine Strength", bg=_BG_SECONDARY, fg=_FG,
            font=("sans-serif", 11, "bold"), anchor="w",
        ).pack(side="left")

        elo_value_frame = tk.Frame(elo_header, bg=_BG_SECONDARY)
        elo_value_frame.pack(side="right")

        elo_value_label = tk.Label(
            elo_value_frame, text=str(initial_elo), bg=_BG_SECONDARY, fg=_ACCENT,
            font=("monospace", 14, "bold"), anchor="e",
        )
        elo_value_label.pack(anchor="e")

        level_value_label = tk.Label(
            elo_value_frame, text="", bg=_BG_SECONDARY, fg=_FG_DIM,
            font=("sans-serif", 8), anchor="e",
        )
        level_value_label.pack(anchor="e")

        def _on_elo_change(*_args: object) -> None:
            snapped_elo = AnalysisConfig.snap_engine_elo(elo_var.get())
            if snapped_elo != elo_var.get():
                elo_var.set(snapped_elo)
                return

            level = AnalysisConfig.engine_level_from_elo(snapped_elo)
            max_level = AnalysisConfig.selectable_engine_levels()
            elo_value_label.config(text=str(snapped_elo))
            level_value_label.config(text=f"Level {level}/{max_level}")

        elo_var.trace_add("write", _on_elo_change)
        _on_elo_change()

        scale = tk.Scale(
            elo_inner,
            from_=AnalysisConfig.MIN_SELECTABLE_ENGINE_ELO,
            to=AnalysisConfig.MAX_SELECTABLE_ENGINE_ELO,
            orient="horizontal",
            resolution=AnalysisConfig.ENGINE_ELO_STEP,
            variable=elo_var,
            bg=_BG_SECONDARY,
            fg=_FG_DIM,
            troughcolor="#2a2a3c",
            highlightthickness=0,
            length=300,
            showvalue=False,
            sliderrelief="flat",
            bd=0,
        )
        scale.pack(fill="x", padx=12, pady=(4, 4))

        # Elo range labels.
        range_frame = tk.Frame(elo_inner, bg=_BG_SECONDARY)
        range_frame.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(
            range_frame, text=str(AnalysisConfig.MIN_SELECTABLE_ENGINE_ELO),
            bg=_BG_SECONDARY, fg=_FG_MUTED, font=("monospace", 8),
        ).pack(side="left")
        tk.Label(
            range_frame, text=str(AnalysisConfig.MAX_SELECTABLE_ENGINE_ELO),
            bg=_BG_SECONDARY, fg=_FG_MUTED, font=("monospace", 8),
        ).pack(side="right")

        # Color choice section.
        tk.Label(
            container, text="Play as", bg=_BG, fg=_FG_DIM,
            font=("sans-serif", 10), anchor="w",
        ).pack(fill="x", pady=(0, 8))

        btn_frame = tk.Frame(container, bg=_BG)
        btn_frame.pack(fill="x")

        def choose(color: PlayerColor) -> None:
            result[0] = (color, int(elo_var.get()))
            dialog.destroy()

        # White button.
        white_btn = _styled_button(
            btn_frame,
            text="\u2654  White",
            command=lambda: choose(PlayerColor.WHITE),
            bg=_BTN_WHITE_BG,
            fg=_BTN_WHITE_FG,
            hover_bg=_BTN_WHITE_HOVER,
            active_bg="#f5ede1",
            font=("sans-serif", 13, "bold"),
            pady=12,
        )
        white_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        # Black button.
        black_btn = _styled_button(
            btn_frame,
            text="\u265a  Black",
            command=lambda: choose(PlayerColor.BLACK),
            bg=_BTN_BLACK_BG,
            fg=_BTN_BLACK_FG,
            hover_bg=_BTN_BLACK_HOVER,
            active_bg="#4a4a5c",
            font=("sans-serif", 13, "bold"),
            pady=12,
        )
        black_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        # Center dialog.
        dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

        parent.wait_window(dialog)
        return result[0]

    # -- Window construction -------------------------------------------------

    def _build_window(self, root: tk.Tk, player_color: PlayerColor) -> None:
        """Build the main window layout."""
        flipped = player_color == PlayerColor.BLACK

        # Main content frame.
        main_frame = tk.Frame(root, bg=_BG)
        main_frame.pack(fill="both", expand=True, padx=6, pady=(6, 0))

        # Left: board with subtle shadow frame.
        board_outer = tk.Frame(main_frame, bg="#0a0a12", bd=0)
        board_outer.pack(side="left", padx=(0, 8))

        board_frame = tk.Frame(board_outer, bg=_BG)
        board_frame.pack(padx=2, pady=2)

        self._board_widget = BoardWidget(
            board_frame,
            square_size=self._config.square_size,
            flipped=flipped,
            on_move=self._on_board_move,
        )
        self._board_widget.pack()

        # Right: panels.
        panel_frame = tk.Frame(main_frame, bg=_BG, width=290)
        panel_frame.pack(side="left", fill="both", expand=True)
        panel_frame.pack_propagate(False)

        self._eval_panel = EvalPanel(panel_frame, bar_height=160)
        self._eval_panel.pack(fill="x", pady=(0, 6))

        self._observe_panel = ObserveLivePanel(panel_frame)

        self._suggestions_panel = SuggestionsPanel(panel_frame)
        self._suggestions_panel.pack(fill="x", pady=(0, 6))

        self._history_panel = MoveHistoryPanel(panel_frame)
        self._history_panel.pack(fill="both", expand=True, pady=(0, 6))

        # Action buttons row.
        self._btn_frame = tk.Frame(panel_frame, bg=_BG)
        self._btn_frame.pack(fill="x", pady=(0, 4))

        self._apply_btn = _styled_button(
            self._btn_frame,
            text="\u25b6  Apply suggestion",
            command=self._on_apply_suggestion,
            bg=_BTN_ACCENT_BG,
            fg=_BTN_ACCENT_FG,
            hover_bg=_BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 10, "bold"),
        )
        self._apply_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self._undo_btn = _styled_button(
            self._btn_frame,
            text="\u21a9  Undo",
            command=self._on_undo,
        )
        self._undo_btn.pack(side="left", fill="x", expand=True, padx=(3, 3))

        self._observe_btn = _styled_button(
            self._btn_frame,
            text="\u25ce  Observe Live",
            command=self._on_toggle_observe,
        )
        self._observe_btn.pack(side="left", fill="x", expand=True, padx=(3, 0))

        # New game button.
        self._new_game_btn = _styled_button(
            panel_frame,
            text="New Game",
            command=self._on_new_game,
            font=("sans-serif", 10),
        )
        self._new_game_btn.pack(fill="x", pady=(0, 4))

        self._progression_btn = _styled_button(
            panel_frame,
            text="Progression",
            command=self._show_progression_dialog,
            font=("sans-serif", 10, "bold"),
        )
        self._progression_btn.pack(fill="x", pady=(0, 4))

        # PGN buttons row.
        self._pgn_frame = tk.Frame(panel_frame, bg=_BG)
        self._pgn_frame.pack(fill="x", pady=(0, 4))

        export_btn = _styled_button(
            self._pgn_frame,
            text="Export PGN",
            command=self._on_export_pgn,
            font=("sans-serif", 9),
        )
        export_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        load_btn = _styled_button(
            self._pgn_frame,
            text="Load PGN",
            command=self._on_load_pgn,
            font=("sans-serif", 9),
        )
        load_btn.pack(side="left", fill="x", expand=True, padx=(3, 3))

        paste_btn = _styled_button(
            self._pgn_frame,
            text="Paste PGN",
            command=self._on_paste_pgn,
            font=("sans-serif", 9),
        )
        paste_btn.pack(side="left", fill="x", expand=True, padx=(3, 3))

        recent_btn = _styled_button(
            self._pgn_frame,
            text="Recent",
            command=self._on_recent_games,
            font=("sans-serif", 9),
        )
        recent_btn.pack(side="left", fill="x", expand=True, padx=(3, 3))

        review_btn = _styled_button(
            self._pgn_frame,
            text="Review",
            command=self._on_review_current,
            font=("sans-serif", 9),
        )
        review_btn.pack(side="left", fill="x", expand=True, padx=(3, 0))

        # Review mode panels (created but not packed yet).
        self._review_panel = ReviewPanel(panel_frame)
        self._review_nav = ReviewNavigationPanel(
            panel_frame,
            on_first=self._review_first,
            on_prev=self._review_prev,
            on_next=self._review_next,
            on_last=self._review_last,
            on_prev_error=self._review_prev_error,
            on_next_error=self._review_next_error,
            on_exit_review=self._exit_review_mode,
            on_back_to_launcher=self._return_to_launcher,
        )
        self._summary_panel = GameSummaryPanel(panel_frame)

        # Track play-mode widgets for hide/show.
        self._play_mode_widgets = [
            self._eval_panel, self._observe_panel, self._suggestions_panel,
            self._btn_frame,
            self._new_game_btn, self._pgn_frame,
        ]
        self._review_mode_widgets = [
            self._review_panel, self._review_nav, self._summary_panel,
        ]

        # Bottom status bar.
        status_bar = tk.Frame(root, bg=_STATUS_BG, height=40)
        status_bar.pack(fill="x", padx=0, pady=0)
        status_bar.pack_propagate(False)

        inner_status = tk.Frame(status_bar, bg=_STATUS_BG)
        inner_status.pack(fill="both", expand=True, padx=8)

        # Move entry area.
        self._entry_frame = tk.Frame(inner_status, bg=_STATUS_BG)
        self._entry_frame.pack(side="left", fill="y", pady=4)
        entry_frame = self._entry_frame

        tk.Label(
            entry_frame, text="Move", bg=_STATUS_BG, fg=_FG_DIM,
            font=("sans-serif", 9),
        ).pack(side="left", padx=(0, 6))

        entry_border = tk.Frame(entry_frame, bg=_ENTRY_BORDER)
        entry_border.pack(side="left")

        self._move_entry = tk.Entry(
            entry_border,
            font=("monospace", 12),
            width=8,
            bg=_ENTRY_BG,
            fg=_ENTRY_FG,
            insertbackground=_ACCENT,
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        self._move_entry.pack(padx=1, pady=1, ipady=2)
        self._move_entry.bind("<Return>", self._on_entry_submit)

        # Focus effects.
        self._move_entry.bind(
            "<FocusIn>",
            lambda e: entry_border.config(bg=_ENTRY_FOCUS_BORDER),
        )
        self._move_entry.bind(
            "<FocusOut>",
            lambda e: entry_border.config(bg=_ENTRY_BORDER),
        )

        self._submit_btn = _styled_button(
            entry_frame,
            text="Submit",
            command=self._on_entry_submit,
            bg=_BTN_ACCENT_BG,
            fg=_BTN_ACCENT_FG,
            hover_bg=_BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 9, "bold"),
            padx=10,
            pady=3,
        )
        self._submit_btn.pack(side="left", padx=(6, 0))

        # Keyboard bindings for review mode.
        root.bind("<Left>", self._on_key_left)
        root.bind("<Right>", self._on_key_right)
        root.bind("<Home>", self._on_key_home)
        root.bind("<End>", self._on_key_end)

        # Status text.
        self._status_icon = tk.Label(
            inner_status, text="\u25cf", bg=_STATUS_BG, fg=_SUCCESS,
            font=("sans-serif", 10),
        )
        self._status_icon.pack(side="left", padx=(16, 4))

        self._status_label = tk.Label(
            inner_status, text="", bg=_STATUS_BG, fg=_FG_DIM,
            font=("sans-serif", 10), anchor="w",
        )
        self._status_label.pack(side="left", fill="x", expand=True)

    # -- Live observation ----------------------------------------------------

    def _show_observe_dialog(
        self,
        parent: tk.Tk,
    ) -> LiveObserverConfig | None:
        """Show a setup-driven dialog for live observation."""
        result: list[LiveObserverConfig | None] = [None]
        assistant: LiveSetupAssistant | None
        assistant_error: str | None = None
        try:
            assistant = self._get_live_setup_assistant()
            setup_options = assistant.list_options()
        except SupportCatalogError as exc:
            assistant = None
            setup_options = ()
            assistant_error = f"Setup catalog unavailable: {exc}"

        dialog = tk.Toplevel(parent)
        dialog.title("Observe Live")
        dialog.configure(bg=_BG)
        dialog.transient(parent)
        dialog.grab_set()
        dialog.resizable(False, False)

        container = tk.Frame(dialog, bg=_BG)
        container.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(
            container, text="Observe Live", bg=_BG, fg=_FG,
            font=("sans-serif", 18, "bold"),
        ).pack(pady=(0, 6))

        tk.Label(
            container,
            text=OBSERVE_LIVE_DIALOG_INTRO,
            bg=_BG,
            fg=_FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=500,
        ).pack(fill="x", pady=(0, 14))

        guardrail_card = tk.Frame(container, bg=_CARD_BORDER)
        guardrail_card.pack(fill="x", pady=(0, 12))
        guardrail_inner = tk.Frame(guardrail_card, bg=_BG_SECONDARY)
        guardrail_inner.pack(fill="both", expand=True, padx=1, pady=1)

        tk.Label(
            guardrail_inner,
            text="Usage Guardrail",
            bg=_BG_SECONDARY,
            fg=_ACCENT,
            font=("sans-serif", 9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))

        tk.Label(
            guardrail_inner,
            text=OBSERVE_LIVE_GUARDRAIL_TITLE,
            bg=_BG_SECONDARY,
            fg=_FG,
            font=("sans-serif", 11, "bold"),
            justify="left",
            wraplength=480,
            anchor="w",
        ).pack(fill="x", padx=12)

        tk.Label(
            guardrail_inner,
            text=OBSERVE_LIVE_GUARDRAIL_BODY,
            bg=_BG_SECONDARY,
            fg=_FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=480,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(6, 8))

        acknowledgement_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            guardrail_inner,
            text=OBSERVE_LIVE_ACKNOWLEDGEMENT,
            variable=acknowledgement_var,
            bg=_BG_SECONDARY,
            fg=_FG,
            selectcolor=_BG_SECONDARY,
            activebackground=_BG_SECONDARY,
            activeforeground=_FG,
            highlightthickness=0,
            justify="left",
            anchor="w",
            wraplength=470,
            font=("sans-serif", 9, "bold"),
        ).pack(fill="x", padx=12, pady=(0, 10))

        # Monitor selector.
        monitor_card = tk.Frame(container, bg=_CARD_BORDER)
        monitor_card.pack(fill="x", pady=(0, 12))
        monitor_inner = tk.Frame(monitor_card, bg=_BG_SECONDARY)
        monitor_inner.pack(fill="both", expand=True, padx=1, pady=1)

        tk.Label(
            monitor_inner, text="Monitor", bg=_BG_SECONDARY, fg=_FG,
            font=("sans-serif", 11, "bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))

        monitor_error: str | None = None
        try:
            monitors = ScreenCapture.list_monitors()
        except CapturePermissionError:
            monitors = []
            monitor_error = (
                "Screen capture permission is blocked. "
                "Allow screen recording for Python/Terminal in your OS settings."
            )
        except DisplayUnavailableError:
            monitors = []
            monitor_error = (
                "No graphical display is available. "
                "Start a desktop session and retry."
            )
        except Exception as exc:
            monitors = []
            monitor_error = f"Monitor discovery failed: {exc}"
            logger.warning("Cannot list monitors: %s", exc)

        monitor_labels = [
            f"Monitor {m.index}: {m.width}x{m.height}" for m in monitors
        ]
        if not monitor_labels:
            monitor_labels = ["No monitors found"]
        default_monitor_label = monitor_labels[0]
        if self._last_observer_config is not None:
            for monitor, label in zip(monitors, monitor_labels):
                if monitor.index == self._last_observer_config.monitor_index:
                    default_monitor_label = label
                    break

        monitor_var = tk.StringVar(value=default_monitor_label)
        monitor_menu = tk.OptionMenu(monitor_inner, monitor_var, *monitor_labels)
        monitor_menu.config(
            bg=_BG_SECONDARY, fg=_FG, font=("sans-serif", 10),
            highlightthickness=0, relief="flat", activebackground=_BTN_HOVER,
            activeforeground=_FG,
        )
        monitor_menu["menu"].config(
            bg=_BG_SECONDARY, fg=_FG, activebackground=_ACCENT,
            activeforeground="#ffffff",
        )
        monitor_menu.pack(fill="x", padx=12, pady=(0, 8))

        if monitor_error is not None:
            tk.Label(
                monitor_inner,
                text=monitor_error,
                bg=_BG_SECONDARY,
                fg=_ERROR,
                font=("sans-serif", 9),
                justify="left",
                wraplength=360,
            ).pack(fill="x", padx=12, pady=(0, 8))

        # Declared setup selector.
        setup_card = tk.Frame(container, bg=_CARD_BORDER)
        setup_card.pack(fill="x", pady=(0, 12))
        setup_inner = tk.Frame(setup_card, bg=_BG_SECONDARY)
        setup_inner.pack(fill="both", expand=True, padx=1, pady=1)

        tk.Label(
            setup_inner, text="Declared Setup", bg=_BG_SECONDARY, fg=_FG,
            font=("sans-serif", 11, "bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))

        option_by_label = {option.label: option for option in setup_options}
        setup_labels = list(option_by_label) or ["No declared setups available"]
        default_setup_label = setup_labels[0]
        if self._last_observer_config is not None and self._last_observer_config.setup_id:
            for option in setup_options:
                if option.profile_id == self._last_observer_config.setup_id:
                    default_setup_label = option.label
                    break

        setup_var = tk.StringVar(value=default_setup_label)
        setup_menu = tk.OptionMenu(setup_inner, setup_var, *setup_labels)
        setup_menu.config(
            bg=_BG_SECONDARY, fg=_FG, font=("sans-serif", 10),
            highlightthickness=0, relief="flat", activebackground=_BTN_HOVER,
            activeforeground=_FG,
        )
        setup_menu["menu"].config(
            bg=_BG_SECONDARY, fg=_FG, activebackground=_ACCENT,
            activeforeground="#ffffff",
        )
        setup_menu.pack(fill="x", padx=12, pady=(0, 8))

        if assistant_error is not None:
            tk.Label(
                setup_inner,
                text=assistant_error,
                bg=_BG_SECONDARY,
                fg=_ERROR,
                font=("sans-serif", 9),
                justify="left",
                wraplength=480,
            ).pack(fill="x", padx=12, pady=(0, 8))

        # Setup details.
        details_card = tk.Frame(container, bg=_CARD_BORDER)
        details_card.pack(fill="x", pady=(0, 12))
        details_inner = tk.Frame(details_card, bg=_BG_SECONDARY)
        details_inner.pack(fill="both", expand=True, padx=1, pady=1)

        support_badge = tk.Label(
            details_inner,
            text="SETUP",
            bg=_BTN_BG,
            fg=_FG,
            font=("sans-serif", 9, "bold"),
            padx=8,
            pady=3,
        )
        support_badge.pack(anchor="w", padx=12, pady=(10, 6))

        summary_label = tk.Label(
            details_inner,
            text="",
            bg=_BG_SECONDARY,
            fg=_FG,
            font=("sans-serif", 11, "bold"),
            justify="left",
            wraplength=480,
            anchor="w",
        )
        summary_label.pack(fill="x", padx=12)

        support_copy_label = tk.Label(
            details_inner,
            text="",
            bg=_BG_SECONDARY,
            fg=_FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=480,
            anchor="w",
        )
        support_copy_label.pack(fill="x", padx=12, pady=(4, 0))

        expected_label = tk.Label(
            details_inner,
            text="",
            bg=_BG_SECONDARY,
            fg=_FG,
            font=("sans-serif", 9),
            justify="left",
            wraplength=480,
            anchor="w",
        )
        expected_label.pack(fill="x", padx=12, pady=(8, 0))

        notes_label = tk.Label(
            details_inner,
            text="",
            bg=_BG_SECONDARY,
            fg=_FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=480,
            anchor="w",
        )
        notes_label.pack(fill="x", padx=12, pady=(6, 10))

        # Preflight results.
        preflight_card = tk.Frame(container, bg=_CARD_BORDER)
        preflight_card.pack(fill="x", pady=(0, 12))
        preflight_inner = tk.Frame(preflight_card, bg=_BG_SECONDARY)
        preflight_inner.pack(fill="both", expand=True, padx=1, pady=1)

        verdict_badge = tk.Label(
            preflight_inner,
            text="PREFLIGHT",
            bg=_BTN_BG,
            fg=_FG,
            font=("sans-serif", 9, "bold"),
            padx=8,
            pady=3,
        )
        verdict_badge.pack(anchor="w", padx=12, pady=(10, 6))

        verdict_title = tk.Label(
            preflight_inner,
            text="",
            bg=_BG_SECONDARY,
            fg=_FG,
            font=("sans-serif", 11, "bold"),
            anchor="w",
            justify="left",
            wraplength=480,
        )
        verdict_title.pack(fill="x", padx=12)

        verdict_body = tk.Label(
            preflight_inner,
            text="",
            bg=_BG_SECONDARY,
            fg=_FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=480,
            anchor="w",
        )
        verdict_body.pack(fill="x", padx=12, pady=(6, 10))

        latest_report: list[CalibrationReport | None] = [None]
        validated_signature: list[tuple[int, str] | None] = [None]

        def selected_monitor_index() -> int | None:
            if not monitors:
                return None
            try:
                selected = monitor_var.get()
                return monitors[monitor_labels.index(selected)].index
            except (ValueError, IndexError):
                return None

        def selected_profile() -> SetupSupportProfile | None:
            if assistant is None:
                return None
            option = option_by_label.get(setup_var.get())
            if option is None:
                return None
            return assistant.get_profile(option.profile_id)

        def set_verdict(
            badge_text: str,
            title: str,
            body: str,
            *,
            badge_bg: str,
            badge_fg: str = "#ffffff",
            title_fg: str = _FG,
            body_fg: str = _FG_DIM,
        ) -> None:
            verdict_badge.config(text=badge_text, bg=badge_bg, fg=badge_fg)
            verdict_title.config(text=title, fg=title_fg)
            verdict_body.config(text=body, fg=body_fg)

        def refresh_details() -> None:
            profile = selected_profile()
            if profile is None:
                support_badge.config(text="SETUP", bg=_BTN_BG, fg=_FG)
                summary_label.config(text="No declared setup available.")
                support_copy_label.config(
                    text=(
                        "Observe Live uses a narrow live perimeter built from "
                        "declared real-template setups."
                    )
                )
                expected_label.config(text="")
                notes_label.config(text=assistant_error or "")
                return

            badge_bg, badge_fg = self._support_badge_palette(
                profile.support_level.value
            )
            support_badge.config(
                text=profile.support_level.value.upper(),
                bg=badge_bg,
                fg=badge_fg,
            )
            summary_label.config(text=self._format_live_setup_summary(profile))
            support_copy_label.config(
                text=self._support_level_copy(profile.support_level.value)
            )
            expected_label.config(
                text=(
                    f"Expected theme: {profile.expected_theme_id}  |  "
                    f"Template: {profile.template_theme}  |  "
                    f"Perspective: {profile.perspective.title()}  |  "
                    f"Template quality: {profile.template_quality}"
                )
            )
            notes_label.config(text=profile.notes)

        def refresh_preflight_state() -> None:
            latest_report[0] = None
            validated_signature[0] = None
            can_preflight = (
                selected_monitor_index() is not None and selected_profile() is not None
            )
            can_start = can_preflight and acknowledgement_var.get()
            preflight_btn.config(state="normal" if can_preflight else "disabled")
            start_btn.config(state="normal" if can_start else "disabled")

            if assistant_error is not None:
                set_verdict(
                    "BLOCKED",
                    "Setup catalog unavailable",
                    assistant_error,
                    badge_bg=_ERROR,
                )
                return

            if selected_profile() is None:
                set_verdict(
                    "BLOCKED",
                    "No declared live setup available",
                    "Add a supported setup profile before starting Observe Live.",
                    badge_bg=_ERROR,
                )
                return

            if monitor_error is not None or selected_monitor_index() is None:
                set_verdict(
                    "BLOCKED",
                    "Monitor selection required",
                    monitor_error or "Pick a valid monitor before running preflight.",
                    badge_bg=_ERROR,
                )
                return

            if not acknowledgement_var.get():
                set_verdict(
                    "ACK",
                    "Confirm the Observe Live guardrail",
                    (
                        "ChessLab only starts Observe Live after you confirm "
                        "the session is passive observation in a context that "
                        "allows outside tools."
                    ),
                    badge_bg="#7a5a00",
                )
                return

            set_verdict(
                "READY",
                "Preflight required before launch",
                (
                    "ChessLab will capture a short burst of live frames and return "
                    "a PASS/FAIL verdict with blockers and corrective actions."
                ),
                badge_bg=_ACCENT_DARK,
            )

        def render_report(profile: SetupSupportProfile, report: CalibrationReport) -> None:
            if report.passed:
                set_verdict(
                    "PASS",
                    "Setup verified for live observation",
                    self._format_preflight_report(profile, report),
                    badge_bg=_SUCCESS,
                )
            else:
                set_verdict(
                    "FAIL",
                    "Fix blockers before starting live observation",
                    self._format_preflight_report(profile, report),
                    badge_bg=_ERROR,
                )

        def run_preflight() -> CalibrationReport | None:
            profile = selected_profile()
            monitor_index = selected_monitor_index()
            if assistant is None or profile is None or monitor_index is None:
                refresh_preflight_state()
                return None

            preflight_btn.config(state="disabled")
            start_btn.config(state="disabled")
            cancel_btn.config(state="disabled")
            set_verdict(
                "RUNNING",
                "Running live preflight",
                (
                    f"Capturing a few frames from monitor {monitor_index} and checking "
                    f"setup '{profile.id}' against the declared support thresholds."
                ),
                badge_bg=_ACCENT,
            )
            dialog.update_idletasks()

            try:
                report = assistant.run_preflight(monitor_index, profile.id)
            except Exception as exc:
                latest_report[0] = None
                validated_signature[0] = None
                set_verdict(
                    "ERROR",
                    "Preflight could not complete",
                    self._format_preflight_error(exc),
                    badge_bg=_ERROR,
                )
                return None
            finally:
                cancel_btn.config(state="normal")
                can_preflight = (
                    selected_monitor_index() is not None
                    and selected_profile() is not None
                )
                can_start = can_preflight and acknowledgement_var.get()
                preflight_btn.config(state="normal" if can_preflight else "disabled")
                start_btn.config(state="normal" if can_start else "disabled")

            latest_report[0] = report
            if report.passed:
                validated_signature[0] = (monitor_index, profile.id)
            else:
                validated_signature[0] = None
            render_report(profile, report)
            return report

        # OK / Cancel buttons.
        ok_frame = tk.Frame(container, bg=_BG)
        ok_frame.pack(fill="x", pady=(4, 0))

        def on_start() -> None:
            profile = selected_profile()
            monitor_index = selected_monitor_index()
            if not acknowledgement_var.get():
                refresh_preflight_state()
                return
            if assistant is None or profile is None or monitor_index is None:
                refresh_preflight_state()
                return

            signature = (monitor_index, profile.id)
            report = latest_report[0]
            if validated_signature[0] != signature or report is None or not report.passed:
                report = run_preflight()
            if report is None or not report.passed:
                return

            result[0] = assistant.build_observer_config(monitor_index, profile)
            dialog.destroy()

        preflight_btn = _styled_button(
            ok_frame,
            text="Run Preflight",
            command=run_preflight,
            font=("sans-serif", 11),
            pady=8,
        )
        preflight_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        start_btn = _styled_button(
            ok_frame,
            text="Preflight & Start",
            command=on_start,
            bg=_BTN_ACCENT_BG,
            fg=_BTN_ACCENT_FG,
            hover_bg=_BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 11, "bold"),
            pady=8,
        )
        start_btn.pack(side="left", fill="x", expand=True, padx=(4, 4))

        cancel_btn = _styled_button(
            ok_frame,
            text="Cancel",
            command=dialog.destroy,
            font=("sans-serif", 11),
            pady=8,
        )
        cancel_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        refresh_details()
        refresh_preflight_state()
        monitor_var.trace_add(
            "write",
            lambda *_args: (refresh_details(), refresh_preflight_state()),
        )
        setup_var.trace_add(
            "write",
            lambda *_args: (refresh_details(), refresh_preflight_state()),
        )
        acknowledgement_var.trace_add(
            "write",
            lambda *_args: refresh_preflight_state(),
        )

        # Center dialog.
        dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

        parent.wait_window(dialog)
        return result[0]

    def _get_live_setup_assistant(self) -> LiveSetupAssistant:
        """Lazily load the Observe Live setup assistant."""
        if self._live_setup_assistant is None:
            self._live_setup_assistant = LiveSetupAssistant()
        return self._live_setup_assistant

    @staticmethod
    def _support_badge_palette(level: str) -> tuple[str, str]:
        """Return badge colours for the given support level."""
        normalized = level.strip().lower()
        if normalized == "official":
            return "#1e6f43", "#ffffff"
        if normalized == "candidate":
            return "#7a5a00", "#ffffff"
        if normalized == "experimental":
            return "#6d2f8b", "#ffffff"
        return _BTN_BG, _FG

    @staticmethod
    def _support_level_copy(level: str) -> str:
        """Human-facing description of the setup maturity level."""
        normalized = level.strip().lower()
        if normalized == "official":
            return (
                "Official reference setup. Recommended for passive live "
                "observation in permitted contexts."
            )
        if normalized == "candidate":
            return (
                "Evidence-backed candidate setup. Real captures exist, but this "
                "path is not official yet. Validate with preflight before every "
                "permitted live session."
            )
        if normalized == "experimental":
            return (
                "Experimental setup. Not recommended for Observe Live routine "
                "use and likely to fail calibration."
            )
        return "Support level unknown. Treat this setup as unverified."

    @staticmethod
    def _format_live_setup_summary(profile: SetupSupportProfile) -> str:
        """Summarize the declared setup in one readable line."""
        return (
            f"{profile.platform} / {profile.site_theme} / "
            f"{profile.perspective.title()} perspective"
        )

    @staticmethod
    def _format_preflight_report(
        profile: SetupSupportProfile,
        report: CalibrationReport,
    ) -> str:
        """Render a readable PASS/FAIL report for the Observe Live dialog."""
        metrics = report.metrics
        lines = [
            (
                "Metrics: "
                f"detection {metrics.detection_rate:.0%}  |  "
                f"theme {metrics.theme_match_rate:.0%}  |  "
                f"ready {metrics.ready_rate:.0%}  |  "
                f"uncertain {metrics.uncertain_rate:.0%}  |  "
                f"p95 {metrics.p95_total_latency_ms:.0f} ms"
            )
        ]

        if report.blockers:
            lines.append("Blockers:")
            lines.extend(f"- {line}" for line in report.blockers)
        elif report.passed:
            lines.append("No blockers detected for this declared setup.")

        if report.corrective_actions:
            lines.append("Corrective actions:")
            lines.extend(f"- {line}" for line in report.corrective_actions)

        if report.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {line}" for line in report.warnings)

        if not report.corrective_actions and profile.support_level.value == "official":
            lines.append("This official setup is ready to launch without reconfiguration.")

        if report.passed:
            lines.append(
                "PASS confirms technical compatibility only. Confirm your "
                "platform or event allows passive observation and outside tools."
            )

        return "\n".join(lines)

    @staticmethod
    def _format_preflight_error(exc: Exception) -> str:
        """Map preflight failures to actionable UI text."""
        if isinstance(exc, CapturePermissionError):
            return (
                "Preflight blocked: screen-capture permission denied. Allow screen "
                "recording in OS settings, then retry."
            )
        if isinstance(exc, DisplayUnavailableError):
            return (
                "Preflight unavailable: no display detected. Start a graphical "
                "desktop session and retry."
            )
        if isinstance(exc, MonitorSelectionError):
            return (
                f"Preflight unavailable: {exc} Select a valid monitor and retry."
            )
        if isinstance(exc, ScreenCaptureError):
            return f"Preflight failed: {exc}"
        return f"Preflight failed: {exc}"

    def _on_toggle_observe(self) -> None:
        """Toggle live observation mode on/off."""
        if self._observing:
            self._stop_observation()
            return

        assert self._root is not None
        config = self._show_observe_dialog(self._root)
        if config is None:
            return
        self._start_observation(config)

    def _start_observation(self, config: LiveObserverConfig) -> bool:
        """Start live observation with explicit lifecycle status handling."""
        if config.setup_id:
            self._set_status(
                f"Starting Observe Live for {config.setup_id}...",
                waiting=True,
            )
        else:
            self._set_status("Starting Observe Live...", waiting=True)

        # Clear the observation queue before starting.
        while not self._observation_queue.empty():
            try:
                self._observation_queue.get_nowait()
            except queue.Empty:
                break

        observer = LiveObserver(config, self._observation_queue)
        try:
            observer.start()
        except Exception as exc:
            self._set_status(self._format_observe_start_error(exc), error=True)
            logger.error("Failed to start live observer: %s", exc)
            return False

        self._observer = observer
        self._observing = True
        self._last_observer_config = config

        self._clear_coaching_outputs()
        self._set_observe_mode_ui(True)
        self._refresh_observe_panel()

        if config.setup_id:
            support = (
                f" ({config.support_level})"
                if config.support_level is not None
                else ""
            )
            self._set_status(
                f"Preflight PASS — Observe Live active for {config.setup_id}"
                f"{support}. Passive sync only."
            )
        else:
            self._set_status("Observe Live active — passive sync only.")
        logger.info("Live observation started.")
        return True

    def _stop_observation(self) -> None:
        """Stop live observation and restore manual input."""
        if self._observer is not None:
            self._observer.stop()
            self._observer = None
        self._observing = False

        self._set_observe_mode_ui(False)
        self._update_turn_status()
        logger.info("Live observation stopped.")

    def _handle_observation_event(self, event: ObservationEvent) -> None:
        """Process an observation event from the live observer."""
        assert self._coach is not None

        if event.type is ObservationEventType.TRACKING_SYNC:
            if event.fen is not None:
                self._sync_coach_to_fen(event.fen)
            self._set_status(event.message or "Live tracking synchronized.")

        elif event.type is ObservationEventType.MOVE_DETECTED:
            # If game tracking provided a move, use it directly.
            if event.tracking is not None and event.tracking.move_san is not None:
                self._submit_san(event.tracking.move_san, passive=True)
                if self._coach.current_fen != event.tracking.fen:
                    logger.warning(
                        "Live tracking drift detected, resynchronizing coach board."
                    )
                    self._sync_coach_to_fen(event.tracking.fen)
                    self._set_status("Live state resynchronized.")
                return

            # Fallback: deduce move from piece placement (no tracker).
            if event.piece_placement is None:
                return
            board = chess.Board(self._coach.current_fen)
            move = deduce_move_from_fen(board, event.piece_placement)
            if move is not None:
                san = board.san(move)
                self._submit_san(san, passive=True)
            else:
                self._set_status(
                    "Position changed \u2014 no legal move matches",
                    error=True,
                )

        elif event.type is ObservationEventType.NEW_GAME:
            # Capture the finished game before syncing to the new position.
            finished_board = self._coach.board
            if finished_board.move_stack:
                finished_board = finished_board.copy()

            if event.fen is not None:
                self._sync_coach_to_fen(event.fen, auto_save_current=True)
            self._set_status(
                event.tracking.message if event.tracking else "New game detected.",
            )

            # Propose review of the just-finished game.
            if finished_board.move_stack:
                self._propose_observe_review(finished_board)

        elif event.type is ObservationEventType.RESYNC:
            if event.fen is not None:
                self._sync_coach_to_fen(event.fen, auto_save_current=True)
            self._set_status(event.message or "Live tracking resynchronized.")

        elif event.type is ObservationEventType.ERROR:
            self._set_status(self._format_observation_error(event), error=True)

        elif event.type is ObservationEventType.STOPPED:
            self._stop_observation()
            self._set_status(
                event.message or "Live observation stopped unexpectedly.",
                error=True,
            )

    @staticmethod
    def _format_observe_start_error(exc: Exception) -> str:
        """Map startup errors to explicit, actionable UI messages."""
        if isinstance(exc, CapturePermissionError):
            return (
                "Observation blocked: screen-capture permission denied. "
                "Allow screen recording in OS settings, then retry."
            )
        if isinstance(exc, DisplayUnavailableError):
            return (
                "Observation unavailable: no display detected. "
                "Start a graphical desktop session and retry."
            )
        if isinstance(exc, MonitorSelectionError):
            return (
                f"Observation unavailable: {exc} "
                "Select a valid monitor in the Observe dialog."
            )
        if isinstance(exc, ScreenCaptureError):
            return f"Observation startup failed: {exc}"
        return f"Observation startup failed: {exc}"

    @staticmethod
    def _format_observation_error(event: ObservationEvent) -> str:
        """Provide actionable status text for runtime observation failures."""
        if event.frame_status is LiveFrameStatus.BOARD_NOT_FOUND:
            return (
                "No board detected. Keep the chessboard fully visible on screen "
                "and verify the selected monitor."
            )
        if event.frame_status is LiveFrameStatus.THEME_UNRECOGNIZED:
            return (
                "Theme not recognized. Pick a supported board theme from Observe settings."
            )
        if event.frame_status is LiveFrameStatus.THEME_UNSUPPORTED:
            return (
                "Theme unsupported. Switch to a built-in theme and restart observation."
            )
        if event.frame_status is LiveFrameStatus.POSITION_UNCERTAIN:
            return (
                "Position uncertain. Improve capture quality (zoom, contrast, board focus)."
            )
        if event.frame_status is LiveFrameStatus.INVALID_POSITION:
            return (
                "Detected position is invalid. Wait for a stable frame or restart observation."
            )
        return event.message or "Live observation error."

    def _drain_analysis_queue(self) -> None:
        """Discard any queued analysis updates."""
        while not self._update_queue.empty():
            try:
                self._update_queue.get_nowait()
            except queue.Empty:
                break

    @staticmethod
    def _best_move_squares(best_move_uci: str | None) -> tuple[str, str] | None:
        """Return source and destination squares for a valid UCI move."""
        if best_move_uci is None or len(best_move_uci) < 4:
            return None

        from_sq = best_move_uci[:2]
        to_sq = best_move_uci[2:4]
        try:
            chess.parse_square(from_sq)
            chess.parse_square(to_sq)
        except ValueError:
            return None
        return from_sq, to_sq

    def _set_board_move_hint(self, best_move_uci: str | None, *, color: str) -> None:
        """Render or clear a move hint overlay on the board."""
        if self._board_widget is None:
            return

        squares = self._best_move_squares(best_move_uci)
        if squares is None:
            self._board_widget.clear_highlights()
            return

        self._board_widget.set_move_hint(*squares, color=color)

    def _update_review_move_hint(self, move: ReviewedMove | None) -> None:
        """Show the review hint only for pedagogical error classes."""
        if move is None:
            self._set_board_move_hint(None, color=_COACH_MOVE_HINT)
            return

        colour = _REVIEW_MOVE_HINTS.get(move.classification)
        if colour is None or not move.best_move_uci or move.best_move_uci == move.uci:
            self._set_board_move_hint(None, color=_COACH_MOVE_HINT)
            return

        self._set_board_move_hint(move.best_move_uci, color=colour)

    def _clear_coaching_outputs(self) -> None:
        """Clear engine-driven UI state and stop any in-flight analysis."""
        if self._coach is not None:
            self._coach.stop_analysis()
        self._current_handle = None
        self._best_move_uci = None
        self._drain_analysis_queue()

        if self._board_widget is not None:
            self._board_widget.clear_highlights()
        if self._eval_panel is not None:
            self._eval_panel.update_eval(None, 0, is_loading=False)
        if self._suggestions_panel is not None:
            self._suggestions_panel.update_suggestions(())

    def _refresh_observe_panel(self) -> None:
        """Refresh passive-session copy while Observe Live is active."""
        if self._observe_panel is None or self._coach is None:
            return
        setup_label = (
            self._last_observer_config.setup_id
            if self._last_observer_config is not None
            else None
        )
        self._observe_panel.update_session(
            setup_label=setup_label,
            move_count=len(self._coach.move_history),
        )

    def _set_observe_mode_ui(self, observing: bool) -> None:
        """Toggle between passive Observe Live UI and Manual Coach UI."""
        if self._in_review_mode:
            return

        if self._board_widget is not None:
            self._board_widget.set_interactive(not observing)

        if self._move_entry is not None:
            if observing:
                self._move_entry.delete(0, "end")
            self._move_entry.config(state="disabled" if observing else "normal")
        if self._submit_btn is not None:
            self._submit_btn.config(state="disabled" if observing else "normal")

        if observing:
            if self._eval_panel is not None:
                self._eval_panel.pack_forget()
            if self._suggestions_panel is not None:
                self._suggestions_panel.pack_forget()
            if self._observe_panel is not None:
                if self._history_panel is not None:
                    self._observe_panel.pack(
                        fill="x",
                        pady=(0, 6),
                        before=self._history_panel,
                    )
                else:
                    self._observe_panel.pack(fill="x", pady=(0, 6))
            if self._btn_frame is not None:
                if self._history_panel is not None:
                    self._btn_frame.pack(
                        fill="x",
                        pady=(0, 4),
                        after=self._history_panel,
                    )
                else:
                    self._btn_frame.pack(fill="x", pady=(0, 4))
            if self._new_game_btn is not None:
                self._new_game_btn.pack_forget()
            if self._pgn_frame is not None:
                self._pgn_frame.pack_forget()
            if self._entry_frame is not None:
                self._entry_frame.pack_forget()

            if self._apply_btn is not None:
                self._apply_btn.config(
                    text="Manual Coach",
                    command=self._exit_observe_to_manual_coach,
                )
            if self._undo_btn is not None:
                self._undo_btn.config(
                    text="Stop Live",
                    command=self._stop_observation_with_review_proposal,
                )
            if self._observe_btn is not None:
                self._observe_btn.config(
                    text="Review",
                    command=self._exit_observe_to_review,
                )
            return

        if self._observe_panel is not None:
            self._observe_panel.pack_forget()
        if self._eval_panel is not None:
            if self._history_panel is not None:
                self._eval_panel.pack(fill="x", pady=(0, 6), before=self._history_panel)
            else:
                self._eval_panel.pack(fill="x", pady=(0, 6))
        if self._suggestions_panel is not None:
            if self._history_panel is not None:
                self._suggestions_panel.pack(
                    fill="x",
                    pady=(0, 6),
                    before=self._history_panel,
                )
            else:
                self._suggestions_panel.pack(fill="x", pady=(0, 6))
        if self._btn_frame is not None:
            if self._history_panel is not None:
                self._btn_frame.pack(fill="x", pady=(0, 4), after=self._history_panel)
            else:
                self._btn_frame.pack(fill="x", pady=(0, 4))
        if self._new_game_btn is not None:
            if self._btn_frame is not None:
                self._new_game_btn.pack(fill="x", pady=(0, 4), after=self._btn_frame)
            else:
                self._new_game_btn.pack(fill="x", pady=(0, 4))
        if self._pgn_frame is not None:
            if self._new_game_btn is not None:
                self._pgn_frame.pack(fill="x", pady=(0, 4), after=self._new_game_btn)
            else:
                self._pgn_frame.pack(fill="x", pady=(0, 4))
        if self._entry_frame is not None:
            self._entry_frame.pack(side="left", fill="y", pady=4)

        if self._apply_btn is not None:
            self._apply_btn.config(
                text="\u25b6  Apply suggestion",
                command=self._on_apply_suggestion,
            )
        if self._undo_btn is not None:
            self._undo_btn.config(
                text="\u21a9  Undo",
                command=self._on_undo,
            )
        if self._observe_btn is not None:
            self._observe_btn.config(
                text="\u25ce  Observe Live",
                command=self._on_toggle_observe,
            )

    def _exit_observe_to_manual_coach(self) -> None:
        """Leave passive observation and return to manual coaching."""
        if not self._observing:
            return
        self._stop_observation()
        self._set_status(
            "Observe Live stopped — Manual Coach restored. Enter moves yourself to continue."
        )

    def _stop_observation_with_review_proposal(self) -> None:
        """Stop observation and propose review if there are captured moves."""
        if self._coach is not None:
            board = self._coach.board
            has_moves = bool(board.move_stack)
            if has_moves:
                board = board.copy()
        else:
            has_moves = False

        self._stop_observation()
        self._set_status("Live observation stopped.")

        if has_moves:
            self._propose_observe_review(board)

    def _exit_observe_to_review(self, board: chess.Board | None = None) -> None:
        """Leave passive observation and start a review.

        Parameters
        ----------
        board:
            If provided, use this board for the review (e.g. the finished game
            captured before a new-game sync).  Otherwise fall back to the
            current coach board.
        """
        if board is None:
            if self._coach is None:
                self._set_status("No game to review", error=True)
                return
            board = self._coach.board

        if not board.move_stack:
            self._set_status(
                "No synced moves to review yet. Keep observing or exit to Manual Coach.",
                error=True,
            )
            return

        # Auto-save the observed game with observation-specific headers.
        try:
            headers = self._build_observe_headers(board)
            path = self._storage.save_game(board, headers)
            if path is not None:
                logger.info("Observed game saved: %s", path)
        except Exception:
            logger.exception("Failed to save observed game before review.")

        if self._observing:
            self._stop_observation()

        self._from_observation = True
        self._start_review(board)

    def _propose_observe_review(self, board: chess.Board) -> None:
        """Show a dialog proposing to review a just-finished observed game."""
        if not board.move_stack:
            return

        assert self._root is not None
        move_count = len(board.move_stack)

        dialog = tk.Toplevel(self._root)
        dialog.title("Game finished")
        dialog.configure(bg=_BG)
        dialog.transient(self._root)
        dialog.grab_set()
        dialog.resizable(False, False)

        container = tk.Frame(dialog, bg=_BG)
        container.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(
            container,
            text="Analyse this game?",
            bg=_BG,
            fg=_FG,
            font=("sans-serif", 13, "bold"),
        ).pack(pady=(0, 6))

        outcome = board.outcome()
        if outcome is not None:
            result_text = board.result()
        else:
            result_text = "incomplete"

        tk.Label(
            container,
            text=f"{move_count} moves captured \u2014 {result_text}",
            bg=_BG,
            fg=_FG_DIM,
            font=("sans-serif", 10),
        ).pack(pady=(0, 14))

        btn_row = tk.Frame(container, bg=_BG)
        btn_row.pack(fill="x")

        def on_review() -> None:
            dialog.destroy()
            self._exit_observe_to_review(board)

        def on_skip() -> None:
            dialog.destroy()

        _styled_button(
            btn_row,
            text="\u25b6  Analyse this game",
            command=on_review,
            bg=_BTN_ACCENT_BG,
            fg=_BTN_ACCENT_FG,
            hover_bg=_BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 10, "bold"),
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        _styled_button(
            btn_row,
            text="Continue observing",
            command=on_skip,
            font=("sans-serif", 10),
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        dialog.update_idletasks()
        x = self._root.winfo_x() + (
            self._root.winfo_width() - dialog.winfo_width()
        ) // 2
        y = self._root.winfo_y() + (
            self._root.winfo_height() - dialog.winfo_height()
        ) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _sync_coach_to_fen(
        self,
        fen: str,
        *,
        auto_save_current: bool = False,
    ) -> bool:
        """Replace coach and GUI board state with a tracked live FEN."""
        assert self._coach is not None
        assert self._board_widget is not None
        assert self._history_panel is not None
        assert self._eval_panel is not None
        assert self._suggestions_panel is not None

        try:
            board = chess.Board(fen)
        except ValueError as exc:
            logger.error("Cannot sync live board from FEN '%s': %s", fen, exc)
            self._set_status(f"Live sync failed: {exc}", error=True)
            return False

        if auto_save_current:
            self._auto_save()

        self._coach.stop_analysis()
        self._current_handle = None
        self._best_move_uci = None
        self._coach.load_board(board)
        self._game_saved = False

        self._board_widget.set_position(self._coach.current_fen)
        self._board_widget.clear_last_move()
        self._board_widget.clear_highlights()
        self._history_panel.set_history(self._coach.move_history)
        self._eval_panel.update_eval(None, 0, is_loading=False)
        self._suggestions_panel.update_suggestions(())
        self._refresh_observe_panel()
        self._update_turn_status()
        return True

    # -- Move handling -------------------------------------------------------

    def _on_board_move(self, from_sq: str, to_sq: str) -> None:
        """Handle a move from board click (source, destination)."""
        if self._in_review_mode or self._observing:
            return
        assert self._coach is not None

        # Build UCI string (to_sq may include promotion char).
        uci_str = from_sq + to_sq

        try:
            move = chess.Move.from_uci(uci_str)
            board = chess.Board(self._coach.current_fen)
            san = board.san(move)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            self._set_status(f"Invalid move: {uci_str}", error=True)
            return

        self._submit_san(san)

    def _on_entry_submit(self, event: tk.Event | None = None) -> None:  # type: ignore[type-arg]
        """Handle SAN move submission from the text entry."""
        if self._observing:
            self._set_status(
                "Observe Live is passive. Exit to Manual Coach before entering moves.",
                error=True,
            )
            return
        assert self._move_entry is not None
        san = self._move_entry.get().strip()
        if not san:
            return
        self._move_entry.delete(0, "end")
        self._submit_san(san)

    def _on_apply_suggestion(self) -> None:
        """Apply the current best move suggestion to the board."""
        if self._observing:
            self._set_status(
                "Observe Live is passive. Exit to Manual Coach before using coaching actions.",
                error=True,
            )
            return
        assert self._coach is not None

        if self._best_move_uci is None:
            self._set_status("No suggestion available", error=True)
            return

        if not self._coach.is_player_turn:
            self._set_status("Not your turn", error=True)
            return

        uci_str = self._best_move_uci
        try:
            move = chess.Move.from_uci(uci_str)
            board = chess.Board(self._coach.current_fen)
            san = board.san(move)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            self._set_status(f"Suggestion invalid: {uci_str}", error=True)
            return

        self._best_move_uci = None
        self._submit_san(san)

    def _on_undo(self) -> None:
        """Undo the last move."""
        if self._observing:
            self._set_status(
                "Observe Live is passive. Exit to Manual Coach before changing moves.",
                error=True,
            )
            return
        assert self._coach is not None

        san = self._coach.undo_move()
        if san is None:
            self._set_status("Nothing to undo", error=True)
            return

        self._best_move_uci = None
        self._current_handle = None

        assert self._board_widget is not None
        assert self._history_panel is not None
        assert self._eval_panel is not None
        assert self._suggestions_panel is not None

        self._board_widget.set_position(self._coach.current_fen)
        self._board_widget.clear_last_move()
        self._board_widget.clear_highlights()
        self._history_panel.set_history(self._coach.move_history)
        self._eval_panel.update_eval(None, 0, is_loading=False)
        self._suggestions_panel.update_suggestions(())

        self._update_turn_status()
        logger.info("Undo: %s", san)

    def _submit_san(self, san: str, *, passive: bool = False) -> None:
        """Submit a SAN move, routing to opponent or player method."""
        assert self._coach is not None

        try:
            if passive:
                self._coach.record_move(san)
                self._clear_coaching_outputs()
                self._refresh_after_move()
                return

            if self._coach.is_player_turn:
                self._coach.submit_player_move(san)
                self._clear_coaching_outputs()
                self._refresh_after_move()
                if self._coach.is_game_over:
                    self._handle_finished_manual_game()
            else:
                # Stop previous analysis.
                self._coach.stop_analysis()
                self._current_handle = None

                handle = self._coach.submit_opponent_move_continuous(
                    san, self._on_analysis_update,
                )
                self._current_handle = handle
                self._refresh_after_move()

                if self._coach.is_game_over:
                    self._handle_finished_manual_game()
                    return

                # Show loading state.
                assert self._eval_panel is not None
                self._eval_panel.update_eval(None, 0, is_loading=True)

        except IllegalMoveError as exc:
            self._set_status(str(exc), error=True)
        except NotYourTurnError as exc:
            self._set_status(str(exc), error=True)
        except GameOverError as exc:
            self._set_status(str(exc), error=True)

    def _on_analysis_update(self, advice: ContinuousAdvice) -> None:
        """Called from analysis thread — enqueue for main thread."""
        self._update_queue.put(advice)

    def _refresh_after_move(self) -> None:
        """Update board and history after any move."""
        assert self._coach is not None
        assert self._board_widget is not None
        assert self._history_panel is not None

        self._board_widget.set_position(self._coach.current_fen)

        # Highlight last move.
        history = self._coach.move_history
        if history:
            board = chess.Board()
            for san_move in history:
                move = board.parse_san(san_move)
                from_sq = chess.square_name(move.from_square)
                to_sq = chess.square_name(move.to_square)
                board.push(move)
            self._board_widget.set_last_move(from_sq, to_sq)

        self._history_panel.set_history(history)
        self._update_turn_status()

    # -- Queue polling -------------------------------------------------------

    def _poll_queue(self) -> None:
        """Drain the update queue and refresh panels."""
        try:
            while True:
                advice = self._update_queue.get_nowait()
                self._apply_advice(advice)
        except queue.Empty:
            pass

        if self._observing:
            try:
                while True:
                    obs_event = self._observation_queue.get_nowait()
                    self._handle_observation_event(obs_event)
            except queue.Empty:
                pass

        # Review progress / completion.
        try:
            while True:
                item = self._review_queue.get_nowait()
                if isinstance(item, int):
                    self._update_review_progress(item)
                elif isinstance(item, GameReview):
                    self._on_review_complete(item)
                elif isinstance(item, Exception):
                    self._on_review_failed(item)
        except queue.Empty:
            pass

        if self._root is not None:
            self._root.after(_POLL_INTERVAL_MS, self._poll_queue)

    def _apply_advice(self, advice: ContinuousAdvice) -> None:
        """Apply a ContinuousAdvice update to the panels."""
        if self._observing:
            return

        update = advice.update

        if self._eval_panel is not None:
            self._eval_panel.update_eval(
                update.evaluation, update.depth_reached, is_loading=False,
            )

        if self._suggestions_panel is not None:
            self._suggestions_panel.update_suggestions(update.variations)

        self._best_move_uci = update.best_move_uci or None
        self._set_board_move_hint(self._best_move_uci, color=_COACH_MOVE_HINT)

    # -- Status management ---------------------------------------------------

    def _update_turn_status(self) -> None:
        """Update the status label to show whose turn it is."""
        assert self._coach is not None

        if self._observing:
            self._refresh_observe_panel()
            self._set_status("Observe Live active \u2014 passive sync only.")
            return

        if self._coach.is_game_over:
            self._auto_save()
            outcome = chess.Board(self._coach.current_fen).outcome()
            if outcome and outcome.winner is not None:
                winner = "White" if outcome.winner else "Black"
                self._set_status(f"Game over \u2014 {winner} wins!", game_over=True)
            else:
                self._set_status("Game over \u2014 Draw!", game_over=True)
            return

        if self._coach.is_player_turn:
            self._set_status(
                "Manual Coach \u2014 your turn: click the board or type a move"
            )
        else:
            self._set_status(
                "Manual Coach \u2014 opponent's turn: enter their move",
                waiting=True,
            )

    def _handle_finished_manual_game(self) -> None:
        """Auto-save a finished manual game and offer a post-game review."""
        if self._coach is None:
            return

        board = self._coach.board
        if not board.move_stack:
            return

        self._clear_coaching_outputs()
        self._auto_save()
        self._propose_manual_review(board)

    def _set_status(
        self,
        text: str,
        error: bool = False,
        waiting: bool = False,
        game_over: bool = False,
    ) -> None:
        """Update the status label."""
        if self._status_label is not None:
            if error:
                fg = _ERROR
            elif waiting:
                fg = _FG_MUTED
            elif game_over:
                fg = _ACCENT
            else:
                fg = _FG_DIM
            self._status_label.config(text=text, fg=fg)

        if self._status_icon is not None:
            if error:
                self._status_icon.config(text="\u25cf", fg=_ERROR)
            elif waiting:
                self._status_icon.config(text="\u25cb", fg=_FG_MUTED)
            elif game_over:
                self._status_icon.config(text="\u2605", fg=_ACCENT)
            else:
                self._status_icon.config(text="\u25cf", fg=_SUCCESS)

    # -- New game / close ----------------------------------------------------

    def _on_new_game(self) -> None:
        """Start a new game."""
        self._close_post_game_review_dialog()
        if self._in_review_mode:
            self._exit_review_mode()
        self._auto_save()

        if self._observing:
            self._stop_observation()

        if self._coach is not None:
            self._coach.stop()

        self._game_saved = False
        self._current_game_file = None

        # Clear queue.
        while not self._update_queue.empty():
            try:
                self._update_queue.get_nowait()
            except queue.Empty:
                break

        self._current_handle = None
        self._best_move_uci = None

        assert self._root is not None
        player_color, self._engine_elo = self._show_color_dialog(
            self._root,
            default_engine_elo=self._engine_elo,
        )
        self._coach = self._create_coach(player_color, self._engine_elo)

        assert self._board_widget is not None
        self._board_widget.set_flipped(player_color == PlayerColor.BLACK)
        self._board_widget.clear_last_move()
        self._board_widget.clear_highlights()
        self._board_widget.set_position(self._coach.current_fen)

        assert self._eval_panel is not None
        self._eval_panel.update_eval(None, 0, is_loading=False)

        assert self._suggestions_panel is not None
        self._suggestions_panel.update_suggestions(())

        assert self._history_panel is not None
        self._history_panel.set_history(())

        self._update_turn_status()
        logger.info(
            "New game started: player=%s elo=%d",
            player_color.value,
            self._engine_elo,
        )

    def _on_close(self) -> None:
        """Clean up resources and close the window."""
        self._close_post_game_review_dialog()
        self._auto_save()
        if self._observer is not None:
            self._observer.stop()
        if self._coach is not None:
            self._coach.stop()
        if self._root is not None:
            self._root.destroy()
            self._root = None
        logger.info("ChessLab GUI closed.")

    # -- Persistence ---------------------------------------------------------

    def _save_review_summary(self, review: GameReview) -> bool:
        """Persist a lightweight summary for longitudinal progression."""
        try:
            self._review_history.record_review(
                review,
                source_path=self._current_game_file,
            )
        except Exception:
            logger.exception("Failed to persist review summary.")
            return False

        if (
            self._progression_dialog is not None
            and self._progression_dialog.winfo_exists()
        ):
            self._refresh_progression_panel()
        return True

    def _build_headers(self) -> dict[str, str]:
        """Build PGN headers from the current game state."""
        assert self._coach is not None
        player_color = self._coach._config.player_color

        outcome = self._coach.board.outcome()
        if outcome is not None:
            result = self._coach.board.result()
        else:
            result = "*"

        return build_headers(
            player_color=player_color,
            engine_elo=self._engine_elo,
            result=result,
        )

    def _build_observe_headers(self, board: chess.Board) -> dict[str, str]:
        """Build PGN headers for a game observed via Observe Live."""
        from datetime import datetime

        outcome = board.outcome()
        result = board.result() if outcome is not None else "*"

        setup_id = (
            self._last_observer_config.setup_id
            if self._last_observer_config is not None
            else None
        )

        headers: dict[str, str] = {
            "Event": "Observed via ChessLab",
            "Site": "ChessLab",
            "Date": datetime.now().strftime("%Y.%m.%d"),
            "White": "?",
            "Black": "?",
            "Result": result,
            "Annotator": "ChessLab Observe Live",
        }
        if setup_id:
            headers["Source"] = f"Observe Live — {setup_id}"

        return headers

    def _auto_save(self) -> None:
        """Auto-save the current game if it has moves."""
        if self._game_saved:
            return
        if self._coach is None:
            return

        board = self._coach.board
        if not board.move_stack:
            return

        try:
            path = self._storage.save_game(board, self._build_headers())
            if path is not None:
                self._game_saved = True
                self._current_game_file = path
                logger.info("Auto-saved game: %s", path)
        except Exception:
            logger.exception("Failed to auto-save game.")

    def _on_export_pgn(self) -> None:
        """Export the current game to a user-chosen PGN file."""
        if self._coach is None or not self._coach.board.move_stack:
            self._set_status("No moves to export", error=True)
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".pgn",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")],
            title="Export PGN",
        )
        if not path:
            return

        try:
            GameStorage.export_game(
                self._coach.board, self._build_headers(), Path(path),
            )
            self._set_status("PGN exported")
        except Exception:
            logger.exception("Failed to export PGN.")
            self._set_status("Export failed", error=True)

    def _on_load_pgn(self) -> None:
        """Load a PGN file and display it on the board."""
        path = filedialog.askopenfilename(
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")],
            title="Load PGN",
        )
        if not path:
            return

        self._load_pgn_file(Path(path))

    def _load_pgn_file(self, file_path: Path) -> None:
        """Load a PGN from the given path into the GUI."""
        self._close_post_game_review_dialog()
        try:
            record = GameStorage.load_game(file_path)
        except Exception as exc:
            self._set_status(f"Load failed: {exc}", error=True)
            logger.exception("Failed to load PGN.")
            return

        assert self._coach is not None
        self._game_saved = False
        self._auto_save()

        self._coach.load_board(record.board)
        self._game_saved = True  # Loaded game doesn't need re-saving.
        self._current_game_file = record.file_path

        self._best_move_uci = None
        self._current_handle = None

        assert self._board_widget is not None
        assert self._history_panel is not None
        assert self._eval_panel is not None
        assert self._suggestions_panel is not None

        self._board_widget.set_position(self._coach.current_fen)
        self._board_widget.clear_last_move()
        self._board_widget.clear_highlights()
        self._history_panel.set_history(self._coach.move_history)
        self._eval_panel.update_eval(None, 0, is_loading=False)
        self._suggestions_panel.update_suggestions(())
        self._update_turn_status()

        self._set_status(f"Loaded: {file_path.name}")
        logger.info("PGN loaded: %s", file_path)

    def _on_recent_games(self) -> None:
        """Show a dialog listing recently saved games."""
        assert self._root is not None
        games = self._storage.list_recent_games()

        dialog = tk.Toplevel(self._root)
        dialog.title("Recent Games")
        dialog.configure(bg=_BG)
        dialog.transient(self._root)
        dialog.grab_set()
        dialog.resizable(False, False)

        container = tk.Frame(dialog, bg=_BG)
        container.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(
            container, text="Recent Games", bg=_BG, fg=_FG,
            font=("sans-serif", 14, "bold"),
        ).pack(pady=(0, 12))

        if not games:
            tk.Label(
                container, text="No saved games", bg=_BG, fg=_FG_DIM,
                font=("sans-serif", 10),
            ).pack(pady=20)
        else:
            list_frame = tk.Frame(container, bg=_CARD_BORDER)
            list_frame.pack(fill="both", expand=True)

            listbox = tk.Listbox(
                list_frame,
                bg=_BG_SECONDARY,
                fg=_FG,
                font=("monospace", 10),
                selectbackground=_ACCENT,
                selectforeground="#ffffff",
                relief="flat",
                bd=0,
                highlightthickness=0,
                height=min(len(games), 12),
                width=50,
            )
            listbox.pack(fill="both", expand=True, padx=1, pady=1)

            for g in games:
                line = (
                    f"{g.date}  {g.white} vs {g.black}"
                    f"  {g.result}  {g.move_count} moves"
                )
                listbox.insert("end", line)

            def on_load() -> None:
                sel = listbox.curselection()
                if not sel:
                    return
                selected_game = games[sel[0]]
                dialog.destroy()
                self._load_pgn_file(selected_game.file_path)

            def on_review() -> None:
                sel = listbox.curselection()
                if not sel:
                    return
                selected_game = games[sel[0]]
                dialog.destroy()
                self._review_from_file(selected_game.file_path)

            listbox.bind("<Double-Button-1>", lambda e: on_load())

            btn_frame = tk.Frame(container, bg=_BG)
            btn_frame.pack(fill="x", pady=(8, 0))

            load_btn = _styled_button(
                btn_frame,
                text="Load",
                command=on_load,
                bg=_BTN_ACCENT_BG,
                fg=_BTN_ACCENT_FG,
                hover_bg=_BTN_ACCENT_HOVER,
                active_bg="#1e88e5",
                font=("sans-serif", 10, "bold"),
            )
            load_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

            review_btn = _styled_button(
                btn_frame,
                text="Review",
                command=on_review,
                font=("sans-serif", 10, "bold"),
            )
            review_btn.pack(side="left", fill="x", expand=True, padx=(4, 4))

            cancel_btn = _styled_button(
                btn_frame,
                text="Cancel",
                command=dialog.destroy,
                font=("sans-serif", 10),
            )
            cancel_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        if not games:
            close_btn = _styled_button(
                container,
                text="Close",
                command=dialog.destroy,
                font=("sans-serif", 10),
            )
            close_btn.pack(pady=(8, 0))

        dialog.update_idletasks()
        x = self._root.winfo_x() + (
            self._root.winfo_width() - dialog.winfo_width()
        ) // 2
        y = self._root.winfo_y() + (
            self._root.winfo_height() - dialog.winfo_height()
        ) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _show_progression_dialog(self) -> None:
        """Show the cross-session progression dashboard."""
        assert self._root is not None

        if (
            self._progression_dialog is not None
            and self._progression_dialog.winfo_exists()
        ):
            self._progression_dialog.deiconify()
            self._progression_dialog.lift()
            self._progression_dialog.focus_force()
            self._refresh_progression_panel()
            return

        dialog = tk.Toplevel(self._root)
        dialog.title("Progression")
        dialog.configure(bg=_BG)
        dialog.transient(self._root)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", self._close_progression_dialog)
        self._progression_dialog = dialog

        container = tk.Frame(dialog, bg=_BG)
        container.pack(fill="both", expand=True, padx=16, pady=16)

        title_row = tk.Frame(container, bg=_BG)
        title_row.pack(fill="x", pady=(0, 10))

        tk.Label(
            title_row,
            text="Progression",
            bg=_BG,
            fg=_FG,
            font=("sans-serif", 16, "bold"),
            anchor="w",
        ).pack(side="left")

        controls = tk.Frame(container, bg=_BG)
        controls.pack(fill="x", pady=(0, 10))

        tk.Label(
            controls,
            text="Last",
            bg=_BG,
            fg=_FG_DIM,
            font=("sans-serif", 9),
        ).pack(side="left")

        self._progression_limit_var = tk.IntVar(value=self._progression_sample_size)
        limit_spin = tk.Spinbox(
            controls,
            from_=3,
            to=100,
            width=5,
            textvariable=self._progression_limit_var,
            justify="center",
            bg=_ENTRY_BG,
            fg=_ENTRY_FG,
            buttonbackground=_BTN_BG,
            insertbackground=_FG,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("monospace", 10),
        )
        limit_spin.pack(side="left", padx=(6, 6))
        limit_spin.bind("<Return>", lambda _event: self._refresh_progression_panel())
        limit_spin.bind("<FocusOut>", lambda _event: self._refresh_progression_panel())

        tk.Label(
            controls,
            text="reviewed games",
            bg=_BG,
            fg=_FG_DIM,
            font=("sans-serif", 9),
        ).pack(side="left")

        _styled_button(
            controls,
            text="Refresh",
            command=self._refresh_progression_panel,
            font=("sans-serif", 9, "bold"),
            padx=10,
            pady=4,
        ).pack(side="right")

        self._progression_panel = ProgressionPanel(container)
        self._progression_panel.pack(fill="both", expand=True)

        footer = tk.Frame(container, bg=_BG)
        footer.pack(fill="x", pady=(10, 0))
        _styled_button(
            footer,
            text="Close",
            command=self._close_progression_dialog,
            font=("sans-serif", 10),
        ).pack(side="right")

        self._refresh_progression_panel()

        dialog.update_idletasks()
        x = self._root.winfo_x() + (
            self._root.winfo_width() - dialog.winfo_width()
        ) // 2
        y = self._root.winfo_y() + (
            self._root.winfo_height() - dialog.winfo_height()
        ) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _refresh_progression_panel(self) -> None:
        """Reload the progression snapshot from persisted review summaries."""
        if self._progression_panel is None:
            return

        limit = self._coerce_progression_limit()
        try:
            snapshot = self._progress_tracker.build_snapshot(limit=limit)
        except Exception as exc:
            logger.exception("Failed to build progression snapshot.")
            self._progression_panel.show_error(
                f"Could not load progression data: {exc}",
            )
            return

        self._progression_panel.update_snapshot(snapshot)

    def _close_progression_dialog(self) -> None:
        """Close the progression dashboard dialog."""
        if self._progression_dialog is not None:
            self._progression_dialog.destroy()
        self._progression_dialog = None
        self._progression_panel = None
        self._progression_limit_var = None

    def _coerce_progression_limit(self) -> int:
        """Return the selected progression sample size."""
        if self._progression_limit_var is None:
            return self._progression_sample_size

        try:
            limit = int(self._progression_limit_var.get())
        except (tk.TclError, ValueError):
            limit = self._progression_sample_size

        limit = max(3, min(limit, 100))
        self._progression_sample_size = limit
        self._progression_limit_var.set(limit)
        return limit

    # -- Review mode ---------------------------------------------------------

    def _on_review_current(self) -> None:
        """Review the current game on the board."""
        if self._observing:
            self._exit_observe_to_review()
            return
        if self._coach is None:
            self._set_status("No game to review", error=True)
            return
        board = self._coach.board
        if not board.move_stack:
            self._set_status("No moves to review", error=True)
            return
        self._start_review(board)

    def _review_from_file(self, file_path: Path) -> None:
        """Load a PGN and immediately start review."""
        try:
            record = GameStorage.load_game(file_path)
        except Exception as exc:
            self._set_status(f"Load failed: {exc}", error=True)
            logger.exception("Failed to load PGN for review.")
            return

        self._review_from_record(record)

    def _review_from_record(self, record: GameRecord) -> None:
        """Start review from an already-loaded :class:`GameRecord`."""
        if not record.board.move_stack:
            self._set_status("No moves to review", error=True)
            return

        # Load into coach for display.
        assert self._coach is not None
        self._game_saved = False
        self._auto_save()
        self._coach.load_board(record.board)
        self._game_saved = True
        self._current_game_file = record.file_path

        self._start_review(record.board)

    def _on_paste_pgn(self) -> None:
        """Show a dialog to paste PGN text and launch review."""
        assert self._root is not None

        dialog = tk.Toplevel(self._root)
        dialog.title("Paste PGN")
        dialog.configure(bg=_BG)
        dialog.transient(self._root)
        dialog.grab_set()
        dialog.resizable(False, False)

        container = tk.Frame(dialog, bg=_BG)
        container.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(
            container,
            text="Paste PGN",
            bg=_BG,
            fg=_FG,
            font=("sans-serif", 14, "bold"),
        ).pack(pady=(0, 8))

        tk.Label(
            container,
            text="Paste a game in PGN format below:",
            bg=_BG,
            fg=_FG_DIM,
            font=("sans-serif", 9),
        ).pack(anchor="w", pady=(0, 4))

        # Text area with border frame.
        text_border = tk.Frame(container, bg=_ENTRY_BORDER)
        text_border.pack(fill="both", expand=True)

        text_widget = tk.Text(
            text_border,
            bg=_ENTRY_BG,
            fg=_ENTRY_FG,
            font=("monospace", 10),
            insertbackground=_FG,
            selectbackground=_ACCENT,
            selectforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=0,
            width=55,
            height=18,
            wrap="word",
        )
        text_widget.pack(fill="both", expand=True, padx=1, pady=1)

        # Error label (hidden by default).
        error_label = tk.Label(
            container,
            text="",
            bg=_BG,
            fg=_ERROR,
            font=("sans-serif", 9),
            wraplength=400,
            justify="left",
        )

        # Buttons.
        btn_frame = tk.Frame(container, bg=_BG)
        btn_frame.pack(fill="x", pady=(8, 0))

        def on_review() -> None:
            pgn_text = text_widget.get("1.0", "end").strip()
            if not pgn_text:
                error_label.config(text="Please paste a PGN game.")
                error_label.pack(anchor="w", pady=(4, 0))
                return

            try:
                record = GameStorage.load_game_from_text(pgn_text)
            except Exception as exc:
                error_label.config(text=f"Invalid PGN: {exc}")
                error_label.pack(anchor="w", pady=(4, 0))
                return

            if not record.board.move_stack:
                error_label.config(text="No moves found in PGN.")
                error_label.pack(anchor="w", pady=(4, 0))
                return

            dialog.destroy()
            self._review_from_record(record)

        review_btn = _styled_button(
            btn_frame,
            text="Review",
            command=on_review,
            bg=_BTN_ACCENT_BG,
            fg=_BTN_ACCENT_FG,
            hover_bg=_BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 10, "bold"),
        )
        review_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        cancel_btn = _styled_button(
            btn_frame,
            text="Cancel",
            command=dialog.destroy,
            font=("sans-serif", 10),
        )
        cancel_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        # Center dialog over main window.
        dialog.update_idletasks()
        x = self._root.winfo_x() + (
            self._root.winfo_width() - dialog.winfo_width()
        ) // 2
        y = self._root.winfo_y() + (
            self._root.winfo_height() - dialog.winfo_height()
        ) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

        text_widget.focus_set()

    def _start_review(self, board: chess.Board) -> None:
        """Begin post-game review: show progress, launch background analysis."""
        self._close_post_game_review_dialog()
        if self._observing:
            self._stop_observation()
        if self._in_review_mode:
            self._exit_review_mode()

        # Stop current analysis.
        if self._coach is not None:
            self._coach.stop_analysis()
        self._current_handle = None

        # Pre-compute FEN at each ply.
        root = board.copy()
        while root.move_stack:
            root.pop()
        replay = root
        self._review_positions = [replay.fen()]
        for move in board.move_stack:
            replay.push(move)
            self._review_positions.append(replay.fen())

        # Clear review queue.
        while not self._review_queue.empty():
            try:
                self._review_queue.get_nowait()
            except queue.Empty:
                break

        self._review_cancel = threading.Event()
        total = len(board.move_stack)

        # Show progress dialog.
        self._show_review_progress(total)

        # Launch background worker.
        board_copy = board.copy()
        worker = threading.Thread(
            target=self._review_worker,
            args=(board_copy,),
            daemon=True,
        )
        worker.start()
        logger.info("Review started: %d positions to analyse.", total)

    def _review_worker(self, board: chess.Board) -> None:
        """Background thread: run GameReviewer and post result to queue."""
        try:
            reviewer = GameReviewer(ReviewConfig())

            def on_progress(current: int, total: int) -> None:
                self._review_queue.put(current)

            review = reviewer.review(
                board,
                progress_callback=on_progress,
                cancel_event=self._review_cancel,
            )
            self._review_queue.put(review)
        except Exception as exc:
            self._review_queue.put(exc)

    def _show_review_progress(self, total: int) -> None:
        """Show a modal progress dialog for analysis."""
        assert self._root is not None

        dialog = tk.Toplevel(self._root)
        dialog.title("Analysing Game")
        dialog.configure(bg=_BG)
        dialog.transient(self._root)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", self._cancel_review)

        container = tk.Frame(dialog, bg=_BG)
        container.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(
            container, text="Analysing game...", bg=_BG, fg=_FG,
            font=("sans-serif", 12, "bold"),
        ).pack(pady=(0, 12))

        self._progress_bar = ttk.Progressbar(
            container, maximum=total, mode="determinate", length=300,
        )
        self._progress_bar.pack(fill="x", pady=(0, 8))

        self._progress_label = tk.Label(
            container, text=f"0 / {total}", bg=_BG, fg=_FG_DIM,
            font=("sans-serif", 9),
        )
        self._progress_label.pack()

        cancel_btn = _styled_button(
            container,
            text="Cancel",
            command=self._cancel_review,
            font=("sans-serif", 10),
        )
        cancel_btn.pack(pady=(8, 0))

        self._progress_dialog = dialog

        dialog.update_idletasks()
        x = self._root.winfo_x() + (
            self._root.winfo_width() - dialog.winfo_width()
        ) // 2
        y = self._root.winfo_y() + (
            self._root.winfo_height() - dialog.winfo_height()
        ) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _cancel_review(self) -> None:
        """Cancel an in-progress review."""
        if self._review_cancel is not None:
            self._review_cancel.set()
        self._close_progress_dialog()
        self._set_status("Review cancelled")

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.destroy()
            self._progress_dialog = None
            self._progress_bar = None

    def _update_review_progress(self, current: int) -> None:
        """Update progress bar (called from poll loop on main thread)."""
        if self._progress_bar is not None:
            self._progress_bar["value"] = current
            total = int(self._progress_bar["maximum"])
            if hasattr(self, "_progress_label"):
                self._progress_label.config(text=f"{current} / {total}")

    def _on_review_complete(self, review: GameReview) -> None:
        """Transition to review mode after analysis finishes."""
        self._close_progress_dialog()
        self._game_review = review
        self._in_review_mode = True
        progression_saved = self._save_review_summary(review)

        # Hide play-mode widgets, show review-mode widgets.
        for w in self._play_mode_widgets:
            w.pack_forget()
        if self._entry_frame is not None:
            self._entry_frame.pack_forget()

        assert self._review_panel is not None
        assert self._review_nav is not None
        assert self._summary_panel is not None
        assert self._history_panel is not None
        assert self._eval_panel is not None

        self._review_panel.pack(fill="x", pady=(0, 6))
        self._review_nav.pack(fill="x", pady=(0, 4))
        self._summary_panel.pack(fill="x", pady=(0, 4))

        # Populate move history with classification colours.
        moves = tuple(m.san for m in review.moves)
        classifications = tuple(m.classification for m in review.moves)
        self._history_panel.set_review_history(
            moves, classifications,
            on_move_click=lambda ply: self._review_navigate_to(ply + 1),
        )

        self._summary_panel.update_summary(
            review,
            on_lesson_click=self._review_jump_to_lesson,
        )
        self._review_navigate_to(0)

        # Show "Back to Launcher" when reviewing a game from Observe Live.
        if self._review_nav is not None:
            self._review_nav.show_launcher_button(self._from_observation)

        self._set_status(
            f"Review mode \u2014 post-game analysis for {len(review.moves)} moves "
            f"in {review.total_elapsed_ms / 1000:.1f}s"
            f"{'' if progression_saved else ' (progression update failed)'}",
            error=not progression_saved,
        )
        logger.info(
            "Review complete: %d moves, %.1fs",
            len(review.moves), review.total_elapsed_ms / 1000,
        )

    def _on_review_failed(self, exc: Exception) -> None:
        """Handle review failure."""
        self._close_progress_dialog()
        if isinstance(exc, ReviewCancelledError):
            return
        self._set_status(f"Review failed: {exc}", error=True)
        logger.exception("Review failed: %s", exc)

    def _exit_review_mode(self) -> None:
        """Return to normal play mode."""
        if not self._in_review_mode:
            return

        self._in_review_mode = False
        self._from_observation = False
        self._game_review = None
        self._review_ply = 0
        self._review_positions = []

        # Hide review widgets, show play-mode widgets.
        for w in self._review_mode_widgets:
            w.pack_forget()

        assert self._suggestions_panel is not None
        assert self._btn_frame is not None
        assert self._new_game_btn is not None
        assert self._pgn_frame is not None
        assert self._history_panel is not None
        assert self._eval_panel is not None

        self._set_observe_mode_ui(self._observing)

        # Restore board to current coach state.
        if self._coach is not None:
            assert self._board_widget is not None
            self._board_widget.set_position(self._coach.current_fen)
            self._board_widget.clear_last_move()
            self._board_widget.clear_highlights()
            self._history_panel.set_history(self._coach.move_history)
            self._eval_panel.update_eval(None, 0, is_loading=False)
            self._suggestions_panel.update_suggestions(())
            self._update_turn_status()

        if self._review_panel is not None:
            self._review_panel.clear()
        if self._review_nav is not None:
            self._review_nav.show_launcher_button(False)
        self._set_status("Exited review mode")
        logger.info("Review mode exited.")

    def _return_to_launcher(self) -> None:
        """Exit review mode and re-show the startup launcher dialog."""
        self._close_post_game_review_dialog()
        self._exit_review_mode()

        if self._root is None:
            return
        mode = self._show_startup_dialog(self._root)
        self._apply_startup_mode(mode)

    def _close_post_game_review_dialog(self) -> None:
        """Dismiss the non-blocking post-game review prompt, if present."""
        if self._post_game_review_dialog is None:
            return

        dialog = self._post_game_review_dialog
        self._post_game_review_dialog = None
        try:
            if dialog.winfo_exists():
                dialog.destroy()
        except tk.TclError:
            pass

    def _propose_manual_review(self, board: chess.Board) -> None:
        """Offer a post-game review for a finished Manual Coach game."""
        if not board.move_stack or self._root is None:
            return

        self._auto_save()
        self._close_post_game_review_dialog()

        move_count = len(board.move_stack)
        result_text = board.result() if board.outcome() is not None else "incomplete"

        dialog = tk.Toplevel(self._root)
        dialog.title("Game finished")
        dialog.configure(bg=_BG)
        dialog.transient(self._root)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", self._close_post_game_review_dialog)
        self._post_game_review_dialog = dialog

        container = tk.Frame(dialog, bg=_BG)
        container.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(
            container,
            text="Review this Manual Coach game?",
            bg=_BG,
            fg=_FG,
            font=("sans-serif", 13, "bold"),
        ).pack(pady=(0, 6))

        tk.Label(
            container,
            text=(
                f"{move_count} moves saved \u2014 {result_text}\n"
                "Launch a full review now, start a new game, or head back to the launcher."
            ),
            bg=_BG,
            fg=_FG_DIM,
            font=("sans-serif", 10),
            justify="center",
        ).pack(pady=(0, 14))

        primary_row = tk.Frame(container, bg=_BG)
        primary_row.pack(fill="x")

        def on_review() -> None:
            self._close_post_game_review_dialog()
            self._start_review(board)

        def on_new_game() -> None:
            self._close_post_game_review_dialog()
            self._on_new_game()

        def on_launcher() -> None:
            self._close_post_game_review_dialog()
            self._return_to_launcher()

        _styled_button(
            primary_row,
            text="\u25b6  Analyse this game",
            command=on_review,
            bg=_BTN_ACCENT_BG,
            fg=_BTN_ACCENT_FG,
            hover_bg=_BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 10, "bold"),
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        _styled_button(
            primary_row,
            text="New game",
            command=on_new_game,
            font=("sans-serif", 10),
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        secondary_row = tk.Frame(container, bg=_BG)
        secondary_row.pack(fill="x", pady=(8, 0))

        _styled_button(
            secondary_row,
            text="Back to launcher",
            command=on_launcher,
            font=("sans-serif", 10),
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        _styled_button(
            secondary_row,
            text="Later",
            command=self._close_post_game_review_dialog,
            font=("sans-serif", 10),
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        dialog.update_idletasks()
        x = self._root.winfo_x() + (
            self._root.winfo_width() - dialog.winfo_width()
        ) // 2
        y = self._root.winfo_y() + (
            self._root.winfo_height() - dialog.winfo_height()
        ) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

    # -- Review navigation ---------------------------------------------------

    def _review_navigate_to(self, ply: int) -> None:
        """Navigate to the given half-move index in review mode."""
        if not self._in_review_mode or self._game_review is None:
            return

        total = len(self._game_review.moves)
        ply = max(0, min(ply, total))
        self._review_ply = ply

        assert self._board_widget is not None
        assert self._eval_panel is not None
        assert self._review_panel is not None
        assert self._review_nav is not None
        assert self._history_panel is not None

        # Update board.
        fen = self._review_positions[ply]
        self._board_widget.set_position(fen)

        # Show last move highlight.
        if ply > 0:
            move = self._game_review.moves[ply - 1]
            from_sq = move.uci[:2]
            to_sq = move.uci[2:4]
            self._board_widget.set_last_move(from_sq, to_sq)
            self._update_review_move_hint(move)
        else:
            self._board_widget.clear_last_move()
            self._update_review_move_hint(None)

        # Update review panel.
        if ply > 0:
            reviewed_move = self._game_review.moves[ply - 1]
            self._review_panel.update_move(reviewed_move)
            self._eval_panel.update_eval(
                reviewed_move.eval_after, 0, is_loading=False,
            )
        else:
            self._review_panel.clear()
            if self._game_review.moves:
                self._eval_panel.update_eval(
                    self._game_review.moves[0].eval_before, 0, is_loading=False,
                )
            else:
                self._eval_panel.update_eval(None, 0, is_loading=False)

        # Highlight in move list.
        if ply > 0:
            self._history_panel.highlight_ply(ply - 1)

        self._review_nav.update_position(ply, total)
        self._set_status(f"Review \u2014 Move {ply} / {total}")

    def _review_first(self) -> None:
        self._review_navigate_to(0)

    def _review_jump_to_lesson(self, focus_ply: int) -> None:
        """Jump to the move highlighted by a summary lesson."""
        self._review_navigate_to(focus_ply + 1)

    def _review_prev(self) -> None:
        self._review_navigate_to(self._review_ply - 1)

    def _review_next(self) -> None:
        self._review_navigate_to(self._review_ply + 1)

    def _review_last(self) -> None:
        if self._game_review is not None:
            self._review_navigate_to(len(self._game_review.moves))

    def _review_prev_error(self) -> None:
        """Jump to the previous mistake or blunder."""
        if self._game_review is None:
            return
        for ply_idx in reversed(self._game_review.critical_moments):
            if ply_idx + 1 < self._review_ply:
                self._review_navigate_to(ply_idx + 1)
                return

    def _review_next_error(self) -> None:
        """Jump to the next mistake or blunder."""
        if self._game_review is None:
            return
        for ply_idx in self._game_review.critical_moments:
            if ply_idx + 1 > self._review_ply:
                self._review_navigate_to(ply_idx + 1)
                return

    # -- Keyboard bindings for review ----------------------------------------

    def _on_key_left(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._in_review_mode:
            self._review_prev()

    def _on_key_right(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._in_review_mode:
            self._review_next()

    def _on_key_home(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._in_review_mode:
            self._review_first()

    def _on_key_end(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._in_review_mode:
            self._review_last()
