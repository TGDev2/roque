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
import queue
import threading
import tkinter as tk
import tkinter.ttk as ttk
from dataclasses import dataclass, field, replace
from pathlib import Path
from tkinter import filedialog

import chess

from chesslab.persistence.game_storage import GameRecord, GameStorage, build_headers

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
from chesslab.gui.board_widget import BoardWidget
from chesslab.gui.live_observer import (
    LiveObserver,
    LiveObserverConfig,
    ObservationEvent,
    ObservationEventType,
)
from chesslab.live.move_deduction import deduce_move_from_fen
from chesslab.gui.panels import (
    EvalPanel,
    GameSummaryPanel,
    MoveHistoryPanel,
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
    """

    player_color: PlayerColor | None = None
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    square_size: int = 64


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

        # Persistence.
        self._storage = GameStorage()
        self._game_saved: bool = False

        # Review mode state.
        self._in_review_mode: bool = False
        self._game_review: GameReview | None = None
        self._review_ply: int = 0
        self._review_positions: list[str] = []
        self._review_cancel: threading.Event | None = None
        self._review_queue: queue.Queue[int | GameReview | Exception] = queue.Queue()
        self._progress_dialog: tk.Toplevel | None = None
        self._progress_bar: ttk.Progressbar | None = None

        # Widgets (populated by _build_window).
        self._observe_btn: tk.Button | None = None
        self._board_widget: BoardWidget | None = None
        self._eval_panel: EvalPanel | None = None
        self._suggestions_panel: SuggestionsPanel | None = None
        self._history_panel: MoveHistoryPanel | None = None
        self._move_entry: tk.Entry | None = None
        self._status_label: tk.Label | None = None
        self._status_icon: tk.Label | None = None

        # Review mode widgets.
        self._review_panel: ReviewPanel | None = None
        self._review_nav: ReviewNavigationPanel | None = None
        self._summary_panel: GameSummaryPanel | None = None

        # Play mode widget groups (for hiding/showing during review).
        self._play_mode_widgets: list[tk.Widget] = []
        self._review_mode_widgets: list[tk.Widget] = []

        # Frame refs for layout management.
        self._btn_frame: tk.Frame | None = None
        self._new_game_btn: tk.Button | None = None
        self._pgn_frame: tk.Frame | None = None
        self._entry_frame: tk.Frame | None = None
        self._submit_btn: tk.Button | None = None

    def run(self) -> None:
        """Build the window and enter the tkinter mainloop (blocking)."""
        root = tk.Tk()
        root.title("ChessLab")
        root.configure(bg=_BG)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root = root

        # Determine player color.
        player_color = self._config.player_color
        if player_color is None:
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

        self._suggestions_panel = SuggestionsPanel(panel_frame)
        self._suggestions_panel.pack(fill="x", pady=(0, 6))

        self._history_panel = MoveHistoryPanel(panel_frame)
        self._history_panel.pack(fill="both", expand=True, pady=(0, 6))

        # Action buttons row.
        self._btn_frame = tk.Frame(panel_frame, bg=_BG)
        self._btn_frame.pack(fill="x", pady=(0, 4))

        apply_btn = _styled_button(
            self._btn_frame,
            text="\u25b6  Apply suggestion",
            command=self._on_apply_suggestion,
            bg=_BTN_ACCENT_BG,
            fg=_BTN_ACCENT_FG,
            hover_bg=_BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 10, "bold"),
        )
        apply_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        undo_btn = _styled_button(
            self._btn_frame,
            text="\u21a9  Undo",
            command=self._on_undo,
        )
        undo_btn.pack(side="left", fill="x", expand=True, padx=(3, 3))

        self._observe_btn = _styled_button(
            self._btn_frame,
            text="\u25ce  Observe",
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
        )
        self._summary_panel = GameSummaryPanel(panel_frame)

        # Track play-mode widgets for hide/show.
        self._play_mode_widgets = [
            self._suggestions_panel, self._btn_frame,
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

    _KNOWN_THEMES = (
        "chess.com/green",
        "chess.com/brown",
        "chess.com/blue",
        "chess.com/purple",
        "chess.com/walnut",
        "lichess/brown",
        "lichess/blue",
        "lichess/green",
    )

    def _show_observe_dialog(
        self,
        parent: tk.Tk,
    ) -> LiveObserverConfig | None:
        """Show a dialog to configure live observation settings."""
        result: list[LiveObserverConfig | None] = [None]

        dialog = tk.Toplevel(parent)
        dialog.title("Live Observation")
        dialog.configure(bg=_BG)
        dialog.transient(parent)
        dialog.grab_set()
        dialog.resizable(False, False)

        container = tk.Frame(dialog, bg=_BG)
        container.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(
            container, text="Live Observation", bg=_BG, fg=_FG,
            font=("sans-serif", 18, "bold"),
        ).pack(pady=(0, 16))

        # Monitor selector.
        monitor_card = tk.Frame(container, bg=_CARD_BORDER)
        monitor_card.pack(fill="x", pady=(0, 12))
        monitor_inner = tk.Frame(monitor_card, bg=_BG_SECONDARY)
        monitor_inner.pack(fill="both", expand=True, padx=1, pady=1)

        tk.Label(
            monitor_inner, text="Monitor", bg=_BG_SECONDARY, fg=_FG,
            font=("sans-serif", 11, "bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))

        try:
            monitors = ScreenCapture.list_monitors()
        except Exception as exc:
            monitors = []
            logger.warning("Cannot list monitors: %s", exc)

        monitor_labels = [
            f"Monitor {m.index}: {m.width}x{m.height}" for m in monitors
        ]
        if not monitor_labels:
            monitor_labels = ["No monitors found"]

        monitor_var = tk.StringVar(value=monitor_labels[0])
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

        # Theme selector.
        theme_card = tk.Frame(container, bg=_CARD_BORDER)
        theme_card.pack(fill="x", pady=(0, 12))
        theme_inner = tk.Frame(theme_card, bg=_BG_SECONDARY)
        theme_inner.pack(fill="both", expand=True, padx=1, pady=1)

        tk.Label(
            theme_inner, text="Board Theme", bg=_BG_SECONDARY, fg=_FG,
            font=("sans-serif", 11, "bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))

        theme_var = tk.StringVar(value=self._KNOWN_THEMES[0])
        theme_menu = tk.OptionMenu(theme_inner, theme_var, *self._KNOWN_THEMES)
        theme_menu.config(
            bg=_BG_SECONDARY, fg=_FG, font=("sans-serif", 10),
            highlightthickness=0, relief="flat", activebackground=_BTN_HOVER,
            activeforeground=_FG,
        )
        theme_menu["menu"].config(
            bg=_BG_SECONDARY, fg=_FG, activebackground=_ACCENT,
            activeforeground="#ffffff",
        )
        theme_menu.pack(fill="x", padx=12, pady=(0, 8))

        # Perspective selector.
        tk.Label(
            container, text="Board perspective", bg=_BG, fg=_FG_DIM,
            font=("sans-serif", 10), anchor="w",
        ).pack(fill="x", pady=(0, 8))

        perspective_var = tk.BooleanVar(value=True)

        persp_frame = tk.Frame(container, bg=_BG)
        persp_frame.pack(fill="x", pady=(0, 12))

        white_btn = _styled_button(
            persp_frame,
            text="\u2654  White",
            command=lambda: perspective_var.set(True),
            bg=_BTN_WHITE_BG,
            fg=_BTN_WHITE_FG,
            hover_bg=_BTN_WHITE_HOVER,
            active_bg="#f5ede1",
            font=("sans-serif", 11, "bold"),
            pady=8,
        )
        white_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        black_btn = _styled_button(
            persp_frame,
            text="\u265a  Black",
            command=lambda: perspective_var.set(False),
            bg=_BTN_BLACK_BG,
            fg=_BTN_BLACK_FG,
            hover_bg=_BTN_BLACK_HOVER,
            active_bg="#4a4a5c",
            font=("sans-serif", 11, "bold"),
            pady=8,
        )
        black_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        # OK / Cancel buttons.
        ok_frame = tk.Frame(container, bg=_BG)
        ok_frame.pack(fill="x", pady=(4, 0))

        def on_ok() -> None:
            if not monitors:
                return
            selected = monitor_var.get()
            monitor_index = monitors[monitor_labels.index(selected)].index
            result[0] = LiveObserverConfig(
                monitor_index=monitor_index,
                theme=theme_var.get(),
                white_perspective=perspective_var.get(),
            )
            dialog.destroy()

        ok_btn = _styled_button(
            ok_frame,
            text="Start",
            command=on_ok,
            bg=_BTN_ACCENT_BG,
            fg=_BTN_ACCENT_FG,
            hover_bg=_BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 11, "bold"),
            pady=8,
        )
        ok_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        if not monitors:
            ok_btn.config(state="disabled")

        cancel_btn = _styled_button(
            ok_frame,
            text="Cancel",
            command=dialog.destroy,
            font=("sans-serif", 11),
            pady=8,
        )
        cancel_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        # Center dialog.
        dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

        parent.wait_window(dialog)
        return result[0]

    def _on_toggle_observe(self) -> None:
        """Toggle live observation mode on/off."""
        if self._observing:
            self._stop_observation()
            return

        assert self._root is not None
        config = self._show_observe_dialog(self._root)
        if config is None:
            return

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
            self._set_status(f"Observation failed: {exc}", error=True)
            logger.error("Failed to start live observer: %s", exc)
            return

        self._observer = observer
        self._observing = True

        if self._observe_btn is not None:
            self._observe_btn.config(text="\u25a0  Stop")
        if self._move_entry is not None:
            self._move_entry.config(state="disabled")

        self._set_status("Live observation active")
        logger.info("Live observation started.")

    def _stop_observation(self) -> None:
        """Stop live observation and restore manual input."""
        if self._observer is not None:
            self._observer.stop()
            self._observer = None
        self._observing = False

        if self._observe_btn is not None:
            self._observe_btn.config(text="\u25ce  Observe")
        if self._move_entry is not None:
            self._move_entry.config(state="normal")

        self._update_turn_status()
        logger.info("Live observation stopped.")

    def _handle_observation_event(self, event: ObservationEvent) -> None:
        """Process an observation event from the live observer."""
        assert self._coach is not None

        if event.type is ObservationEventType.MOVE_DETECTED:
            # If game tracking provided a move, use it directly.
            if event.tracking is not None and event.tracking.move_san is not None:
                self._submit_san(event.tracking.move_san)
                return

            # Fallback: deduce move from piece placement (no tracker).
            if event.piece_placement is None:
                return
            board = chess.Board(self._coach.current_fen)
            move = deduce_move_from_fen(board, event.piece_placement)
            if move is not None:
                san = board.san(move)
                self._submit_san(san)
            else:
                self._set_status(
                    "Position changed \u2014 no legal move matches",
                    error=True,
                )

        elif event.type is ObservationEventType.NEW_GAME:
            self._set_status(
                event.tracking.message if event.tracking else "New game detected.",
            )

        elif event.type is ObservationEventType.ERROR:
            self._set_status(event.message, error=True)

        elif event.type is ObservationEventType.STOPPED:
            self._stop_observation()
            self._set_status(event.message, error=True)

    # -- Move handling -------------------------------------------------------

    def _on_board_move(self, from_sq: str, to_sq: str) -> None:
        """Handle a move from board click (source, destination)."""
        if self._in_review_mode:
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
        assert self._move_entry is not None
        san = self._move_entry.get().strip()
        if not san:
            return
        self._move_entry.delete(0, "end")
        self._submit_san(san)

    def _on_apply_suggestion(self) -> None:
        """Apply the current best move suggestion to the board."""
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

    def _submit_san(self, san: str) -> None:
        """Submit a SAN move, routing to opponent or player method."""
        assert self._coach is not None

        try:
            if self._coach.is_player_turn:
                self._coach.submit_player_move(san)
                self._refresh_after_move()
            else:
                # Stop previous analysis.
                self._coach.stop_analysis()
                self._current_handle = None

                handle = self._coach.submit_opponent_move_continuous(
                    san, self._on_analysis_update,
                )
                self._current_handle = handle
                self._refresh_after_move()

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
        update = advice.update

        if self._eval_panel is not None:
            self._eval_panel.update_eval(
                update.evaluation, update.depth_reached, is_loading=False,
            )

        if self._suggestions_panel is not None:
            self._suggestions_panel.update_suggestions(update.variations)

        if update.best_move_uci:
            self._best_move_uci = update.best_move_uci
            if self._board_widget is not None:
                # Show arrow from source to destination.
                from_sq = update.best_move_uci[:2]
                to_sq = update.best_move_uci[2:4]
                self._board_widget.highlight_squares([from_sq, to_sq])

    # -- Status management ---------------------------------------------------

    def _update_turn_status(self) -> None:
        """Update the status label to show whose turn it is."""
        assert self._coach is not None

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
            self._set_status("Your turn \u2014 click the board or type a move")
        else:
            self._set_status("Opponent's turn \u2014 enter their move", waiting=True)

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
        if self._in_review_mode:
            self._exit_review_mode()
        self._auto_save()

        if self._observing:
            self._stop_observation()

        if self._coach is not None:
            self._coach.stop()

        self._game_saved = False

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

    # -- Review mode ---------------------------------------------------------

    def _on_review_current(self) -> None:
        """Review the current game on the board."""
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

        self._summary_panel.update_summary(review)
        self._review_navigate_to(0)

        self._set_status(
            f"Review mode \u2014 {len(review.moves)} moves analysed "
            f"in {review.total_elapsed_ms / 1000:.1f}s",
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

        self._suggestions_panel.pack(fill="x", pady=(0, 6))
        self._btn_frame.pack(fill="x", pady=(0, 4))
        self._new_game_btn.pack(fill="x", pady=(0, 4))
        self._pgn_frame.pack(fill="x", pady=(0, 4))
        if self._entry_frame is not None:
            self._entry_frame.pack(side="left", fill="y", pady=4)

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
        self._set_status("Exited review mode")
        logger.info("Review mode exited.")

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
        self._board_widget.clear_highlights()

        # Show last move highlight.
        if ply > 0:
            move = self._game_review.moves[ply - 1]
            from_sq = move.uci[:2]
            to_sq = move.uci[2:4]
            self._board_widget.set_last_move(from_sq, to_sq)

            # Show best move arrow if different from played move.
            if move.best_move_uci and move.best_move_uci != move.uci:
                best_from = move.best_move_uci[:2]
                best_to = move.best_move_uci[2:4]
                self._board_widget.highlight_squares([best_from, best_to])
        else:
            self._board_widget.clear_last_move()

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
