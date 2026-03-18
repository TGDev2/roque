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
import tkinter as tk
from dataclasses import dataclass, field, replace

import chess

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
from chesslab.gui.board_widget import BoardWidget
from chesslab.gui.panels import EvalPanel, MoveHistoryPanel, SuggestionsPanel

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

        # Widgets (populated by _build_window).
        self._board_widget: BoardWidget | None = None
        self._eval_panel: EvalPanel | None = None
        self._suggestions_panel: SuggestionsPanel | None = None
        self._history_panel: MoveHistoryPanel | None = None
        self._move_entry: tk.Entry | None = None
        self._status_label: tk.Label | None = None
        self._status_icon: tk.Label | None = None

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

        elo_value_label = tk.Label(
            elo_header, text=str(initial_elo), bg=_BG_SECONDARY, fg=_ACCENT,
            font=("monospace", 14, "bold"), anchor="e",
        )
        elo_value_label.pack(side="right")

        def _on_elo_change(*_args: object) -> None:
            elo_value_label.config(text=str(elo_var.get()))

        elo_var.trace_add("write", _on_elo_change)

        scale = tk.Scale(
            elo_inner,
            from_=AnalysisConfig.MIN_ENGINE_ELO,
            to=AnalysisConfig.MAX_ENGINE_ELO,
            orient="horizontal",
            resolution=10,
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
            range_frame, text=str(AnalysisConfig.MIN_ENGINE_ELO),
            bg=_BG_SECONDARY, fg=_FG_MUTED, font=("monospace", 8),
        ).pack(side="left")
        tk.Label(
            range_frame, text=str(AnalysisConfig.MAX_ENGINE_ELO),
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
        btn_frame = tk.Frame(panel_frame, bg=_BG)
        btn_frame.pack(fill="x", pady=(0, 4))

        apply_btn = _styled_button(
            btn_frame,
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
            btn_frame,
            text="\u21a9  Undo",
            command=self._on_undo,
        )
        undo_btn.pack(side="left", fill="x", expand=True, padx=(3, 0))

        # New game button.
        new_game_btn = _styled_button(
            panel_frame,
            text="New Game",
            command=self._on_new_game,
            font=("sans-serif", 10),
        )
        new_game_btn.pack(fill="x", pady=(0, 4))

        # Bottom status bar.
        status_bar = tk.Frame(root, bg=_STATUS_BG, height=40)
        status_bar.pack(fill="x", padx=0, pady=0)
        status_bar.pack_propagate(False)

        inner_status = tk.Frame(status_bar, bg=_STATUS_BG)
        inner_status.pack(fill="both", expand=True, padx=8)

        # Move entry area.
        entry_frame = tk.Frame(inner_status, bg=_STATUS_BG)
        entry_frame.pack(side="left", fill="y", pady=4)

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

        submit_btn = _styled_button(
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
        submit_btn.pack(side="left", padx=(6, 0))

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

    # -- Move handling -------------------------------------------------------

    def _on_board_move(self, from_sq: str, to_sq: str) -> None:
        """Handle a move from board click (source, destination)."""
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
        if self._coach is not None:
            self._coach.stop()

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
        if self._coach is not None:
            self._coach.stop()
        if self._root is not None:
            self._root.destroy()
            self._root = None
        logger.info("ChessLab GUI closed.")
