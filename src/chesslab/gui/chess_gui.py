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
from dataclasses import dataclass, field

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
# Colour palette
# ---------------------------------------------------------------------------

_BG = "#1e1e1e"
_FG = "#e0e0e0"
_FG_DIM = "#888888"
_ACCENT = "#4fc3f7"
_ERROR = "#ef5350"

_POLL_INTERVAL_MS = 100


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
        self._root: tk.Tk | None = None
        self._coach: GameCoach | None = None
        self._current_handle: ContinuousAnalysisHandle | None = None
        self._update_queue: queue.Queue[ContinuousAdvice] = queue.Queue()

        # Widgets (populated by _build_window).
        self._board_widget: BoardWidget | None = None
        self._eval_panel: EvalPanel | None = None
        self._suggestions_panel: SuggestionsPanel | None = None
        self._history_panel: MoveHistoryPanel | None = None
        self._move_entry: tk.Entry | None = None
        self._status_label: tk.Label | None = None

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
            player_color = self._show_color_dialog(root)

        # Create and start coach.
        coach_config = CoachConfig(
            player_color=player_color,
            analysis=self._config.analysis,
        )
        self._coach = GameCoach(coach_config)
        self._coach.start()

        self._build_window(root, player_color)

        # Start polling.
        root.after(_POLL_INTERVAL_MS, self._poll_queue)

        # Update status for initial turn.
        self._update_turn_status()

        logger.info("ChessLab GUI started: player=%s", player_color.value)
        root.mainloop()

    # -- Color selection dialog ----------------------------------------------

    def _show_color_dialog(self, parent: tk.Tk) -> PlayerColor:
        """Show a dialog to choose player color."""
        result: list[PlayerColor] = [PlayerColor.WHITE]

        dialog = tk.Toplevel(parent)
        dialog.title("ChessLab — Choose Color")
        dialog.configure(bg=_BG)
        dialog.transient(parent)
        dialog.grab_set()
        dialog.resizable(False, False)

        tk.Label(
            dialog, text="Choose your color", bg=_BG, fg=_FG,
            font=("sans-serif", 14, "bold"), padx=20, pady=10,
        ).pack()

        btn_frame = tk.Frame(dialog, bg=_BG)
        btn_frame.pack(pady=10, padx=20)

        def choose(color: PlayerColor) -> None:
            result[0] = color
            dialog.destroy()

        tk.Button(
            btn_frame, text="\u2654  Play as White", font=("sans-serif", 12),
            command=lambda: choose(PlayerColor.WHITE),
            width=16, pady=8,
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame, text="\u265a  Play as Black", font=("sans-serif", 12),
            command=lambda: choose(PlayerColor.BLACK),
            width=16, pady=8,
        ).pack(side="left", padx=8)

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
        main_frame.pack(fill="both", expand=True, padx=4, pady=4)

        # Left: board.
        board_frame = tk.Frame(main_frame, bg=_BG)
        board_frame.pack(side="left", padx=(0, 8))

        self._board_widget = BoardWidget(
            board_frame,
            square_size=self._config.square_size,
            flipped=flipped,
            on_move=self._on_board_move,
        )
        self._board_widget.pack()

        # Right: panels.
        panel_frame = tk.Frame(main_frame, bg=_BG, width=280)
        panel_frame.pack(side="left", fill="both", expand=True)
        panel_frame.pack_propagate(False)

        self._eval_panel = EvalPanel(panel_frame, bar_height=140)
        self._eval_panel.pack(fill="x", pady=(0, 4))

        self._suggestions_panel = SuggestionsPanel(panel_frame)
        self._suggestions_panel.pack(fill="x", pady=(0, 4))

        self._history_panel = MoveHistoryPanel(panel_frame)
        self._history_panel.pack(fill="both", expand=True, pady=(0, 4))

        # New game button.
        tk.Button(
            panel_frame, text="New Game", font=("sans-serif", 10),
            command=self._on_new_game,
        ).pack(fill="x", padx=6, pady=(0, 4))

        # Bottom: move entry + status.
        bottom_frame = tk.Frame(root, bg=_BG)
        bottom_frame.pack(fill="x", padx=4, pady=(0, 4))

        tk.Label(
            bottom_frame, text="Move (SAN):", bg=_BG, fg=_FG,
            font=("sans-serif", 10),
        ).pack(side="left", padx=(0, 4))

        self._move_entry = tk.Entry(
            bottom_frame, font=("monospace", 12), width=12,
        )
        self._move_entry.pack(side="left", padx=(0, 4))
        self._move_entry.bind("<Return>", self._on_entry_submit)

        tk.Button(
            bottom_frame, text="Submit", font=("sans-serif", 10),
            command=self._on_entry_submit,
        ).pack(side="left", padx=(0, 8))

        self._status_label = tk.Label(
            bottom_frame, text="", bg=_BG, fg=_FG_DIM,
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
        self._history_panel.set_history(self._coach.move_history)
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

        if self._board_widget is not None and update.best_move_uci:
            # Highlight best move source and destination.
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
                self._set_status(f"Game over — {winner} wins!")
            else:
                self._set_status("Game over — Draw!")
            return

        if self._coach.is_player_turn:
            self._set_status("Your turn — click the board or type a move")
        else:
            self._set_status("Opponent's turn — enter their move")

    def _set_status(self, text: str, error: bool = False) -> None:
        """Update the status label."""
        if self._status_label is not None:
            self._status_label.config(
                text=text,
                fg=_ERROR if error else _FG_DIM,
            )

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

        assert self._root is not None
        player_color = self._show_color_dialog(self._root)

        coach_config = CoachConfig(
            player_color=player_color,
            analysis=self._config.analysis,
        )
        self._coach = GameCoach(coach_config)
        self._coach.start()

        assert self._board_widget is not None
        self._board_widget.set_flipped(player_color == PlayerColor.BLACK)
        self._board_widget.set_position(self._coach.current_fen)

        assert self._eval_panel is not None
        self._eval_panel.update_eval(None, 0, is_loading=False)

        assert self._suggestions_panel is not None
        self._suggestions_panel.update_suggestions(())

        assert self._history_panel is not None
        self._history_panel.set_history(())

        self._update_turn_status()
        logger.info("New game started: player=%s", player_color.value)

    def _on_close(self) -> None:
        """Clean up resources and close the window."""
        if self._coach is not None:
            self._coach.stop()
        if self._root is not None:
            self._root.destroy()
            self._root = None
        logger.info("ChessLab GUI closed.")
