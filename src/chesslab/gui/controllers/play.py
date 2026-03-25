"""Manual play controller, including Observe Live as a play sub-state."""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import chess

from chesslab.analysis import ContinuousAnalysisHandle
from chesslab.coach.game_coach import (
    ContinuousAdvice,
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
from chesslab.gui import theme
from chesslab.gui.controllers.base import (
    AppMode,
    ControllerBase,
    PlayModeRequest,
    ProgressionModeRequest,
    ReviewModeRequest,
)
from chesslab.gui.dialog_utils import (
    center_dialog,
    create_scrollable_dialog,
    finalize_scrollable_dialog,
)
from chesslab.gui.live_observer import (
    LiveObserver,
    LiveObserverConfig,
    ObservationEvent,
    ObservationEventType,
)
from chesslab.gui.live_setup_assistant import LiveSetupAssistant
from chesslab.gui.panels import (
    EvalPanel,
    MoveHistoryPanel,
    ObserveLivePanel,
    SuggestionsPanel,
)
from chesslab.live.live_position_processor import LiveFrameStatus
from chesslab.live.move_deduction import deduce_move_from_fen
from chesslab.persistence.game_storage import GameRecord, GameStorage
from chesslab.product_usage import (
    OBSERVE_LIVE_ACKNOWLEDGEMENT,
    OBSERVE_LIVE_DIALOG_INTRO,
    OBSERVE_LIVE_GUARDRAIL_BODY,
    OBSERVE_LIVE_GUARDRAIL_TITLE,
)
from chesslab.validation import CalibrationReport, SetupSupportProfile, SupportCatalogError

logger = logging.getLogger(__name__)


class PlayController(ControllerBase):
    """Own manual coaching, game persistence, and observe-live control."""

    mode = AppMode.PLAY

    def __init__(
        self,
        host: object,
        *,
        initial_player_color: PlayerColor,
    ) -> None:
        super().__init__(host)
        self._initial_player_color = initial_player_color
        self._coach = None
        self._current_handle: ContinuousAnalysisHandle | None = None
        self._update_queue: queue.Queue[ContinuousAdvice] = queue.Queue()
        self._best_move_uci: str | None = None

        self._observer: LiveObserver | None = None
        self._observation_queue: queue.Queue[ObservationEvent] = queue.Queue()
        self._observing = False
        self._last_observer_config: LiveObserverConfig | None = None
        self._live_setup_assistant: LiveSetupAssistant | None = None

        self._game_saved = False
        self._current_game_file: Path | None = None
        self._post_game_review_dialog: tk.Toplevel | None = None

        self._container: tk.Frame | None = None
        self._eval_panel: EvalPanel | None = None
        self._observe_panel: ObserveLivePanel | None = None
        self._suggestions_panel: SuggestionsPanel | None = None
        self._history_panel: MoveHistoryPanel | None = None
        self._btn_frame: tk.Frame | None = None
        self._observe_btn: tk.Button | None = None
        self._apply_btn: tk.Button | None = None
        self._undo_btn: tk.Button | None = None
        self._resign_frame: tk.Frame | None = None
        self._new_game_btn: tk.Button | None = None
        self._pgn_frame: tk.Frame | None = None

        self._entry_frame: tk.Frame | None = None
        self._move_entry: tk.Entry | None = None
        self._submit_btn: tk.Button | None = None

    @property
    def current_game_file(self) -> Path | None:
        return self._current_game_file

    def activate(self, request: object | None = None) -> None:
        self._build_widgets()
        self._mount_status_widgets()
        assert self._container is not None

        self._host.set_board_visible(True)
        self._container.pack(fill="both", expand=True)

        if self._coach is None:
            self._coach = self._host.create_coach(
                self._initial_player_color,
                self._host.engine_elo,
            )
            self.widgets.board_widget.set_flipped(
                self._coach._config.player_color == PlayerColor.BLACK
            )
            self.widgets.board_widget.set_position(self._coach.current_fen)
            self.widgets.board_widget.clear_last_move()
            self.widgets.board_widget.clear_highlights()
            if self._history_panel is not None:
                self._history_panel.set_history(())

        if isinstance(request, PlayModeRequest) and request.record is not None:
            self._load_record(request.record, status_message=request.status_message)
        else:
            self._restore_play_mode_view()
            if isinstance(request, PlayModeRequest) and request.status_message is not None:
                self._host.set_status(request.status_message)
            else:
                self._update_turn_status()

        if isinstance(request, PlayModeRequest) and request.start_observe_config is not None:
            self._start_observation(request.start_observe_config)

    def deactivate(self) -> None:
        if self._observing:
            self._stop_observation()
        self._clear_coaching_outputs()
        self._close_post_game_review_dialog()
        if self._container is not None:
            self._container.pack_forget()
        if self._entry_frame is not None:
            self._entry_frame.pack_forget()

    def poll(self) -> None:
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

    def on_board_move(self, from_sq: str, to_sq: str) -> None:
        if self._observing or self._coach is None:
            return

        uci_str = from_sq + to_sq
        try:
            move = chess.Move.from_uci(uci_str)
            board = chess.Board(self._coach.current_fen)
            san = board.san(move)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            self._host.set_status(f"Invalid move: {uci_str}", error=True)
            return

        self._submit_san(san)

    def shutdown(self) -> None:
        self._close_post_game_review_dialog()
        if self._observer is not None:
            self._observer.stop()
            self._observer = None
        if self._coach is not None:
            self._coach.stop()
            self._coach = None

    def open_observe_live_startup(self) -> LiveObserverConfig | None:
        return self._show_observe_dialog(self.widgets.root)

    def start_observation(self, config: LiveObserverConfig) -> bool:
        if self._host.current_mode is not AppMode.PLAY:
            self._host.switch_mode(AppMode.PLAY)
        return self._start_observation(config)

    def _build_widgets(self) -> None:
        if self._container is not None:
            return

        self._container = tk.Frame(self.widgets.side_host, bg=theme.BG, width=290)
        self._container.pack_propagate(False)

        self._eval_panel = EvalPanel(self._container, bar_height=160)
        self._eval_panel.pack(fill="x", pady=(0, 6))

        self._observe_panel = ObserveLivePanel(self._container)

        self._suggestions_panel = SuggestionsPanel(self._container)
        self._suggestions_panel.pack(fill="x", pady=(0, 6))

        self._history_panel = MoveHistoryPanel(self._container)
        self._history_panel.pack(fill="both", expand=True, pady=(0, 6))

        self._btn_frame = tk.Frame(self._container, bg=theme.BG)
        self._btn_frame.pack(fill="x", pady=(0, 4))

        self._apply_btn = theme.styled_button(
            self._btn_frame,
            text="▶  Apply suggestion",
            command=self._on_apply_suggestion,
            bg=theme.BTN_ACCENT_BG,
            fg=theme.BTN_ACCENT_FG,
            hover_bg=theme.BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 10, "bold"),
        )
        self._apply_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self._undo_btn = theme.styled_button(
            self._btn_frame,
            text="↩  Undo",
            command=self._on_undo,
        )
        self._undo_btn.pack(side="left", fill="x", expand=True, padx=(3, 3))

        self._observe_btn = theme.styled_button(
            self._btn_frame,
            text="◎  Observe Live",
            command=self._on_toggle_observe,
        )
        self._observe_btn.pack(side="left", fill="x", expand=True, padx=(3, 0))

        self._resign_frame = tk.Frame(self._container, bg=theme.BG)
        self._resign_frame.pack(fill="x", pady=(0, 4))

        theme.styled_button(
            self._resign_frame,
            text="White resigns",
            command=lambda: self._on_resign(PlayerColor.WHITE),
            bg=theme.BTN_WHITE_BG,
            fg=theme.BTN_WHITE_FG,
            hover_bg=theme.BTN_WHITE_HOVER,
            active_bg="#f5ede1",
            font=("sans-serif", 9, "bold"),
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))

        theme.styled_button(
            self._resign_frame,
            text="Black resigns",
            command=lambda: self._on_resign(PlayerColor.BLACK),
            bg=theme.BTN_BLACK_BG,
            fg=theme.BTN_BLACK_FG,
            hover_bg=theme.BTN_BLACK_HOVER,
            active_bg="#4a4a5c",
            font=("sans-serif", 9, "bold"),
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))

        self._new_game_btn = theme.styled_button(
            self._container,
            text="New Game",
            command=self._on_new_game,
            font=("sans-serif", 10),
        )
        self._new_game_btn.pack(fill="x", pady=(0, 4))

        theme.styled_button(
            self._container,
            text="Progression",
            command=self._open_progression_mode,
            font=("sans-serif", 10, "bold"),
        ).pack(fill="x", pady=(0, 4))

        self._pgn_frame = tk.Frame(self._container, bg=theme.BG)
        self._pgn_frame.pack(fill="x", pady=(0, 4))

        theme.styled_button(
            self._pgn_frame,
            text="Export PGN",
            command=self._on_export_pgn,
            font=("sans-serif", 9),
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))

        theme.styled_button(
            self._pgn_frame,
            text="Load PGN",
            command=self._on_load_pgn,
            font=("sans-serif", 9),
        ).pack(side="left", fill="x", expand=True, padx=(3, 3))

        theme.styled_button(
            self._pgn_frame,
            text="Paste PGN",
            command=self._open_paste_review_flow,
            font=("sans-serif", 9),
        ).pack(side="left", fill="x", expand=True, padx=(3, 3))

        theme.styled_button(
            self._pgn_frame,
            text="Recent",
            command=self._open_recent_review_flow,
            font=("sans-serif", 9),
        ).pack(side="left", fill="x", expand=True, padx=(3, 3))

        theme.styled_button(
            self._pgn_frame,
            text="Review",
            command=self._on_review_current,
            font=("sans-serif", 9),
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))

        self._entry_frame = tk.Frame(self.widgets.status_host, bg=theme.STATUS_BG)
        tk.Label(
            self._entry_frame,
            text="Move",
            bg=theme.STATUS_BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
        ).pack(side="left", padx=(0, 6))

        entry_border = tk.Frame(self._entry_frame, bg=theme.ENTRY_BORDER)
        entry_border.pack(side="left")

        self._move_entry = tk.Entry(
            entry_border,
            font=("monospace", 12),
            width=8,
            bg=theme.ENTRY_BG,
            fg=theme.ENTRY_FG,
            insertbackground=theme.ACCENT,
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        self._move_entry.pack(padx=1, pady=1, ipady=2)
        self._move_entry.bind("<Return>", self._on_entry_submit)
        self._move_entry.bind(
            "<FocusIn>",
            lambda _event: entry_border.config(bg=theme.ENTRY_FOCUS_BORDER),
        )
        self._move_entry.bind(
            "<FocusOut>",
            lambda _event: entry_border.config(bg=theme.ENTRY_BORDER),
        )

        self._submit_btn = theme.styled_button(
            self._entry_frame,
            text="Submit",
            command=self._on_entry_submit,
            bg=theme.BTN_ACCENT_BG,
            fg=theme.BTN_ACCENT_FG,
            hover_bg=theme.BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 9, "bold"),
            padx=10,
            pady=3,
        )
        self._submit_btn.pack(side="left", padx=(6, 0))

    def _mount_status_widgets(self) -> None:
        assert self._entry_frame is not None
        self._entry_frame.pack(side="left", fill="y", pady=4)

    def _restore_play_mode_view(self) -> None:
        if self._coach is None:
            return

        self._set_observe_mode_ui(self._observing)
        self.widgets.board_widget.set_position(self._coach.current_fen)
        self.widgets.board_widget.clear_last_move()
        self.widgets.board_widget.clear_highlights()
        self.widgets.board_widget.set_interactive(not self._observing)

        if self._history_panel is not None:
            self._history_panel.set_history(self._coach.move_history)
        if self._eval_panel is not None:
            self._eval_panel.update_eval(None, 0, is_loading=False)
        if self._suggestions_panel is not None:
            self._suggestions_panel.update_suggestions(())
        self._update_turn_status()

    def _open_recent_review_flow(self) -> None:
        self._close_post_game_review_dialog()
        self._host.switch_mode(
            AppMode.REVIEW,
            ReviewModeRequest(
                action="open_recent",
                return_mode=AppMode.PLAY,
            ),
        )

    def _open_paste_review_flow(self) -> None:
        self._close_post_game_review_dialog()
        self._host.switch_mode(
            AppMode.REVIEW,
            ReviewModeRequest(
                action="open_paste",
                return_mode=AppMode.PLAY,
            ),
        )

    def _open_progression_mode(self) -> None:
        self._host.switch_mode(
            AppMode.PROGRESSION,
            ProgressionModeRequest(
                sample_size=self._host.progression_sample_size,
                return_mode=AppMode.PLAY,
            ),
        )

    def _on_review_current(self) -> None:
        if self._coach is None:
            self._host.set_status("No game to review", error=True)
            return

        if self._observing:
            self._exit_observe_to_review()
            return

        board = self._coach.board
        if not board.move_stack:
            self._host.set_status("No moves to review", error=True)
            return

        self._close_post_game_review_dialog()
        self._host.switch_mode(
            AppMode.REVIEW,
            ReviewModeRequest(
                action="review_board",
                board=board.copy(),
                return_mode=AppMode.PLAY,
                source_path=self._current_game_file,
            ),
        )

    def _on_entry_submit(self, event: tk.Event | None = None) -> None:  # type: ignore[type-arg]
        del event
        if self._observing:
            self._host.set_status(
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
        if self._observing:
            self._host.set_status(
                "Observe Live is passive. Exit to Manual Coach before using coaching actions.",
                error=True,
            )
            return
        if self._coach is None:
            return
        if self._best_move_uci is None:
            self._host.set_status("No suggestion available", error=True)
            return
        if not self._coach.is_player_turn:
            self._host.set_status("Not your turn", error=True)
            return

        try:
            move = chess.Move.from_uci(self._best_move_uci)
            board = chess.Board(self._coach.current_fen)
            san = board.san(move)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            self._host.set_status(f"Suggestion invalid: {self._best_move_uci}", error=True)
            return

        self._best_move_uci = None
        self._submit_san(san)

    def _on_resign(self, resigning_color: PlayerColor) -> None:
        if self._observing:
            self._host.set_status(
                "Observe Live is passive. Exit to Manual Coach before ending the game.",
                error=True,
            )
            return
        if self._coach is None:
            return

        color_label = "White" if resigning_color == PlayerColor.WHITE else "Black"
        confirmed = messagebox.askyesno(
            "Confirm resignation",
            f"End this Manual Coach game with {color_label} resigning?",
            parent=self.widgets.root,
        )
        if not confirmed:
            return

        try:
            self._coach.record_resignation(resigning_color)
        except GameOverError as exc:
            self._host.set_status(str(exc), error=True)
            return

        self._clear_coaching_outputs()
        self._update_turn_status()
        self._handle_finished_manual_game()

    def _on_undo(self) -> None:
        if self._observing:
            self._host.set_status(
                "Observe Live is passive. Exit to Manual Coach before changing moves.",
                error=True,
            )
            return
        if self._coach is None:
            return

        san = self._coach.undo_move()
        if san is None:
            self._host.set_status("Nothing to undo", error=True)
            return

        self._best_move_uci = None
        self._current_handle = None
        self.widgets.board_widget.set_position(self._coach.current_fen)
        self.widgets.board_widget.clear_last_move()
        self.widgets.board_widget.clear_highlights()
        if self._history_panel is not None:
            self._history_panel.set_history(self._coach.move_history)
        if self._eval_panel is not None:
            self._eval_panel.update_eval(None, 0, is_loading=False)
        if self._suggestions_panel is not None:
            self._suggestions_panel.update_suggestions(())

        self._update_turn_status()
        logger.info("Undo: %s", san or "manual game termination")

    def _submit_san(self, san: str, *, passive: bool = False) -> None:
        if self._coach is None:
            return

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
                self._coach.stop_analysis()
                self._current_handle = None

                handle = self._coach.submit_opponent_move_continuous(
                    san,
                    self._on_analysis_update,
                )
                self._current_handle = handle
                self._refresh_after_move()

                if self._coach.is_game_over:
                    self._handle_finished_manual_game()
                    return

                if self._eval_panel is not None:
                    self._eval_panel.update_eval(None, 0, is_loading=True)

        except IllegalMoveError as exc:
            self._host.set_status(str(exc), error=True)
        except NotYourTurnError as exc:
            self._host.set_status(str(exc), error=True)
        except GameOverError as exc:
            self._host.set_status(str(exc), error=True)

    def _on_analysis_update(self, advice: ContinuousAdvice) -> None:
        self._update_queue.put(advice)

    def _refresh_after_move(self) -> None:
        if self._coach is None or self._history_panel is None:
            return

        self.widgets.board_widget.set_position(self._coach.current_fen)

        history = self._coach.move_history
        if history:
            board = chess.Board()
            for san_move in history:
                move = board.parse_san(san_move)
                from_sq = chess.square_name(move.from_square)
                to_sq = chess.square_name(move.to_square)
                board.push(move)
            self.widgets.board_widget.set_last_move(from_sq, to_sq)

        self._history_panel.set_history(history)
        self._update_turn_status()

    def _apply_advice(self, advice: ContinuousAdvice) -> None:
        if self._observing:
            return

        update = advice.update
        if self._eval_panel is not None:
            self._eval_panel.update_eval(
                update.evaluation,
                update.depth_reached,
                is_loading=False,
            )
        if self._suggestions_panel is not None:
            self._suggestions_panel.update_suggestions(update.variations)

        self._best_move_uci = update.best_move_uci or None
        self._set_board_move_hint(self._best_move_uci, color=theme.COACH_MOVE_HINT)

    def _update_turn_status(self) -> None:
        if self._coach is None:
            return

        if self._observing:
            self._refresh_observe_panel()
            self._host.set_status("Observe Live active — passive sync only.")
            return

        if self._coach.is_game_over:
            self._auto_save()
            winner = self._coach.game_winner
            termination = self._coach.game_termination
            if winner is not None:
                winner_label = "White" if winner == chess.WHITE else "Black"
                if termination is not None and "resignation" in termination.lower():
                    self._host.set_status(
                        f"Game over — {winner_label} wins by resignation!",
                        game_over=True,
                    )
                else:
                    self._host.set_status(
                        f"Game over — {winner_label} wins!",
                        game_over=True,
                    )
            else:
                self._host.set_status("Game over — Draw!", game_over=True)
            return

        if self._coach.is_player_turn:
            self._host.set_status("Manual Coach — your turn: click the board or type a move")
        else:
            self._host.set_status(
                "Manual Coach — opponent's turn: enter their move",
                waiting=True,
            )

    def _handle_finished_manual_game(self) -> None:
        if self._coach is None:
            return
        board = self._coach.board
        result_text = self._current_game_result_text()
        if not board.move_stack:
            return

        self._clear_coaching_outputs()
        self._auto_save()
        self._propose_manual_review(board, result_text=result_text)

    def _drain_analysis_queue(self) -> None:
        while not self._update_queue.empty():
            try:
                self._update_queue.get_nowait()
            except queue.Empty:
                break

    def _clear_coaching_outputs(self) -> None:
        if self._coach is not None:
            self._coach.stop_analysis()
        self._current_handle = None
        self._best_move_uci = None
        self._drain_analysis_queue()
        self.widgets.board_widget.clear_highlights()
        if self._eval_panel is not None:
            self._eval_panel.update_eval(None, 0, is_loading=False)
        if self._suggestions_panel is not None:
            self._suggestions_panel.update_suggestions(())

    def _set_board_move_hint(self, best_move_uci: str | None, *, color: str) -> None:
        squares = self._best_move_squares(best_move_uci)
        if squares is None:
            self.widgets.board_widget.clear_highlights()
            return
        self.widgets.board_widget.set_move_hint(*squares, color=color)

    @staticmethod
    def _best_move_squares(best_move_uci: str | None) -> tuple[str, str] | None:
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

    def _on_new_game(self) -> None:
        self._close_post_game_review_dialog()
        self._auto_save()
        if self._observing:
            self._stop_observation()

        if self._coach is not None:
            self._coach.stop()

        self._game_saved = False
        self._current_game_file = None
        while not self._update_queue.empty():
            try:
                self._update_queue.get_nowait()
            except queue.Empty:
                break

        self._current_handle = None
        self._best_move_uci = None

        player_color, engine_elo = self._host.show_color_dialog(
            default_engine_elo=self._host.engine_elo,
        )
        self._host.engine_elo = engine_elo
        self._coach = self._host.create_coach(player_color, engine_elo)

        self.widgets.board_widget.set_flipped(player_color == PlayerColor.BLACK)
        self.widgets.board_widget.clear_last_move()
        self.widgets.board_widget.clear_highlights()
        self.widgets.board_widget.set_position(self._coach.current_fen)

        if self._eval_panel is not None:
            self._eval_panel.update_eval(None, 0, is_loading=False)
        if self._suggestions_panel is not None:
            self._suggestions_panel.update_suggestions(())
        if self._history_panel is not None:
            self._history_panel.set_history(())

        self._update_turn_status()
        logger.info(
            "New game started: player=%s elo=%d",
            player_color.value,
            engine_elo,
        )

    def _load_record(
        self,
        record: GameRecord,
        *,
        status_message: str | None = None,
    ) -> None:
        if self._coach is None:
            return

        self._close_post_game_review_dialog()
        self._auto_save()
        self._coach.load_board(record.board)
        self._game_saved = record.file_path is not None
        self._current_game_file = record.file_path
        self._best_move_uci = None
        self._current_handle = None

        self.widgets.board_widget.set_position(self._coach.current_fen)
        self.widgets.board_widget.clear_last_move()
        self.widgets.board_widget.clear_highlights()
        if self._history_panel is not None:
            self._history_panel.set_history(self._coach.move_history)
        if self._eval_panel is not None:
            self._eval_panel.update_eval(None, 0, is_loading=False)
        if self._suggestions_panel is not None:
            self._suggestions_panel.update_suggestions(())
        self._update_turn_status()

        if status_message is not None:
            self._host.set_status(status_message)

    def _on_load_pgn(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")],
            title="Load PGN",
        )
        if not path:
            return

        try:
            record = GameStorage.load_game(Path(path))
        except Exception as exc:
            self._host.set_status(f"Load failed: {exc}", error=True)
            logger.exception("Failed to load PGN.")
            return

        self._load_record(record, status_message=f"Loaded: {Path(path).name}")

    def _on_export_pgn(self) -> None:
        if self._coach is None or not self._coach.board.move_stack:
            self._host.set_status("No moves to export", error=True)
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".pgn",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")],
            title="Export PGN",
        )
        if not path:
            return

        try:
            GameStorage.export_game(self._coach.board, self._build_headers(), Path(path))
            self._host.set_status("PGN exported")
        except Exception:
            logger.exception("Failed to export PGN.")
            self._host.set_status("Export failed", error=True)

    def _build_headers(self) -> dict[str, str]:
        assert self._coach is not None
        player_color = self._coach._config.player_color
        from chesslab.persistence.game_storage import build_headers

        headers = build_headers(
            player_color=player_color,
            engine_elo=self._host.engine_elo,
            result=self._coach.game_result,
        )
        if self._coach.game_termination is not None:
            headers["Termination"] = self._coach.game_termination
        return headers

    def _build_observe_headers(self, board: chess.Board) -> dict[str, str]:
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
        if self._game_saved or self._coach is None:
            return
        board = self._coach.board
        if not board.move_stack:
            return

        try:
            path = self.services.storage.save_game(board, self._build_headers())
            if path is not None:
                self._game_saved = True
                self._current_game_file = path
                logger.info("Auto-saved game: %s", path)
        except Exception:
            logger.exception("Failed to auto-save game.")

    def _current_game_result_text(self) -> str:
        if self._coach is None:
            return "incomplete"

        result = self._coach.game_result
        termination = self._coach.game_termination
        if termination is not None:
            return f"{result} ({termination})"
        return result if result != "*" else "incomplete"

    def _show_observe_dialog(self, parent: tk.Tk) -> LiveObserverConfig | None:
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

        layout = create_scrollable_dialog(
            parent,
            title="Observe Live",
            padx=24,
            pady=20,
        )
        dialog = layout.dialog
        container = layout.container

        tk.Label(
            container,
            text="Observe Live",
            bg=theme.BG,
            fg=theme.FG,
            font=("sans-serif", 18, "bold"),
        ).pack(pady=(0, 6))

        tk.Label(
            container,
            text=OBSERVE_LIVE_DIALOG_INTRO,
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=500,
        ).pack(fill="x", pady=(0, 14))

        guardrail_card = tk.Frame(container, bg=theme.CARD_BORDER)
        guardrail_card.pack(fill="x", pady=(0, 12))
        guardrail_inner = tk.Frame(guardrail_card, bg=theme.BG_SECONDARY)
        guardrail_inner.pack(fill="both", expand=True, padx=1, pady=1)

        tk.Label(
            guardrail_inner,
            text="Usage Guardrail",
            bg=theme.BG_SECONDARY,
            fg=theme.ACCENT,
            font=("sans-serif", 9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))

        tk.Label(
            guardrail_inner,
            text=OBSERVE_LIVE_GUARDRAIL_TITLE,
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("sans-serif", 11, "bold"),
            justify="left",
            wraplength=480,
            anchor="w",
        ).pack(fill="x", padx=12)

        tk.Label(
            guardrail_inner,
            text=OBSERVE_LIVE_GUARDRAIL_BODY,
            bg=theme.BG_SECONDARY,
            fg=theme.FG_DIM,
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
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            selectcolor=theme.BG_SECONDARY,
            activebackground=theme.BG_SECONDARY,
            activeforeground=theme.FG,
            highlightthickness=0,
            justify="left",
            anchor="w",
            wraplength=470,
            font=("sans-serif", 9, "bold"),
        ).pack(fill="x", padx=12, pady=(0, 10))

        monitor_card = tk.Frame(container, bg=theme.CARD_BORDER)
        monitor_card.pack(fill="x", pady=(0, 12))
        monitor_inner = tk.Frame(monitor_card, bg=theme.BG_SECONDARY)
        monitor_inner.pack(fill="both", expand=True, padx=1, pady=1)

        tk.Label(
            monitor_inner,
            text="Monitor",
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("sans-serif", 11, "bold"),
            anchor="w",
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

        monitor_labels = [f"Monitor {m.index}: {m.width}x{m.height}" for m in monitors]
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
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("sans-serif", 10),
            highlightthickness=0,
            relief="flat",
            activebackground=theme.BTN_HOVER,
            activeforeground=theme.FG,
        )
        monitor_menu["menu"].config(
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            activebackground=theme.ACCENT,
            activeforeground="#ffffff",
        )
        monitor_menu.pack(fill="x", padx=12, pady=(0, 8))

        if monitor_error is not None:
            tk.Label(
                monitor_inner,
                text=monitor_error,
                bg=theme.BG_SECONDARY,
                fg=theme.ERROR,
                font=("sans-serif", 9),
                justify="left",
                wraplength=360,
            ).pack(fill="x", padx=12, pady=(0, 8))

        setup_card = tk.Frame(container, bg=theme.CARD_BORDER)
        setup_card.pack(fill="x", pady=(0, 12))
        setup_inner = tk.Frame(setup_card, bg=theme.BG_SECONDARY)
        setup_inner.pack(fill="both", expand=True, padx=1, pady=1)

        tk.Label(
            setup_inner,
            text="Declared Setup",
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("sans-serif", 11, "bold"),
            anchor="w",
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
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("sans-serif", 10),
            highlightthickness=0,
            relief="flat",
            activebackground=theme.BTN_HOVER,
            activeforeground=theme.FG,
        )
        setup_menu["menu"].config(
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            activebackground=theme.ACCENT,
            activeforeground="#ffffff",
        )
        setup_menu.pack(fill="x", padx=12, pady=(0, 8))

        if assistant_error is not None:
            tk.Label(
                setup_inner,
                text=assistant_error,
                bg=theme.BG_SECONDARY,
                fg=theme.ERROR,
                font=("sans-serif", 9),
                justify="left",
                wraplength=480,
            ).pack(fill="x", padx=12, pady=(0, 8))

        details_card = tk.Frame(container, bg=theme.CARD_BORDER)
        details_card.pack(fill="x", pady=(0, 12))
        details_inner = tk.Frame(details_card, bg=theme.BG_SECONDARY)
        details_inner.pack(fill="both", expand=True, padx=1, pady=1)

        support_badge = tk.Label(
            details_inner,
            text="SETUP",
            bg=theme.BTN_BG,
            fg=theme.FG,
            font=("sans-serif", 9, "bold"),
            padx=8,
            pady=3,
        )
        support_badge.pack(anchor="w", padx=12, pady=(10, 6))

        summary_label = tk.Label(
            details_inner,
            text="",
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("sans-serif", 11, "bold"),
            justify="left",
            wraplength=480,
            anchor="w",
        )
        summary_label.pack(fill="x", padx=12)

        support_copy_label = tk.Label(
            details_inner,
            text="",
            bg=theme.BG_SECONDARY,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=480,
            anchor="w",
        )
        support_copy_label.pack(fill="x", padx=12, pady=(4, 0))

        expected_label = tk.Label(
            details_inner,
            text="",
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("sans-serif", 9),
            justify="left",
            wraplength=480,
            anchor="w",
        )
        expected_label.pack(fill="x", padx=12, pady=(8, 0))

        notes_label = tk.Label(
            details_inner,
            text="",
            bg=theme.BG_SECONDARY,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=480,
            anchor="w",
        )
        notes_label.pack(fill="x", padx=12, pady=(6, 10))

        preflight_card = tk.Frame(container, bg=theme.CARD_BORDER)
        preflight_card.pack(fill="x", pady=(0, 12))
        preflight_inner = tk.Frame(preflight_card, bg=theme.BG_SECONDARY)
        preflight_inner.pack(fill="both", expand=True, padx=1, pady=1)

        verdict_badge = tk.Label(
            preflight_inner,
            text="PREFLIGHT",
            bg=theme.BTN_BG,
            fg=theme.FG,
            font=("sans-serif", 9, "bold"),
            padx=8,
            pady=3,
        )
        verdict_badge.pack(anchor="w", padx=12, pady=(10, 6))

        verdict_title = tk.Label(
            preflight_inner,
            text="",
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("sans-serif", 11, "bold"),
            anchor="w",
            justify="left",
            wraplength=480,
        )
        verdict_title.pack(fill="x", padx=12)

        verdict_body = tk.Label(
            preflight_inner,
            text="",
            bg=theme.BG_SECONDARY,
            fg=theme.FG_DIM,
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
            title_fg: str = theme.FG,
            body_fg: str = theme.FG_DIM,
        ) -> None:
            verdict_badge.config(text=badge_text, bg=badge_bg, fg=badge_fg)
            verdict_title.config(text=title, fg=title_fg)
            verdict_body.config(text=body, fg=body_fg)

        def refresh_details() -> None:
            profile = selected_profile()
            if profile is None:
                support_badge.config(text="SETUP", bg=theme.BTN_BG, fg=theme.FG)
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

            badge_bg, badge_fg = self._support_badge_palette(profile.support_level.value)
            support_badge.config(
                text=profile.support_level.value.upper(),
                bg=badge_bg,
                fg=badge_fg,
            )
            summary_label.config(text=self._format_live_setup_summary(profile))
            support_copy_label.config(text=self._support_level_copy(profile.support_level.value))
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
            can_preflight = selected_monitor_index() is not None and selected_profile() is not None
            can_start = can_preflight and acknowledgement_var.get()
            preflight_btn.config(state="normal" if can_preflight else "disabled")
            start_btn.config(state="normal" if can_start else "disabled")

            if assistant_error is not None:
                set_verdict("BLOCKED", "Setup catalog unavailable", assistant_error, badge_bg=theme.ERROR)
                return

            if selected_profile() is None:
                set_verdict(
                    "BLOCKED",
                    "No declared live setup available",
                    "Add a supported setup profile before starting Observe Live.",
                    badge_bg=theme.ERROR,
                )
                return

            if monitor_error is not None or selected_monitor_index() is None:
                set_verdict(
                    "BLOCKED",
                    "Monitor selection required",
                    monitor_error or "Pick a valid monitor before running preflight.",
                    badge_bg=theme.ERROR,
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
                badge_bg=theme.ACCENT_DARK,
            )

        def render_report(profile: SetupSupportProfile, report: CalibrationReport) -> None:
            if report.passed:
                set_verdict(
                    "PASS",
                    "Setup verified for live observation",
                    self._format_preflight_report(profile, report),
                    badge_bg=theme.SUCCESS,
                )
            else:
                set_verdict(
                    "FAIL",
                    "Fix blockers before starting live observation",
                    self._format_preflight_report(profile, report),
                    badge_bg=theme.ERROR,
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
                badge_bg=theme.ACCENT,
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
                    badge_bg=theme.ERROR,
                )
                return None
            finally:
                cancel_btn.config(state="normal")
                can_preflight = selected_monitor_index() is not None and selected_profile() is not None
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

        ok_frame = tk.Frame(container, bg=theme.BG)
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

        preflight_btn = theme.styled_button(
            ok_frame,
            text="Run Preflight",
            command=run_preflight,
            font=("sans-serif", 11),
            pady=8,
        )
        preflight_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        start_btn = theme.styled_button(
            ok_frame,
            text="Preflight & Start",
            command=on_start,
            bg=theme.BTN_ACCENT_BG,
            fg=theme.BTN_ACCENT_FG,
            hover_bg=theme.BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 11, "bold"),
            pady=8,
        )
        start_btn.pack(side="left", fill="x", expand=True, padx=(4, 4))

        cancel_btn = theme.styled_button(
            ok_frame,
            text="Cancel",
            command=dialog.destroy,
            font=("sans-serif", 11),
            pady=8,
        )
        cancel_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        refresh_details()
        refresh_preflight_state()
        monitor_var.trace_add("write", lambda *_args: (refresh_details(), refresh_preflight_state()))
        setup_var.trace_add("write", lambda *_args: (refresh_details(), refresh_preflight_state()))
        acknowledgement_var.trace_add("write", lambda *_args: refresh_preflight_state())

        finalize_scrollable_dialog(parent, layout)
        parent.wait_window(dialog)
        return result[0]

    def _get_live_setup_assistant(self) -> LiveSetupAssistant:
        if self._live_setup_assistant is None:
            self._live_setup_assistant = LiveSetupAssistant()
        return self._live_setup_assistant

    @staticmethod
    def _support_badge_palette(level: str) -> tuple[str, str]:
        normalized = level.strip().lower()
        if normalized == "official":
            return "#1e6f43", "#ffffff"
        if normalized == "candidate":
            return "#7a5a00", "#ffffff"
        if normalized == "experimental":
            return "#6d2f8b", "#ffffff"
        return theme.BTN_BG, theme.FG

    @staticmethod
    def _support_level_copy(level: str) -> str:
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
        return (
            f"{profile.platform} / {profile.site_theme} / "
            f"{profile.perspective.title()} perspective"
        )

    @staticmethod
    def _format_preflight_report(
        profile: SetupSupportProfile,
        report: CalibrationReport,
    ) -> str:
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
            return f"Preflight unavailable: {exc} Select a valid monitor and retry."
        if isinstance(exc, ScreenCaptureError):
            return f"Preflight failed: {exc}"
        return f"Preflight failed: {exc}"

    def _on_toggle_observe(self) -> None:
        if self._observing:
            self._stop_observation()
            return
        config = self._show_observe_dialog(self.widgets.root)
        if config is not None:
            self._start_observation(config)

    def _start_observation(self, config: LiveObserverConfig) -> bool:
        if config.setup_id:
            self._host.set_status(
                f"Starting Observe Live for {config.setup_id}...",
                waiting=True,
            )
        else:
            self._host.set_status("Starting Observe Live...", waiting=True)

        while not self._observation_queue.empty():
            try:
                self._observation_queue.get_nowait()
            except queue.Empty:
                break

        observer = LiveObserver(config, self._observation_queue)
        try:
            observer.start()
        except Exception as exc:
            self._host.set_status(self._format_observe_start_error(exc), error=True)
            logger.error("Failed to start live observer: %s", exc)
            return False

        self._observer = observer
        self._observing = True
        self._last_observer_config = config

        self._clear_coaching_outputs()
        self._set_observe_mode_ui(True)
        self._refresh_observe_panel()

        if config.setup_id:
            support = f" ({config.support_level})" if config.support_level is not None else ""
            self._host.set_status(
                f"Preflight PASS — Observe Live active for {config.setup_id}{support}. Passive sync only."
            )
        else:
            self._host.set_status("Observe Live active — passive sync only.")
        logger.info("Live observation started.")
        return True

    def _stop_observation(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer = None
        self._observing = False
        self._set_observe_mode_ui(False)
        self._update_turn_status()
        logger.info("Live observation stopped.")

    @staticmethod
    def _format_observe_start_error(exc: Exception) -> str:
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
            return f"Observation unavailable: {exc} Select a valid monitor in the Observe dialog."
        if isinstance(exc, ScreenCaptureError):
            return f"Observation startup failed: {exc}"
        return f"Observation startup failed: {exc}"

    @staticmethod
    def _format_observation_error(event: ObservationEvent) -> str:
        if event.frame_status is LiveFrameStatus.BOARD_NOT_FOUND:
            return (
                "No board detected. Keep the chessboard fully visible on screen "
                "and verify the selected monitor."
            )
        if event.frame_status is LiveFrameStatus.THEME_UNRECOGNIZED:
            return "Theme not recognized. Pick a supported board theme from Observe settings."
        if event.frame_status is LiveFrameStatus.THEME_UNSUPPORTED:
            return "Theme unsupported. Switch to a built-in theme and restart observation."
        if event.frame_status is LiveFrameStatus.POSITION_UNCERTAIN:
            return "Position uncertain. Improve capture quality (zoom, contrast, board focus)."
        if event.frame_status is LiveFrameStatus.INVALID_POSITION:
            return "Detected position is invalid. Wait for a stable frame or restart observation."
        return event.message or "Live observation error."

    def _handle_observation_event(self, event: ObservationEvent) -> None:
        if self._coach is None:
            return

        if event.type is ObservationEventType.TRACKING_SYNC:
            if event.fen is not None:
                self._sync_coach_to_fen(event.fen)
            self._host.set_status(event.message or "Live tracking synchronized.")
            return

        if event.type is ObservationEventType.MOVE_DETECTED:
            if event.tracking is not None and event.tracking.move_san is not None:
                self._submit_san(event.tracking.move_san, passive=True)
                if self._coach.current_fen != event.tracking.fen:
                    logger.warning("Live tracking drift detected, resynchronizing coach board.")
                    self._sync_coach_to_fen(event.tracking.fen)
                    self._host.set_status("Live state resynchronized.")
                return

            if event.piece_placement is None:
                return
            board = chess.Board(self._coach.current_fen)
            move = deduce_move_from_fen(board, event.piece_placement)
            if move is not None:
                san = board.san(move)
                self._submit_san(san, passive=True)
            else:
                self._host.set_status("Position changed — no legal move matches", error=True)
            return

        if event.type is ObservationEventType.NEW_GAME:
            finished_board = self._coach.board
            if finished_board.move_stack:
                finished_board = finished_board.copy()

            if event.fen is not None:
                self._sync_coach_to_fen(event.fen, auto_save_current=True)
            self._host.set_status(event.tracking.message if event.tracking else "New game detected.")

            if finished_board.move_stack:
                self._propose_observe_review(finished_board)
            return

        if event.type is ObservationEventType.RESYNC:
            if event.fen is not None:
                self._sync_coach_to_fen(event.fen, auto_save_current=True)
            self._host.set_status(event.message or "Live tracking resynchronized.")
            return

        if event.type is ObservationEventType.ERROR:
            self._host.set_status(self._format_observation_error(event), error=True)
            return

        if event.type is ObservationEventType.STOPPED:
            self._stop_observation()
            self._host.set_status(
                event.message or "Live observation stopped unexpectedly.",
                error=True,
            )

    def _refresh_observe_panel(self) -> None:
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
        self.widgets.board_widget.set_interactive(not observing)

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
            if self._observe_panel is not None and self._history_panel is not None:
                self._observe_panel.pack(fill="x", pady=(0, 6), before=self._history_panel)
            if self._btn_frame is not None and self._history_panel is not None:
                self._btn_frame.pack(fill="x", pady=(0, 4), after=self._history_panel)
            if self._resign_frame is not None:
                self._resign_frame.pack_forget()
            if self._new_game_btn is not None:
                self._new_game_btn.pack_forget()
            if self._pgn_frame is not None:
                self._pgn_frame.pack_forget()
            if self._entry_frame is not None:
                self._entry_frame.pack_forget()

            if self._apply_btn is not None:
                self._apply_btn.config(text="Manual Coach", command=self._exit_observe_to_manual_coach)
            if self._undo_btn is not None:
                self._undo_btn.config(text="Stop Live", command=self._stop_observation_with_review_proposal)
            if self._observe_btn is not None:
                self._observe_btn.config(text="Review", command=self._exit_observe_to_review)
            return

        if self._observe_panel is not None:
            self._observe_panel.pack_forget()
        if self._eval_panel is not None and self._history_panel is not None:
            self._eval_panel.pack(fill="x", pady=(0, 6), before=self._history_panel)
        if self._suggestions_panel is not None and self._history_panel is not None:
            self._suggestions_panel.pack(fill="x", pady=(0, 6), before=self._history_panel)
        if self._btn_frame is not None and self._history_panel is not None:
            self._btn_frame.pack(fill="x", pady=(0, 4), after=self._history_panel)
        if self._resign_frame is not None and self._btn_frame is not None:
            self._resign_frame.pack(fill="x", pady=(0, 4), after=self._btn_frame)
        if self._new_game_btn is not None and self._resign_frame is not None:
            self._new_game_btn.pack(fill="x", pady=(0, 4), after=self._resign_frame)
        if self._pgn_frame is not None and self._new_game_btn is not None:
            self._pgn_frame.pack(fill="x", pady=(0, 4), after=self._new_game_btn)
        self._mount_status_widgets()

        if self._apply_btn is not None:
            self._apply_btn.config(text="▶  Apply suggestion", command=self._on_apply_suggestion)
        if self._undo_btn is not None:
            self._undo_btn.config(text="↩  Undo", command=self._on_undo)
        if self._observe_btn is not None:
            self._observe_btn.config(text="◎  Observe Live", command=self._on_toggle_observe)

    def _exit_observe_to_manual_coach(self) -> None:
        if not self._observing:
            return
        self._stop_observation()
        self._host.set_status(
            "Observe Live stopped — Manual Coach restored. Enter moves yourself to continue."
        )

    def _stop_observation_with_review_proposal(self) -> None:
        if self._coach is not None:
            board = self._coach.board
            has_moves = bool(board.move_stack)
            if has_moves:
                board = board.copy()
        else:
            has_moves = False

        self._stop_observation()
        self._host.set_status("Live observation stopped.")

        if has_moves:
            self._propose_observe_review(board)

    def _exit_observe_to_review(self, board: chess.Board | None = None) -> None:
        if board is None:
            if self._coach is None:
                self._host.set_status("No game to review", error=True)
                return
            board = self._coach.board

        if not board.move_stack:
            self._host.set_status(
                "No synced moves to review yet. Keep observing or exit to Manual Coach.",
                error=True,
            )
            return

        source_path: Path | None = None
        try:
            headers = self._build_observe_headers(board)
            path = self.services.storage.save_game(board, headers)
            if path is not None:
                source_path = path
                logger.info("Observed game saved: %s", path)
        except Exception:
            logger.exception("Failed to save observed game before review.")

        if self._observing:
            self._stop_observation()

        self._host.switch_mode(
            AppMode.REVIEW,
            ReviewModeRequest(
                action="review_board",
                board=board.copy(),
                return_mode=AppMode.PLAY,
                from_observation=True,
                source_path=source_path,
            ),
        )

    def _propose_observe_review(self, board: chess.Board) -> None:
        if not board.move_stack:
            return

        dialog = tk.Toplevel(self.widgets.root)
        dialog.title("Game finished")
        dialog.configure(bg=theme.BG)
        dialog.transient(self.widgets.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        container = tk.Frame(dialog, bg=theme.BG)
        container.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(
            container,
            text="Analyse this game?",
            bg=theme.BG,
            fg=theme.FG,
            font=("sans-serif", 13, "bold"),
        ).pack(pady=(0, 6))

        outcome = board.outcome()
        result_text = board.result() if outcome is not None else "incomplete"
        tk.Label(
            container,
            text=f"{len(board.move_stack)} moves captured — {result_text}",
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 10),
        ).pack(pady=(0, 14))

        btn_row = tk.Frame(container, bg=theme.BG)
        btn_row.pack(fill="x")

        theme.styled_button(
            btn_row,
            text="▶  Analyse this game",
            command=lambda: (dialog.destroy(), self._exit_observe_to_review(board)),
            bg=theme.BTN_ACCENT_BG,
            fg=theme.BTN_ACCENT_FG,
            hover_bg=theme.BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 10, "bold"),
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        theme.styled_button(
            btn_row,
            text="Continue observing",
            command=dialog.destroy,
            font=("sans-serif", 10),
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        center_dialog(self.widgets.root, dialog)

    def _sync_coach_to_fen(
        self,
        fen: str,
        *,
        auto_save_current: bool = False,
    ) -> bool:
        if self._coach is None:
            return False

        try:
            board = chess.Board(fen)
        except ValueError as exc:
            logger.error("Cannot sync live board from FEN '%s': %s", fen, exc)
            self._host.set_status(f"Live sync failed: {exc}", error=True)
            return False

        if auto_save_current:
            self._auto_save()

        self._coach.stop_analysis()
        self._current_handle = None
        self._best_move_uci = None
        self._coach.load_board(board)
        self._game_saved = False

        self.widgets.board_widget.set_position(self._coach.current_fen)
        self.widgets.board_widget.clear_last_move()
        self.widgets.board_widget.clear_highlights()
        if self._history_panel is not None:
            self._history_panel.set_history(self._coach.move_history)
        if self._eval_panel is not None:
            self._eval_panel.update_eval(None, 0, is_loading=False)
        if self._suggestions_panel is not None:
            self._suggestions_panel.update_suggestions(())
        self._refresh_observe_panel()
        self._update_turn_status()
        return True

    def _propose_manual_review(
        self,
        board: chess.Board,
        *,
        result_text: str | None = None,
    ) -> None:
        if not board.move_stack:
            return

        self._close_post_game_review_dialog()

        move_count = len(board.move_stack)
        summary_text = result_text if result_text is not None else (
            board.result() if board.outcome() is not None else "incomplete"
        )

        dialog = tk.Toplevel(self.widgets.root)
        dialog.title("Game finished")
        dialog.configure(bg=theme.BG)
        dialog.transient(self.widgets.root)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", self._close_post_game_review_dialog)
        self._post_game_review_dialog = dialog

        container = tk.Frame(dialog, bg=theme.BG)
        container.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(
            container,
            text="Review this Manual Coach game?",
            bg=theme.BG,
            fg=theme.FG,
            font=("sans-serif", 13, "bold"),
        ).pack(pady=(0, 6))

        tk.Label(
            container,
            text=(
                f"{move_count} moves saved — {summary_text}\n"
                "Launch a full review now, start a new game, or head back to the launcher."
            ),
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 10),
            justify="center",
        ).pack(pady=(0, 14))

        primary_row = tk.Frame(container, bg=theme.BG)
        primary_row.pack(fill="x")

        theme.styled_button(
            primary_row,
            text="▶  Analyse this game",
            command=lambda: (
                self._close_post_game_review_dialog(),
                self._host.switch_mode(
                    AppMode.REVIEW,
                    ReviewModeRequest(
                        action="review_board",
                        board=board.copy(),
                        return_mode=AppMode.PLAY,
                        source_path=self._current_game_file,
                    ),
                ),
            ),
            bg=theme.BTN_ACCENT_BG,
            fg=theme.BTN_ACCENT_FG,
            hover_bg=theme.BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 10, "bold"),
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        theme.styled_button(
            primary_row,
            text="New game",
            command=lambda: (self._close_post_game_review_dialog(), self._on_new_game()),
            font=("sans-serif", 10),
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        secondary_row = tk.Frame(container, bg=theme.BG)
        secondary_row.pack(fill="x", pady=(8, 0))

        theme.styled_button(
            secondary_row,
            text="Back to launcher",
            command=lambda: (self._close_post_game_review_dialog(), self._host.relaunch_startup()),
            font=("sans-serif", 10),
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        theme.styled_button(
            secondary_row,
            text="Later",
            command=self._close_post_game_review_dialog,
            font=("sans-serif", 10),
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        center_dialog(self.widgets.root, dialog)

    def _close_post_game_review_dialog(self) -> None:
        if self._post_game_review_dialog is None:
            return
        dialog = self._post_game_review_dialog
        self._post_game_review_dialog = None
        try:
            if dialog.winfo_exists():
                dialog.destroy()
        except tk.TclError:
            pass
