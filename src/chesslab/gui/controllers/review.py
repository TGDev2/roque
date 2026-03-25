"""Review mode controller and its related workflows."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
import tkinter.ttk as ttk
from dataclasses import dataclass, replace
from pathlib import Path

import chess

from chesslab.gui import theme
from chesslab.gui.controllers.base import (
    AppMode,
    ControllerBase,
    PlayModeRequest,
    ProgressionModeRequest,
    ReviewModeRequest,
    TrainingModeRequest,
)
from chesslab.gui.dialog_utils import center_dialog
from chesslab.gui.panels import (
    EvalPanel,
    GameSummaryPanel,
    MoveHistoryPanel,
    ReviewNavigationPanel,
    ReviewPanel,
)
from chesslab.persistence.game_storage import GameRecord, GameStorage, GameSummary
from chesslab.persistence.remote_import import (
    RemoteGameCandidate,
    RemoteGameSource,
    RemoteImportBatchResult,
)
from chesslab.review.batch_review import (
    BatchReviewItemResult,
    BatchReviewItemStatus,
    BatchReviewResult,
    BatchReviewTarget,
)
from chesslab.review.game_reviewer import (
    GameReview,
    GameReviewer,
    ReviewCancelledError,
    ReviewedMove,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BatchReviewStartedEvent:
    index: int
    target: BatchReviewTarget


@dataclass(frozen=True)
class _BatchReviewProgressEvent:
    index: int
    target: BatchReviewTarget
    current: int
    total: int


@dataclass(frozen=True)
class _BatchReviewFinishedEvent:
    index: int
    result: BatchReviewItemResult


class ReviewController(ControllerBase):
    """Own post-game review, PGN intake, import, and batch review workflows."""

    mode = AppMode.REVIEW

    def __init__(self, host: object) -> None:
        super().__init__(host)
        self._container: tk.Frame | None = None
        self._eval_panel: EvalPanel | None = None
        self._history_panel: MoveHistoryPanel | None = None
        self._review_panel: ReviewPanel | None = None
        self._review_nav: ReviewNavigationPanel | None = None
        self._summary_panel: GameSummaryPanel | None = None

        self._return_mode = AppMode.PLAY
        self._from_observation = False
        self._game_review: GameReview | None = None
        self._review_ply = 0
        self._review_positions: list[str] = []
        self._current_source_path: Path | None = None
        self._play_return_record: GameRecord | None = None

        self._review_cancel: threading.Event | None = None
        self._review_queue: queue.Queue[int | GameReview | Exception] = queue.Queue()
        self._progress_dialog: tk.Toplevel | None = None
        self._progress_bar: ttk.Progressbar | None = None
        self._progress_label: tk.Label | None = None

        self._batch_review_cancel: threading.Event | None = None
        self._batch_review_queue: queue.Queue[
            _BatchReviewStartedEvent
            | _BatchReviewProgressEvent
            | _BatchReviewFinishedEvent
            | BatchReviewResult
            | Exception
        ] = queue.Queue()
        self._batch_review_dialog: tk.Toplevel | None = None
        self._batch_review_progress_bar: ttk.Progressbar | None = None
        self._batch_review_overall_label: tk.Label | None = None
        self._batch_review_current_label: tk.Label | None = None
        self._batch_review_listbox: tk.Listbox | None = None
        self._batch_review_close_btn: tk.Button | None = None
        self._batch_review_progression_btn: tk.Button | None = None
        self._batch_review_training_btn: tk.Button | None = None
        self._batch_review_targets: tuple[BatchReviewTarget, ...] = ()
        self._batch_review_results: dict[int, BatchReviewItemResult] = {}
        self._batch_review_current_progress: tuple[int, int, int] | None = None
        self._batch_review_result: BatchReviewResult | None = None

    def activate(self, request: object | None = None) -> None:
        self._build_widgets()
        assert self._container is not None

        self._host.set_board_visible(True)
        self._container.pack(fill="both", expand=True)

        if isinstance(request, ReviewModeRequest):
            self._return_mode = request.return_mode
            self._from_observation = request.from_observation
            self._handle_request(request)
        elif self._game_review is not None:
            self._render_review_state()

    def deactivate(self) -> None:
        if self._container is not None:
            self._container.pack_forget()

    def poll(self) -> None:
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

        try:
            while True:
                item = self._batch_review_queue.get_nowait()
                if isinstance(item, _BatchReviewStartedEvent):
                    self._on_batch_review_item_started(item)
                elif isinstance(item, _BatchReviewProgressEvent):
                    self._on_batch_review_item_progress(item)
                elif isinstance(item, _BatchReviewFinishedEvent):
                    self._on_batch_review_item_finished(item)
                elif isinstance(item, BatchReviewResult):
                    self._on_batch_review_complete(item)
                elif isinstance(item, Exception):
                    self._on_batch_review_failed(item)
        except queue.Empty:
            pass

    def on_key_left(self) -> None:
        self._review_prev()

    def on_key_right(self) -> None:
        self._review_next()

    def on_key_home(self) -> None:
        self._review_first()

    def on_key_end(self) -> None:
        self._review_last()

    def shutdown(self) -> None:
        if self._review_cancel is not None:
            self._review_cancel.set()
        if self._batch_review_cancel is not None:
            self._batch_review_cancel.set()
        self._close_progress_dialog()
        self._close_batch_review_dialog()

    def _build_widgets(self) -> None:
        if self._container is not None:
            return

        self._container = tk.Frame(self.widgets.side_host, bg=theme.BG, width=290)
        self._container.pack_propagate(False)

        self._eval_panel = EvalPanel(self._container, bar_height=160)
        self._eval_panel.pack(fill="x", pady=(0, 6))

        self._history_panel = MoveHistoryPanel(self._container)
        self._history_panel.pack(fill="both", expand=True, pady=(0, 6))

        self._review_panel = ReviewPanel(self._container)
        self._review_panel.pack(fill="x", pady=(0, 6))

        self._review_nav = ReviewNavigationPanel(
            self._container,
            on_first=self._review_first,
            on_prev=self._review_prev,
            on_next=self._review_next,
            on_last=self._review_last,
            on_prev_error=self._review_prev_error,
            on_next_error=self._review_next_error,
            on_training=self._start_training_from_review,
            on_import_lichess=lambda: self._open_remote_import_dialog(
                RemoteGameSource.LICHESS,
                return_mode=AppMode.REVIEW,
            ),
            on_import_chess_com=lambda: self._open_remote_import_dialog(
                RemoteGameSource.CHESS_COM,
                return_mode=AppMode.REVIEW,
            ),
            on_exit_review=self._exit_review_mode,
            on_back_to_launcher=self._return_to_launcher,
        )
        self._review_nav.pack(fill="x", pady=(0, 4))

        self._summary_panel = GameSummaryPanel(self._container)
        self._summary_panel.pack(fill="x", pady=(0, 4))

        self._clear_review_widgets()

    def _clear_review_widgets(self) -> None:
        if self._review_panel is not None:
            self._review_panel.clear()
        if self._review_nav is not None:
            self._review_nav.update_position(0, 0)
            self._review_nav.show_launcher_button(False)
        if self._summary_panel is not None:
            self._summary_panel.clear()
        if self._history_panel is not None:
            self._history_panel.set_history(())
        if self._eval_panel is not None:
            self._eval_panel.update_eval(None, 0, is_loading=False)

    def _handle_request(self, request: ReviewModeRequest) -> None:
        if request.action == "review_board" and request.board is not None:
            self._current_source_path = request.source_path
            self._play_return_record = None
            self._start_review(request.board)
            return

        if request.action == "review_file" and request.file_path is not None:
            self._review_from_file(request.file_path)
            return

        if request.action == "review_record" and request.record is not None:
            self._review_from_record(
                request.record,
                source_path=request.source_path,
            )
            return

        if request.action == "open_recent":
            self._open_recent_games_dialog(return_mode=request.return_mode)
            if self._game_review is None and self._host.current_mode is AppMode.REVIEW:
                self._host.switch_mode(request.return_mode)
            return

        if request.action == "open_paste":
            self._open_paste_pgn_dialog(return_mode=request.return_mode)
            if self._game_review is None and self._host.current_mode is AppMode.REVIEW:
                self._host.switch_mode(request.return_mode)
            return

        if request.action == "open_import" and request.remote_source is not None:
            self._open_remote_import_dialog(
                request.remote_source,
                return_mode=request.return_mode,
            )
            if self._game_review is None and self._host.current_mode is AppMode.REVIEW:
                self._host.switch_mode(request.return_mode)

    def _review_from_file(self, file_path: Path) -> None:
        try:
            record = GameStorage.load_game(file_path)
        except Exception as exc:
            self._host.set_status(f"Load failed: {exc}", error=True)
            logger.exception("Failed to load PGN for review.")
            return

        self._review_from_record(record, source_path=record.file_path)

    def _review_from_record(
        self,
        record: GameRecord,
        *,
        source_path: Path | None,
    ) -> None:
        if not record.board.move_stack:
            self._host.set_status("No moves to review", error=True)
            return

        self._play_return_record = record
        self._current_source_path = source_path
        self._start_review(record.board)

    def _open_paste_pgn_dialog(self, *, return_mode: AppMode) -> None:
        dialog = tk.Toplevel(self.widgets.root)
        dialog.title("Paste PGN")
        dialog.configure(bg=theme.BG)
        dialog.transient(self.widgets.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        container = tk.Frame(dialog, bg=theme.BG)
        container.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(
            container,
            text="Paste PGN",
            bg=theme.BG,
            fg=theme.FG,
            font=("sans-serif", 14, "bold"),
        ).pack(pady=(0, 8))

        tk.Label(
            container,
            text="Paste a game in PGN format below:",
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
        ).pack(anchor="w", pady=(0, 4))

        text_border = tk.Frame(container, bg=theme.ENTRY_BORDER)
        text_border.pack(fill="both", expand=True)

        text_widget = tk.Text(
            text_border,
            bg=theme.ENTRY_BG,
            fg=theme.ENTRY_FG,
            font=("monospace", 10),
            insertbackground=theme.FG,
            selectbackground=theme.ACCENT,
            selectforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=0,
            width=55,
            height=18,
            wrap="word",
        )
        text_widget.pack(fill="both", expand=True, padx=1, pady=1)

        error_label = tk.Label(
            container,
            text="",
            bg=theme.BG,
            fg=theme.ERROR,
            font=("sans-serif", 9),
            wraplength=400,
            justify="left",
        )

        btn_frame = tk.Frame(container, bg=theme.BG)
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
            self._review_from_record(record, source_path=record.file_path)

        theme.styled_button(
            btn_frame,
            text="Review",
            command=on_review,
            bg=theme.BTN_ACCENT_BG,
            fg=theme.BTN_ACCENT_FG,
            hover_bg=theme.BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 10, "bold"),
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        def on_cancel() -> None:
            dialog.destroy()
            if self._game_review is None and self._host.current_mode is AppMode.REVIEW:
                self._host.switch_mode(return_mode)

        theme.styled_button(
            btn_frame,
            text="Cancel",
            command=on_cancel,
            font=("sans-serif", 10),
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        center_dialog(self.widgets.root, dialog)
        text_widget.focus_set()

    def _start_review(self, board: chess.Board) -> None:
        self._clear_review_widgets()
        self.widgets.board_widget.clear_highlights()

        root = board.copy()
        while root.move_stack:
            root.pop()
        replay = root
        self._review_positions = [replay.fen()]
        for move in board.move_stack:
            replay.push(move)
            self._review_positions.append(replay.fen())

        while not self._review_queue.empty():
            try:
                self._review_queue.get_nowait()
            except queue.Empty:
                break

        self._review_cancel = threading.Event()
        total = len(board.move_stack)
        self._show_review_progress(total)

        board_copy = board.copy()
        worker = threading.Thread(
            target=self._review_worker,
            args=(board_copy,),
            daemon=True,
        )
        worker.start()
        logger.info("Review started: %d positions to analyse.", total)

    def _review_worker(self, board: chess.Board) -> None:
        reviewer = GameReviewer(self.services.review_config)
        try:
            def on_progress(current: int, total: int) -> None:
                del total
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
        dialog = tk.Toplevel(self.widgets.root)
        dialog.title("Analysing Game")
        dialog.configure(bg=theme.BG)
        dialog.transient(self.widgets.root)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", self._cancel_review)

        container = tk.Frame(dialog, bg=theme.BG)
        container.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(
            container,
            text="Analysing game...",
            bg=theme.BG,
            fg=theme.FG,
            font=("sans-serif", 12, "bold"),
        ).pack(pady=(0, 12))

        self._progress_bar = ttk.Progressbar(
            container,
            maximum=total,
            mode="determinate",
            length=300,
        )
        self._progress_bar.pack(fill="x", pady=(0, 8))

        self._progress_label = tk.Label(
            container,
            text=f"0 / {total}",
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
        )
        self._progress_label.pack()

        theme.styled_button(
            container,
            text="Cancel",
            command=self._cancel_review,
            font=("sans-serif", 10),
        ).pack(pady=(8, 0))

        self._progress_dialog = dialog
        center_dialog(self.widgets.root, dialog)

    def _cancel_review(self) -> None:
        if self._review_cancel is not None:
            self._review_cancel.set()
        self._close_progress_dialog()
        self._host.set_status("Review cancelled")

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.destroy()
            self._progress_dialog = None
            self._progress_bar = None
            self._progress_label = None

    def _update_review_progress(self, current: int) -> None:
        if self._progress_bar is not None:
            self._progress_bar["value"] = current
            total = int(self._progress_bar["maximum"])
            if self._progress_label is not None:
                self._progress_label.config(text=f"{current} / {total}")

    def _on_review_complete(self, review: GameReview) -> None:
        self._close_progress_dialog()
        self._game_review = review
        self._review_ply = 0
        progression_saved = self._save_review_summary(review)
        self._render_review_state()
        self._host.set_status(
            f"Review mode — post-game analysis for {len(review.moves)} moves "
            f"in {review.total_elapsed_ms / 1000:.1f}s"
            f"{'' if progression_saved else ' (progression update failed)'}",
            error=not progression_saved,
        )
        logger.info(
            "Review complete: %d moves, %.1fs",
            len(review.moves),
            review.total_elapsed_ms / 1000,
        )

    def _render_review_state(self) -> None:
        if self._game_review is None:
            self._clear_review_widgets()
            return

        assert self._history_panel is not None
        assert self._summary_panel is not None
        assert self._review_nav is not None

        moves = tuple(move.san for move in self._game_review.moves)
        classifications = tuple(move.classification for move in self._game_review.moves)
        self._history_panel.set_review_history(
            moves,
            classifications,
            on_move_click=lambda ply: self._review_navigate_to(ply + 1),
        )
        self._summary_panel.update_summary(
            self._game_review,
            on_lesson_click=self._review_jump_to_lesson,
        )
        self._review_nav.show_launcher_button(self._from_observation)
        self._review_navigate_to(self._review_ply)

    def _on_review_failed(self, exc: Exception) -> None:
        self._close_progress_dialog()
        if isinstance(exc, ReviewCancelledError):
            return
        self._host.set_status(f"Review failed: {exc}", error=True)
        logger.exception("Review failed: %s", exc)

    def _save_review_summary(self, review: GameReview) -> bool:
        try:
            self.services.review_history.record_review(
                review,
                source_path=self._current_source_path,
            )
        except Exception:
            logger.exception("Failed to persist review summary.")
            return False
        return True

    def _exit_review_mode(self) -> None:
        self._from_observation = False
        self._review_ply = 0
        self._review_positions = []
        self._game_review = None
        self.widgets.board_widget.clear_highlights()
        self._clear_review_widgets()

        request: object | None = None
        if self._play_return_record is not None:
            request = PlayModeRequest(
                record=self._play_return_record,
                status_message=(
                    f"Loaded: {self._play_return_record.file_path.name}"
                    if self._play_return_record.file_path is not None
                    else None
                ),
            )
        self._play_return_record = None
        self._host.switch_mode(self._return_mode, request)
        self._host.set_status("Exited review mode")

    def _return_to_launcher(self) -> None:
        self._from_observation = False
        self._review_ply = 0
        self._review_positions = []
        self._game_review = None
        self._play_return_record = None
        self.widgets.board_widget.clear_highlights()
        self._clear_review_widgets()
        self._host.switch_mode(AppMode.PLAY)
        self._host.relaunch_startup()

    def _review_navigate_to(self, ply: int) -> None:
        if self._game_review is None:
            return

        total = len(self._game_review.moves)
        ply = max(0, min(ply, total))
        self._review_ply = ply

        fen = self._review_positions[ply]
        self.widgets.board_widget.set_position(fen)

        assert self._eval_panel is not None
        assert self._review_panel is not None
        assert self._review_nav is not None
        assert self._history_panel is not None

        if ply > 0:
            move = self._game_review.moves[ply - 1]
            from_sq = move.uci[:2]
            to_sq = move.uci[2:4]
            self.widgets.board_widget.set_last_move(from_sq, to_sq)
            self._update_review_move_hint(move)
        else:
            self.widgets.board_widget.clear_last_move()
            self._update_review_move_hint(None)

        if ply > 0:
            reviewed_move = self._game_review.moves[ply - 1]
            self._review_panel.update_move(reviewed_move)
            self._eval_panel.update_eval(
                reviewed_move.eval_after,
                0,
                is_loading=False,
            )
            self._history_panel.highlight_ply(ply - 1)
        else:
            self._review_panel.clear()
            if self._game_review.moves:
                self._eval_panel.update_eval(
                    self._game_review.moves[0].eval_before,
                    0,
                    is_loading=False,
                )
            else:
                self._eval_panel.update_eval(None, 0, is_loading=False)

        self._review_nav.update_position(ply, total)
        self._host.set_status(f"Review — Move {ply} / {total}")

    def _update_review_move_hint(self, move: ReviewedMove | None) -> None:
        if move is None:
            self._set_board_move_hint(None, color=theme.COACH_MOVE_HINT)
            return

        colour = theme.REVIEW_MOVE_HINTS.get(move.classification)
        if colour is None or not move.best_move_uci or move.best_move_uci == move.uci:
            self._set_board_move_hint(None, color=theme.COACH_MOVE_HINT)
            return

        self._set_board_move_hint(move.best_move_uci, color=colour)

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

    def _review_first(self) -> None:
        self._review_navigate_to(0)

    def _review_jump_to_lesson(self, focus_ply: int) -> None:
        self._review_navigate_to(focus_ply + 1)

    def _review_prev(self) -> None:
        self._review_navigate_to(self._review_ply - 1)

    def _review_next(self) -> None:
        self._review_navigate_to(self._review_ply + 1)

    def _review_last(self) -> None:
        if self._game_review is not None:
            self._review_navigate_to(len(self._game_review.moves))

    def _review_prev_error(self) -> None:
        if self._game_review is None:
            return
        for ply_idx in reversed(self._game_review.critical_moments):
            if ply_idx + 1 < self._review_ply:
                self._review_navigate_to(ply_idx + 1)
                return

    def _review_next_error(self) -> None:
        if self._game_review is None:
            return
        for ply_idx in self._game_review.critical_moments:
            if ply_idx + 1 > self._review_ply:
                self._review_navigate_to(ply_idx + 1)
                return

    def _start_training_from_review(self) -> None:
        self._host.switch_mode(
            AppMode.TRAINING,
            TrainingModeRequest(
                sample_size=self._host.progression_sample_size,
                focus_review=self._game_review,
                return_mode=AppMode.REVIEW,
            ),
        )

    def _open_progression_from_batch_review(self) -> None:
        self._close_batch_review_dialog()
        self._host.switch_mode(
            AppMode.PROGRESSION,
            ProgressionModeRequest(
                sample_size=self._host.progression_sample_size,
                focus_review=self._game_review,
                return_mode=AppMode.REVIEW,
            ),
        )

    def _start_training_from_batch_review(self) -> None:
        self._close_batch_review_dialog()
        self._host.switch_mode(
            AppMode.TRAINING,
            TrainingModeRequest(
                sample_size=self._host.progression_sample_size,
                focus_review=self._game_review,
                return_mode=AppMode.REVIEW,
            ),
        )

    def _open_recent_games_dialog(self, *, return_mode: AppMode) -> None:
        games = self.services.storage.list_recent_games()
        reviewed_paths = self._reviewed_source_paths()
        dialog = tk.Toplevel(self.widgets.root)
        dialog.title("Recent Games")
        dialog.configure(bg=theme.BG)
        dialog.transient(self.widgets.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        container = tk.Frame(dialog, bg=theme.BG)
        container.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(
            container,
            text="Recent Games",
            bg=theme.BG,
            fg=theme.FG,
            font=("sans-serif", 14, "bold"),
        ).pack(pady=(0, 12))

        if not games:
            tk.Label(
                container,
                text="No saved games",
                bg=theme.BG,
                fg=theme.FG_DIM,
                font=("sans-serif", 10),
            ).pack(pady=20)

            theme.styled_button(
                container,
                text="Close",
                command=lambda: (
                    dialog.destroy(),
                    self._host.switch_mode(return_mode),
                ),
                font=("sans-serif", 10),
            ).pack(pady=(8, 0))
            center_dialog(self.widgets.root, dialog)
            return

        list_frame = tk.Frame(container, bg=theme.CARD_BORDER)
        list_frame.pack(fill="both", expand=True)

        listbox = tk.Listbox(
            list_frame,
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("monospace", 10),
            selectmode="extended",
            selectbackground=theme.ACCENT,
            selectforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=0,
            height=min(len(games), 12),
            width=50,
        )
        listbox.pack(fill="both", expand=True, padx=1, pady=1)

        for index, game in enumerate(games):
            line = self._format_recent_game_label(
                game,
                reviewed=str(game.file_path.resolve()) in reviewed_paths,
            )
            listbox.insert("end", line)
            if str(game.file_path.resolve()) not in reviewed_paths:
                listbox.selection_set(index)

        def on_load() -> None:
            sel = listbox.curselection()
            if not sel:
                return
            selected_game = games[sel[0]]
            try:
                record = GameStorage.load_game(selected_game.file_path)
            except Exception as exc:
                self._host.set_status(f"Load failed: {exc}", error=True)
                logger.exception("Failed to load PGN.")
                return

            dialog.destroy()
            self._host.switch_mode(
                AppMode.PLAY,
                PlayModeRequest(
                    record=record,
                    status_message=f"Loaded: {selected_game.file_path.name}",
                ),
            )

        def on_review() -> None:
            sel = listbox.curselection()
            if not sel:
                return
            dialog.destroy()
            selected_games = [games[int(index)] for index in sel]
            self._start_batch_review(
                self._build_batch_review_recent_targets(selected_games),
                origin_label="Recent games",
            )

        listbox.bind("<Double-Button-1>", lambda _event: on_load())

        btn_frame = tk.Frame(container, bg=theme.BG)
        btn_frame.pack(fill="x", pady=(8, 0))

        theme.styled_button(
            btn_frame,
            text="Load",
            command=on_load,
            bg=theme.BTN_ACCENT_BG,
            fg=theme.BTN_ACCENT_FG,
            hover_bg=theme.BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 10, "bold"),
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        theme.styled_button(
            btn_frame,
            text="Review Selected",
            command=on_review,
            font=("sans-serif", 10, "bold"),
        ).pack(side="left", fill="x", expand=True, padx=(4, 4))

        theme.styled_button(
            btn_frame,
            text="Cancel",
            command=lambda: (
                dialog.destroy(),
                self._host.switch_mode(return_mode),
            ),
            font=("sans-serif", 10),
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        center_dialog(self.widgets.root, dialog)

    def _open_remote_import_dialog(
        self,
        source: RemoteGameSource,
        *,
        return_mode: AppMode,
    ) -> None:
        dialog = tk.Toplevel(self.widgets.root)
        dialog.title(f"Import from {source.label}")
        dialog.configure(bg=theme.BG)
        dialog.transient(self.widgets.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        container = tk.Frame(dialog, bg=theme.BG)
        container.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(
            container,
            text=f"Import from {source.label}",
            bg=theme.BG,
            fg=theme.FG,
            font=("sans-serif", 14, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        tk.Label(
            container,
            text=(
                "Fetch your latest public games by username. Only the platform "
                "username is sent to the selected API."
            ),
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=430,
        ).pack(anchor="w", pady=(0, 10))

        controls = tk.Frame(container, bg=theme.BG)
        controls.pack(fill="x", pady=(0, 8))

        tk.Label(
            controls,
            text="Username",
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
        ).pack(side="left")

        username_border = tk.Frame(controls, bg=theme.ENTRY_BORDER)
        username_border.pack(side="left", padx=(6, 10))

        username_entry = tk.Entry(
            username_border,
            bg=theme.ENTRY_BG,
            fg=theme.ENTRY_FG,
            insertbackground=theme.FG,
            relief="flat",
            bd=0,
            highlightthickness=0,
            width=20,
            font=("monospace", 10),
        )
        username_entry.pack(padx=1, pady=1, ipady=2)

        tk.Label(
            controls,
            text="Last",
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
        ).pack(side="left")

        limit_var = tk.IntVar(value=10)
        limit_spin = tk.Spinbox(
            controls,
            from_=1,
            to=50,
            width=4,
            textvariable=limit_var,
            justify="center",
            bg=theme.ENTRY_BG,
            fg=theme.ENTRY_FG,
            buttonbackground=theme.BTN_BG,
            insertbackground=theme.FG,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("monospace", 10),
        )
        limit_spin.pack(side="left", padx=(6, 6))

        tk.Label(
            controls,
            text="games",
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
        ).pack(side="left")

        games: list[RemoteGameCandidate] = []

        list_frame = tk.Frame(container, bg=theme.CARD_BORDER)
        list_frame.pack(fill="both", expand=True)

        listbox = tk.Listbox(
            list_frame,
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("monospace", 10),
            selectmode="extended",
            selectbackground=theme.ACCENT,
            selectforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=0,
            height=12,
            width=58,
        )
        listbox.pack(fill="both", expand=True, padx=1, pady=1)

        feedback_label = tk.Label(
            container,
            text="Enter a username, then fetch the latest public games.",
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=430,
        )
        feedback_label.pack(anchor="w", pady=(8, 0))

        def set_feedback(
            message: str,
            *,
            error: bool = False,
            success: bool = False,
        ) -> None:
            colour = theme.ERROR if error else (theme.SUCCESS if success else theme.FG_DIM)
            feedback_label.config(text=message, fg=colour)

        def refresh_listbox() -> None:
            listbox.delete(0, "end")
            listbox.selection_clear(0, "end")
            for index, game in enumerate(games):
                listbox.insert("end", game.display_label())
                if not game.already_imported:
                    listbox.selection_set(index)

        def on_fetch() -> None:
            nonlocal games
            username = username_entry.get().strip()
            try:
                limit = self._coerce_remote_import_limit(limit_var.get())
            except (tk.TclError, ValueError) as exc:
                set_feedback(f"Invalid game count: {exc}", error=True)
                return

            set_feedback(f"Loading recent games from {source.label}...")
            dialog.update_idletasks()
            try:
                games = self.services.remote_import.list_recent_games(
                    source,
                    username=username,
                    limit=limit,
                )
            except Exception as exc:
                games = []
                listbox.delete(0, "end")
                set_feedback(str(exc), error=True)
                self._host.set_status(f"Import failed: {exc}", error=True)
                return

            refresh_listbox()
            importable_count = sum(not game.already_imported for game in games)
            if importable_count:
                message = f"Fetched {len(games)} game(s). {importable_count} ready to import."
                set_feedback(message, success=True)
                self._host.set_status(f"{source.label}: {message}")
            else:
                message = f"Fetched {len(games)} game(s). Everything is already imported."
                set_feedback(message)
                self._host.set_status(f"{source.label}: {message}")

        def on_import(*, review_after_import: bool = False) -> None:
            nonlocal games
            selection = listbox.curselection()
            if not games:
                set_feedback("Fetch games first.", error=True)
                return
            if not selection:
                set_feedback("Select at least one game to import.", error=True)
                return

            selected_games = [games[int(index)] for index in selection]
            action_label = (
                "Importing and queuing reviews"
                if review_after_import
                else "Importing selected games"
            )
            set_feedback(f"{action_label}...")
            dialog.update_idletasks()

            result = self.services.remote_import.import_games(selected_games)
            failed_ids = {failure.game.game_id for failure in result.failures}
            completed_ids = {
                game.game_id for game in selected_games if game.game_id not in failed_ids
            }
            games = [
                replace(game, already_imported=True)
                if game.game_id in completed_ids
                else game
                for game in games
            ]
            refresh_listbox()

            summary = self._format_remote_import_summary(source, result)
            if result.failed_count:
                detail = result.failures[0].message
                if result.failed_count > 1:
                    detail += f" (+{result.failed_count - 1} more)"
                set_feedback(f"{summary}\n{detail}", error=result.imported_count == 0)
            else:
                set_feedback(summary, success=result.imported_count > 0)
            self._host.set_status(
                summary,
                error=result.failed_count > 0 and result.imported_count == 0,
            )

            if review_after_import:
                review_targets = self._build_batch_review_import_targets(result)
                if not review_targets:
                    set_feedback(
                        f"{summary}\nNo local PGNs are available to review from this selection.",
                        error=True,
                    )
                    return
                dialog.destroy()
                self._start_batch_review(
                    review_targets,
                    origin_label=f"{source.label} import",
                )

        btn_frame = tk.Frame(container, bg=theme.BG)
        btn_frame.pack(fill="x", pady=(10, 0))

        theme.styled_button(
            btn_frame,
            text="Fetch",
            command=on_fetch,
            bg=theme.BTN_ACCENT_BG,
            fg=theme.BTN_ACCENT_FG,
            hover_bg=theme.BTN_ACCENT_HOVER,
            active_bg="#1e88e5",
            font=("sans-serif", 10, "bold"),
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        theme.styled_button(
            btn_frame,
            text="Import Selected",
            command=on_import,
            font=("sans-serif", 10, "bold"),
        ).pack(side="left", fill="x", expand=True, padx=(4, 4))

        theme.styled_button(
            btn_frame,
            text="Import + Review",
            command=lambda: on_import(review_after_import=True),
            font=("sans-serif", 10, "bold"),
        ).pack(side="left", fill="x", expand=True, padx=(4, 4))

        theme.styled_button(
            btn_frame,
            text="Close",
            command=lambda: (
                dialog.destroy(),
                self._host.switch_mode(return_mode) if (
                    self._game_review is None and self._host.current_mode is AppMode.REVIEW
                ) else None,
            ),
            font=("sans-serif", 10),
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        username_entry.bind("<Return>", lambda _event: on_fetch())
        limit_spin.bind("<Return>", lambda _event: on_fetch())
        center_dialog(self.widgets.root, dialog)
        username_entry.focus_set()

    @staticmethod
    def _coerce_remote_import_limit(raw_limit: int) -> int:
        return max(1, min(int(raw_limit), 50))

    @staticmethod
    def _format_remote_import_summary(
        source: RemoteGameSource,
        result: RemoteImportBatchResult,
    ) -> str:
        fragments: list[str] = []
        if result.imported_count:
            fragments.append(f"{result.imported_count} imported")
        if result.skipped_count:
            fragments.append(f"{result.skipped_count} already present")
        if result.failed_count:
            fragments.append(f"{result.failed_count} failed")
        if not fragments:
            return f"{source.label}: nothing was imported."
        return f"{source.label}: {', '.join(fragments)}."

    def _reviewed_source_paths(self) -> set[str]:
        try:
            return {
                summary.source_path
                for summary in self.services.review_history.list_recent_reviews(limit=None)
                if summary.source_path
            }
        except Exception:
            logger.exception("Failed to load reviewed game paths.")
            return set()

    @staticmethod
    def _format_recent_game_label(
        game: GameSummary,
        *,
        reviewed: bool = False,
    ) -> str:
        line = (
            f"{game.date}  {game.white} vs {game.black}"
            f"  {game.result}  {game.move_count} moves"
        )
        return f"{line}  reviewed" if reviewed else line

    @staticmethod
    def _format_batch_review_line(
        target: BatchReviewTarget,
        *,
        status: str,
        detail: str | None = None,
    ) -> str:
        parts = [status, target.display_label]
        if detail:
            parts.append(detail)
        return "  |  ".join(parts)

    @classmethod
    def _format_batch_review_result_line(
        cls,
        result: BatchReviewItemResult,
    ) -> str:
        if result.status is BatchReviewItemStatus.COMPLETED:
            detail = result.message
            if result.summary is not None:
                detail = f"avg {result.summary.avg_cp_loss:.0f}cp"
            return cls._format_batch_review_line(result.target, status="Done", detail=detail)
        if result.status is BatchReviewItemStatus.SKIPPED_ALREADY_REVIEWED:
            return cls._format_batch_review_line(
                result.target,
                status="Skipped",
                detail="already reviewed",
            )
        if result.status is BatchReviewItemStatus.SKIPPED_DUPLICATE_SELECTION:
            return cls._format_batch_review_line(
                result.target,
                status="Skipped",
                detail="duplicate selection",
            )
        if result.status is BatchReviewItemStatus.CANCELLED:
            return cls._format_batch_review_line(
                result.target,
                status="Cancelled",
                detail=result.message,
            )
        return cls._format_batch_review_line(
            result.target,
            status="Failed",
            detail=result.message,
        )

    def _build_batch_review_recent_targets(
        self,
        games: tuple[GameSummary, ...] | list[GameSummary],
    ) -> tuple[BatchReviewTarget, ...]:
        return tuple(
            BatchReviewTarget(
                file_path=game.file_path,
                label=self._format_recent_game_label(game),
            )
            for game in games
        )

    @staticmethod
    def _build_batch_review_import_targets(
        result: RemoteImportBatchResult,
    ) -> tuple[BatchReviewTarget, ...]:
        return tuple(
            BatchReviewTarget(
                file_path=item.file_path,
                label=item.game.display_label(),
            )
            for item in result.resolved_games
        )

    def _start_batch_review(
        self,
        targets: tuple[BatchReviewTarget, ...],
        *,
        origin_label: str,
    ) -> bool:
        filtered_targets = tuple(targets)
        if not filtered_targets:
            self._host.set_status(f"{origin_label}: no games available to review.")
            return False

        if (
            self._batch_review_dialog is not None
            and self._batch_review_dialog.winfo_exists()
        ):
            self._batch_review_dialog.deiconify()
            self._batch_review_dialog.lift()
            self._batch_review_dialog.focus_force()
            return False

        while not self._batch_review_queue.empty():
            try:
                self._batch_review_queue.get_nowait()
            except queue.Empty:
                break

        self._batch_review_targets = filtered_targets
        self._batch_review_results = {}
        self._batch_review_current_progress = None
        self._batch_review_result = None
        self._batch_review_cancel = threading.Event()

        self._show_batch_review_dialog(origin_label)

        worker = threading.Thread(
            target=self._batch_review_worker,
            args=(filtered_targets,),
            daemon=True,
        )
        worker.start()
        self._host.set_status(
            f"{origin_label}: reviewing {len(filtered_targets)} selected game"
            f"{'' if len(filtered_targets) == 1 else 's'}...",
            waiting=True,
        )
        logger.info("Batch review started for %d game(s).", len(filtered_targets))
        return True

    def _batch_review_worker(
        self,
        targets: tuple[BatchReviewTarget, ...],
    ) -> None:
        try:
            result = self.services.batch_review.run(
                targets,
                on_item_started=(
                    lambda index, target: self._batch_review_queue.put(
                        _BatchReviewStartedEvent(index=index, target=target),
                    )
                ),
                on_item_progress=(
                    lambda index, target, current, total: self._batch_review_queue.put(
                        _BatchReviewProgressEvent(
                            index=index,
                            target=target,
                            current=current,
                            total=total,
                        ),
                    )
                ),
                on_item_finished=(
                    lambda index, item_result: self._batch_review_queue.put(
                        _BatchReviewFinishedEvent(index=index, result=item_result),
                    )
                ),
                cancel_event=self._batch_review_cancel,
            )
            self._batch_review_queue.put(result)
        except Exception as exc:
            self._batch_review_queue.put(exc)

    def _show_batch_review_dialog(self, origin_label: str) -> None:
        dialog = tk.Toplevel(self.widgets.root)
        dialog.title("Batch Review")
        dialog.configure(bg=theme.BG)
        dialog.transient(self.widgets.root)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", self._cancel_batch_review)
        self._batch_review_dialog = dialog

        container = tk.Frame(dialog, bg=theme.BG)
        container.pack(fill="both", expand=True, padx=20, pady=18)

        tk.Label(
            container,
            text="Batch Review",
            bg=theme.BG,
            fg=theme.FG,
            font=("sans-serif", 13, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        tk.Label(
            container,
            text=(
                f"{origin_label}: each completed review is saved immediately for "
                "progression and training."
            ),
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=460,
        ).pack(anchor="w", pady=(0, 10))

        self._batch_review_progress_bar = ttk.Progressbar(
            container,
            maximum=max(1, len(self._batch_review_targets)),
            mode="determinate",
            length=340,
        )
        self._batch_review_progress_bar.pack(fill="x", pady=(0, 8))

        self._batch_review_overall_label = tk.Label(
            container,
            text=f"0 / {len(self._batch_review_targets)} finished",
            bg=theme.BG,
            fg=theme.FG,
            font=("sans-serif", 9, "bold"),
            anchor="w",
        )
        self._batch_review_overall_label.pack(fill="x")

        self._batch_review_current_label = tk.Label(
            container,
            text="Waiting for the first game...",
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=460,
            anchor="w",
        )
        self._batch_review_current_label.pack(fill="x", pady=(4, 8))

        list_frame = tk.Frame(container, bg=theme.CARD_BORDER)
        list_frame.pack(fill="both", expand=True)

        self._batch_review_listbox = tk.Listbox(
            list_frame,
            bg=theme.BG_SECONDARY,
            fg=theme.FG,
            font=("monospace", 10),
            selectbackground=theme.BG_SECONDARY,
            selectforeground=theme.FG,
            activestyle="none",
            exportselection=False,
            relief="flat",
            bd=0,
            highlightthickness=0,
            height=min(max(len(self._batch_review_targets), 6), 12),
            width=72,
        )
        self._batch_review_listbox.pack(fill="both", expand=True, padx=1, pady=1)

        for target in self._batch_review_targets:
            self._batch_review_listbox.insert(
                "end",
                self._format_batch_review_line(target, status="Pending"),
            )

        btn_frame = tk.Frame(container, bg=theme.BG)
        btn_frame.pack(fill="x", pady=(10, 0))

        self._batch_review_progression_btn = theme.styled_button(
            btn_frame,
            text="Open Progression",
            command=self._open_progression_from_batch_review,
            font=("sans-serif", 10, "bold"),
        )
        self._batch_review_progression_btn.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 4),
        )
        self._batch_review_progression_btn.config(state="disabled")

        self._batch_review_training_btn = theme.styled_button(
            btn_frame,
            text="Start Training",
            command=self._start_training_from_batch_review,
            font=("sans-serif", 10, "bold"),
        )
        self._batch_review_training_btn.pack(
            side="left",
            fill="x",
            expand=True,
            padx=4,
        )
        self._batch_review_training_btn.config(state="disabled")

        self._batch_review_close_btn = theme.styled_button(
            btn_frame,
            text="Cancel",
            command=self._cancel_batch_review,
            font=("sans-serif", 10),
        )
        self._batch_review_close_btn.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(4, 0),
        )

        center_dialog(self.widgets.root, dialog)

    def _cancel_batch_review(self) -> None:
        if self._batch_review_cancel is None:
            self._close_batch_review_workflow()
            return
        self._batch_review_cancel.set()
        if self._batch_review_current_label is not None:
            self._batch_review_current_label.config(
                text="Stopping after the current engine call...",
                fg=theme.LOADING,
            )
        self._host.set_status("Batch review cancellation requested.", waiting=True)

    def _close_batch_review_dialog(self) -> None:
        if self._batch_review_dialog is not None:
            self._batch_review_dialog.destroy()
        self._batch_review_dialog = None
        self._batch_review_progress_bar = None
        self._batch_review_overall_label = None
        self._batch_review_current_label = None
        self._batch_review_listbox = None
        self._batch_review_close_btn = None
        self._batch_review_progression_btn = None
        self._batch_review_training_btn = None
        self._batch_review_cancel = None
        self._batch_review_current_progress = None

    def _close_batch_review_workflow(self) -> None:
        self._close_batch_review_dialog()
        if self._game_review is None and self._host.current_mode is AppMode.REVIEW:
            self._host.switch_mode(self._return_mode)

    def _batch_review_processed_counts(self) -> tuple[int, int, int, int, int]:
        completed = 0
        failed = 0
        skipped = 0
        cancelled = 0
        for item in self._batch_review_results.values():
            if item.status is BatchReviewItemStatus.COMPLETED:
                completed += 1
            elif item.status is BatchReviewItemStatus.FAILED:
                failed += 1
            elif item.status in {
                BatchReviewItemStatus.SKIPPED_ALREADY_REVIEWED,
                BatchReviewItemStatus.SKIPPED_DUPLICATE_SELECTION,
            }:
                skipped += 1
            elif item.status is BatchReviewItemStatus.CANCELLED:
                cancelled += 1
        processed = len(self._batch_review_results)
        return processed, completed, skipped, failed, cancelled

    def _refresh_batch_review_progress(self) -> None:
        total = len(self._batch_review_targets)
        processed, completed, skipped, failed, cancelled = self._batch_review_processed_counts()

        if self._batch_review_progress_bar is not None:
            value = float(processed)
            if self._batch_review_result is None and self._batch_review_current_progress is not None:
                _, current, item_total = self._batch_review_current_progress
                if item_total > 0:
                    value += current / item_total
            self._batch_review_progress_bar["maximum"] = max(1, total)
            self._batch_review_progress_bar["value"] = min(float(total), value)

        if self._batch_review_overall_label is not None:
            fragments = [f"{processed} / {total} finished"]
            if completed:
                fragments.append(f"{completed} done")
            if skipped:
                fragments.append(f"{skipped} skipped")
            if failed:
                fragments.append(f"{failed} failed")
            if cancelled:
                fragments.append(f"{cancelled} cancelled")
            self._batch_review_overall_label.config(text="  |  ".join(fragments))

    def _on_batch_review_item_started(self, event: _BatchReviewStartedEvent) -> None:
        self._batch_review_current_progress = (event.index, 0, 0)
        if self._batch_review_listbox is not None:
            self._batch_review_listbox.delete(event.index)
            self._batch_review_listbox.insert(
                event.index,
                self._format_batch_review_line(event.target, status="Reviewing"),
            )
        if self._batch_review_current_label is not None:
            self._batch_review_current_label.config(
                text=(
                    f"Current game {event.index + 1} / {len(self._batch_review_targets)}"
                    f" — {event.target.display_label}"
                ),
                fg=theme.FG_DIM,
            )
        self._refresh_batch_review_progress()

    def _on_batch_review_item_progress(self, event: _BatchReviewProgressEvent) -> None:
        self._batch_review_current_progress = (event.index, event.current, event.total)
        if self._batch_review_listbox is not None:
            self._batch_review_listbox.delete(event.index)
            self._batch_review_listbox.insert(
                event.index,
                self._format_batch_review_line(
                    event.target,
                    status="Reviewing",
                    detail=f"{event.current} / {event.total}",
                ),
            )
        if self._batch_review_current_label is not None:
            self._batch_review_current_label.config(
                text=(
                    f"Current game {event.index + 1} / {len(self._batch_review_targets)}"
                    f" — {event.target.display_label} — {event.current} / {event.total} plies"
                ),
                fg=theme.FG_DIM,
            )
        self._refresh_batch_review_progress()

    def _on_batch_review_item_finished(self, event: _BatchReviewFinishedEvent) -> None:
        self._batch_review_results[event.index] = event.result
        if self._batch_review_listbox is not None:
            self._batch_review_listbox.delete(event.index)
            self._batch_review_listbox.insert(
                event.index,
                self._format_batch_review_result_line(event.result),
            )
        self._refresh_batch_review_progress()

    def _build_batch_review_completion_text(
        self,
        result: BatchReviewResult,
    ) -> tuple[str, bool, bool]:
        if result.completed_count == 0:
            if result.cancelled:
                return ("Batch review stopped before any new review was saved.", False, False)
            return ("Batch review finished without adding new review history.", False, False)

        try:
            snapshot = self.services.progress_tracker.build_snapshot(
                limit=self._host.progression_sample_size,
            )
            summaries = self.services.review_history.list_recent_reviews(
                limit=self._host.progression_sample_size,
            )
            attempts = self.services.training_history.list_recent_attempts(limit=None)
            training_ready = bool(
                self.services.training_service.build_session(
                    summaries,
                    sample_size=self._host.progression_sample_size,
                    focus_review=self._game_review,
                    session_size=1,
                    training_attempts=attempts,
                )
            )
        except Exception as exc:
            logger.exception("Failed to compute batch-review activation state.")
            return (
                f"Batch review complete, but follow-up insights could not be refreshed: {exc}",
                False,
                False,
            )

        progression_ready = snapshot.has_enough_games
        if progression_ready and training_ready:
            return ("Batch review complete. Progression and targeted training are ready.", True, True)
        if progression_ready:
            return (
                "Batch review complete. Progression is ready; targeted training still needs more reusable mistakes.",
                True,
                False,
            )
        if training_ready:
            return (
                "Batch review complete. Targeted training is ready, and "
                f"{snapshot.games_remaining} more reviewed game"
                f"{'' if snapshot.games_remaining == 1 else 's'} are needed for the full progression dashboard.",
                False,
                True,
            )
        return (
            "Batch review complete. "
            f"{snapshot.games_remaining} more reviewed game"
            f"{'' if snapshot.games_remaining == 1 else 's'} needed for the full progression dashboard.",
            False,
            False,
        )

    def _on_batch_review_complete(self, result: BatchReviewResult) -> None:
        self._batch_review_result = result
        self._batch_review_current_progress = None
        self._refresh_batch_review_progress()

        message, progression_ready, training_ready = self._build_batch_review_completion_text(result)
        if self._batch_review_current_label is not None:
            self._batch_review_current_label.config(
                text=message,
                fg=theme.SUCCESS if result.completed_count else theme.FG_DIM,
            )

        if self._batch_review_progression_btn is not None:
            self._batch_review_progression_btn.config(
                state="normal" if progression_ready else "disabled",
            )
        if self._batch_review_training_btn is not None:
            self._batch_review_training_btn.config(
                state="normal" if training_ready else "disabled",
            )
        if self._batch_review_close_btn is not None:
            self._batch_review_close_btn.config(
                text="Close",
                command=self._close_batch_review_workflow,
            )

        status = (
            f"Batch review: {result.completed_count} completed, "
            f"{result.failed_count} failed, "
            f"{result.skipped_reviewed_count + result.skipped_duplicate_count} skipped"
        )
        if result.cancelled:
            status += ", cancelled"
        if progression_ready:
            status += " — progression ready"
        if training_ready:
            status += " — training ready"
        self._host.set_status(status, error=result.completed_count == 0 and result.failed_count > 0)

    def _on_batch_review_failed(self, exc: Exception) -> None:
        logger.exception("Batch review failed: %s", exc)
        if self._batch_review_current_label is not None:
            self._batch_review_current_label.config(
                text=f"Batch review failed: {exc}",
                fg=theme.ERROR,
            )
        if self._batch_review_close_btn is not None:
            self._batch_review_close_btn.config(
                text="Close",
                command=self._close_batch_review_workflow,
            )
        if self._batch_review_progression_btn is not None:
            self._batch_review_progression_btn.config(state="disabled")
        if self._batch_review_training_btn is not None:
            self._batch_review_training_btn.config(state="disabled")
        self._host.set_status(f"Batch review failed: {exc}", error=True)
