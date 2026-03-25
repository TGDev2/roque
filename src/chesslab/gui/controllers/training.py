"""Targeted training mode controller."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import messagebox

import chess

from chesslab.gui.controllers.base import (
    AppMode,
    ControllerBase,
    TrainingModeRequest,
)
from chesslab.gui.panels import TrainingPanel
from chesslab.training import (
    TrainingExercise,
    TrainingFeedbackLabel,
    TrainingMoveFeedback,
    build_training_attempt,
)

logger = logging.getLogger(__name__)


class TrainingController(ControllerBase):
    """Own the focused targeted-training flow."""

    mode = AppMode.TRAINING

    def __init__(self, host: object) -> None:
        super().__init__(host)
        self._container: tk.Frame | None = None
        self._panel: TrainingPanel | None = None
        self._session: tuple[TrainingExercise, ...] = ()
        self._index = 0
        self._pending = False
        self._request_id = 0
        self._queue: queue.Queue[tuple[int, TrainingMoveFeedback | Exception]] = queue.Queue()
        self._return_mode = AppMode.PLAY

    def activate(self, request: object | None = None) -> None:
        training_request = self._coerce_request(request)
        if training_request is None:
            return

        session = self._build_session(training_request)
        if not session:
            return

        self._return_mode = training_request.return_mode
        self._session = session
        self._index = 0
        self._pending = False
        self._request_id += 1

        self._build_widgets()
        assert self._container is not None

        self._host.set_board_visible(True)
        self._container.pack(fill="both", expand=True)
        self._show_training_exercise()

    def deactivate(self) -> None:
        if self._container is not None:
            self._container.pack_forget()

    def poll(self) -> None:
        try:
            while True:
                request_id, item = self._queue.get_nowait()
                if request_id != self._request_id:
                    continue
                if isinstance(item, Exception):
                    self._on_training_failed(item)
                else:
                    self._on_training_feedback(item)
        except queue.Empty:
            pass

    def on_board_move(self, from_sq: str, to_sq: str) -> None:
        if self._pending:
            return

        exercise = self._current_training_exercise()
        if exercise is None:
            return

        try:
            board = chess.Board(exercise.position.fen)
            uci_str = from_sq + to_sq
            move = chess.Move.from_uci(uci_str)
            san = board.san(move)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            self._host.set_status(f"Invalid training move: {from_sq}{to_sq}", error=True)
            return

        board.push(move)
        self._pending = True
        self._request_id += 1
        request_id = self._request_id

        self.widgets.board_widget.set_interactive(False)
        self.widgets.board_widget.set_position(board.fen())
        self.widgets.board_widget.set_last_move(from_sq, to_sq[:2])
        assert self._panel is not None
        self._panel.show_loading(san)
        self._host.set_status("Training — checking your move...")

        worker = threading.Thread(
            target=self._training_worker,
            args=(request_id, exercise.position.fen, uci_str),
            daemon=True,
        )
        worker.start()

    def shutdown(self) -> None:
        self._request_id += 1
        self._pending = False
        self._session = ()

    def _coerce_request(self, request: object | None) -> TrainingModeRequest | None:
        if isinstance(request, TrainingModeRequest):
            return request
        return None

    def _build_widgets(self) -> None:
        if self._container is not None:
            return

        self._container = tk.Frame(self.widgets.side_host, bg="#16161e", width=290)
        self._container.pack_propagate(False)
        self._panel = TrainingPanel(
            self._container,
            on_next=self._training_next,
            on_exit=self._exit_training_mode,
        )
        self._panel.pack(fill="both", expand=True)

    def _build_session(
        self,
        request: TrainingModeRequest,
    ) -> tuple[TrainingExercise, ...]:
        try:
            summaries = self.services.review_history.list_recent_reviews(
                limit=request.sample_size,
            )
            attempts = self.services.training_history.list_recent_attempts(limit=None)
            session = self.services.training_service.build_session(
                summaries,
                sample_size=request.sample_size,
                focus_review=request.focus_review,
                training_attempts=attempts,
                opening_filter_key=request.opening_filter_key,
            )
        except Exception as exc:
            logger.exception("Failed to build training session.")
            self._host.set_status(f"Training unavailable: {exc}", error=True)
            self._host.switch_mode(request.return_mode)
            return ()

        if session:
            return session

        self._show_training_unavailable(
            self._build_training_unavailable_message(
                summaries,
                opening_filter_key=request.opening_filter_key,
            ),
        )
        self._host.switch_mode(request.return_mode)
        return ()

    @staticmethod
    def _build_training_unavailable_message(
        summaries: list[object],
        *,
        opening_filter_key: str | None = None,
    ) -> str:
        if not summaries:
            return (
                "No training positions are available yet. Run a few post-game reviews "
                "to generate reusable mistakes, blunders, and lesson positions."
            )
        if opening_filter_key is not None:
            return (
                "No reusable training positions were found for the selected opening "
                "in your recent review history."
            )
        return (
            "No reusable training positions were found in your recent review history. "
            "Review a few more games to generate targeted exercises from critical moments."
        )

    def _show_training_unavailable(self, message: str) -> None:
        self._host.set_status(message)
        messagebox.showinfo(
            title="Training unavailable",
            message=message,
            parent=self.widgets.root,
        )

    def _show_training_exercise(self) -> None:
        exercise = self._current_training_exercise()
        if exercise is None:
            self._exit_training_mode()
            return

        self._pending = False
        self.widgets.board_widget.set_position(exercise.position.fen)
        self.widgets.board_widget.clear_last_move()
        self.widgets.board_widget.clear_highlights()
        self.widgets.board_widget.set_interactive(True)
        assert self._panel is not None
        self._panel.show_exercise(
            exercise,
            index=self._index,
            total=len(self._session),
        )
        self._host.set_status(
            f"Training — exercise {self._index + 1} / {len(self._session)}",
        )

    def _current_training_exercise(self) -> TrainingExercise | None:
        if not self._session:
            return None
        if not (0 <= self._index < len(self._session)):
            return None
        return self._session[self._index]

    def _training_worker(
        self,
        request_id: int,
        fen: str,
        move_uci: str,
    ) -> None:
        try:
            feedback = self.services.training_service.evaluate_move(fen, move_uci)
            self._queue.put((request_id, feedback))
        except Exception as exc:
            self._queue.put((request_id, exc))

    def _on_training_feedback(self, feedback: TrainingMoveFeedback) -> None:
        exercise = self._current_training_exercise()
        if exercise is None or self._panel is None:
            return

        self._pending = False
        self._panel.show_feedback(
            feedback,
            is_last=self._index + 1 >= len(self._session),
        )

        try:
            self.services.training_history.record_attempt(
                build_training_attempt(exercise, feedback),
            )
        except Exception:
            logger.exception("Failed to persist training attempt.")

        if feedback.label is TrainingFeedbackLabel.GOOD:
            status = "Training — good move"
        elif feedback.label is TrainingFeedbackLabel.BETTER_AVAILABLE:
            status = "Training — there was a better move"
        else:
            status = "Training — significant error"
        self._host.set_status(status)

    def _on_training_failed(self, exc: Exception) -> None:
        logger.exception("Training move evaluation failed: %s", exc)
        self._pending = False
        self._host.set_status(f"Training failed: {exc}", error=True)
        self._show_training_exercise()

    def _training_next(self) -> None:
        if self._index + 1 >= len(self._session):
            self._exit_training_mode(
                status_message=(
                    f"Training session complete — {len(self._session)} "
                    "position(s) practised."
                ),
            )
            return

        self._index += 1
        self._show_training_exercise()

    def _exit_training_mode(self, *, status_message: str | None = None) -> None:
        self._request_id += 1
        self._pending = False
        self._session = ()
        self._index = 0
        self.widgets.board_widget.clear_highlights()

        return_mode = self._return_mode
        self._return_mode = AppMode.PLAY
        self._host.switch_mode(return_mode)
        if status_message is not None:
            self._host.set_status(status_message)
