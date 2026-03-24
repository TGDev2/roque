"""Side panels for evaluation, suggestions, move history, and review.

These are pure display widgets with no business logic.  They receive
pre-formatted data from the GUI controller.
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Callable

import chess

from chesslab.analysis import Evaluation, VariationInfo
from chesslab.overlay.overlay_display import eval_to_bar_fraction, format_pv
from chesslab.progression import (
    ProgressionSnapshot,
    ProgressionTrendPoint,
    WeaknessMasteryState,
    WeaknessTrend,
)
from chesslab.review.game_reviewer import (
    ErrorMotif,
    GamePhase,
    GameReview,
    LessonTheme,
    MoveClassification,
    ReviewLesson,
    ReviewedMove,
)
from chesslab.training import (
    TrainingExercise,
    TrainingFeedbackLabel,
    TrainingMoveFeedback,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BG = "#16161e"
_CARD_BG = "#1e1e2e"
_CARD_BORDER = "#2a2a3c"
_FG = "#e0e0e8"
_FG_DIM = "#7a7a8c"
_FG_MUTED = "#555566"
_ACCENT = "#64b5f6"
_ACCENT_DIM = "#3d7ab8"
_BAR_WHITE = "#f0f0f0"
_BAR_BLACK = "#1a1a24"
_BAR_BORDER = "#3a3a4c"
_LOADING = "#ffb74d"
_EVAL_POSITIVE = "#66bb6a"
_EVAL_NEGATIVE = "#ef5350"
_EVAL_NEUTRAL = "#b0b0b8"
_MOVE_NUM_FG = "#5a5a6c"
_WHITE_MOVE_FG = "#e0e0e8"
_BLACK_MOVE_FG = "#b0b0c0"
_SCROLLBAR_BG = "#2a2a3c"
_SCROLLBAR_FG = "#4a4a5c"

# Classification colours.
_CLS_COLORS: dict[MoveClassification, str] = {
    MoveClassification.EXCELLENT: "#66bb6a",
    MoveClassification.GOOD: "#a5d6a7",
    MoveClassification.INACCURACY: "#ffee58",
    MoveClassification.MISTAKE: "#ffb74d",
    MoveClassification.BLUNDER: "#ef5350",
    MoveClassification.FORCED: "#7a7a8c",
}
_CLS_LABELS: dict[MoveClassification, str] = {
    MoveClassification.EXCELLENT: "Excellent",
    MoveClassification.GOOD: "Good",
    MoveClassification.INACCURACY: "Inaccuracy",
    MoveClassification.MISTAKE: "Mistake",
    MoveClassification.BLUNDER: "Blunder",
    MoveClassification.FORCED: "Forced",
}
_ERROR_CLASSIFICATIONS = frozenset({
    MoveClassification.INACCURACY,
    MoveClassification.MISTAKE,
    MoveClassification.BLUNDER,
})
_HIGHLIGHT_BG = "#2a3a4c"
_LESSON_BG = "#23283a"
_TREND_LINE = "#80cbc4"
_TREND_POINT = "#26a69a"

_THEME_LABELS: dict[LessonTheme, str] = {
    LessonTheme.PHASE_PATTERN: "Phase pattern",
    LessonTheme.TURNING_POINT: "Turning point",
    LessonTheme.ADVANTAGE_MANAGEMENT: "Advantage management",
}
_PHASE_LABELS: dict[GamePhase, str] = {
    GamePhase.OPENING: "Opening",
    GamePhase.MIDDLEGAME: "Middlegame",
    GamePhase.ENDGAME: "Endgame",
}
_ERROR_MOTIF_LABELS: dict[ErrorMotif, str] = {
    ErrorMotif.MISSED_TACTIC: "Tactique ratee",
    ErrorMotif.MATERIAL_LOSS: "Perte de materiel",
    ErrorMotif.PAWN_STRUCTURE: "Structure de pions",
    ErrorMotif.PIECE_ACTIVITY: "Activite des pieces",
    ErrorMotif.KING_SAFETY: "Securite du roi",
    ErrorMotif.ENDGAME_ERROR: "Erreur de finale",
    ErrorMotif.UNKNOWN: "Motif inconnu",
}
_TRAINING_FEEDBACK_COLORS: dict[TrainingFeedbackLabel, str] = {
    TrainingFeedbackLabel.GOOD: _EVAL_POSITIVE,
    TrainingFeedbackLabel.BETTER_AVAILABLE: _LOADING,
    TrainingFeedbackLabel.SIGNIFICANT_ERROR: _EVAL_NEGATIVE,
}
_TRAINING_FEEDBACK_LABELS: dict[TrainingFeedbackLabel, str] = {
    TrainingFeedbackLabel.GOOD: "Good move",
    TrainingFeedbackLabel.BETTER_AVAILABLE: "There was better",
    TrainingFeedbackLabel.SIGNIFICANT_ERROR: "Significant error",
}
_WEAKNESS_STATE_LABELS: dict[WeaknessMasteryState, str] = {
    WeaknessMasteryState.UNTRAINED: "Not trained",
    WeaknessMasteryState.ACTIVE: "Still fragile",
    WeaknessMasteryState.IMPROVING: "Improving",
    WeaknessMasteryState.MASTERED: "Mastered",
    WeaknessMasteryState.RELAPSING: "Repeating",
}
_WEAKNESS_TREND_LABELS: dict[WeaknessTrend, str] = {
    WeaknessTrend.IMPROVING: "Improving",
    WeaknessTrend.STABLE: "Stable",
    WeaknessTrend.REPEATING: "Repeating",
}
_WEAKNESS_TREND_COLORS: dict[WeaknessTrend, str] = {
    WeaknessTrend.IMPROVING: _EVAL_POSITIVE,
    WeaknessTrend.STABLE: _FG,
    WeaknessTrend.REPEATING: _EVAL_NEGATIVE,
}


def _card_frame(parent: tk.Widget, **kwargs: object) -> tk.Frame:
    """Create a card-style frame with subtle border."""
    outer = tk.Frame(parent, bg=_CARD_BORDER, bd=0, **kwargs)  # type: ignore[arg-type]
    inner = tk.Frame(outer, bg=_CARD_BG, bd=0)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    # Store inner ref so callers pack into inner.
    outer._inner = inner  # type: ignore[attr-defined]
    return outer


# ---------------------------------------------------------------------------
# Evaluation panel
# ---------------------------------------------------------------------------


class EvalPanel(tk.Frame):
    """Displays evaluation bar, numeric eval, and search depth."""

    def __init__(self, parent: tk.Widget, *, bar_height: int = 200) -> None:
        super().__init__(parent, bg=_BG)
        self._bar_height = bar_height

        card = _card_frame(self)
        card.pack(fill="x")
        inner = card._inner  # type: ignore[attr-defined]

        # Header row.
        header = tk.Frame(inner, bg=_CARD_BG)
        header.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(
            header, text="EVALUATION", bg=_CARD_BG, fg=_ACCENT,
            font=("sans-serif", 9, "bold"), anchor="w",
        ).pack(side="left")

        self._depth_label = tk.Label(
            header, text="", bg=_CARD_BG, fg=_FG_DIM,
            font=("monospace", 8), anchor="e",
        )
        self._depth_label.pack(side="right")

        # Content row: bar + eval text.
        content = tk.Frame(inner, bg=_CARD_BG)
        content.pack(fill="x", padx=8, pady=(0, 8))

        # Eval bar canvas — wider, with border.
        bar_frame = tk.Frame(content, bg=_BAR_BORDER)
        bar_frame.pack(side="left", padx=(0, 10))

        self._canvas = tk.Canvas(
            bar_frame, width=28, height=bar_height,
            bg=_BAR_BLACK, highlightthickness=0, bd=0,
        )
        self._canvas.pack(padx=1, pady=1)

        self._white_rect = self._canvas.create_rectangle(
            0, bar_height // 2, 28, bar_height,
            fill=_BAR_WHITE, outline="",
        )
        self._black_rect = self._canvas.create_rectangle(
            0, 0, 28, bar_height // 2,
            fill=_BAR_BLACK, outline="",
        )

        # Divider line at center.
        self._divider = self._canvas.create_line(
            0, bar_height // 2, 28, bar_height // 2,
            fill="#666677", width=1, dash=(2, 2),
        )

        # Text area next to bar.
        text_frame = tk.Frame(content, bg=_CARD_BG)
        text_frame.pack(side="left", fill="both", expand=True)

        self._eval_label = tk.Label(
            text_frame, text="- -", bg=_CARD_BG, fg=_FG,
            font=("sans-serif", 26, "bold"), anchor="w",
        )
        self._eval_label.pack(fill="x", pady=(8, 0))

        self._status_label = tk.Label(
            text_frame, text="", bg=_CARD_BG, fg=_FG_DIM,
            font=("sans-serif", 9), anchor="w",
        )
        self._status_label.pack(fill="x", pady=(2, 0))

    def update_eval(
        self,
        evaluation: Evaluation | None,
        depth: int,
        is_loading: bool,
    ) -> None:
        """Update the evaluation display."""
        if evaluation is None:
            eval_text = "..." if is_loading else "- -"
            bar_fraction = 0.5
            eval_colour = _LOADING if is_loading else _EVAL_NEUTRAL
        else:
            eval_text = str(evaluation)
            bar_fraction = eval_to_bar_fraction(evaluation)
            # Colour based on side advantage.
            if evaluation.is_mate:
                eval_colour = (
                    _EVAL_POSITIVE
                    if (evaluation.mate_in is not None and evaluation.mate_in > 0)
                    else _EVAL_NEGATIVE
                )
            elif evaluation.centipawns is not None:
                if evaluation.centipawns > 30:
                    eval_colour = _EVAL_POSITIVE
                elif evaluation.centipawns < -30:
                    eval_colour = _EVAL_NEGATIVE
                else:
                    eval_colour = _EVAL_NEUTRAL
            else:
                eval_colour = _EVAL_NEUTRAL

        self._eval_label.config(text=eval_text, fg=eval_colour)

        # Depth info.
        depth_text = f"d{depth}" if depth > 0 else ""
        self._depth_label.config(text=depth_text)

        # Status.
        if is_loading:
            self._status_label.config(text="Analysing...", fg=_LOADING)
        elif depth > 0:
            self._status_label.config(text=f"Depth {depth}", fg=_FG_DIM)
        else:
            self._status_label.config(text="")

        # Update bar.
        h = self._bar_height
        white_h = int(bar_fraction * h)
        black_h = h - white_h
        self._canvas.coords(self._black_rect, 0, 0, 28, black_h)
        self._canvas.coords(self._white_rect, 0, black_h, 28, h)
        self._canvas.coords(self._divider, 0, black_h, 28, black_h)


# ---------------------------------------------------------------------------
# Observe Live panel
# ---------------------------------------------------------------------------


class ObserveLivePanel(tk.Frame):
    """Passive-session panel shown while Observe Live is active."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=_BG)

        card = _card_frame(self)
        card.pack(fill="x")
        inner = card._inner  # type: ignore[attr-defined]

        tk.Label(
            inner, text="OBSERVE LIVE", bg=_CARD_BG, fg=_ACCENT,
            font=("sans-serif", 9, "bold"), anchor="w", padx=8,
        ).pack(fill="x", pady=(6, 4))

        self._badge = tk.Label(
            inner, text="PASSIVE SYNC", bg=_ACCENT_DIM, fg="#ffffff",
            font=("sans-serif", 9, "bold"), anchor="w", padx=8, pady=3,
        )
        self._badge.pack(anchor="w", padx=8, pady=(0, 6))

        self._headline = tk.Label(
            inner, text="", bg=_CARD_BG, fg=_FG,
            font=("sans-serif", 12, "bold"), anchor="w",
            justify="left", wraplength=240,
        )
        self._headline.pack(fill="x", padx=8)

        self._body = tk.Label(
            inner, text="", bg=_CARD_BG, fg=_FG_DIM,
            font=("sans-serif", 9), anchor="w",
            justify="left", wraplength=240,
        )
        self._body.pack(fill="x", padx=8, pady=(6, 10))

        self.update_session()

    def update_session(
        self,
        *,
        setup_label: str | None = None,
        move_count: int = 0,
    ) -> None:
        """Refresh passive-session copy for the current observation state."""
        if setup_label:
            headline = f"Passive observation active for {setup_label}"
        else:
            headline = "Passive observation is active"

        move_text = (
            "Waiting for the first synced move."
            if move_count == 0
            else f"{move_count} synced {'move' if move_count == 1 else 'moves'} captured."
        )
        body = (
            "ChessLab is syncing the visible board and move list only. "
            "Manual move entry, undo, and engine coaching stay unavailable "
            "until you explicitly exit to Manual Coach or Review.\n\n"
            f"{move_text}"
        )

        self._headline.config(text=headline)
        self._body.config(text=body)


# ---------------------------------------------------------------------------
# Suggestions panel
# ---------------------------------------------------------------------------


class SuggestionsPanel(tk.Frame):
    """Displays the top variations from Stockfish."""

    def __init__(self, parent: tk.Widget, *, max_lines: int = 3) -> None:
        super().__init__(parent, bg=_BG)
        self._max_lines = max_lines

        card = _card_frame(self)
        card.pack(fill="x")
        inner = card._inner  # type: ignore[attr-defined]

        tk.Label(
            inner, text="SUGGESTIONS", bg=_CARD_BG, fg=_ACCENT,
            font=("sans-serif", 9, "bold"), anchor="w", padx=8,
        ).pack(fill="x", pady=(6, 2))

        self._rows: list[dict[str, tk.Label]] = []
        for _ in range(max_lines):
            row_frame = tk.Frame(inner, bg=_CARD_BG)
            row_frame.pack(fill="x", padx=8, pady=1)

            rank_lbl = tk.Label(
                row_frame, text="", bg=_CARD_BG, fg=_FG_MUTED,
                font=("monospace", 9), anchor="w", width=2,
            )
            rank_lbl.pack(side="left")

            move_lbl = tk.Label(
                row_frame, text="", bg=_CARD_BG, fg=_FG,
                font=("sans-serif", 11, "bold"), anchor="w", width=6,
            )
            move_lbl.pack(side="left", padx=(0, 4))

            eval_lbl = tk.Label(
                row_frame, text="", bg=_CARD_BG, fg=_FG_DIM,
                font=("monospace", 9), anchor="w", width=7,
            )
            eval_lbl.pack(side="left", padx=(0, 4))

            pv_lbl = tk.Label(
                row_frame, text="", bg=_CARD_BG, fg=_FG_DIM,
                font=("monospace", 9), anchor="w",
            )
            pv_lbl.pack(side="left", fill="x", expand=True)

            self._rows.append({
                "frame": row_frame,
                "rank": rank_lbl,
                "move": move_lbl,
                "eval": eval_lbl,
                "pv": pv_lbl,
            })

        # Bottom padding.
        tk.Frame(inner, bg=_CARD_BG, height=4).pack()

    def update_suggestions(
        self,
        variations: tuple[VariationInfo, ...],
    ) -> None:
        """Update the displayed variations."""
        for i, row in enumerate(self._rows):
            if i < len(variations):
                v = variations[i]
                pv_text = format_pv(v.pv_san, max_moves=4)
                row["rank"].config(text=f"{v.rank}.")
                row["move"].config(
                    text=v.move_san,
                    fg=_ACCENT if v.rank == 1 else _FG,
                )

                # Colour eval.
                eval_str = str(v.evaluation)
                if v.evaluation.is_mate:
                    ec = _EVAL_POSITIVE if (
                        v.evaluation.mate_in is not None and v.evaluation.mate_in > 0
                    ) else _EVAL_NEGATIVE
                elif v.evaluation.centipawns is not None:
                    ec = (
                        _EVAL_POSITIVE if v.evaluation.centipawns > 30
                        else _EVAL_NEGATIVE if v.evaluation.centipawns < -30
                        else _EVAL_NEUTRAL
                    )
                else:
                    ec = _EVAL_NEUTRAL
                row["eval"].config(text=eval_str, fg=ec)
                row["pv"].config(text=pv_text)
            else:
                row["rank"].config(text="")
                row["move"].config(text="")
                row["eval"].config(text="")
                row["pv"].config(text="")


# ---------------------------------------------------------------------------
# Move history panel
# ---------------------------------------------------------------------------


class MoveHistoryPanel(tk.Frame):
    """Scrollable move history in paired-move format."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=_BG)

        card = _card_frame(self)
        card.pack(fill="both", expand=True)
        inner = card._inner  # type: ignore[attr-defined]

        tk.Label(
            inner, text="MOVES", bg=_CARD_BG, fg=_ACCENT,
            font=("sans-serif", 9, "bold"), anchor="w", padx=8,
        ).pack(fill="x", pady=(6, 2))

        # Scrollable text area with custom styling.
        text_frame = tk.Frame(inner, bg=_CARD_BG)
        text_frame.pack(fill="both", expand=True, padx=(8, 0), pady=(0, 6))

        self._text = tk.Text(
            text_frame, bg=_CARD_BG, fg=_FG, font=("monospace", 11),
            wrap="word", height=8, state="disabled",
            highlightthickness=0, borderwidth=0, padx=2, pady=2,
            insertbackground=_FG, selectbackground=_ACCENT_DIM,
            selectforeground="#ffffff",
        )
        scrollbar = tk.Scrollbar(
            text_frame, command=self._text.yview,
            bg=_SCROLLBAR_BG, troughcolor=_CARD_BG,
            activebackground=_SCROLLBAR_FG,
            highlightthickness=0, bd=0, width=8,
        )
        self._text.config(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._text.pack(fill="both", expand=True)

        # Configure text tags for styled move numbers.
        self._text.tag_configure("move_num", foreground=_MOVE_NUM_FG,
                                 font=("monospace", 10))
        self._text.tag_configure("white_move", foreground=_WHITE_MOVE_FG,
                                 font=("sans-serif", 11, "bold"))
        self._text.tag_configure("black_move", foreground=_BLACK_MOVE_FG,
                                 font=("sans-serif", 11))

    def set_history(self, moves: tuple[str, ...]) -> None:
        """Replace the entire move history display."""
        self._on_move_click: Callable[[int], None] | None = None
        self._text.config(state="normal")
        self._text.delete("1.0", "end")

        for i in range(0, len(moves), 2):
            move_num = i // 2 + 1
            white_move = moves[i]
            black_move = moves[i + 1] if i + 1 < len(moves) else ""

            if i > 0:
                self._text.insert("end", "\n")

            self._text.insert("end", f"{move_num}. ", "move_num")
            self._text.insert("end", f"{white_move}", "white_move")
            if black_move:
                self._text.insert("end", "  ")
                self._text.insert("end", f"{black_move}", "black_move")

        self._text.config(state="disabled")
        self._text.see("end")

    def set_review_history(
        self,
        moves: tuple[str, ...],
        classifications: tuple[MoveClassification, ...],
        on_move_click: Callable[[int], None] | None = None,
    ) -> None:
        """Display move history with classification colour-coding.

        Each move SAN is coloured according to its classification.  If
        *on_move_click* is provided, clicking a move navigates to that ply.
        """
        self._on_move_click = on_move_click
        self._text.config(state="normal")
        self._text.delete("1.0", "end")

        # Ensure classification tags exist.
        for cls, colour in _CLS_COLORS.items():
            tag = f"cls_{cls.value}"
            self._text.tag_configure(
                tag, foreground=colour, font=("sans-serif", 11, "bold"),
            )
        self._text.tag_configure("highlight", background=_HIGHLIGHT_BG)

        for i in range(0, len(moves), 2):
            move_num = i // 2 + 1
            white_move = moves[i]
            white_cls = classifications[i] if i < len(classifications) else None
            black_move = moves[i + 1] if i + 1 < len(moves) else ""
            black_cls = classifications[i + 1] if i + 1 < len(classifications) else None

            if i > 0:
                self._text.insert("end", "\n")

            self._text.insert("end", f"{move_num}. ", "move_num")

            white_tag = f"cls_{white_cls.value}" if white_cls else "white_move"
            move_tag = f"ply_{i}"
            self._text.tag_configure(move_tag)  # click target
            self._text.insert("end", white_move, (white_tag, move_tag))
            if on_move_click is not None:
                ply_idx = i
                self._text.tag_bind(
                    move_tag, "<Button-1>",
                    lambda e, p=ply_idx: self._on_move_click(p) if self._on_move_click else None,
                )

            if black_move:
                self._text.insert("end", "  ")
                black_tag = f"cls_{black_cls.value}" if black_cls else "black_move"
                move_tag_b = f"ply_{i + 1}"
                self._text.tag_configure(move_tag_b)
                self._text.insert("end", black_move, (black_tag, move_tag_b))
                if on_move_click is not None:
                    ply_idx_b = i + 1
                    self._text.tag_bind(
                        move_tag_b, "<Button-1>",
                        lambda e, p=ply_idx_b: self._on_move_click(p) if self._on_move_click else None,
                    )

        self._text.config(state="disabled")
        self._text.see("end")

    def highlight_ply(self, ply: int) -> None:
        """Highlight the move at the given half-move index."""
        self._text.tag_remove("highlight", "1.0", "end")
        tag = f"ply_{ply}"
        ranges = self._text.tag_ranges(tag)
        if ranges:
            self._text.tag_add("highlight", ranges[0], ranges[1])
            self._text.see(ranges[0])


# ---------------------------------------------------------------------------
# Review panel — per-move classification details
# ---------------------------------------------------------------------------


class ReviewPanel(tk.Frame):
    """Displays classification details for the currently selected review move."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=_BG)

        card = _card_frame(self)
        card.pack(fill="x")
        inner = card._inner  # type: ignore[attr-defined]

        tk.Label(
            inner, text="REVIEW", bg=_CARD_BG, fg=_ACCENT,
            font=("sans-serif", 9, "bold"), anchor="w", padx=8,
        ).pack(fill="x", pady=(6, 4))

        # Classification badge.
        self._badge = tk.Label(
            inner, text="", bg=_CARD_BG, fg=_FG,
            font=("sans-serif", 12, "bold"), anchor="w", padx=8,
        )
        self._badge.pack(fill="x")

        # Details grid.
        details = tk.Frame(inner, bg=_CARD_BG)
        details.pack(fill="x", padx=8, pady=(4, 6))

        self._played_label = self._detail_row(details, "Played")
        self._best_label = self._detail_row(details, "Best")
        self._loss_label = self._detail_row(details, "Loss")
        self._pv_label = self._detail_row(details, "Line")

    @staticmethod
    def _detail_row(parent: tk.Frame, title: str) -> tk.Label:
        row = tk.Frame(parent, bg=_CARD_BG)
        row.pack(fill="x", pady=1)
        tk.Label(
            row, text=f"{title}:", bg=_CARD_BG, fg=_FG_DIM,
            font=("sans-serif", 9), anchor="w", width=6,
        ).pack(side="left")
        value = tk.Label(
            row, text="", bg=_CARD_BG, fg=_FG,
            font=("sans-serif", 10), anchor="w",
        )
        value.pack(side="left", fill="x", expand=True)
        return value

    def update_move(self, move: ReviewedMove | None) -> None:
        """Display classification details for a reviewed move."""
        if move is None:
            self.clear()
            return

        cls_color = _CLS_COLORS.get(move.classification, _FG)
        cls_label = _CLS_LABELS.get(move.classification, "")
        if move.classification in _ERROR_CLASSIFICATIONS:
            motif_label = _ERROR_MOTIF_LABELS.get(move.error_motif, "")
            badge_label = f"{cls_label} · {motif_label}" if motif_label else cls_label
        else:
            badge_label = cls_label
        self._badge.config(text=badge_label, fg=cls_color)

        self._played_label.config(
            text=f"{move.san}  ({move.eval_after})",
        )
        self._best_label.config(
            text=f"{move.best_move_san}  ({move.best_eval})",
        )
        self._loss_label.config(
            text=f"{move.cp_loss} cp",
            fg=cls_color,
        )
        pv_text = format_pv(move.best_pv_san, max_moves=4)
        self._pv_label.config(text=pv_text)

    def clear(self) -> None:
        """Reset to empty state."""
        self._badge.config(text="")
        self._played_label.config(text="")
        self._best_label.config(text="")
        self._loss_label.config(text="")
        self._pv_label.config(text="")


# ---------------------------------------------------------------------------
# Review navigation panel
# ---------------------------------------------------------------------------


class ReviewNavigationPanel(tk.Frame):
    """Navigation buttons for review mode."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        on_first: Callable[[], None],
        on_prev: Callable[[], None],
        on_next: Callable[[], None],
        on_last: Callable[[], None],
        on_prev_error: Callable[[], None],
        on_next_error: Callable[[], None],
        on_training: Callable[[], None] | None,
        on_import_lichess: Callable[[], None] | None = None,
        on_import_chess_com: Callable[[], None] | None = None,
        on_exit_review: Callable[[], None],
        on_back_to_launcher: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, bg=_BG)

        self._on_back_to_launcher = on_back_to_launcher

        nav_row = tk.Frame(self, bg=_BG)
        nav_row.pack(fill="x")

        btn_data = [
            ("\u23ee", on_first),     # |<
            ("\u25c0", on_prev),      # <
            ("\u25b6", on_next),      # >
            ("\u23ed", on_last),      # >|
        ]
        for text, cmd in btn_data:
            b = tk.Button(
                nav_row, text=text, command=cmd,
                font=("sans-serif", 12), bg=_CARD_BG, fg=_FG,
                activebackground=_ACCENT_DIM, activeforeground="#ffffff",
                relief="flat", bd=0, padx=8, pady=4,
                cursor="hand2", highlightthickness=0,
            )
            b.pack(side="left", fill="x", expand=True, padx=1)

        err_row = tk.Frame(self, bg=_BG)
        err_row.pack(fill="x", pady=(2, 0))

        prev_err = tk.Button(
            err_row, text="\u25c0 Error", command=on_prev_error,
            font=("sans-serif", 9), bg=_CARD_BG, fg=_EVAL_NEGATIVE,
            activebackground=_ACCENT_DIM, activeforeground="#ffffff",
            relief="flat", bd=0, padx=6, pady=3,
            cursor="hand2", highlightthickness=0,
        )
        prev_err.pack(side="left", fill="x", expand=True, padx=(0, 1))

        next_err = tk.Button(
            err_row, text="Error \u25b6", command=on_next_error,
            font=("sans-serif", 9), bg=_CARD_BG, fg=_EVAL_NEGATIVE,
            activebackground=_ACCENT_DIM, activeforeground="#ffffff",
            relief="flat", bd=0, padx=6, pady=3,
            cursor="hand2", highlightthickness=0,
        )
        next_err.pack(side="left", fill="x", expand=True, padx=(1, 1))

        if on_training is not None:
            train_btn = tk.Button(
                err_row, text="Train weakness", command=on_training,
                font=("sans-serif", 9), bg=_CARD_BG, fg=_ACCENT,
                activebackground=_ACCENT_DIM, activeforeground="#ffffff",
                relief="flat", bd=0, padx=6, pady=3,
                cursor="hand2", highlightthickness=0,
            )
            train_btn.pack(side="left", fill="x", expand=True, padx=(1, 1))

        exit_btn = tk.Button(
            err_row, text="Exit Review", command=on_exit_review,
            font=("sans-serif", 9), bg=_CARD_BG, fg=_ACCENT,
            activebackground=_ACCENT_DIM, activeforeground="#ffffff",
            relief="flat", bd=0, padx=6, pady=3,
            cursor="hand2", highlightthickness=0,
        )
        exit_btn.pack(side="left", fill="x", expand=True, padx=(1, 0))

        if on_import_lichess is not None or on_import_chess_com is not None:
            import_row = tk.Frame(self, bg=_BG)
            import_row.pack(fill="x", pady=(2, 0))

            if on_import_lichess is not None:
                lichess_btn = tk.Button(
                    import_row,
                    text="Import Lichess",
                    command=on_import_lichess,
                    font=("sans-serif", 9),
                    bg=_CARD_BG,
                    fg=_FG,
                    activebackground=_ACCENT_DIM,
                    activeforeground="#ffffff",
                    relief="flat",
                    bd=0,
                    padx=6,
                    pady=3,
                    cursor="hand2",
                    highlightthickness=0,
                )
                lichess_btn.pack(side="left", fill="x", expand=True, padx=(0, 1))

            if on_import_chess_com is not None:
                chess_com_btn = tk.Button(
                    import_row,
                    text="Import Chess.com",
                    command=on_import_chess_com,
                    font=("sans-serif", 9),
                    bg=_CARD_BG,
                    fg=_FG,
                    activebackground=_ACCENT_DIM,
                    activeforeground="#ffffff",
                    relief="flat",
                    bd=0,
                    padx=6,
                    pady=3,
                    cursor="hand2",
                    highlightthickness=0,
                )
                chess_com_btn.pack(side="left", fill="x", expand=True, padx=(1, 0))

        # Optional "Back to Launcher" row — shown/hidden dynamically.
        self._launcher_row = tk.Frame(self, bg=_BG)
        self._launcher_btn = tk.Button(
            self._launcher_row,
            text="\u2302  Back to Launcher",
            command=self._on_back_to_launcher or (lambda: None),
            font=("sans-serif", 9),
            bg=_CARD_BG,
            fg=_FG_DIM,
            activebackground=_ACCENT_DIM,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=6,
            pady=3,
            cursor="hand2",
            highlightthickness=0,
        )
        self._launcher_btn.pack(fill="x")

        # Position indicator.
        self._pos_label = tk.Label(
            self, text="", bg=_BG, fg=_FG_DIM,
            font=("sans-serif", 9),
        )
        self._pos_label.pack(fill="x", pady=(2, 0))

    def update_position(self, current_ply: int, total_plies: int) -> None:
        """Update the position indicator text."""
        self._pos_label.config(text=f"Move {current_ply} / {total_plies}")

    def show_launcher_button(self, visible: bool) -> None:
        """Show or hide the 'Back to Launcher' button."""
        if visible and self._on_back_to_launcher is not None:
            self._launcher_row.pack(fill="x", pady=(2, 0), before=self._pos_label)
        else:
            self._launcher_row.pack_forget()


# ---------------------------------------------------------------------------
# Game summary panel
# ---------------------------------------------------------------------------


class GameSummaryPanel(tk.Frame):
    """Displays aggregated review statistics."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=_BG)

        card = _card_frame(self)
        card.pack(fill="x")
        self._inner = card._inner  # type: ignore[attr-defined]

        tk.Label(
            self._inner, text="SUMMARY", bg=_CARD_BG, fg=_ACCENT,
            font=("sans-serif", 9, "bold"), anchor="w", padx=8,
        ).pack(fill="x", pady=(6, 4))

        self._content = tk.Frame(self._inner, bg=_CARD_BG)
        self._content.pack(fill="x", padx=8, pady=(0, 6))

    def update_summary(
        self,
        review: GameReview,
        *,
        on_lesson_click: Callable[[int], None] | None = None,
    ) -> None:
        """Populate summary from a completed review."""
        for w in self._content.winfo_children():
            w.destroy()

        self._render_lessons(review, on_lesson_click=on_lesson_click)

        divider = tk.Frame(self._content, bg=_CARD_BORDER, height=1)
        divider.pack(fill="x", pady=(6, 6))

        # Classification distribution.
        cls_frame = tk.Frame(self._content, bg=_CARD_BG)
        cls_frame.pack(fill="x", pady=(0, 4))

        for cls in (
            MoveClassification.EXCELLENT,
            MoveClassification.GOOD,
            MoveClassification.INACCURACY,
            MoveClassification.MISTAKE,
            MoveClassification.BLUNDER,
        ):
            w_count = review.white_classification_counts.get(cls, 0)
            b_count = review.black_classification_counts.get(cls, 0)
            total = w_count + b_count
            if total == 0:
                continue
            row = tk.Frame(cls_frame, bg=_CARD_BG)
            row.pack(fill="x", pady=0)
            tk.Label(
                row, text=_CLS_LABELS[cls], bg=_CARD_BG,
                fg=_CLS_COLORS[cls], font=("sans-serif", 9),
                anchor="w", width=10,
            ).pack(side="left")
            tk.Label(
                row, text=str(total), bg=_CARD_BG, fg=_FG,
                font=("sans-serif", 9, "bold"), anchor="w",
            ).pack(side="left")

        # Average cp loss.
        avg_frame = tk.Frame(self._content, bg=_CARD_BG)
        avg_frame.pack(fill="x", pady=(4, 0))

        tk.Label(
            avg_frame, text="Avg loss:", bg=_CARD_BG, fg=_FG_DIM,
            font=("sans-serif", 9), anchor="w",
        ).pack(side="left")
        tk.Label(
            avg_frame,
            text=f"  W {review.white_avg_cp_loss:.0f}cp  B {review.black_avg_cp_loss:.0f}cp",
            bg=_CARD_BG, fg=_FG,
            font=("sans-serif", 9), anchor="w",
        ).pack(side="left")

        # Phase summaries.
        for ps in review.phase_summaries:
            ph_frame = tk.Frame(self._content, bg=_CARD_BG)
            ph_frame.pack(fill="x", pady=(2, 0))
            tk.Label(
                ph_frame,
                text=f"{ps.phase.value.capitalize()}: {ps.move_count} moves, avg {ps.avg_cp_loss:.0f}cp",
                bg=_CARD_BG, fg=_FG_DIM, font=("sans-serif", 8),
                anchor="w",
            ).pack(side="left")

    def _render_lessons(
        self,
        review: GameReview,
        *,
        on_lesson_click: Callable[[int], None] | None,
    ) -> None:
        section = tk.Frame(self._content, bg=_CARD_BG)
        section.pack(fill="x")

        tk.Label(
            section, text="Lessons", bg=_CARD_BG, fg=_ACCENT,
            font=("sans-serif", 9, "bold"), anchor="w",
        ).pack(fill="x")

        if not review.lessons:
            tk.Label(
                section,
                text=(
                    "No reliable lesson extracted for this game. This usually means "
                    "the sample was too short or too low-swing to support a clear takeaway."
                ),
                bg=_CARD_BG,
                fg=_FG_DIM,
                font=("sans-serif", 8),
                justify="left",
                anchor="w",
                wraplength=340,
            ).pack(fill="x", pady=(4, 0))
            return

        for lesson in review.lessons:
            self._render_lesson_card(
                section,
                review,
                lesson,
                on_lesson_click=on_lesson_click,
            )

    def _render_lesson_card(
        self,
        parent: tk.Frame,
        review: GameReview,
        lesson: ReviewLesson,
        *,
        on_lesson_click: Callable[[int], None] | None,
    ) -> None:
        outer = tk.Frame(parent, bg=_CARD_BORDER)
        outer.pack(fill="x", pady=(6, 0))

        inner = tk.Frame(outer, bg=_LESSON_BG)
        inner.pack(fill="x", padx=1, pady=1)

        header = tk.Frame(inner, bg=_LESSON_BG)
        header.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(
            header,
            text=f"#{lesson.priority}",
            bg=_ACCENT_DIM,
            fg="#ffffff",
            font=("sans-serif", 8, "bold"),
            padx=6,
            pady=2,
        ).pack(side="left")

        tk.Label(
            header,
            text=lesson.title,
            bg=_LESSON_BG,
            fg=_FG,
            font=("sans-serif", 9, "bold"),
            anchor="w",
            justify="left",
            wraplength=220,
        ).pack(side="left", fill="x", expand=True, padx=(8, 8))

        anchor = self._format_lesson_anchor(review, lesson)
        if on_lesson_click is not None:
            tk.Button(
                header,
                text=f"Go to {anchor}",
                command=lambda focus_ply=lesson.focus_ply: on_lesson_click(focus_ply),
                font=("sans-serif", 8),
                bg=_CARD_BG,
                fg=_ACCENT,
                activebackground=_ACCENT_DIM,
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                padx=6,
                pady=3,
                cursor="hand2",
                highlightthickness=0,
            ).pack(side="right")
        else:
            tk.Label(
                header,
                text=anchor,
                bg=_LESSON_BG,
                fg=_FG_DIM,
                font=("sans-serif", 8),
                anchor="e",
            ).pack(side="right")

        self._lesson_row(inner, "Fact", lesson.finding, _FG)
        self._lesson_row(inner, "Lesson", lesson.takeaway, _FG_DIM)

    @staticmethod
    def _lesson_row(
        parent: tk.Frame,
        label: str,
        text: str,
        value_fg: str,
    ) -> None:
        row = tk.Frame(parent, bg=_LESSON_BG)
        row.pack(fill="x", padx=8, pady=(0, 6))

        tk.Label(
            row,
            text=f"{label}:",
            bg=_LESSON_BG,
            fg=_ACCENT if label == "Lesson" else _FG_DIM,
            font=("sans-serif", 8, "bold"),
            anchor="nw",
            width=7,
        ).pack(side="left")

        tk.Label(
            row,
            text=text,
            bg=_LESSON_BG,
            fg=value_fg,
            font=("sans-serif", 8),
            justify="left",
            anchor="w",
            wraplength=300,
        ).pack(side="left", fill="x", expand=True)

    @staticmethod
    def _format_lesson_anchor(review: GameReview, lesson: ReviewLesson) -> str:
        if 0 <= lesson.focus_ply < len(review.moves):
            move = review.moves[lesson.focus_ply]
            move_num = move.ply // 2 + 1
            dot = "." if move.is_white else "..."
            phase = lesson.phase.value.capitalize() if lesson.phase is not None else "Game"
            return f"{phase} {move_num}{dot} {move.san}"

        if lesson.phase is not None:
            return lesson.phase.value.capitalize()
        return "position"


# ---------------------------------------------------------------------------
# Training panel
# ---------------------------------------------------------------------------


class TrainingPanel(tk.Frame):
    """Displays a targeted training exercise and move feedback."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        on_next: Callable[[], None],
        on_exit: Callable[[], None],
    ) -> None:
        super().__init__(parent, bg=_BG)

        card = _card_frame(self)
        card.pack(fill="both", expand=True)
        inner = card._inner  # type: ignore[attr-defined]

        header = tk.Frame(inner, bg=_CARD_BG)
        header.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(
            header,
            text="TRAINING",
            bg=_CARD_BG,
            fg=_ACCENT,
            font=("sans-serif", 9, "bold"),
            anchor="w",
        ).pack(side="left")

        self._progress_label = tk.Label(
            header,
            text="",
            bg=_CARD_BG,
            fg=_FG_DIM,
            font=("sans-serif", 8),
            anchor="e",
        )
        self._progress_label.pack(side="right")

        self._content = tk.Frame(inner, bg=_CARD_BG)
        self._content.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._prompt_label = tk.Label(
            self._content,
            text="",
            bg=_CARD_BG,
            fg=_FG,
            font=("sans-serif", 12, "bold"),
            anchor="w",
            justify="left",
            wraplength=320,
        )
        self._prompt_label.pack(fill="x", pady=(0, 6))

        self._focus_label = tk.Label(
            self._content,
            text="",
            bg=_CARD_BG,
            fg=_ACCENT,
            font=("sans-serif", 9, "bold"),
            anchor="w",
            justify="left",
            wraplength=320,
        )
        self._focus_label.pack(fill="x")

        self._source_label = tk.Label(
            self._content,
            text="",
            bg=_CARD_BG,
            fg=_FG_DIM,
            font=("sans-serif", 8),
            anchor="w",
            justify="left",
            wraplength=320,
        )
        self._source_label.pack(fill="x", pady=(2, 6))

        self._context_label = tk.Label(
            self._content,
            text="",
            bg=_LESSON_BG,
            fg=_FG,
            font=("sans-serif", 9),
            anchor="w",
            justify="left",
            wraplength=320,
            padx=8,
            pady=8,
        )
        self._context_label.pack(fill="x", pady=(0, 8))

        self._feedback_badge = tk.Label(
            self._content,
            text="Awaiting your move",
            bg=_ACCENT_DIM,
            fg="#ffffff",
            font=("sans-serif", 9, "bold"),
            anchor="w",
            padx=8,
            pady=3,
        )
        self._feedback_badge.pack(anchor="w", pady=(0, 6))

        self._feedback_body = tk.Label(
            self._content,
            text="Play the move you would choose on the board.",
            bg=_CARD_BG,
            fg=_FG,
            font=("sans-serif", 9),
            anchor="w",
            justify="left",
            wraplength=320,
        )
        self._feedback_body.pack(fill="x")

        self._best_label = tk.Label(
            self._content,
            text="",
            bg=_CARD_BG,
            fg=_FG,
            font=("sans-serif", 9),
            anchor="w",
            justify="left",
            wraplength=320,
        )
        self._best_label.pack(fill="x", pady=(8, 2))

        self._top_label = tk.Label(
            self._content,
            text="",
            bg=_CARD_BG,
            fg=_FG_DIM,
            font=("sans-serif", 8),
            anchor="w",
            justify="left",
            wraplength=320,
        )
        self._top_label.pack(fill="x")

        button_row = tk.Frame(self._content, bg=_CARD_BG)
        button_row.pack(fill="x", pady=(10, 0))

        self._next_btn = tk.Button(
            button_row,
            text="Next position",
            command=on_next,
            font=("sans-serif", 9, "bold"),
            bg=_CARD_BG,
            fg=_ACCENT,
            activebackground=_ACCENT_DIM,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=8,
            pady=4,
            cursor="hand2",
            highlightthickness=0,
        )
        self._next_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        tk.Button(
            button_row,
            text="Exit training",
            command=on_exit,
            font=("sans-serif", 9),
            bg=_CARD_BG,
            fg=_FG_DIM,
            activebackground=_ACCENT_DIM,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=8,
            pady=4,
            cursor="hand2",
            highlightthickness=0,
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))

    def show_exercise(
        self,
        exercise: TrainingExercise,
        *,
        index: int,
        total: int,
    ) -> None:
        """Render the current training exercise before the user plays a move."""
        board = chess.Board(exercise.position.fen)
        side = "White" if board.turn == chess.WHITE else "Black"
        reviewed_date = exercise.reviewed_at.split("T", 1)[0]
        classification = exercise.position.classification.value.replace("_", " ")

        self._progress_label.config(text=f"{index + 1} / {total}")
        self._prompt_label.config(
            text=f"{side} to move. Play the move you would choose.",
        )
        self._focus_label.config(text=exercise.focus_reason)
        self._source_label.config(
            text=(
                f"From your review history on {reviewed_date}. "
                f"{exercise.mastery_summary}"
            ),
        )
        self._context_label.config(
            text=(
                f"In the original game, {exercise.position.played_san} was marked as a "
                f"{classification} and cost about {exercise.position.cp_loss}cp."
            ),
        )
        self._feedback_badge.config(
            text="Awaiting your move",
            bg=_ACCENT_DIM,
            fg="#ffffff",
        )
        self._feedback_body.config(
            text="Play a move on the board to get immediate engine feedback.",
            fg=_FG,
        )
        self._best_label.config(text="")
        self._top_label.config(text="")
        self._next_btn.config(text="Next position" if index + 1 < total else "Finish")

    def show_loading(self, played_san: str) -> None:
        """Show a short loading state while the engine checks the move."""
        self._feedback_badge.config(
            text="Checking move",
            bg=_LOADING,
            fg="#1a1a1a",
        )
        self._feedback_body.config(
            text=f"Stockfish is evaluating {played_san} against the top 3 choices.",
            fg=_FG,
        )

    def show_feedback(
        self,
        feedback: TrainingMoveFeedback,
        *,
        is_last: bool,
    ) -> None:
        """Render feedback after the player submits a move."""
        colour = _TRAINING_FEEDBACK_COLORS[feedback.label]
        badge_label = _TRAINING_FEEDBACK_LABELS[feedback.label]
        self._feedback_badge.config(
            text=badge_label,
            bg=colour,
            fg="#ffffff" if feedback.label is not TrainingFeedbackLabel.BETTER_AVAILABLE else "#1a1a1a",
        )
        self._feedback_body.config(text=feedback.message, fg=_FG)

        best_text = (
            f"Best move: {feedback.best_move_san}"
            if feedback.best_move_san
            else "Best move unavailable"
        )
        if feedback.best_line_san:
            best_text += f"  |  line: {format_pv(feedback.best_line_san, max_moves=4)}"
        self._best_label.config(text=best_text)

        if feedback.top_move_sans:
            self._top_label.config(
                text="Top 3: " + ", ".join(feedback.top_move_sans),
            )
        else:
            self._top_label.config(text="")

        self._next_btn.config(text="Finish" if is_last else "Next position")


# ---------------------------------------------------------------------------
# Progression panel
# ---------------------------------------------------------------------------


class ProgressionPanel(tk.Frame):
    """Displays longitudinal metrics across persisted game reviews."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=_BG)

        card = _card_frame(self)
        card.pack(fill="both", expand=True)
        self._inner = card._inner  # type: ignore[attr-defined]

        header = tk.Frame(self._inner, bg=_CARD_BG)
        header.pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(
            header,
            text="PROGRESSION",
            bg=_CARD_BG,
            fg=_ACCENT,
            font=("sans-serif", 9, "bold"),
            anchor="w",
        ).pack(side="left")

        self._sample_label = tk.Label(
            header,
            text="",
            bg=_CARD_BG,
            fg=_FG_DIM,
            font=("sans-serif", 8),
            anchor="e",
        )
        self._sample_label.pack(side="right")

        self._content = tk.Frame(self._inner, bg=_CARD_BG)
        self._content.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def show_error(self, message: str) -> None:
        """Display a storage or aggregation error message."""
        self._sample_label.config(text="Unavailable")
        self._clear_content()
        tk.Label(
            self._content,
            text=message,
            bg=_CARD_BG,
            fg=_EVAL_NEGATIVE,
            font=("sans-serif", 9),
            justify="left",
            wraplength=360,
            anchor="w",
        ).pack(fill="x", pady=(4, 0))

    def update_snapshot(self, snapshot: ProgressionSnapshot) -> None:
        """Render a progression snapshot."""
        self._clear_content()
        self._sample_label.config(
            text=f"{snapshot.analysed_games} games / window {snapshot.sample_size}",
        )

        if not snapshot.has_enough_games:
            self._render_empty_state(snapshot)
            if snapshot.tracked_weaknesses:
                self._divider()
                self._render_training_follow_up(snapshot)
                self._divider()
                self._render_urgent_weaknesses(snapshot)
                self._divider()
                self._render_training_themes(snapshot)
            return

        self._render_classifications(snapshot)
        self._divider()
        self._render_cp_loss_trend(snapshot)
        self._divider()
        self._render_themes(snapshot)
        self._divider()
        self._render_error_motifs(snapshot)
        self._divider()
        self._render_error_phases(snapshot)
        self._divider()
        self._render_training_follow_up(snapshot)
        self._divider()
        self._render_urgent_weaknesses(snapshot)
        self._divider()
        self._render_training_themes(snapshot)

    def _render_empty_state(self, snapshot: ProgressionSnapshot) -> None:
        tk.Label(
            self._content,
            text=(
                f"Analyse at least {snapshot.minimum_games} games to unlock the "
                f"progression dashboard. {snapshot.analysed_games} review"
                f"{'' if snapshot.analysed_games == 1 else 's'} recorded so far; "
                f"{snapshot.games_remaining} more to go."
            ),
            bg=_CARD_BG,
            fg=_FG,
            font=("sans-serif", 10, "bold"),
            justify="left",
            wraplength=360,
            anchor="w",
        ).pack(fill="x", pady=(6, 6))

        tk.Label(
            self._content,
            text=(
                "Run more post-game reviews to track recurring lessons, error "
                "motifs, error phases, whether your average centipawn loss is trending down, "
                "and how your targeted practice is holding up."
            ),
            bg=_CARD_BG,
            fg=_FG_DIM,
            font=("sans-serif", 9),
            justify="left",
            wraplength=360,
            anchor="w",
        ).pack(fill="x")

    def _render_classifications(self, snapshot: ProgressionSnapshot) -> None:
        self._section_title("Move classifications")
        for cls in (
            MoveClassification.EXCELLENT,
            MoveClassification.GOOD,
            MoveClassification.INACCURACY,
            MoveClassification.MISTAKE,
            MoveClassification.BLUNDER,
        ):
            count = snapshot.classification_counts.get(cls, 0)
            percentage = snapshot.classification_percentages.get(cls, 0.0)
            self._metric_row(
                label=_CLS_LABELS[cls],
                value=f"{percentage:.1f}%  ({count})",
                label_fg=_CLS_COLORS[cls],
            )

    def _render_cp_loss_trend(self, snapshot: ProgressionSnapshot) -> None:
        self._section_title("Average cp loss per game")

        trend_label = "Stable"
        trend_colour = _FG
        if snapshot.avg_cp_loss_delta <= -10:
            trend_label = "Improving"
            trend_colour = _EVAL_POSITIVE
        elif snapshot.avg_cp_loss_delta >= 10:
            trend_label = "Worsening"
            trend_colour = _EVAL_NEGATIVE

        tk.Label(
            self._content,
            text=(
                f"{trend_label}  |  sample avg {snapshot.sample_avg_cp_loss:.0f}cp  "
                f"|  delta {snapshot.avg_cp_loss_delta:+.0f}cp"
            ),
            bg=_CARD_BG,
            fg=trend_colour,
            font=("sans-serif", 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 4))

        if not snapshot.cp_loss_trend:
            tk.Label(
                self._content,
                text="No reviewed games yet.",
                bg=_CARD_BG,
                fg=_FG_DIM,
                font=("sans-serif", 9),
                anchor="w",
            ).pack(fill="x")
            return

        self._draw_sparkline(snapshot.cp_loss_trend)

    def _render_themes(self, snapshot: ProgressionSnapshot) -> None:
        self._section_title("Recurring lesson themes")
        if not snapshot.top_themes:
            tk.Label(
                self._content,
                text="No recurring lesson theme extracted yet in this sample.",
                bg=_CARD_BG,
                fg=_FG_DIM,
                font=("sans-serif", 9),
                anchor="w",
                justify="left",
                wraplength=360,
            ).pack(fill="x")
            return

        for stat in snapshot.top_themes:
            self._metric_row(
                label=_THEME_LABELS[stat.theme],
                value=f"{stat.count} lesson{'s' if stat.count != 1 else ''}",
            )

    def _render_error_motifs(self, snapshot: ProgressionSnapshot) -> None:
        self._section_title("Error motifs")
        if not snapshot.top_error_motifs:
            tk.Label(
                self._content,
                text="No reliable error motif extracted yet in this sample.",
                bg=_CARD_BG,
                fg=_FG_DIM,
                font=("sans-serif", 9),
                anchor="w",
                justify="left",
                wraplength=360,
            ).pack(fill="x")
            return

        for stat in snapshot.top_error_motifs:
            self._metric_row(
                label=_ERROR_MOTIF_LABELS[stat.motif],
                value=f"{stat.percentage:.1f}%  ({stat.count})",
            )

    def _render_error_phases(self, snapshot: ProgressionSnapshot) -> None:
        self._section_title("Errors by phase")
        tk.Label(
            self._content,
            text="Errors = inaccuracies, mistakes, and blunders.",
            bg=_CARD_BG,
            fg=_FG_DIM,
            font=("sans-serif", 8),
            anchor="w",
        ).pack(fill="x", pady=(0, 4))

        for stat in snapshot.error_phases:
            self._metric_row(
                label=_PHASE_LABELS[stat.phase],
                value=f"{stat.percentage:.1f}%  ({stat.count})",
            )

    def _render_training_follow_up(self, snapshot: ProgressionSnapshot) -> None:
        self._section_title("Training follow-up")

        trend_colour = _FG
        if snapshot.relapsing_weaknesses:
            trend_colour = _EVAL_NEGATIVE
        elif snapshot.mastered_weaknesses:
            trend_colour = _EVAL_POSITIVE

        tk.Label(
            self._content,
            text=(
                f"{snapshot.tracked_weaknesses} tracked weakness"
                f"{'' if snapshot.tracked_weaknesses == 1 else 'es'}  |  "
                f"{snapshot.training_attempt_count} attempt"
                f"{'' if snapshot.training_attempt_count == 1 else 's'}  |  "
                f"{snapshot.training_success_rate:.0f}% good answers"
            ),
            bg=_CARD_BG,
            fg=trend_colour,
            font=("sans-serif", 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 4))

        self._metric_row(
            label="Mastered weaknesses",
            value=str(snapshot.mastered_weaknesses),
        )
        self._metric_row(
            label="Not mastered yet",
            value=str(max(0, snapshot.tracked_weaknesses - snapshot.mastered_weaknesses)),
        )
        self._metric_row(
            label="Repeating lately",
            value=str(snapshot.relapsing_weaknesses),
        )

    def _render_urgent_weaknesses(self, snapshot: ProgressionSnapshot) -> None:
        self._section_title("Weaknesses to revisit")
        if not snapshot.urgent_weaknesses:
            tk.Label(
                self._content,
                text="No urgent weakness stands out in this sample right now.",
                bg=_CARD_BG,
                fg=_FG_DIM,
                font=("sans-serif", 9),
                anchor="w",
                justify="left",
                wraplength=360,
            ).pack(fill="x")
            return

        for stat in snapshot.urgent_weaknesses:
            self._metric_row(
                label=stat.label,
                value=(
                    f"{_WEAKNESS_STATE_LABELS[stat.mastery_state]}  |  "
                    f"{stat.attempt_count} attempt"
                    f"{'' if stat.attempt_count == 1 else 's'}  |  "
                    f"{stat.success_rate:.0f}%"
                ),
                label_fg=_WEAKNESS_TREND_COLORS[stat.trend],
            )

    def _render_training_themes(self, snapshot: ProgressionSnapshot) -> None:
        self._section_title("Practice by theme")
        if not snapshot.training_themes:
            tk.Label(
                self._content,
                text="No lesson theme has enough reusable training positions yet.",
                bg=_CARD_BG,
                fg=_FG_DIM,
                font=("sans-serif", 9),
                anchor="w",
                justify="left",
                wraplength=360,
            ).pack(fill="x")
            return

        for stat in snapshot.training_themes:
            attempt_text = (
                "No attempts yet"
                if stat.attempt_count == 0
                else f"{stat.success_rate:.0f}% good"
            )
            self._metric_row(
                label=_THEME_LABELS[stat.theme],
                value=(
                    f"{_WEAKNESS_TREND_LABELS[stat.trend]}  |  "
                    f"{stat.urgent_count} urgent  |  {attempt_text}"
                ),
                label_fg=_WEAKNESS_TREND_COLORS[stat.trend],
            )

    def _draw_sparkline(
        self,
        points: tuple[ProgressionTrendPoint, ...],
    ) -> None:
        width = 360
        height = 92
        pad_x = 12
        pad_y = 14
        canvas = tk.Canvas(
            self._content,
            width=width,
            height=height,
            bg=_LESSON_BG,
            bd=0,
            highlightthickness=1,
            highlightbackground=_CARD_BORDER,
        )
        canvas.pack(fill="x", pady=(0, 4))

        values = [point.avg_cp_loss for point in points]
        low = min(values)
        high = max(values)
        spread = high - low
        usable_w = width - 2 * pad_x
        usable_h = height - 2 * pad_y

        coords: list[float] = []
        for idx, value in enumerate(values):
            x = (
                width / 2
                if len(values) == 1
                else pad_x + usable_w * idx / (len(values) - 1)
            )
            if spread == 0:
                y = pad_y + usable_h / 2
            else:
                y = pad_y + usable_h * (1 - ((value - low) / spread))
            coords.extend([x, y])

        if len(coords) >= 4:
            canvas.create_line(*coords, fill=_TREND_LINE, width=2, smooth=True)

        for idx in range(0, len(coords), 2):
            x = coords[idx]
            y = coords[idx + 1]
            canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3,
                fill=_TREND_POINT,
                outline="",
            )

        canvas.create_text(
            pad_x,
            height - 8,
            text=self._format_date(points[0].reviewed_at),
            fill=_FG_DIM,
            font=("sans-serif", 8),
            anchor="sw",
        )
        canvas.create_text(
            width - pad_x,
            height - 8,
            text=self._format_date(points[-1].reviewed_at),
            fill=_FG_DIM,
            font=("sans-serif", 8),
            anchor="se",
        )
        canvas.create_text(
            width - pad_x,
            8,
            text=f"{high:.0f}cp",
            fill=_FG_DIM,
            font=("sans-serif", 8),
            anchor="ne",
        )
        canvas.create_text(
            width - pad_x,
            height - 20,
            text=f"{low:.0f}cp",
            fill=_FG_DIM,
            font=("sans-serif", 8),
            anchor="se",
        )

    def _section_title(self, text: str) -> None:
        tk.Label(
            self._content,
            text=text,
            bg=_CARD_BG,
            fg=_ACCENT,
            font=("sans-serif", 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 4))

    def _metric_row(
        self,
        *,
        label: str,
        value: str,
        label_fg: str = _FG,
    ) -> None:
        row = tk.Frame(self._content, bg=_CARD_BG)
        row.pack(fill="x", pady=(0, 2))

        tk.Label(
            row,
            text=label,
            bg=_CARD_BG,
            fg=label_fg,
            font=("sans-serif", 9),
            anchor="w",
        ).pack(side="left")

        tk.Label(
            row,
            text=value,
            bg=_CARD_BG,
            fg=_FG,
            font=("sans-serif", 9, "bold"),
            anchor="e",
        ).pack(side="right")

    def _divider(self) -> None:
        tk.Frame(self._content, bg=_CARD_BORDER, height=1).pack(fill="x", pady=8)

    def _clear_content(self) -> None:
        for child in self._content.winfo_children():
            child.destroy()

    @staticmethod
    def _format_date(timestamp: str) -> str:
        return timestamp[:10]
