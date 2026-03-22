"""Side panels for evaluation, suggestions, move history, and review.

These are pure display widgets with no business logic.  They receive
pre-formatted data from the GUI controller.
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Callable

from chesslab.analysis import Evaluation, VariationInfo
from chesslab.overlay.overlay_display import eval_to_bar_fraction, format_pv
from chesslab.review.game_reviewer import (
    GamePhase,
    GameReview,
    MoveClassification,
    ReviewedMove,
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
_HIGHLIGHT_BG = "#2a3a4c"


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
        self._badge.config(text=cls_label, fg=cls_color)

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
        on_exit_review: Callable[[], None],
    ) -> None:
        super().__init__(parent, bg=_BG)

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

        exit_btn = tk.Button(
            err_row, text="Exit Review", command=on_exit_review,
            font=("sans-serif", 9), bg=_CARD_BG, fg=_ACCENT,
            activebackground=_ACCENT_DIM, activeforeground="#ffffff",
            relief="flat", bd=0, padx=6, pady=3,
            cursor="hand2", highlightthickness=0,
        )
        exit_btn.pack(side="left", fill="x", expand=True, padx=(1, 0))

        # Position indicator.
        self._pos_label = tk.Label(
            self, text="", bg=_BG, fg=_FG_DIM,
            font=("sans-serif", 9),
        )
        self._pos_label.pack(fill="x", pady=(2, 0))

    def update_position(self, current_ply: int, total_plies: int) -> None:
        """Update the position indicator text."""
        self._pos_label.config(text=f"Move {current_ply} / {total_plies}")


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

    def update_summary(self, review: GameReview) -> None:
        """Populate summary from a completed review."""
        for w in self._content.winfo_children():
            w.destroy()

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
