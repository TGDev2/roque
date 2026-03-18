"""Side panels for evaluation, suggestions, and move history.

These are pure display widgets with no business logic.  They receive
pre-formatted data from the GUI controller.
"""

from __future__ import annotations

import logging
import tkinter as tk

from chesslab.analysis import Evaluation, VariationInfo
from chesslab.overlay.overlay_display import eval_to_bar_fraction, format_pv

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
