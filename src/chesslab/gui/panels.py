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
# Colour palette (consistent with overlay)
# ---------------------------------------------------------------------------

_BG = "#1e1e1e"
_FG = "#e0e0e0"
_FG_DIM = "#888888"
_ACCENT = "#4fc3f7"
_BAR_WHITE = "#f0f0f0"
_BAR_BLACK = "#2a2a2a"
_LOADING = "#ffa726"


# ---------------------------------------------------------------------------
# Evaluation panel
# ---------------------------------------------------------------------------


class EvalPanel(tk.Frame):
    """Displays evaluation bar, numeric eval, and search depth."""

    def __init__(self, parent: tk.Widget, *, bar_height: int = 180) -> None:
        super().__init__(parent, bg=_BG)
        self._bar_height = bar_height

        # Header.
        tk.Label(
            self, text="Evaluation", bg=_BG, fg=_ACCENT,
            font=("sans-serif", 11, "bold"), anchor="w", padx=6,
        ).pack(fill="x")

        # Eval bar canvas.
        self._canvas = tk.Canvas(
            self, width=40, height=bar_height,
            bg=_BAR_BLACK, highlightthickness=0,
        )
        self._canvas.pack(side="left", padx=(6, 4), pady=4)
        self._white_rect = self._canvas.create_rectangle(
            0, bar_height // 2, 40, bar_height,
            fill=_BAR_WHITE, outline="",
        )
        self._black_rect = self._canvas.create_rectangle(
            0, 0, 40, bar_height // 2,
            fill=_BAR_BLACK, outline="",
        )

        # Text area next to bar.
        text_frame = tk.Frame(self, bg=_BG)
        text_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self._eval_label = tk.Label(
            text_frame, text="--", bg=_BG, fg=_FG,
            font=("monospace", 18, "bold"), anchor="w",
        )
        self._eval_label.pack(fill="x", pady=(4, 0))

        self._depth_label = tk.Label(
            text_frame, text="", bg=_BG, fg=_FG_DIM,
            font=("monospace", 10), anchor="w",
        )
        self._depth_label.pack(fill="x")

    def update_eval(
        self,
        evaluation: Evaluation | None,
        depth: int,
        is_loading: bool,
    ) -> None:
        """Update the evaluation display."""
        if evaluation is None:
            eval_text = "..."  if is_loading else "--"
            bar_fraction = 0.5
        else:
            eval_text = str(evaluation)
            bar_fraction = eval_to_bar_fraction(evaluation)

        self._eval_label.config(
            text=eval_text,
            fg=_LOADING if is_loading else _FG,
        )

        depth_text = f"depth {depth}" if depth > 0 else ""
        if is_loading:
            depth_text += "  analysing..." if depth_text else "analysing..."
        self._depth_label.config(text=depth_text)

        # Update bar.
        h = self._bar_height
        white_h = int(bar_fraction * h)
        black_h = h - white_h
        self._canvas.coords(self._black_rect, 0, 0, 40, black_h)
        self._canvas.coords(self._white_rect, 0, black_h, 40, h)


# ---------------------------------------------------------------------------
# Suggestions panel
# ---------------------------------------------------------------------------


class SuggestionsPanel(tk.Frame):
    """Displays the top variations from Stockfish."""

    def __init__(self, parent: tk.Widget, *, max_lines: int = 3) -> None:
        super().__init__(parent, bg=_BG)
        self._max_lines = max_lines

        tk.Label(
            self, text="Suggestions", bg=_BG, fg=_ACCENT,
            font=("sans-serif", 11, "bold"), anchor="w", padx=6,
        ).pack(fill="x")

        self._labels: list[tk.Label] = []
        for _ in range(max_lines):
            lbl = tk.Label(
                self, text="", bg=_BG, fg=_FG_DIM,
                font=("monospace", 11), anchor="w", padx=6, pady=1,
            )
            lbl.pack(fill="x")
            self._labels.append(lbl)

    def update_suggestions(
        self,
        variations: tuple[VariationInfo, ...],
    ) -> None:
        """Update the displayed variations."""
        for i, lbl in enumerate(self._labels):
            if i < len(variations):
                v = variations[i]
                pv_text = format_pv(v.pv_san, max_moves=5)
                lbl.config(
                    text=f"{v.rank}. {v.move_san}  {v.evaluation}  {pv_text}",
                    fg=_FG if v.rank == 1 else _FG_DIM,
                )
            else:
                lbl.config(text="")


# ---------------------------------------------------------------------------
# Move history panel
# ---------------------------------------------------------------------------


class MoveHistoryPanel(tk.Frame):
    """Scrollable move history in paired-move format."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=_BG)

        tk.Label(
            self, text="Moves", bg=_BG, fg=_ACCENT,
            font=("sans-serif", 11, "bold"), anchor="w", padx=6,
        ).pack(fill="x")

        # Scrollable text area.
        self._text = tk.Text(
            self, bg=_BG, fg=_FG, font=("monospace", 11),
            wrap="word", height=8, state="disabled",
            highlightthickness=0, borderwidth=0, padx=6, pady=4,
        )
        scrollbar = tk.Scrollbar(self, command=self._text.yview)
        self._text.config(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._text.pack(fill="both", expand=True)

    def set_history(self, moves: tuple[str, ...]) -> None:
        """Replace the entire move history display."""
        self._text.config(state="normal")
        self._text.delete("1.0", "end")

        lines: list[str] = []
        for i in range(0, len(moves), 2):
            move_num = i // 2 + 1
            white_move = moves[i]
            black_move = moves[i + 1] if i + 1 < len(moves) else ""
            if black_move:
                lines.append(f"{move_num}. {white_move}  {black_move}")
            else:
                lines.append(f"{move_num}. {white_move}")

        self._text.insert("1.0", "\n".join(lines))
        self._text.config(state="disabled")
        self._text.see("end")
