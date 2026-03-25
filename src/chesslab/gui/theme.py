"""Shared Tkinter theme tokens for ChessLab GUI components."""

from __future__ import annotations

import tkinter as tk

from chesslab.progression import WeaknessTrend
from chesslab.review.game_reviewer import MoveClassification
from chesslab.training import TrainingFeedbackLabel

BG = "#16161e"
BG_SECONDARY = "#1e1e2e"
CARD_BG = BG_SECONDARY
CARD_BORDER = "#2a2a3c"
FG = "#e0e0e8"
FG_DIM = "#7a7a8c"
FG_MUTED = "#555566"
ACCENT = "#64b5f6"
ACCENT_HOVER = "#90caf9"
ACCENT_DARK = "#3d7ab8"
ERROR = "#ef5350"
SUCCESS = "#66bb6a"
LOADING = "#ffb74d"
STATUS_BG = "#12121a"

BTN_BG = "#2a2a3c"
BTN_FG = "#e0e0e8"
BTN_HOVER = "#3a3a4c"
BTN_ACTIVE = "#4a4a5c"
BTN_ACCENT_BG = "#2962ff"
BTN_ACCENT_FG = "#ffffff"
BTN_ACCENT_HOVER = "#448aff"
BTN_WHITE_BG = "#e8e0d4"
BTN_WHITE_FG = "#1a1a1a"
BTN_WHITE_HOVER = "#f0e8dc"
BTN_BLACK_BG = "#2a2a3c"
BTN_BLACK_FG = "#e0e0e8"
BTN_BLACK_HOVER = "#3a3a4c"

ENTRY_BG = "#1a1a2e"
ENTRY_FG = "#e0e0e8"
ENTRY_BORDER = "#3a3a4c"
ENTRY_FOCUS_BORDER = ACCENT

BAR_WHITE = "#f0f0f0"
BAR_BLACK = "#1a1a24"
BAR_BORDER = "#3a3a4c"
EVAL_POSITIVE = SUCCESS
EVAL_NEGATIVE = ERROR
EVAL_NEUTRAL = "#b0b0b8"
MOVE_NUM_FG = "#5a5a6c"
WHITE_MOVE_FG = "#e0e0e8"
BLACK_MOVE_FG = "#b0b0c0"
SCROLLBAR_BG = "#2a2a3c"
SCROLLBAR_FG = "#4a4a5c"
HIGHLIGHT_BG = "#2a3a4c"
LESSON_BG = "#23283a"
TREND_LINE = "#80cbc4"
TREND_POINT = "#26a69a"

BOARD_SHADOW_BG = "#0a0a12"
LIGHT_SQUARE = "#ecd8b8"
DARK_SQUARE = "#ae8a68"
SELECTED_LIGHT = "#f7ec5d"
SELECTED_DARK = "#d6cc3a"
LEGAL_DOT_COLOUR = "#4a4a4a"
LEGAL_CAPTURE_COLOUR = "#e05555"
LAST_MOVE_LIGHT = "#cdd26a"
LAST_MOVE_DARK = "#a9a238"
CHECK_COLOUR = "#e74c3c"
HIGHLIGHT_ARROW = "#3b82f6"
MOVE_HINT_STIPPLE = "gray50"
PROMOTION_BG = ENTRY_BG
PROMOTION_FG = "#e8e8e8"
PROMOTION_BTN_BG = "#2d2d44"
PROMOTION_BTN_HOVER = "#3d3d5c"
PROMOTION_BORDER = "#444466"
WHITE_PIECE_FILL = "#ffffff"
WHITE_PIECE_OUTLINE = "#222222"
BLACK_PIECE_FILL = "#1a1a1a"
BLACK_PIECE_OUTLINE = "#1a1a1a"
COORD_LIGHT_FG = "#a07850"
COORD_DARK_FG = "#d4bc98"

COACH_MOVE_HINT = HIGHLIGHT_ARROW
REVIEW_MOVE_HINTS: dict[MoveClassification, str] = {
    MoveClassification.INACCURACY: "#ffee58",
    MoveClassification.MISTAKE: "#ffb74d",
    MoveClassification.BLUNDER: "#ef5350",
}
CLASSIFICATION_COLOURS: dict[MoveClassification, str] = {
    MoveClassification.EXCELLENT: SUCCESS,
    MoveClassification.GOOD: "#a5d6a7",
    MoveClassification.INACCURACY: "#ffee58",
    MoveClassification.MISTAKE: "#ffb74d",
    MoveClassification.BLUNDER: ERROR,
    MoveClassification.FORCED: FG_DIM,
}
TRAINING_FEEDBACK_COLOURS: dict[TrainingFeedbackLabel, str] = {
    TrainingFeedbackLabel.GOOD: EVAL_POSITIVE,
    TrainingFeedbackLabel.BETTER_AVAILABLE: LOADING,
    TrainingFeedbackLabel.SIGNIFICANT_ERROR: EVAL_NEGATIVE,
}
WEAKNESS_TREND_COLOURS: dict[WeaknessTrend, str] = {
    WeaknessTrend.IMPROVING: EVAL_POSITIVE,
    WeaknessTrend.STABLE: FG,
    WeaknessTrend.REPEATING: EVAL_NEGATIVE,
}


def styled_button(
    parent: tk.Widget,
    text: str,
    command: object,
    *,
    bg: str = BTN_BG,
    fg: str = BTN_FG,
    hover_bg: str = BTN_HOVER,
    active_bg: str = BTN_ACTIVE,
    font: tuple[str, ...] = ("sans-serif", 10),
    padx: int = 12,
    pady: int = 6,
    cursor: str = "hand2",
    **kwargs: object,
) -> tk.Button:
    """Create a flat-styled button with a hover state."""
    button = tk.Button(
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
    button.bind("<Enter>", lambda _event: button.config(bg=hover_bg))
    button.bind("<Leave>", lambda _event: button.config(bg=bg))
    return button
