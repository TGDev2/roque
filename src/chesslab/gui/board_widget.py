"""Interactive 8x8 chess board widget using tkinter Canvas.

Renders a board with Unicode chess pieces and supports two-click move
entry: click a source square then a destination square to make a move.
Legal destinations are highlighted after selecting a source square.

The board can be flipped for playing as Black via :meth:`set_flipped`.
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Callable, ClassVar

import chess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNICODE_PIECES: dict[str, str] = {
    "K": "\u2654",
    "Q": "\u2655",
    "R": "\u2656",
    "B": "\u2657",
    "N": "\u2658",
    "P": "\u2659",
    "k": "\u265a",
    "q": "\u265b",
    "r": "\u265c",
    "b": "\u265d",
    "n": "\u265e",
    "p": "\u265f",
}

# Board colours — warm, modern palette.
_LIGHT_SQUARE = "#ecd8b8"
_DARK_SQUARE = "#ae8a68"
_SELECTED_COLOUR = "#f7ec5d"
_SELECTED_DARK = "#d6cc3a"
_LEGAL_DOT_COLOUR = "#4a4a4a"
_LEGAL_CAPTURE_COLOUR = "#e05555"
_LAST_MOVE_LIGHT = "#cdd26a"
_LAST_MOVE_DARK = "#a9a238"
_HIGHLIGHT_ARROW = "#3b82f6"
_CHECK_COLOUR = "#e74c3c"

_PROMOTION_BG = "#1a1a2e"
_PROMOTION_FG = "#e8e8e8"
_PROMOTION_BTN_BG = "#2d2d44"
_PROMOTION_BTN_HOVER = "#3d3d5c"

# Piece colours.
_WHITE_PIECE_FILL = "#ffffff"
_WHITE_PIECE_OUTLINE = "#222222"
_BLACK_PIECE_FILL = "#1a1a1a"
_BLACK_PIECE_OUTLINE = "#1a1a1a"

_COORD_LIGHT_FG = "#a07850"
_COORD_DARK_FG = "#d4bc98"

MoveCallback = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# Pure helpers (testable without display)
# ---------------------------------------------------------------------------


def square_to_coords(
    square_name: str,
    square_size: int,
    flipped: bool,
) -> tuple[int, int]:
    """Return the top-left (x, y) pixel of *square_name* on the canvas."""
    file_idx = ord(square_name[0]) - ord("a")
    rank_idx = int(square_name[1]) - 1
    if flipped:
        col = 7 - file_idx
        row = rank_idx
    else:
        col = file_idx
        row = 7 - rank_idx
    return col * square_size, row * square_size


def coords_to_square(
    x: int,
    y: int,
    square_size: int,
    flipped: bool,
) -> str:
    """Return the algebraic square name for pixel (x, y)."""
    col = x // square_size
    row = y // square_size
    if flipped:
        file_idx = 7 - col
        rank_idx = row
    else:
        file_idx = col
        rank_idx = 7 - row
    file_idx = max(0, min(7, file_idx))
    rank_idx = max(0, min(7, rank_idx))
    return f"{chr(ord('a') + file_idx)}{rank_idx + 1}"


def parse_fen_placement(placement: str) -> dict[str, str]:
    """Parse FEN piece-placement into ``{square_name: fen_char}`` dict."""
    pieces: dict[str, str] = {}
    ranks = placement.split("/")
    for rank_idx, rank_str in enumerate(ranks):
        file_idx = 0
        rank_number = 8 - rank_idx
        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                sq = f"{chr(ord('a') + file_idx)}{rank_number}"
                pieces[sq] = ch
                file_idx += 1
    return pieces


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


class BoardWidget(tk.Canvas):
    """8x8 interactive chess board canvas.

    Parameters
    ----------
    parent:
        Parent tkinter widget.
    square_size:
        Size of each square in pixels.
    flipped:
        If ``True`` the board is rendered from Black's perspective.
    on_move:
        Callback ``(from_square, to_square)`` fired when the user clicks
        a two-click move.  The squares are algebraic (e.g. ``"e2"``,
        ``"e4"``).
    """

    PROMOTION_PIECES: ClassVar[list[str]] = ["q", "r", "b", "n"]

    def __init__(
        self,
        parent: tk.Widget,
        *,
        square_size: int = 64,
        flipped: bool = False,
        on_move: MoveCallback | None = None,
    ) -> None:
        size = square_size * 8
        super().__init__(
            parent,
            width=size,
            height=size,
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self._square_size = square_size
        self._flipped = flipped
        self._on_move = on_move

        self._board = chess.Board()
        self._selected_square: str | None = None
        self._last_move: tuple[str, str] | None = None
        self._highlighted_squares: list[str] = []

        self.bind("<Button-1>", self._on_click)
        self._draw()

    # -- Public API ----------------------------------------------------------

    def set_position(self, fen: str) -> None:
        """Update the displayed position from a FEN string."""
        self._board = chess.Board(fen)
        self._selected_square = None
        self._draw()

    def set_flipped(self, flipped: bool) -> None:
        """Flip the board orientation."""
        self._flipped = flipped
        self._selected_square = None
        self._draw()

    def set_last_move(self, from_sq: str, to_sq: str) -> None:
        """Set and highlight the last move played."""
        self._last_move = (from_sq, to_sq)
        self._draw()

    def clear_last_move(self) -> None:
        """Clear last move highlight."""
        self._last_move = None
        self._draw()

    def highlight_squares(self, squares: list[str]) -> None:
        """Highlight a list of squares (e.g. for best move arrow)."""
        self._highlighted_squares = squares
        self._draw()

    def clear_highlights(self) -> None:
        """Redraw the board without extra highlights."""
        self._highlighted_squares = []
        self._draw()

    # -- Drawing -------------------------------------------------------------

    def _draw(self) -> None:
        """Redraw the entire board: squares, highlights, pieces, labels."""
        self.delete("all")
        sz = self._square_size
        placement = parse_fen_placement(self._board.board_fen())

        # Compute legal destinations for selected piece.
        legal_dests: set[str] = set()
        legal_captures: set[str] = set()
        if self._selected_square is not None:
            sq_obj = chess.parse_square(self._selected_square)
            for move in self._board.legal_moves:
                if move.from_square == sq_obj:
                    dest_name = chess.square_name(move.to_square)
                    legal_dests.add(dest_name)
                    if self._board.piece_at(move.to_square) is not None:
                        legal_captures.add(dest_name)

        # Compute last-move squares.
        last_move_sqs: set[str] = set()
        if self._last_move:
            last_move_sqs = {self._last_move[0], self._last_move[1]}

        # Check detection.
        check_square: str | None = None
        if self._board.is_check():
            king_sq = self._board.king(self._board.turn)
            if king_sq is not None:
                check_square = chess.square_name(king_sq)

        # Draw squares.
        for rank in range(8):
            for file in range(8):
                sq_name = f"{chr(ord('a') + file)}{rank + 1}"
                x, y = square_to_coords(sq_name, sz, self._flipped)
                is_light = (file + rank) % 2 != 0

                # Base square colour.
                if sq_name == self._selected_square:
                    colour = _SELECTED_COLOUR if is_light else _SELECTED_DARK
                elif sq_name in last_move_sqs:
                    colour = _LAST_MOVE_LIGHT if is_light else _LAST_MOVE_DARK
                elif is_light:
                    colour = _LIGHT_SQUARE
                else:
                    colour = _DARK_SQUARE

                self.create_rectangle(
                    x, y, x + sz, y + sz,
                    fill=colour, outline="",
                )

                # Check indicator (red glow behind king).
                if sq_name == check_square:
                    margin = sz // 8
                    self.create_rectangle(
                        x + margin, y + margin,
                        x + sz - margin, y + sz - margin,
                        fill=_CHECK_COLOUR, outline="", stipple="gray50",
                    )

        # Draw legal move indicators (dots and capture rings).
        for sq_name in legal_dests:
            x, y = square_to_coords(sq_name, sz, self._flipped)
            cx, cy = x + sz // 2, y + sz // 2
            if sq_name in legal_captures:
                # Ring around capturable piece.
                r = int(sz * 0.46)
                self.create_oval(
                    cx - r, cy - r, cx + r, cy + r,
                    outline=_LEGAL_CAPTURE_COLOUR, width=3,
                )
            else:
                # Small dot for empty legal destination.
                r = int(sz * 0.14)
                self.create_oval(
                    cx - r, cy - r, cx + r, cy + r,
                    fill=_LEGAL_DOT_COLOUR, outline="",
                    stipple="gray50",
                )

        # Draw pieces.
        for rank in range(8):
            for file in range(8):
                sq_name = f"{chr(ord('a') + file)}{rank + 1}"
                fen_char = placement.get(sq_name)
                if not fen_char or fen_char not in UNICODE_PIECES:
                    continue
                x, y = square_to_coords(sq_name, sz, self._flipped)
                cx, cy = x + sz // 2, y + sz // 2
                is_white_piece = fen_char.isupper()

                # Draw piece with proper contrast.
                font_size = int(sz * 0.58)
                if is_white_piece:
                    # White piece: white fill with dark outline for contrast.
                    # Draw outline by rendering text offset in 4 directions.
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        self.create_text(
                            cx + dx, cy + dy,
                            text=UNICODE_PIECES[fen_char],
                            font=("sans-serif", font_size, "bold"),
                            fill=_WHITE_PIECE_OUTLINE,
                        )
                    self.create_text(
                        cx, cy,
                        text=UNICODE_PIECES[fen_char],
                        font=("sans-serif", font_size, "bold"),
                        fill=_WHITE_PIECE_FILL,
                    )
                else:
                    # Black piece: solid dark.
                    self.create_text(
                        cx, cy,
                        text=UNICODE_PIECES[fen_char],
                        font=("sans-serif", font_size, "bold"),
                        fill=_BLACK_PIECE_FILL,
                    )

        # Draw best-move arrow highlight.
        if len(self._highlighted_squares) >= 2:
            from_sq = self._highlighted_squares[0]
            to_sq = self._highlighted_squares[1]
            fx, fy = square_to_coords(from_sq, sz, self._flipped)
            tx, ty = square_to_coords(to_sq, sz, self._flipped)
            half = sz // 2
            self.create_line(
                fx + half, fy + half, tx + half, ty + half,
                fill=_HIGHLIGHT_ARROW, width=max(3, sz // 12),
                arrow="last", arrowshape=(sz // 4, sz // 3, sz // 8),
            )

        # Coordinate labels.
        font_size = max(8, sz // 6)
        font = ("sans-serif", font_size, "bold")
        for i in range(8):
            # File labels (bottom row).
            file_label = chr(ord("a") + (7 - i if self._flipped else i))
            bx = i * sz + sz - 3
            by = 8 * sz - 3
            is_light = (((7 - i if self._flipped else i) + 0) % 2 != 0)
            self.create_text(
                bx, by,
                text=file_label,
                font=font,
                fill=_COORD_DARK_FG if is_light else _COORD_LIGHT_FG,
                anchor="se",
            )
            # Rank labels (left column).
            rank_label = str((i + 1) if self._flipped else (8 - i))
            rx, ry = 3, i * sz + 3
            is_light = (
                (0 + (7 - i if not self._flipped else i)) % 2 != 0
            )
            self.create_text(
                rx, ry,
                text=rank_label,
                font=font,
                fill=_COORD_DARK_FG if is_light else _COORD_LIGHT_FG,
                anchor="nw",
            )

    # -- Click handling ------------------------------------------------------

    def _on_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Handle a click on the board canvas."""
        sq = coords_to_square(event.x, event.y, self._square_size, self._flipped)

        if self._selected_square is None:
            # First click — select source.
            piece = self._board.piece_at(chess.parse_square(sq))
            if piece is not None:
                self._selected_square = sq
                self._draw()
        elif sq == self._selected_square:
            # Click same square — deselect.
            self._selected_square = None
            self._draw()
        else:
            # Second click — attempt move.
            from_sq = self._selected_square
            self._selected_square = None

            # Check for promotion.
            promotion = self._check_promotion(from_sq, sq)

            move = chess.Move(
                chess.parse_square(from_sq),
                chess.parse_square(sq),
                promotion=promotion,
            )

            if move in self._board.legal_moves:
                self._last_move = (from_sq, sq)
                if self._on_move is not None:
                    self._on_move(from_sq, sq + (
                        chess.piece_symbol(promotion) if promotion else ""
                    ))
            else:
                # Try selecting the new square instead.
                piece = self._board.piece_at(chess.parse_square(sq))
                if piece is not None:
                    self._selected_square = sq
            self._draw()

    def _check_promotion(self, from_sq: str, to_sq: str) -> int | None:
        """Check if a pawn move requires promotion and show dialog."""
        from_square = chess.parse_square(from_sq)
        to_square = chess.parse_square(to_sq)

        piece = self._board.piece_at(from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            return None

        to_rank = int(to_sq[1])
        if (piece.color == chess.WHITE and to_rank == 8) or (
            piece.color == chess.BLACK and to_rank == 1
        ):
            # Ask for promotion only when at least one promotion move is legal.
            promotion_moves = [
                chess.Move(from_square, to_square, promotion=piece_type)
                for piece_type in (
                    chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT
                )
            ]
            if not any(move in self._board.legal_moves for move in promotion_moves):
                return None
            return self._show_promotion_dialog(piece.color)
        return None

    def _show_promotion_dialog(self, color: chess.Color) -> int:
        """Show a dialog to choose promotion piece. Returns piece type."""
        result: list[int] = [chess.QUEEN]
        if not self.winfo_exists():
            return result[0]

        try:
            parent = self.winfo_toplevel()
            if not parent.winfo_exists() or not parent.winfo_viewable():
                return result[0]

            dialog = tk.Toplevel(parent)
            dialog.withdraw()
            dialog.title("Promote")
            dialog.transient(parent)
            dialog.resizable(False, False)
            dialog.configure(bg=_PROMOTION_BG)
            dialog.overrideredirect(True)

            inner = tk.Frame(dialog, bg=_PROMOTION_BG, bd=2, relief="solid",
                             highlightbackground="#444466",
                             highlightthickness=1)
            inner.pack(fill="both", expand=True)

            tk.Label(
                inner,
                text="Promote to",
                bg=_PROMOTION_BG,
                fg=_PROMOTION_FG,
                font=("sans-serif", 10, "bold"),
                padx=10,
                pady=6,
            ).pack()

            btn_frame = tk.Frame(inner, bg=_PROMOTION_BG)
            btn_frame.pack(padx=6, pady=(0, 8))

            pieces = [
                (chess.QUEEN, "Q" if color == chess.WHITE else "q"),
                (chess.ROOK, "R" if color == chess.WHITE else "r"),
                (chess.BISHOP, "B" if color == chess.WHITE else "b"),
                (chess.KNIGHT, "N" if color == chess.WHITE else "n"),
            ]

            def make_handler(pt: int) -> Callable[[], None]:
                def handler() -> None:
                    result[0] = pt
                    if dialog.winfo_exists():
                        dialog.destroy()
                return handler

            for piece_type, fen_char in pieces:
                symbol = UNICODE_PIECES[fen_char]
                btn = tk.Button(
                    btn_frame,
                    text=symbol,
                    font=("sans-serif", 28),
                    command=make_handler(piece_type),
                    width=2,
                    height=1,
                    bg=_PROMOTION_BTN_BG,
                    fg=_PROMOTION_FG,
                    activebackground=_PROMOTION_BTN_HOVER,
                    activeforeground="#ffffff",
                    relief="flat",
                    bd=0,
                    cursor="hand2",
                )
                btn.pack(side="left", padx=3, pady=2)

            dialog.protocol(
                "WM_DELETE_WINDOW",
                lambda: dialog.destroy() if dialog.winfo_exists() else None,
            )

            # Center on parent before making the dialog visible.
            parent.update_idletasks()
            dialog.update_idletasks()
            x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_width()) // 2
            y = parent.winfo_rooty() + (
                parent.winfo_height() - dialog.winfo_height()
            ) // 2
            dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

            dialog.deiconify()
            dialog.lift(parent)
            dialog.focus_set()
            dialog.wait_visibility()
            try:
                dialog.grab_set()
            except tk.TclError:
                logger.debug(
                    "Promotion dialog grab skipped because window is not viewable.",
                    exc_info=True,
                )

            self.wait_window(dialog)
        except tk.TclError:
            logger.debug(
                "Promotion dialog skipped because GUI is closing.",
                exc_info=True,
            )
        return result[0]
