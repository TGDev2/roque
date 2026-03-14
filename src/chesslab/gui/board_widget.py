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

_LIGHT_SQUARE = "#f0d9b5"
_DARK_SQUARE = "#b58863"
_SELECTED_COLOUR = "#f6f669"
_LEGAL_DEST_COLOUR = "#829769"

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
        )
        self._square_size = square_size
        self._flipped = flipped
        self._on_move = on_move

        self._board = chess.Board()
        self._selected_square: str | None = None

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

    def highlight_squares(self, squares: list[str]) -> None:
        """Highlight a list of squares (e.g. for best move)."""
        self._draw()
        for sq in squares:
            x, y = square_to_coords(sq, self._square_size, self._flipped)
            self.create_rectangle(
                x, y, x + self._square_size, y + self._square_size,
                outline="#ff0000", width=3,
            )

    def clear_highlights(self) -> None:
        """Redraw the board without extra highlights."""
        self._draw()

    # -- Drawing -------------------------------------------------------------

    def _draw(self) -> None:
        """Redraw the entire board: squares, highlights, pieces, labels."""
        self.delete("all")
        sz = self._square_size
        placement = parse_fen_placement(self._board.board_fen())

        legal_dests: set[str] = set()
        if self._selected_square is not None:
            sq_obj = chess.parse_square(self._selected_square)
            for move in self._board.legal_moves:
                if move.from_square == sq_obj:
                    legal_dests.add(chess.square_name(move.to_square))

        for rank in range(8):
            for file in range(8):
                sq_name = f"{chr(ord('a') + file)}{rank + 1}"
                x, y = square_to_coords(sq_name, sz, self._flipped)

                # Square colour.
                is_light = (file + rank) % 2 != 0
                if sq_name == self._selected_square:
                    colour = _SELECTED_COLOUR
                elif sq_name in legal_dests:
                    colour = _LEGAL_DEST_COLOUR
                elif is_light:
                    colour = _LIGHT_SQUARE
                else:
                    colour = _DARK_SQUARE

                self.create_rectangle(
                    x, y, x + sz, y + sz,
                    fill=colour, outline="",
                )

                # Piece.
                fen_char = placement.get(sq_name)
                if fen_char and fen_char in UNICODE_PIECES:
                    self.create_text(
                        x + sz // 2, y + sz // 2,
                        text=UNICODE_PIECES[fen_char],
                        font=("sans-serif", int(sz * 0.55), "bold"),
                        fill="#1a1a1a" if fen_char.isupper() else "#1a1a1a",
                    )

        # Coordinate labels.
        font = ("sans-serif", max(8, sz // 6))
        for i in range(8):
            # File labels (bottom row).
            file_label = chr(ord("a") + (7 - i if self._flipped else i))
            bx = i * sz + sz - 4
            by = 8 * sz - 4
            lbl_light = (((7 - i if self._flipped else i) + 0) % 2 != 0)
            self.create_text(
                bx, by,
                text=file_label,
                font=font,
                fill=_LIGHT_SQUARE if not lbl_light else _DARK_SQUARE,
                anchor="se",
            )
            # Rank labels (left column).
            rank_label = str((i + 1) if self._flipped else (8 - i))
            rx, ry = 4, i * sz + 4
            rlbl_light = (
                (0 + (7 - i if not self._flipped else i)) % 2 != 0
            )
            self.create_text(
                rx, ry,
                text=rank_label,
                font=font,
                fill=_LIGHT_SQUARE if not rlbl_light else _DARK_SQUARE,
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
        piece = self._board.piece_at(chess.parse_square(from_sq))
        if piece is None or piece.piece_type != chess.PAWN:
            return None

        to_rank = int(to_sq[1])
        if (piece.color == chess.WHITE and to_rank == 8) or (
            piece.color == chess.BLACK and to_rank == 1
        ):
            return self._show_promotion_dialog(piece.color)
        return None

    def _show_promotion_dialog(self, color: chess.Color) -> int:
        """Show a dialog to choose promotion piece. Returns piece type."""
        result: list[int] = [chess.QUEEN]

        dialog = tk.Toplevel(self)
        dialog.title("Promotion")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)

        pieces = [
            (chess.QUEEN, "Q" if color == chess.WHITE else "q"),
            (chess.ROOK, "R" if color == chess.WHITE else "r"),
            (chess.BISHOP, "B" if color == chess.WHITE else "b"),
            (chess.KNIGHT, "N" if color == chess.WHITE else "n"),
        ]

        for piece_type, fen_char in pieces:
            symbol = UNICODE_PIECES[fen_char]

            def make_handler(pt: int) -> Callable[[], None]:
                def handler() -> None:
                    result[0] = pt
                    dialog.destroy()
                return handler

            tk.Button(
                dialog,
                text=symbol,
                font=("sans-serif", 32),
                command=make_handler(piece_type),
                width=3,
                height=1,
            ).pack(side="left", padx=4, pady=4)

        dialog.wait_window()
        return result[0]
