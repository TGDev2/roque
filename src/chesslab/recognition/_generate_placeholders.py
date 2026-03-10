"""Generate synthetic placeholder template PNGs for the 'green' theme.

Each piece gets a unique geometric pattern drawn on the appropriate
background colour (chess.com Green light/dark).  These placeholders allow
the recognition pipeline to function end-to-end; replace with real
templates extracted from chess.com screenshots for production accuracy.

Run directly::

    python -m chesslab.recognition._generate_placeholders
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

# chess.com "Green" theme colours (BGR)
LIGHT_BG = (208, 236, 235)  # #EEEED2 → BGR
DARK_BG = (82, 149, 115)  # #769656 → BGR

WHITE_PIECE = (255, 255, 255)
BLACK_PIECE = (30, 30, 30)

SIZE = 80  # template side length in pixels


def _background(bg: str) -> tuple[int, int, int]:
    return LIGHT_BG if bg == "light" else DARK_BG


def _piece_colour(piece_key: str) -> tuple[int, int, int]:
    return WHITE_PIECE if piece_key[0] == "w" else BLACK_PIECE


# ---------------------------------------------------------------------------
# Drawing functions — one unique pattern per piece type
# ---------------------------------------------------------------------------


def _draw_king(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    """Cross shape for king."""
    c = SIZE // 2
    t = max(SIZE // 10, 2)
    cv2.line(img, (c, SIZE // 5), (c, 4 * SIZE // 5), colour, t)
    cv2.line(img, (SIZE // 5, c), (4 * SIZE // 5, c), colour, t)


def _draw_queen(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    """Star / asterisk for queen."""
    c = SIZE // 2
    r = SIZE // 3
    t = max(SIZE // 12, 2)
    for angle_deg in (0, 60, 120):
        rad = np.radians(angle_deg)
        dx, dy = int(r * np.cos(rad)), int(r * np.sin(rad))
        cv2.line(img, (c - dx, c - dy), (c + dx, c + dy), colour, t)


def _draw_rook(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    """Rectangle for rook."""
    m = SIZE // 5
    cv2.rectangle(img, (m, m), (SIZE - m, SIZE - m), colour, -1)


def _draw_bishop(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    """Diamond for bishop."""
    c = SIZE // 2
    r = SIZE // 3
    pts = np.array([[c, c - r], [c + r, c], [c, c + r], [c - r, c]], dtype=np.int32)
    cv2.fillPoly(img, [pts], colour)


def _draw_knight(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    """L-shape for knight."""
    m = SIZE // 5
    t = max(SIZE // 8, 2)
    cv2.line(img, (m, SIZE - m), (m, m), colour, t)
    cv2.line(img, (m, m), (SIZE // 2, m), colour, t)


def _draw_pawn(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    """Circle for pawn."""
    c = SIZE // 2
    r = SIZE // 4
    cv2.circle(img, (c, c), r, colour, -1)


_DRAW_FNS = {
    "K": _draw_king,
    "Q": _draw_queen,
    "R": _draw_rook,
    "B": _draw_bishop,
    "N": _draw_knight,
    "P": _draw_pawn,
}


def generate_template(piece_key: str, bg: str) -> NDArray[np.uint8]:
    """Return a synthetic template image for *piece_key* on *bg*."""
    img = np.full((SIZE, SIZE, 3), _background(bg), dtype=np.uint8)
    piece_letter = piece_key[1]  # e.g. "K" from "wK"
    colour = _piece_colour(piece_key)
    _DRAW_FNS[piece_letter](img, colour)
    return img


def generate_theme(output_dir: Path) -> None:
    """Generate all placeholder templates and theme.json in *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pieces = ["wK", "wQ", "wR", "wB", "wN", "wP",
              "bK", "bQ", "bR", "bB", "bN", "bP"]
    templates: dict[str, str] = {}

    for piece in pieces:
        for bg in ("light", "dark"):
            key = f"{piece}_{bg}"
            filename = f"{key}.png"
            img = generate_template(piece, bg)
            cv2.imwrite(str(output_dir / filename), img)
            templates[key] = filename

    manifest = {
        "name": "green",
        "description": "chess.com Green theme (synthetic placeholders)",
        "templates": templates,
    }
    with (output_dir / "theme.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    _dir = Path(__file__).parent / "templates" / "green"
    generate_theme(_dir)
    print(f"Generated placeholder templates in {_dir}")
