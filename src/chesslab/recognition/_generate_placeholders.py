"""Generate synthetic placeholder template PNGs for any registered theme.

Each piece gets a unique geometric pattern drawn on the appropriate
background colour.  These placeholders allow the recognition pipeline to
function end-to-end; replace with real templates extracted from screenshots
for production accuracy.

Run directly::

    python -m chesslab.recognition._generate_placeholders          # all themes
    python -m chesslab.recognition._generate_placeholders --theme chess.com-green
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

WHITE_PIECE = (255, 255, 255)
BLACK_PIECE = (30, 30, 30)

PIECES = ("wK", "wQ", "wR", "wB", "wN", "wP",
          "bK", "bQ", "bR", "bB", "bN", "bP")

_THEMES_JSON = Path(__file__).parent / "themes.json"
_TEMPLATES_ROOT = Path(__file__).parent / "templates"


def _piece_colour(piece_key: str) -> tuple[int, int, int]:
    return WHITE_PIECE if piece_key[0] == "w" else BLACK_PIECE


# ---------------------------------------------------------------------------
# Drawing functions — one unique pattern per piece type
# ---------------------------------------------------------------------------


def _draw_king(img: NDArray[np.uint8], colour: tuple[int, int, int], size: int) -> None:
    """Cross shape for king."""
    c = size // 2
    t = max(size // 10, 2)
    cv2.line(img, (c, size // 5), (c, 4 * size // 5), colour, t)
    cv2.line(img, (size // 5, c), (4 * size // 5, c), colour, t)


def _draw_queen(img: NDArray[np.uint8], colour: tuple[int, int, int], size: int) -> None:
    """Star / asterisk for queen."""
    c = size // 2
    r = size // 3
    t = max(size // 12, 2)
    for angle_deg in (0, 60, 120):
        rad = np.radians(angle_deg)
        dx, dy = int(r * np.cos(rad)), int(r * np.sin(rad))
        cv2.line(img, (c - dx, c - dy), (c + dx, c + dy), colour, t)


def _draw_rook(img: NDArray[np.uint8], colour: tuple[int, int, int], size: int) -> None:
    """Rectangle for rook."""
    m = size // 5
    cv2.rectangle(img, (m, m), (size - m, size - m), colour, -1)


def _draw_bishop(img: NDArray[np.uint8], colour: tuple[int, int, int], size: int) -> None:
    """Diamond for bishop."""
    c = size // 2
    r = size // 3
    pts = np.array([[c, c - r], [c + r, c], [c, c + r], [c - r, c]], dtype=np.int32)
    cv2.fillPoly(img, [pts], colour)


def _draw_knight(img: NDArray[np.uint8], colour: tuple[int, int, int], size: int) -> None:
    """L-shape for knight."""
    m = size // 5
    t = max(size // 8, 2)
    cv2.line(img, (m, size - m), (m, m), colour, t)
    cv2.line(img, (m, m), (size // 2, m), colour, t)


def _draw_pawn(img: NDArray[np.uint8], colour: tuple[int, int, int], size: int) -> None:
    """Circle for pawn."""
    c = size // 2
    r = size // 4
    cv2.circle(img, (c, c), r, colour, -1)


_DRAW_FNS = {
    "K": _draw_king,
    "Q": _draw_queen,
    "R": _draw_rook,
    "B": _draw_bishop,
    "N": _draw_knight,
    "P": _draw_pawn,
}


def generate_template(
    piece_key: str,
    bg: str,
    *,
    light_bgr: tuple[int, int, int],
    dark_bgr: tuple[int, int, int],
    size: int = 80,
) -> NDArray[np.uint8]:
    """Return a synthetic template image for *piece_key* on *bg*."""
    bg_colour = light_bgr if bg == "light" else dark_bgr
    img = np.full((size, size, 3), bg_colour, dtype=np.uint8)
    piece_letter = piece_key[1]  # e.g. "K" from "wK"
    colour = _piece_colour(piece_key)
    _DRAW_FNS[piece_letter](img, colour, size)
    return img


def generate_theme(
    output_dir: Path,
    *,
    theme_name: str,
    light_bgr: tuple[int, int, int],
    dark_bgr: tuple[int, int, int],
    size: int = 80,
    synthetic: bool = True,
) -> None:
    """Generate all placeholder templates and theme.json in *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    templates: dict[str, str] = {}
    for piece in PIECES:
        for bg in ("light", "dark"):
            key = f"{piece}_{bg}"
            filename = f"{key}.png"
            img = generate_template(
                piece, bg,
                light_bgr=light_bgr, dark_bgr=dark_bgr, size=size,
            )
            cv2.imwrite(str(output_dir / filename), img)
            templates[key] = filename

    manifest: dict[str, object] = {
        "name": theme_name,
        "description": f"{theme_name} theme (synthetic placeholders)",
        "templates": templates,
    }
    if synthetic:
        manifest["synthetic"] = True

    with (output_dir / "theme.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)


def _load_theme_signatures(
    themes_file: Path | None = None,
) -> list[dict[str, object]]:
    """Load theme signatures from themes.json."""
    path = themes_file or _THEMES_JSON
    with path.open() as fh:
        data = json.load(fh)
    return data["themes"]


def _qualified_dir_name(source: str, name: str) -> str:
    """Build canonical directory name from source and name."""
    return f"{source}-{name}"


def generate_all_themes(
    themes_file: Path | None = None,
    output_root: Path | None = None,
    *,
    size: int = 80,
) -> list[str]:
    """Generate synthetic templates for every theme in *themes_file*.

    Returns the list of generated theme directory names.
    """
    root = output_root or _TEMPLATES_ROOT
    signatures = _load_theme_signatures(themes_file)
    generated: list[str] = []

    for sig in signatures:
        dir_name = _qualified_dir_name(str(sig["source"]), str(sig["name"]))
        output_dir = root / dir_name

        # Skip if a non-synthetic theme already exists (real templates).
        manifest_path = output_dir / "theme.json"
        if manifest_path.exists():
            with manifest_path.open() as fh:
                existing = json.load(fh)
            if not existing.get("synthetic", False):
                print(f"  skipping {dir_name} (real templates exist)")
                continue

        light = tuple(int(v) for v in sig["light_bgr"])  # type: ignore[union-attr]
        dark = tuple(int(v) for v in sig["dark_bgr"])  # type: ignore[union-attr]
        generate_theme(
            output_dir,
            theme_name=dir_name,
            light_bgr=light,  # type: ignore[arg-type]
            dark_bgr=dark,  # type: ignore[arg-type]
            size=size,
        )
        generated.append(dir_name)
        print(f"  generated {dir_name}")

    return generated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic placeholder templates.",
    )
    parser.add_argument(
        "--theme",
        default=None,
        help="Generate only this theme (e.g. 'chess.com-green'). Default: all.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=80,
        help="Template side length in pixels (default: 80).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.theme:
        # Generate a single theme by looking up its signature.
        sigs = _load_theme_signatures()
        match = None
        for sig in sigs:
            dir_name = _qualified_dir_name(str(sig["source"]), str(sig["name"]))
            if dir_name == args.theme:
                match = sig
                break
        if match is None:
            available = [
                _qualified_dir_name(str(s["source"]), str(s["name"]))
                for s in sigs
            ]
            raise SystemExit(
                f"Theme '{args.theme}' not found. Available: {', '.join(available)}"
            )
        light = tuple(int(v) for v in match["light_bgr"])  # type: ignore[union-attr]
        dark = tuple(int(v) for v in match["dark_bgr"])  # type: ignore[union-attr]
        out = _TEMPLATES_ROOT / args.theme
        generate_theme(
            out,
            theme_name=args.theme,
            light_bgr=light,  # type: ignore[arg-type]
            dark_bgr=dark,  # type: ignore[arg-type]
            size=args.size,
        )
        print(f"Generated placeholder templates in {out}")
    else:
        print("Generating synthetic templates for all themes...")
        generated = generate_all_themes(size=args.size)
        print(f"Done. Generated {len(generated)} theme(s).")
