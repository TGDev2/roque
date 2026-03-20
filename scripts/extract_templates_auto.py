"""Extract a complete template set from a starting-position screenshot.

.. deprecated::
    Use ``python -m chesslab.extract`` instead.  This script is kept for
    backwards compatibility and delegates to the new module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

# Allow running as `python scripts/extract_templates_auto.py` without install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np

from chesslab.detection.board_detector import BoardDetector
from chesslab.splitting.board_splitter import BoardSplitter, SquareImage
from chesslab.splitting.square_color import square_background

DEFAULT_IMAGE = Path("scripts/data/test_board.png")
DEFAULT_OUTPUT = Path("src/chesslab/recognition/templates/chess.com-green")
BACKGROUNDS = ("light", "dark")
PIECES = (
    "wK", "wQ", "wR", "wB", "wN", "wP",
    "bK", "bQ", "bR", "bB", "bN", "bP",
)

# Known labels for the standard start position.
START_POSITION: dict[str, str] = {
    "a1": "wR", "b1": "wN", "c1": "wB", "d1": "wQ",
    "e1": "wK", "f1": "wB", "g1": "wN", "h1": "wR",
    "a2": "wP", "b2": "wP", "c2": "wP", "d2": "wP",
    "e2": "wP", "f2": "wP", "g2": "wP", "h2": "wP",
    "a8": "bR", "b8": "bN", "c8": "bB", "d8": "bQ",
    "e8": "bK", "f8": "bB", "g8": "bN", "h8": "bR",
    "a7": "bP", "b7": "bP", "c7": "bP", "d7": "bP",
    "e7": "bP", "f7": "bP", "g7": "bP", "h7": "bP",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract templates for all piece/background combinations from "
            "a standard starting-position screenshot."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help="Path to the input screenshot (default: scripts/data/test_board.png).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Target template directory (default: src/chesslab/recognition/templates/green).",
    )
    parser.add_argument(
        "--theme-name",
        default="chess.com-green",
        help="Theme name written to theme.json (default: chess.com-green).",
    )
    return parser.parse_args()


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got {image.shape}.")
    if image.shape[2] == 4:
        return image[:, :, :3]
    if image.shape[2] == 3:
        return image
    raise ValueError(f"Expected 3 or 4 channels, got {image.shape}.")


def _estimate_background_colours(
    squares: dict[str, SquareImage],
) -> dict[str, np.ndarray]:
    samples: dict[str, list[np.ndarray]] = {"light": [], "dark": []}

    for name, square in squares.items():
        # Empty middle-board squares provide clean background colours.
        if name in START_POSITION:
            continue
        image = _ensure_bgr(square.image)
        h, w = image.shape[:2]
        patch = max(2, min(h, w) // 8)
        corners = np.concatenate(
            [
                image[:patch, :patch].reshape(-1, 3),
                image[:patch, w - patch :].reshape(-1, 3),
                image[h - patch :, :patch].reshape(-1, 3),
                image[h - patch :, w - patch :].reshape(-1, 3),
            ],
            axis=0,
        ).astype(np.float32)
        bg = square_background(square.file, square.rank)
        samples[bg].append(np.median(corners, axis=0))

    colours: dict[str, np.ndarray] = {}
    for bg in BACKGROUNDS:
        if not samples[bg]:
            raise RuntimeError(
                f"Could not estimate '{bg}' board colour from empty squares."
            )
        colours[bg] = np.median(np.stack(samples[bg]), axis=0).astype(np.uint8)
    return colours


def _synthesise_background(
    source: np.ndarray,
    *,
    source_bg: np.ndarray,
    target_bg: np.ndarray,
) -> np.ndarray:
    """Repaint source background while preserving the piece silhouette."""
    bgr = _ensure_bgr(source)

    src = bgr.astype(np.int32)
    bg = source_bg.astype(np.int32)
    dist = np.sqrt(np.sum((src - bg) ** 2, axis=2))

    piece_mask = dist > 12.0
    out = np.empty_like(bgr)
    out[:] = target_bg
    out[piece_mask] = bgr[piece_mask]
    return out


def _expected_keys() -> set[str]:
    return {f"{piece}_{bg}" for piece in PIECES for bg in BACKGROUNDS}


def main() -> None:
    warnings.warn(
        "extract_templates_auto.py is deprecated. "
        "Use 'python -m chesslab.extract' instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    args = _parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading screenshot: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    detector = BoardDetector()
    splitter = BoardSplitter()

    print("detecting board...")
    detection = detector.detect(image)
    if not detection.detected or detection.board is None:
        raise RuntimeError("Board not detected.")

    print(
        "detected board:",
        f"x={detection.board.x}",
        f"y={detection.board.y}",
        f"w={detection.board.width}",
        f"h={detection.board.height}",
    )

    print("splitting squares...")
    split = splitter.split(image, detection.board)
    bg_colours = _estimate_background_colours(split.squares)
    print(
        "estimated backgrounds:",
        f"light={tuple(int(v) for v in bg_colours['light'])}",
        f"dark={tuple(int(v) for v in bg_colours['dark'])}",
    )

    templates: dict[str, np.ndarray] = {}
    print("extracting templates from known start squares...")
    for name, square in split.squares.items():
        piece = START_POSITION.get(name)
        if piece is None:
            continue
        bg = square_background(square.file, square.rank)
        key = f"{piece}_{bg}"
        if key not in templates:
            templates[key] = _ensure_bgr(square.image)

    print(f"directly extracted: {len(templates)}")

    for key in sorted(_expected_keys()):
        if key in templates:
            continue
        piece, bg = key.rsplit("_", 1)
        other_bg = "dark" if bg == "light" else "light"
        source_key = f"{piece}_{other_bg}"
        source = templates.get(source_key)
        if source is None:
            raise RuntimeError(
                f"Cannot synthesise missing template '{key}': source '{source_key}' not found."
            )
        templates[key] = _synthesise_background(
            source,
            source_bg=bg_colours[other_bg],
            target_bg=bg_colours[bg],
        )
        print("synthesised", key)

    missing = sorted(_expected_keys() - set(templates.keys()))
    if missing:
        raise RuntimeError(f"Missing templates after synthesis: {', '.join(missing)}")

    print("saving templates...")
    for key in sorted(_expected_keys()):
        out_path = output_dir / f"{key}.png"
        cv2.imwrite(str(out_path), templates[key])
        print("saved", out_path)

    manifest = {
        "name": args.theme_name,
        "description": (
            "Auto-extracted from a starting-position screenshot "
            f"({image_path.name})."
        ),
        "templates": {
            key: f"{key}.png" for key in sorted(_expected_keys())
        },
    }
    manifest_path = output_dir / "theme.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("saved", manifest_path)
    print("done.")


if __name__ == "__main__":
    main()
