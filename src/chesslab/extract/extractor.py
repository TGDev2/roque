"""Core template extraction logic for chess board screenshots.

Extracts a complete set of 24 piece templates (12 pieces x 2 backgrounds)
from a standard starting-position screenshot.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from chesslab.detection.board_detector import BoardDetector
from chesslab.recognition.theme_detector import ThemeDetectionResult, ThemeDetector
from chesslab.splitting.board_splitter import BoardSplitter, SquareImage
from chesslab.splitting.square_color import square_background

logger = logging.getLogger(__name__)

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

_TEMPLATES_ROOT = Path(__file__).resolve().parents[1] / "recognition" / "templates"


class ExtractionError(Exception):
    """Raised when template extraction fails."""


@dataclass(frozen=True)
class ExtractionResult:
    """Result of a template extraction run."""

    theme_name: str
    output_dir: Path
    template_count: int
    synthesised_count: int
    light_bgr: tuple[int, int, int]
    dark_bgr: tuple[int, int, int]


def _expected_keys() -> set[str]:
    return {f"{piece}_{bg}" for piece in PIECES for bg in BACKGROUNDS}


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ExtractionError(f"Expected image with 3 dimensions, got {image.shape}.")
    if image.shape[2] == 4:
        return image[:, :, :3]
    if image.shape[2] == 3:
        return image
    raise ExtractionError(f"Expected 3 or 4 channels, got {image.shape}.")


def _estimate_background_colours(
    squares: dict[str, SquareImage],
) -> dict[str, np.ndarray]:
    """Estimate light/dark background colours from empty squares."""
    samples: dict[str, list[np.ndarray]] = {"light": [], "dark": []}

    for name, square in squares.items():
        if name in START_POSITION:
            continue
        image = _ensure_bgr(square.image)
        h, w = image.shape[:2]
        patch = max(2, min(h, w) // 8)
        corners = np.concatenate(
            [
                image[:patch, :patch].reshape(-1, 3),
                image[:patch, w - patch:].reshape(-1, 3),
                image[h - patch:, :patch].reshape(-1, 3),
                image[h - patch:, w - patch:].reshape(-1, 3),
            ],
            axis=0,
        ).astype(np.float32)
        bg = square_background(square.file, square.rank)
        samples[bg].append(np.median(corners, axis=0))

    colours: dict[str, np.ndarray] = {}
    for bg in BACKGROUNDS:
        if not samples[bg]:
            raise ExtractionError(
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


class TemplateExtractor:
    """Extracts piece templates from a starting-position chess screenshot."""

    def __init__(
        self,
        image_path: Path,
        *,
        white_perspective: bool = True,
    ) -> None:
        self._image_path = image_path
        self._white_perspective = white_perspective

        self._image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if self._image is None:
            raise ExtractionError(f"Failed to read image: {image_path}")

        self._detector = BoardDetector()
        self._splitter = BoardSplitter()
        self._squares: dict[str, SquareImage] | None = None

    def _ensure_squares(self) -> dict[str, SquareImage]:
        """Detect board and split into squares (cached)."""
        if self._squares is not None:
            return self._squares

        detection = self._detector.detect(self._image)
        if not detection.detected or detection.board is None:
            raise ExtractionError("Board not detected in screenshot.")

        logger.info(
            "Detected board: x=%d y=%d w=%d h=%d",
            detection.board.x, detection.board.y,
            detection.board.width, detection.board.height,
        )

        split = self._splitter.split(
            self._image, detection.board,
            white_perspective=self._white_perspective,
        )
        self._squares = split.squares
        return self._squares

    def detect_theme(self) -> ThemeDetectionResult:
        """Auto-detect the board theme from the screenshot."""
        squares = self._ensure_squares()
        detector = ThemeDetector()
        return detector.detect(squares)

    def extract(
        self,
        theme_name: str,
        output_dir: Path | None = None,
    ) -> ExtractionResult:
        """Extract all 24 templates and save to *output_dir*.

        Args:
            theme_name: Name for the theme (e.g. "chess.com-brown").
            output_dir: Target directory. Defaults to the templates root.

        Returns:
            ExtractionResult with extraction metadata.
        """
        out = output_dir or (_TEMPLATES_ROOT / theme_name)
        out.mkdir(parents=True, exist_ok=True)

        squares = self._ensure_squares()
        bg_colours = _estimate_background_colours(squares)

        light_bgr = tuple(int(v) for v in bg_colours["light"])
        dark_bgr = tuple(int(v) for v in bg_colours["dark"])

        logger.info(
            "Estimated backgrounds: light=%s dark=%s",
            light_bgr, dark_bgr,
        )

        # Extract templates from known start position squares.
        templates: dict[str, np.ndarray] = {}
        for name, square in squares.items():
            piece = START_POSITION.get(name)
            if piece is None:
                continue
            bg = square_background(square.file, square.rank)
            key = f"{piece}_{bg}"
            if key not in templates:
                templates[key] = _ensure_bgr(square.image)

        # Synthesise missing background variants.
        synthesised = 0
        for key in sorted(_expected_keys()):
            if key in templates:
                continue
            piece, bg = key.rsplit("_", 1)
            other_bg = "dark" if bg == "light" else "light"
            source_key = f"{piece}_{other_bg}"
            source = templates.get(source_key)
            if source is None:
                raise ExtractionError(
                    f"Cannot synthesise missing template '{key}': "
                    f"source '{source_key}' not found."
                )
            templates[key] = _synthesise_background(
                source,
                source_bg=bg_colours[other_bg],
                target_bg=bg_colours[bg],
            )
            synthesised += 1
            logger.debug("Synthesised %s", key)

        missing = sorted(_expected_keys() - set(templates.keys()))
        if missing:
            raise ExtractionError(
                f"Missing templates after synthesis: {', '.join(missing)}"
            )

        # Save template images.
        for key in sorted(_expected_keys()):
            cv2.imwrite(str(out / f"{key}.png"), templates[key])

        # Write manifest.
        manifest = {
            "name": theme_name,
            "description": (
                f"Auto-extracted from a starting-position screenshot "
                f"({self._image_path.name})."
            ),
            "templates": {
                key: f"{key}.png" for key in sorted(_expected_keys())
            },
        }
        (out / "theme.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "Saved %d templates to %s (%d synthesised).",
            len(templates), out, synthesised,
        )

        return ExtractionResult(
            theme_name=theme_name,
            output_dir=out,
            template_count=len(templates),
            synthesised_count=synthesised,
            light_bgr=light_bgr,  # type: ignore[arg-type]
            dark_bgr=dark_bgr,  # type: ignore[arg-type]
        )
