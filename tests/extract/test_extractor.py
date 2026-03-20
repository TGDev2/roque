"""Tests for the template extraction module."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from chesslab.extract.extractor import (
    ExtractionError,
    ExtractionResult,
    TemplateExtractor,
    _estimate_background_colours,
    _expected_keys,
    _synthesise_background,
)
from chesslab.splitting.board_splitter import SquareImage
from chesslab.splitting.square_color import square_background

_FILES = "abcdefgh"
SIZE = 80
LIGHT_BG = (208, 236, 235)
DARK_BG = (82, 149, 115)


def _make_empty_board_image(size: int = 640) -> NDArray[np.uint8]:
    """Create a synthetic board image (8x8 grid, no pieces)."""
    sq = size // 8
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(8):
        for f in range(8):
            bg = square_background(f, r)
            colour = LIGHT_BG if bg == "light" else DARK_BG
            y0, y1 = (7 - r) * sq, (8 - r) * sq
            x0, x1 = f * sq, (f + 1) * sq
            img[y0:y1, x0:x1] = colour
    return img


def _make_starting_board_image(
    board_size: int = 640, border: int = 40,
) -> NDArray[np.uint8]:
    """Create a synthetic starting position board image.

    Embeds the board in a larger image with a contrasting border so that
    ``BoardDetector`` can find the board via contour detection.
    """
    sq = board_size // 8
    board = _make_empty_board_image(board_size)

    # Draw simple markers for pieces.
    for f in range(8):
        x = f * sq
        # White back rank (rank 0 → row 7)
        y = 7 * sq
        cv2.rectangle(
            board,
            (x + sq // 4, y + sq // 4),
            (x + 3 * sq // 4, y + 3 * sq // 4),
            (255, 255, 255), -1,
        )
        # White pawns (rank 1 → row 6)
        y = 6 * sq
        cv2.circle(board, (x + sq // 2, y + sq // 2), sq // 4, (255, 255, 255), -1)
        # Black back rank (rank 7 → row 0)
        y = 0
        cv2.rectangle(
            board,
            (x + sq // 4, y + sq // 4),
            (x + 3 * sq // 4, y + 3 * sq // 4),
            (30, 30, 30), -1,
        )
        # Black pawns (rank 6 → row 1)
        y = 1 * sq
        cv2.circle(board, (x + sq // 2, y + sq // 2), sq // 4, (30, 30, 30), -1)

    # Embed in a larger image with a contrasting surround.
    total = board_size + 2 * border
    img = np.full((total, total, 3), (40, 40, 40), dtype=np.uint8)
    img[border : border + board_size, border : border + board_size] = board
    return img


# ---------------------------------------------------------------------------
# Background estimation
# ---------------------------------------------------------------------------


class TestBackgroundEstimation:
    """Verify background colour estimation from empty squares."""

    def test_solid_board_exact_colours(self) -> None:
        """Empty board returns exact light/dark BGR values."""
        squares: dict[str, SquareImage] = {}
        for f_idx in range(8):
            for r_idx in range(8):
                name = f"{_FILES[f_idx]}{r_idx + 1}"
                bg = square_background(f_idx, r_idx)
                colour = LIGHT_BG if bg == "light" else DARK_BG
                img = np.full((SIZE, SIZE, 3), colour, dtype=np.uint8)
                squares[name] = SquareImage(
                    name=name, image=img, file=f_idx, rank=r_idx,
                )
        colours = _estimate_background_colours(squares)
        np.testing.assert_array_equal(colours["light"], list(LIGHT_BG))
        np.testing.assert_array_equal(colours["dark"], list(DARK_BG))


# ---------------------------------------------------------------------------
# Background synthesis
# ---------------------------------------------------------------------------


class TestBackgroundSynthesis:
    """Verify background colour swap logic."""

    def test_uniform_source_swaps_completely(self) -> None:
        """A uniform-colour square becomes entirely target-bg."""
        source = np.full((80, 80, 3), LIGHT_BG, dtype=np.uint8)
        result = _synthesise_background(
            source,
            source_bg=np.array(LIGHT_BG, dtype=np.uint8),
            target_bg=np.array(DARK_BG, dtype=np.uint8),
        )
        np.testing.assert_array_equal(result, np.full_like(result, DARK_BG))

    def test_piece_pixels_preserved(self) -> None:
        """Piece pixels (far from background) are kept intact."""
        source = np.full((80, 80, 3), LIGHT_BG, dtype=np.uint8)
        # Draw a white piece region
        source[30:50, 30:50] = (255, 255, 255)
        result = _synthesise_background(
            source,
            source_bg=np.array(LIGHT_BG, dtype=np.uint8),
            target_bg=np.array(DARK_BG, dtype=np.uint8),
        )
        # Piece pixels should still be white
        expected_piece = np.full((20, 20, 3), (255, 255, 255), dtype=np.uint8)
        np.testing.assert_array_equal(result[30:50, 30:50], expected_piece)
        # Background pixels should be dark
        np.testing.assert_array_equal(result[0:5, 0:5], np.full((5, 5, 3), DARK_BG))


# ---------------------------------------------------------------------------
# Expected keys
# ---------------------------------------------------------------------------


class TestExpectedKeys:
    def test_24_keys(self) -> None:
        keys = _expected_keys()
        assert len(keys) == 24

    def test_all_pieces_present(self) -> None:
        keys = _expected_keys()
        for piece in ("wK", "wQ", "wR", "wB", "wN", "wP",
                       "bK", "bQ", "bR", "bB", "bN", "bP"):
            assert f"{piece}_light" in keys
            assert f"{piece}_dark" in keys


# ---------------------------------------------------------------------------
# TemplateExtractor integration
# ---------------------------------------------------------------------------


class TestTemplateExtractor:
    """Integration test: extract templates from a synthetic board image."""

    def test_extraction_produces_24_templates(self, tmp_path: Path) -> None:
        """Full extraction from a synthetic screenshot."""
        img = _make_starting_board_image(640)
        img_path = tmp_path / "board.png"
        cv2.imwrite(str(img_path), img)

        output_dir = tmp_path / "output"
        extractor = TemplateExtractor(img_path)
        result = extractor.extract("test-theme", output_dir)

        assert result.template_count == 24
        assert result.output_dir == output_dir
        assert (output_dir / "theme.json").exists()

        # Check all 24 PNG files exist.
        for key in _expected_keys():
            assert (output_dir / f"{key}.png").exists(), f"Missing {key}.png"

        # Verify manifest.
        with (output_dir / "theme.json").open() as fh:
            manifest = json.load(fh)
        assert manifest["name"] == "test-theme"
        assert len(manifest["templates"]) == 24

    def test_extraction_invalid_image_raises(self, tmp_path: Path) -> None:
        """Non-existent image raises ExtractionError."""
        with pytest.raises(ExtractionError, match="Failed to read"):
            TemplateExtractor(tmp_path / "nonexistent.png")

    def test_detect_theme_runs_without_error(self, tmp_path: Path) -> None:
        """Theme detection runs on extracted squares and returns a result."""
        img = _make_starting_board_image(640)
        img_path = tmp_path / "board.png"
        cv2.imwrite(str(img_path), img)

        extractor = TemplateExtractor(img_path)
        result = extractor.detect_theme()
        # Synthetic images may not match exactly due to border effects,
        # but the call must complete and return a valid result.
        assert result.elapsed_ms >= 0
        assert len(result.available_themes) > 0
