"""Tests for the board splitting module.

Covers acceptance criteria:
- 64 square images produced, indexed by algebraic notation (a1–h8)
- Homogeneous square sizes
- Correct content per square
- Split latency < 50 ms
- Orientation parameter (white/black perspective)
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from chesslab.detection.board_detector import BoardRegion
from chesslab.splitting.board_splitter import (
    BoardSplitter,
    SplitResult,
    SquareImage,
)
from chesslab.splitting.square_color import is_light_square

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LIGHT = (235, 236, 208)  # BGR
DARK = (115, 149, 82)


def _square_colour(file_idx: int, rank_idx: int) -> tuple[int, int, int]:
    """Return expected colour for a square given its file/rank indices."""
    return LIGHT if is_light_square(file_idx, rank_idx) else DARK


def _make_board_image(
    size: int = 640,
    light: tuple[int, int, int] = LIGHT,
    dark: tuple[int, int, int] = DARK,
) -> NDArray[np.uint8]:
    """Create a standalone 8x8 chessboard image (no surrounding scene).

    The board is drawn in white perspective: top-left pixel is a8.
    Colour assignment follows chess convention: a1 is dark, so
    square (file, rank) is light when (file + rank) is odd.
    In the image: row 0 = rank 8, col 0 = file a.
    """
    cell = size // 8
    img_size = cell * 8  # ensure exact divisibility
    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            file_idx = col
            rank_idx = 7 - row
            colour = light if is_light_square(file_idx, rank_idx) else dark
            y0, x0 = row * cell, col * cell
            cv2.rectangle(image, (x0, y0), (x0 + cell, y0 + cell), colour, -1)
    return image


def _make_scene_with_board(
    board_size: int = 640,
    board_x: int = 100,
    board_y: int = 50,
    scene_w: int = 1920,
    scene_h: int = 1080,
) -> tuple[NDArray[np.uint8], BoardRegion]:
    """Create a scene with an embedded chessboard and the matching BoardRegion."""
    scene = np.full((scene_h, scene_w, 3), 48, dtype=np.uint8)
    board_img = _make_board_image(board_size)
    actual_size = (board_size // 8) * 8
    scene[board_y : board_y + actual_size, board_x : board_x + actual_size] = board_img
    region = BoardRegion(
        x=board_x, y=board_y,
        width=actual_size, height=actual_size,
        confidence=0.95,
    )
    return scene, region


# ---------------------------------------------------------------------------
# Scenario: Nominal split — 64 squares, algebraic indexing
# ---------------------------------------------------------------------------


class TestNominalSplit:
    """AC: 64 images indexed a1–h8, homogeneous sizes."""

    @pytest.fixture
    def result(self) -> SplitResult:
        scene, region = _make_scene_with_board()
        return BoardSplitter().split(scene, region)

    def test_64_squares_produced(self, result: SplitResult) -> None:
        assert len(result.squares) == 64

    def test_all_algebraic_names_present(self, result: SplitResult) -> None:
        expected = {f"{f}{r}" for f in "abcdefgh" for r in range(1, 9)}
        assert set(result.squares.keys()) == expected

    def test_uniform_size(self, result: SplitResult) -> None:
        shapes = {sq.image.shape for sq in result.squares.values()}
        assert len(shapes) == 1, f"Non-uniform shapes: {shapes}"

    def test_square_dimensions(self, result: SplitResult) -> None:
        sq = result.squares["a1"]
        cell = 640 // 8
        assert sq.image.shape[0] == cell
        assert sq.image.shape[1] == cell

    def test_file_rank_indices(self, result: SplitResult) -> None:
        sq = result.squares["e4"]
        assert sq.file == 4  # e = index 4
        assert sq.rank == 3  # 4 - 1 = 3


# ---------------------------------------------------------------------------
# Scenario: Correct content per square
# ---------------------------------------------------------------------------


class TestSquareContent:
    """AC: each square contains exactly the content of the corresponding cell."""

    def test_square_colours_white_perspective(self) -> None:
        scene, region = _make_scene_with_board()
        result = BoardSplitter().split(scene, region, white_perspective=True)

        for name, sq in result.squares.items():
            expected = _square_colour(sq.file, sq.rank)
            # Check centre pixel of the square
            cy, cx = sq.image.shape[0] // 2, sq.image.shape[1] // 2
            pixel = tuple(int(v) for v in sq.image[cy, cx])
            assert pixel == expected, (
                f"Square {name}: expected {expected}, got {pixel}"
            )


# ---------------------------------------------------------------------------
# Scenario: Orientation — black perspective
# ---------------------------------------------------------------------------


class TestOrientation:
    """AC: orientation parameter correctly flips indexing."""

    def test_square_colours_black_perspective(self) -> None:
        scene, region = _make_scene_with_board()
        result = BoardSplitter().split(scene, region, white_perspective=False)

        assert len(result.squares) == 64

        for name, sq in result.squares.items():
            expected = _square_colour(sq.file, sq.rank)
            cy, cx = sq.image.shape[0] // 2, sq.image.shape[1] // 2
            pixel = tuple(int(v) for v in sq.image[cy, cx])
            assert pixel == expected, (
                f"Square {name} (black view): expected {expected}, got {pixel}"
            )

    def test_grid_position_differs_by_orientation(self) -> None:
        """Verify that orientation flips which grid cell maps to a given name.

        We build a board with unique per-cell intensity values so we can
        distinguish which physical grid cell was assigned to each name.
        """
        size = 640
        cell = size // 8
        # Create a board where each cell has a unique brightness
        image = np.zeros((size, size, 3), dtype=np.uint8)
        for row in range(8):
            for col in range(8):
                val = row * 8 + col + 10  # unique per cell
                y0, x0 = row * cell, col * cell
                image[y0 : y0 + cell, x0 : x0 + cell] = val

        scene = np.full((1080, 1920, 3), 0, dtype=np.uint8)
        scene[50 : 50 + size, 100 : 100 + size] = image
        region = BoardRegion(x=100, y=50, width=size, height=size, confidence=0.9)

        splitter = BoardSplitter()
        white = splitter.split(scene, region, white_perspective=True)
        black = splitter.split(scene, region, white_perspective=False)

        # a1 should reference different physical cells in each perspective
        assert not np.array_equal(
            white.squares["a1"].image,
            black.squares["a1"].image,
        )

    def test_h8_white_equals_a1_white_content_check(self) -> None:
        """In white view, a8 is top-left. In black view, a8 is bottom-right.
        Both should have the same pixel content (same physical square)."""
        scene, region = _make_scene_with_board()
        splitter = BoardSplitter()
        white = splitter.split(scene, region, white_perspective=True)
        black = splitter.split(scene, region, white_perspective=False)

        # Same algebraic square should have same colour regardless of perspective
        for name in white.squares:
            w_sq = white.squares[name]
            b_sq = black.squares[name]
            w_cy, w_cx = w_sq.image.shape[0] // 2, w_sq.image.shape[1] // 2
            b_cy, b_cx = b_sq.image.shape[0] // 2, b_sq.image.shape[1] // 2
            w_pixel = tuple(int(v) for v in w_sq.image[w_cy, w_cx])
            b_pixel = tuple(int(v) for v in b_sq.image[b_cy, b_cx])
            assert w_pixel == b_pixel, (
                f"Square {name}: white={w_pixel}, black={b_pixel}"
            )


# ---------------------------------------------------------------------------
# Scenario: Performance — < 50 ms
# ---------------------------------------------------------------------------


class TestPerformance:
    """AC: split completes in < 50 ms."""

    def test_latency_under_50ms(self) -> None:
        scene, region = _make_scene_with_board(board_size=800)
        splitter = BoardSplitter()
        start = time.perf_counter()
        splitter.split(scene, region)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50, f"Split took {elapsed_ms:.1f} ms (limit: 50)"


# ---------------------------------------------------------------------------
# Dataclass checks
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_square_image_frozen(self) -> None:
        sq = SquareImage(
            name="a1",
            image=np.zeros((80, 80, 3), dtype=np.uint8),
            file=0,
            rank=0,
        )
        with pytest.raises(AttributeError):
            sq.name = "b2"  # type: ignore[misc]

    def test_split_result_frozen(self) -> None:
        result = SplitResult(squares={}, elapsed_ms=1.0)
        with pytest.raises(AttributeError):
            result.elapsed_ms = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_non_divisible_by_8_board_size(self) -> None:
        """Board size not perfectly divisible by 8 still produces 64 squares."""
        board_size = 645
        cell = board_size // 8

        # Build an explicit 645x645 board region so remainder trimming is exercised.
        board = np.zeros((board_size, board_size, 3), dtype=np.uint8)
        for row in range(8):
            for col in range(8):
                y0, x0 = row * cell, col * cell
                board[y0 : y0 + cell, x0 : x0 + cell] = (row * 8 + col + 10)

        scene = np.full((1080, 1920, 3), 48, dtype=np.uint8)
        board_x, board_y = 100, 50
        scene[board_y : board_y + board_size, board_x : board_x + board_size] = board
        region = BoardRegion(
            x=board_x,
            y=board_y,
            width=board_size,
            height=board_size,
            confidence=0.95,
        )

        result = BoardSplitter().split(scene, region)
        assert len(result.squares) == 64
        assert result.squares["a1"].image.shape[:2] == (cell, cell)

    def test_out_of_bounds_region_raises(self) -> None:
        scene = np.zeros((100, 100, 3), dtype=np.uint8)
        region = BoardRegion(x=60, y=60, width=50, height=50, confidence=0.9)
        with pytest.raises(ValueError, match="out of frame bounds"):
            BoardSplitter().split(scene, region)

    def test_too_small_board_raises(self) -> None:
        scene = np.zeros((100, 100, 3), dtype=np.uint8)
        region = BoardRegion(x=10, y=10, width=7, height=7, confidence=0.9)
        with pytest.raises(ValueError, match="too small"):
            BoardSplitter().split(scene, region)

    def test_bgra_frame(self) -> None:
        """BGRA frames are handled (numpy slicing works on any channel count)."""
        board_size = 640
        board_x, board_y = 100, 50
        scene = np.full((1080, 1920, 4), 48, dtype=np.uint8)
        scene[:, :, 3] = 255
        board_img = _make_board_image(board_size)
        # Insert BGR into BGRA scene
        scene[board_y : board_y + board_size, board_x : board_x + board_size, :3] = (
            board_img
        )
        region = BoardRegion(
            x=board_x, y=board_y,
            width=board_size, height=board_size,
            confidence=0.95,
        )
        result = BoardSplitter().split(scene, region)
        assert len(result.squares) == 64
        # Each square should have 4 channels
        assert result.squares["a1"].image.shape[2] == 4
