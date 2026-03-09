"""Tests for the board detection module.

Covers acceptance criteria:
- Detection on chess.com-style board (1920x1080, precision >= 95%)
- Detection on lichess-style board (precision >= 90%)
- No board detected scenario (no exception, detected=False)
- Partially occluded board (< 20% masked, still detected with warning)
- Detection latency < 300 ms
- Cache behaviour (re-detection skipped when board hasn't moved)
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from chesslab.detection.board_detector import (
    BoardDetector,
    BoardRegion,
    DetectionResult,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic chessboard generation
# ---------------------------------------------------------------------------

# chess.com default green theme
CHESSCOM_LIGHT = (235, 236, 208)  # BGR
CHESSCOM_DARK = (115, 149, 82)

# lichess brown theme
LICHESS_LIGHT = (240, 217, 181)
LICHESS_DARK = (181, 136, 99)


def _draw_board(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    size: int,
    light: tuple[int, int, int],
    dark: tuple[int, int, int],
) -> None:
    """Draw an 8x8 chessboard at (x, y) with the given cell colours."""
    cell = size // 8
    for r in range(8):
        for c in range(8):
            colour = light if (r + c) % 2 == 0 else dark
            x0, y0 = x + c * cell, y + r * cell
            cv2.rectangle(image, (x0, y0), (x0 + cell, y0 + cell), colour, -1)


def _make_scene(
    width: int,
    height: int,
    board_x: int,
    board_y: int,
    board_size: int,
    light: tuple[int, int, int] = CHESSCOM_LIGHT,
    dark: tuple[int, int, int] = CHESSCOM_DARK,
    bg_colour: tuple[int, int, int] = (48, 46, 43),
    bgra: bool = False,
) -> NDArray[np.uint8]:
    """Create a synthetic screen image with a chessboard."""
    channels = 4 if bgra else 3
    image = np.full((height, width, channels), 0, dtype=np.uint8)
    # background
    image[:, :, :3] = bg_colour
    if bgra:
        image[:, :, 3] = 255
    _draw_board(image, board_x, board_y, board_size, light, dark)
    return image


def _iou(
    region: BoardRegion, x: int, y: int, size: int,
) -> float:
    """Intersection-over-union between detected region and ground truth."""
    x1 = max(region.x, x)
    y1 = max(region.y, y)
    x2 = min(region.x + region.width, x + size)
    y2 = min(region.y + region.height, y + size)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_det = region.width * region.height
    area_gt = size * size
    union = area_det + area_gt - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Scenario: Detection on chess.com (1920x1080, precision >= 95%)
# ---------------------------------------------------------------------------


class TestDetectChesscom:
    """AC: chess.com board at 1920x1080, precision >= 95%, < 300 ms."""

    BOARD_X = 430
    BOARD_Y = 60
    BOARD_SIZE = 640

    @pytest.fixture
    def scene(self) -> NDArray[np.uint8]:
        return _make_scene(
            1920, 1080, self.BOARD_X, self.BOARD_Y, self.BOARD_SIZE,
            CHESSCOM_LIGHT, CHESSCOM_DARK,
        )

    def test_board_detected(self, scene: NDArray[np.uint8]) -> None:
        result = BoardDetector().detect(scene)
        assert result.detected is True
        assert result.board is not None

    def test_coordinates_returned(self, scene: NDArray[np.uint8]) -> None:
        result = BoardDetector().detect(scene)
        b = result.board
        assert b is not None
        assert isinstance(b.x, int)
        assert isinstance(b.y, int)
        assert isinstance(b.width, int)
        assert isinstance(b.height, int)

    def test_precision_gte_95(self, scene: NDArray[np.uint8]) -> None:
        result = BoardDetector().detect(scene)
        assert result.board is not None
        iou = _iou(result.board, self.BOARD_X, self.BOARD_Y, self.BOARD_SIZE)
        assert iou >= 0.95, f"IoU {iou:.3f} < 0.95"

    def test_latency_under_300ms(self, scene: NDArray[np.uint8]) -> None:
        detector = BoardDetector()
        start = time.perf_counter()
        detector.detect(scene)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 300, f"Detection took {elapsed_ms:.1f} ms (limit: 300)"

    def test_bgra_input_accepted(self) -> None:
        scene = _make_scene(
            1920, 1080, self.BOARD_X, self.BOARD_Y, self.BOARD_SIZE,
            CHESSCOM_LIGHT, CHESSCOM_DARK, bgra=True,
        )
        result = BoardDetector().detect(scene)
        assert result.detected is True


# ---------------------------------------------------------------------------
# Scenario: Detection on lichess (precision >= 90%)
# ---------------------------------------------------------------------------


class TestDetectLichess:
    """AC: lichess board, precision >= 90%."""

    BOARD_X = 200
    BOARD_Y = 100
    BOARD_SIZE = 720

    @pytest.fixture
    def scene(self) -> NDArray[np.uint8]:
        return _make_scene(
            1920, 1080, self.BOARD_X, self.BOARD_Y, self.BOARD_SIZE,
            LICHESS_LIGHT, LICHESS_DARK,
            bg_colour=(50, 48, 44),
        )

    def test_board_detected(self, scene: NDArray[np.uint8]) -> None:
        result = BoardDetector().detect(scene)
        assert result.detected is True

    def test_precision_gte_90(self, scene: NDArray[np.uint8]) -> None:
        result = BoardDetector().detect(scene)
        assert result.board is not None
        iou = _iou(result.board, self.BOARD_X, self.BOARD_Y, self.BOARD_SIZE)
        assert iou >= 0.90, f"IoU {iou:.3f} < 0.90"


# ---------------------------------------------------------------------------
# Scenario: No board detected
# ---------------------------------------------------------------------------


class TestNoBoard:
    """AC: no exception, detected=False, warning-friendly result."""

    def test_plain_background(self) -> None:
        image = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        result = BoardDetector().detect(image)
        assert result.detected is False
        assert result.board is None

    def test_no_exception_raised(self) -> None:
        image = np.full((1080, 1920, 3), 64, dtype=np.uint8)
        result = BoardDetector().detect(image)
        assert isinstance(result, DetectionResult)
        assert result.detected is False

    def test_random_noise(self) -> None:
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        result = BoardDetector().detect(image)
        assert result.detected is False


# ---------------------------------------------------------------------------
# Scenario: Partially occluded board (< 20%)
# ---------------------------------------------------------------------------


class TestPartiallyOccluded:
    """AC: < 20% masked => still detected, optional warning logged."""

    BOARD_X = 430
    BOARD_Y = 60
    BOARD_SIZE = 640

    @pytest.fixture
    def occluded_scene(self) -> NDArray[np.uint8]:
        scene = _make_scene(
            1920, 1080, self.BOARD_X, self.BOARD_Y, self.BOARD_SIZE,
            CHESSCOM_LIGHT, CHESSCOM_DARK,
        )
        # Overlay a rectangle covering ~15% of the board (top-right corner)
        overlay_w = int(self.BOARD_SIZE * 0.15)
        overlay_h = int(self.BOARD_SIZE * 0.15)
        ox = self.BOARD_X + self.BOARD_SIZE - overlay_w
        oy = self.BOARD_Y
        cv2.rectangle(
            scene, (ox, oy), (ox + overlay_w, oy + overlay_h),
            (200, 200, 200), -1,
        )
        return scene

    def test_still_detected(self, occluded_scene: NDArray[np.uint8]) -> None:
        result = BoardDetector().detect(occluded_scene)
        assert result.detected is True
        assert result.board is not None

    def test_reasonable_precision(
        self, occluded_scene: NDArray[np.uint8]
    ) -> None:
        result = BoardDetector().detect(occluded_scene)
        assert result.board is not None
        iou = _iou(
            result.board, self.BOARD_X, self.BOARD_Y, self.BOARD_SIZE,
        )
        assert iou >= 0.80, f"IoU {iou:.3f} < 0.80 on occluded board"


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


class TestCache:
    """AC: cache avoids full re-detection when board hasn't moved."""

    BOARD_X = 430
    BOARD_Y = 60
    BOARD_SIZE = 640

    @pytest.fixture
    def scene(self) -> NDArray[np.uint8]:
        return _make_scene(
            1920, 1080, self.BOARD_X, self.BOARD_Y, self.BOARD_SIZE,
            CHESSCOM_LIGHT, CHESSCOM_DARK,
        )

    def test_second_detect_uses_cache(self, scene: NDArray[np.uint8]) -> None:
        detector = BoardDetector()
        r1 = detector.detect(scene)
        assert r1.detected is True

        r2 = detector.detect(scene)
        assert r2.detected is True
        assert r2.elapsed_ms <= r1.elapsed_ms or r2.elapsed_ms < 50

    def test_cache_invalidated_on_reset(
        self, scene: NDArray[np.uint8]
    ) -> None:
        detector = BoardDetector()
        detector.detect(scene)
        detector.reset_cache()
        assert detector.last_result is None

    def test_cache_miss_on_different_scene(self) -> None:
        detector = BoardDetector()
        s1 = _make_scene(
            1920, 1080, 430, 60, 640,
            CHESSCOM_LIGHT, CHESSCOM_DARK,
        )
        detector.detect(s1)

        s2 = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        r2 = detector.detect(s2)
        assert r2.detected is False


# ---------------------------------------------------------------------------
# Configuration / data classes
# ---------------------------------------------------------------------------


class TestBoardRegion:
    def test_frozen(self) -> None:
        r = BoardRegion(x=0, y=0, width=640, height=640, confidence=0.9)
        with pytest.raises(AttributeError):
            r.x = 10  # type: ignore[misc]


class TestDetectionResult:
    def test_no_board(self) -> None:
        r = DetectionResult(board=None, detected=False, elapsed_ms=5.0)
        assert r.warning is None

    def test_with_warning(self) -> None:
        r = DetectionResult(
            board=None, detected=False, elapsed_ms=5.0,
            warning="partial",
        )
        assert r.warning == "partial"


# ---------------------------------------------------------------------------
# Multiple board positions (robustness)
# ---------------------------------------------------------------------------


class TestVariousPositions:
    """Verify detection works with different board positions and sizes."""

    @pytest.mark.parametrize(
        "bx, by, bsize",
        [
            (0, 0, 400),       # top-left corner
            (1280, 0, 640),    # right side
            (640, 220, 640),   # centred
            (100, 100, 800),   # large board
            (800, 500, 300),   # smaller board
        ],
        ids=["top-left", "right", "centred", "large", "small"],
    )
    def test_detected(self, bx: int, by: int, bsize: int) -> None:
        scene = _make_scene(1920, 1080, bx, by, bsize)
        result = BoardDetector().detect(scene)
        assert result.detected is True, (
            f"Board at ({bx},{by}) size {bsize} not detected"
        )
        assert result.board is not None
        iou = _iou(result.board, bx, by, bsize)
        assert iou >= 0.85, f"IoU {iou:.3f} too low for board at ({bx},{by})"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_board_too_small_ignored(self) -> None:
        scene = _make_scene(1920, 1080, 500, 500, 100)
        result = BoardDetector().detect(scene)
        assert result.detected is False

    def test_invalid_frame_shape_raises(self) -> None:
        bad = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unexpected frame shape"):
            BoardDetector().detect(bad)
