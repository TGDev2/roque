"""Board detection module using OpenCV for chessboard localization.

Detects a chessboard in a screen capture image using edge detection,
contour analysis, and gradient-based grid validation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoardRegion:
    """Detected chessboard region coordinates."""

    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass(frozen=True)
class DetectionResult:
    """Result of a board detection attempt."""

    board: BoardRegion | None
    detected: bool
    elapsed_ms: float
    warning: str | None = None


class BoardDetector:
    """Detects a chessboard in a screen capture image.

    Uses edge detection and contour analysis to locate the board,
    then validates candidates using gradient-based grid scoring.
    Falls back to Hough line detection when contour analysis fails.
    Caches the last known position for faster re-detection.
    """

    MIN_BOARD_SIZE: int = 200
    MAX_BOARD_SIZE_RATIO: float = 0.95
    ASPECT_RATIO_TOLERANCE: float = 0.15
    GRID_SCORE_THRESHOLD: float = 0.4

    def __init__(self) -> None:
        self._last_result: DetectionResult | None = None

    @property
    def last_result(self) -> DetectionResult | None:
        return self._last_result

    def detect(self, frame: NDArray[np.uint8]) -> DetectionResult:
        """Detect a chessboard in the given frame.

        Args:
            frame: Image as numpy array in BGR (H, W, 3) or BGRA (H, W, 4).

        Returns:
            DetectionResult with board coordinates if found, or detected=False.
        """
        start = time.perf_counter()

        if frame.ndim == 3 and frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            bgr = frame
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Try cache first
        cached = self._try_cache(gray)
        if cached is not None:
            elapsed = (time.perf_counter() - start) * 1000
            result = DetectionResult(
                board=cached, detected=True, elapsed_ms=elapsed
            )
            self._last_result = result
            return result

        board, warning = self._full_detect(gray)
        if board is not None:
            board = self._snap_to_grid(board, frame_w=gray.shape[1], frame_h=gray.shape[0])
        elapsed = (time.perf_counter() - start) * 1000

        result = DetectionResult(
            board=board,
            detected=board is not None,
            elapsed_ms=elapsed,
            warning=warning,
        )
        self._last_result = result

        if board:
            logger.debug(
                "Board detected at (%d, %d) size %dx%d "
                "(confidence=%.2f, %.1f ms).",
                board.x, board.y, board.width, board.height,
                board.confidence, elapsed,
            )
        else:
            logger.debug("No chessboard detected (%.1f ms).", elapsed)

        return result

    def reset_cache(self) -> None:
        """Clear the cached detection result."""
        self._last_result = None

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _try_cache(self, gray: NDArray[np.uint8]) -> BoardRegion | None:
        if self._last_result is None or self._last_result.board is None:
            return None

        prev = self._last_result.board
        h, w = gray.shape[:2]
        if prev.x + prev.width > w or prev.y + prev.height > h:
            return None

        score = self._grid_score(gray, prev.x, prev.y, prev.width, prev.height)
        if score >= self.GRID_SCORE_THRESHOLD:
            return BoardRegion(
                x=prev.x, y=prev.y,
                width=prev.width, height=prev.height,
                confidence=score,
            )
        return None

    # ------------------------------------------------------------------
    # Full detection pipeline
    # ------------------------------------------------------------------

    def _full_detect(
        self, gray: NDArray[np.uint8]
    ) -> tuple[BoardRegion | None, str | None]:
        h, w = gray.shape[:2]
        warning: str | None = None

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = self.MIN_BOARD_SIZE ** 2
        max_side = min(h, w) * self.MAX_BOARD_SIZE_RATIO

        candidates: list[tuple[float, int, int, int, int, float]] = []

        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)

            if bw * bh < min_area or bw > max_side or bh > max_side:
                continue

            aspect = bw / bh if bh > 0 else 0.0
            if abs(aspect - 1.0) > self.ASPECT_RATIO_TOLERANCE:
                continue

            # Normalise to square
            side = max(bw, bh)
            cx, cy = bx + bw // 2, by + bh // 2
            sq_x = max(0, cx - side // 2)
            sq_y = max(0, cy - side // 2)
            if sq_x + side > w:
                sq_x = w - side
            if sq_y + side > h:
                sq_y = h - side

            score = self._grid_score(gray, sq_x, sq_y, side, side)
            if score >= self.GRID_SCORE_THRESHOLD:
                contour_area = cv2.contourArea(contour)
                fill_ratio = contour_area / (bw * bh) if bw * bh > 0 else 1.0
                candidates.append(
                    (score, sq_x, sq_y, side, side, fill_ratio)
                )

        if not candidates:
            board = self._detect_via_lines(gray, edges)
            return board, warning

        candidates.sort(key=lambda c: (c[0], c[3] * c[4]), reverse=True)
        score, bx, by, bw, bh, fill_ratio = candidates[0]

        if fill_ratio < 0.80:
            warning = "Board may be partially occluded"
            logger.warning(warning)

        return (
            BoardRegion(x=bx, y=by, width=bw, height=bh, confidence=score),
            warning,
        )

    # ------------------------------------------------------------------
    # Grid snapping helpers
    # ------------------------------------------------------------------

    def _snap_to_grid(
        self, board: BoardRegion, *, frame_w: int, frame_h: int,
    ) -> BoardRegion:
        """Adjust board bounds so width/height align with an 8x8 grid.

        Detection can return off-by-1 (or a few) pixels around the board.
        Snapping dimensions to multiples of 8 keeps square crops aligned.
        """
        x, w = self._snap_axis(board.x, board.width, frame_w)
        y, h = self._snap_axis(board.y, board.height, frame_h)

        side = min(w, h)
        if side % 8 != 0:
            side -= side % 8

        if side < 8:
            return board

        if w != side:
            x += (w - side) // 2
        if h != side:
            y += (h - side) // 2

        x = max(0, min(x, frame_w - side))
        y = max(0, min(y, frame_h - side))

        return BoardRegion(
            x=x,
            y=y,
            width=side,
            height=side,
            confidence=board.confidence,
        )

    @staticmethod
    def _snap_axis(start: int, length: int, limit: int) -> tuple[int, int]:
        """Snap one axis length to the nearest multiple of 8.

        Preference order:
        1) nearest multiple,
        2) expand to include neighbouring pixels if possible,
        3) shrink with centred crop.
        """
        if length < 8:
            return start, length

        lower = (length // 8) * 8
        upper = lower + 8
        candidates = sorted(
            {c for c in (lower, upper) if c >= 8},
            key=lambda c: (abs(c - length), -c),
        )

        for target in candidates:
            if target == length:
                return start, length

            if target > length:
                new_start = start
                new_end = start + target
                if new_end <= limit:
                    return new_start, target

                overflow = new_end - limit
                if start - overflow >= 0:
                    return start - overflow, target
                continue

            trim = length - target
            new_start = start + trim // 2
            new_start = max(0, min(new_start, limit - target))
            return new_start, target

        return start, length

    # ------------------------------------------------------------------
    # Fallback: Hough line detection
    # ------------------------------------------------------------------

    def _detect_via_lines(
        self, gray: NDArray[np.uint8], edges: NDArray[np.uint8]
    ) -> BoardRegion | None:
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=80, minLineLength=100, maxLineGap=10,
        )
        if lines is None:
            return None

        horizontals: list[int] = []
        verticals: list[int] = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            if angle < 10 or angle > 170:
                horizontals.append((y1 + y2) // 2)
            elif 80 < angle < 100:
                verticals.append((x1 + x2) // 2)

        if len(horizontals) < 7 or len(verticals) < 7:
            return None

        h_clusters = self._cluster_positions(sorted(horizontals))
        v_clusters = self._cluster_positions(sorted(verticals))

        h_grid = self._find_evenly_spaced(h_clusters)
        v_grid = self._find_evenly_spaced(v_clusters)

        if h_grid is None or v_grid is None:
            return None

        x = int(v_grid[0])
        y = int(h_grid[0])
        bw = int(v_grid[-1] - v_grid[0])
        bh = int(h_grid[-1] - h_grid[0])

        if bw < self.MIN_BOARD_SIZE or bh < self.MIN_BOARD_SIZE:
            return None

        aspect = bw / bh if bh > 0 else 0.0
        if abs(aspect - 1.0) > self.ASPECT_RATIO_TOLERANCE:
            return None

        side = max(bw, bh)
        score = self._grid_score(gray, x, y, side, side)
        if score >= self.GRID_SCORE_THRESHOLD:
            return BoardRegion(x=x, y=y, width=side, height=side, confidence=score)

        return None

    # ------------------------------------------------------------------
    # Grid scoring via gradient periodicity
    # ------------------------------------------------------------------

    def _grid_score(
        self, gray: NDArray[np.uint8], x: int, y: int, w: int, h: int,
    ) -> float:
        """Score how well a region matches an 8x8 chessboard grid.

        Analyses horizontal and vertical gradient profiles for periodic
        peaks at cell boundaries.
        """
        if w <= 0 or h <= 0:
            return 0.0

        roi = gray[y : y + h, x : x + w]
        if roi.size == 0:
            return 0.0

        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

        # Sum absolute gradients along each axis
        v_profile = np.abs(grad_x).sum(axis=0)  # vertical-line profile
        h_profile = np.abs(grad_y).sum(axis=1)  # horizontal-line profile

        cell_w = w / 8.0
        cell_h = h / 8.0

        v_score = self._periodicity_score(v_profile, cell_w)
        h_score = self._periodicity_score(h_profile, cell_h)

        return (v_score + h_score) / 2.0

    @staticmethod
    def _periodicity_score(profile: NDArray[np.float64], period: float) -> float:
        """Check if *profile* has peaks at multiples of *period*."""
        if len(profile) < 16 or period < 2:
            return 0.0

        threshold = float(np.mean(profile) + np.std(profile) * 0.5)
        tolerance = max(1, int(period * 0.25))

        found = 0
        for i in range(1, 8):  # 7 internal grid lines
            pos = int(i * period)
            lo = max(0, pos - tolerance)
            hi = min(len(profile), pos + tolerance + 1)
            if float(profile[lo:hi].max()) > threshold:
                found += 1

        return found / 7.0

    # ------------------------------------------------------------------
    # Line clustering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_positions(
        positions: list[int], min_gap: int = 15,
    ) -> list[float]:
        if not positions:
            return []
        clusters: list[list[int]] = [[positions[0]]]
        for p in positions[1:]:
            if p - clusters[-1][-1] < min_gap:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [sum(c) / len(c) for c in clusters]

    @staticmethod
    def _find_evenly_spaced(
        positions: list[float],
        target_count: int = 9,
        tolerance: float = 0.3,
    ) -> list[float] | None:
        if len(positions) < target_count:
            return None

        best: list[float] | None = None

        for i in range(len(positions) - 1):
            spacing = positions[i + 1] - positions[i]
            if spacing < 20:
                continue

            group = [positions[i]]
            expected = positions[i] + spacing

            for j in range(i + 1, len(positions)):
                if abs(positions[j] - expected) < spacing * tolerance:
                    group.append(positions[j])
                    expected = positions[j] + spacing

            if len(group) >= target_count:
                best = group[:target_count]
                return best

        return best
