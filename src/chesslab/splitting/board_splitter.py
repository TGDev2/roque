"""Board splitting module for dividing a detected chessboard into 64 squares.

Splits the board region into individual square images indexed by algebraic
notation (a1–h8), supporting both white and black perspective orientations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chesslab.detection.board_detector import BoardRegion

logger = logging.getLogger(__name__)

_FILES = "abcdefgh"


@dataclass(frozen=True)
class SquareImage:
    """A single chessboard square with its image content."""

    name: str
    image: NDArray[np.uint8]
    file: int
    rank: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SquareImage):
            return NotImplemented
        return (
            self.name == other.name
            and self.file == other.file
            and self.rank == other.rank
            and np.array_equal(self.image, other.image)
        )

    def __hash__(self) -> int:
        return hash((self.name, self.file, self.rank))


@dataclass(frozen=True)
class SplitResult:
    """Result of a board split operation."""

    squares: dict[str, SquareImage]
    elapsed_ms: float


class BoardSplitter:
    """Splits a detected chessboard region into 64 individual square images."""

    def split(
        self,
        frame: NDArray[np.uint8],
        board: BoardRegion,
        *,
        white_perspective: bool = True,
    ) -> SplitResult:
        """Split the board region into 64 square images.

        Args:
            frame: Full screen image (BGR or BGRA).
            board: Detected board region coordinates.
            white_perspective: If True, a1 is bottom-left. If False, a1 is
                top-right (black's view).

        Returns:
            SplitResult with 64 SquareImage entries keyed by algebraic name.
        """
        start = time.perf_counter()
        self._validate_inputs(frame, board)

        roi = frame[
            board.y : board.y + board.height,
            board.x : board.x + board.width,
        ]

        cell_h, rem_h = divmod(board.height, 8)
        cell_w, rem_w = divmod(board.width, 8)
        if rem_h or rem_w:
            logger.debug(
                "Board dimensions (%dx%d) not divisible by 8; trimming %d px bottom and %d px right.",
                board.width,
                board.height,
                rem_h,
                rem_w,
            )

        squares: dict[str, SquareImage] = {}

        for row in range(8):
            for col in range(8):
                y0 = row * cell_h
                x0 = col * cell_w
                cell_img = roi[y0 : y0 + cell_h, x0 : x0 + cell_w].copy()

                if white_perspective:
                    file_idx = col
                    rank_idx = 7 - row
                else:
                    file_idx = 7 - col
                    rank_idx = row

                name = f"{_FILES[file_idx]}{rank_idx + 1}"
                squares[name] = SquareImage(
                    name=name,
                    image=cell_img,
                    file=file_idx,
                    rank=rank_idx,
                )

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("Board split into 64 squares (%.1f ms).", elapsed)

        return SplitResult(squares=squares, elapsed_ms=elapsed)

    @staticmethod
    def _validate_inputs(frame: NDArray[np.uint8], board: BoardRegion) -> None:
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError(
                f"Expected frame shape (H, W, 3|4), got {frame.shape}."
            )

        frame_h, frame_w = frame.shape[:2]
        if board.width < 8 or board.height < 8:
            raise ValueError(
                f"Board region too small for 8x8 split: {board.width}x{board.height}."
            )
        if board.x < 0 or board.y < 0:
            raise ValueError(
                f"Board coordinates must be non-negative, got ({board.x}, {board.y})."
            )
        if board.x + board.width > frame_w or board.y + board.height > frame_h:
            raise ValueError(
                "Board region is out of frame bounds: "
                f"board=({board.x}, {board.y}, {board.width}, {board.height}), "
                f"frame=({frame_w}x{frame_h})."
            )
