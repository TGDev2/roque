"""Tests for the live frame processor used by remote Windows-host capture."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from chesslab.detection.board_detector import BoardRegion, DetectionResult
from chesslab.fen.fen_generator import FenConfig, FenResult, InvalidPositionError
from chesslab.live import LiveFrameStatus, LivePositionProcessor
from chesslab.recognition import (
    RecognitionFlowResult,
    RecognitionFlowStatus,
    RecognitionResult,
)
from chesslab.recognition.piece_recognizer import PieceType, SquareClassification
from chesslab.splitting.board_splitter import SplitResult, SquareImage


def _make_frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _make_squares() -> dict[str, SquareImage]:
    squares: dict[str, SquareImage] = {}
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    for rank in range(1, 9):
        for file_idx, file_char in enumerate("abcdefgh"):
            name = f"{file_char}{rank}"
            squares[name] = SquareImage(
                name=name,
                image=image.copy(),
                file=file_idx,
                rank=rank - 1,
            )
    return squares


def _make_classifications() -> dict[str, SquareClassification]:
    classifications: dict[str, SquareClassification] = {}
    for rank in range(1, 9):
        for file_char in "abcdefgh":
            name = f"{file_char}{rank}"
            classifications[name] = SquareClassification(
                square_name=name,
                piece=PieceType.EMPTY,
                confidence=0.95,
                low_confidence=False,
            )
    return classifications


class StubDetector:
    def __init__(self, result: DetectionResult) -> None:
        self._result = result

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return self._result


class StubSplitter:
    def __init__(self, result: SplitResult) -> None:
        self._result = result

    def split(
        self,
        frame: np.ndarray,
        board: BoardRegion,
        *,
        white_perspective: bool = True,
    ) -> SplitResult:
        return self._result


class StubOrchestrator:
    def __init__(self, result: RecognitionFlowResult) -> None:
        self._result = result
        self.manual_theme_calls: list[str | None] = []

    def recognize_position(
        self,
        squares: dict[str, SquareImage],
        *,
        manual_theme: str | None = None,
    ) -> RecognitionFlowResult:
        self.manual_theme_calls.append(manual_theme)
        return self._result


class StubStateInferrer:
    def __init__(self, fen_config: FenConfig | None = None) -> None:
        self._fen_config = fen_config or FenConfig(active_color="w", castling="-")

    def infer(
        self,
        classifications: dict[str, SquareClassification],
    ) -> SimpleNamespace:
        return SimpleNamespace(fen_config=self._fen_config, elapsed_ms=1.5)


class StubFenGenerator:
    def __init__(self, result: FenResult | None = None, *, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc

    def generate(
        self,
        classifications: dict[str, SquareClassification],
        config: FenConfig | None = None,
    ) -> FenResult:
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        return self._result


def test_process_frame_reports_missing_board() -> None:
    processor = LivePositionProcessor(
        detector=StubDetector(
            DetectionResult(
                board=None,
                detected=False,
                elapsed_ms=12.0,
                warning="No board here",
            )
        )
    )

    result = processor.process_frame(_make_frame())

    assert result.status is LiveFrameStatus.BOARD_NOT_FOUND
    assert result.message == "No board here"
    assert result.fen is None


def test_process_frame_blocks_uncertain_positions_by_default() -> None:
    board = BoardRegion(x=10, y=10, width=80, height=80, confidence=0.9)
    recognition = RecognitionResult(
        classifications=_make_classifications(),
        elapsed_ms=4.0,
        uncertain=True,
        uncertain_squares=("e4", "e5"),
    )
    orchestrator = StubOrchestrator(
        RecognitionFlowResult(
            status=RecognitionFlowStatus.POSITION_UNCERTAIN,
            suspended=True,
            message="Recognition confidence is too low.",
            selected_theme="chess.com-green",
            available_themes=(),
            theme_detection=None,
            recognition=recognition,
        )
    )
    processor = LivePositionProcessor(
        detector=StubDetector(
            DetectionResult(board=board, detected=True, elapsed_ms=5.0)
        ),
        splitter=StubSplitter(SplitResult(squares=_make_squares(), elapsed_ms=2.0)),
        orchestrator=orchestrator,
    )

    result = processor.process_frame(_make_frame())

    assert result.status is LiveFrameStatus.POSITION_UNCERTAIN
    assert result.uncertain_squares == ("e4", "e5")
    assert orchestrator.manual_theme_calls == ["chess.com-green"]


def test_process_frame_marks_fen_changes_only_once() -> None:
    board = BoardRegion(x=10, y=10, width=80, height=80, confidence=0.9)
    recognition = RecognitionResult(
        classifications=_make_classifications(),
        elapsed_ms=4.0,
        uncertain=False,
        uncertain_squares=(),
    )
    orchestrator = StubOrchestrator(
        RecognitionFlowResult(
            status=RecognitionFlowStatus.READY,
            suspended=False,
            message="ok",
            selected_theme="chess.com-green",
            available_themes=(),
            theme_detection=None,
            recognition=recognition,
        )
    )
    processor = LivePositionProcessor(
        detector=StubDetector(
            DetectionResult(board=board, detected=True, elapsed_ms=5.0)
        ),
        splitter=StubSplitter(SplitResult(squares=_make_squares(), elapsed_ms=2.0)),
        orchestrator=orchestrator,
        state_inferrer=StubStateInferrer(),
        fen_generator=StubFenGenerator(
            FenResult(
                fen="8/8/8/8/8/8/8/8 w - - 0 1",
                piece_placement="8/8/8/8/8/8/8/8",
                elapsed_ms=1.0,
            )
        ),
    )

    first = processor.process_frame(_make_frame())
    second = processor.process_frame(_make_frame())

    assert first.status is LiveFrameStatus.READY
    assert first.changed is True
    assert second.status is LiveFrameStatus.READY
    assert second.changed is False


def test_process_frame_maps_invalid_position_error() -> None:
    board = BoardRegion(x=10, y=10, width=80, height=80, confidence=0.9)
    recognition = RecognitionResult(
        classifications=_make_classifications(),
        elapsed_ms=4.0,
        uncertain=False,
        uncertain_squares=(),
    )
    processor = LivePositionProcessor(
        detector=StubDetector(
            DetectionResult(board=board, detected=True, elapsed_ms=5.0)
        ),
        splitter=StubSplitter(SplitResult(squares=_make_squares(), elapsed_ms=2.0)),
        orchestrator=StubOrchestrator(
            RecognitionFlowResult(
                status=RecognitionFlowStatus.READY,
                suspended=False,
                message="ok",
                selected_theme="chess.com-green",
                available_themes=(),
                theme_detection=None,
                recognition=recognition,
            )
        ),
        state_inferrer=StubStateInferrer(),
        fen_generator=StubFenGenerator(
            exc=InvalidPositionError("Position must have exactly 1 white king.")
        ),
    )

    result = processor.process_frame(_make_frame())

    assert result.status is LiveFrameStatus.INVALID_POSITION
    assert "white king" in result.message
