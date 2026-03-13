"""Reusable live-frame pipeline for remote or local screen capture."""

from __future__ import annotations

import enum
import time
from dataclasses import asdict, dataclass

from numpy.typing import NDArray

from chesslab.detection.board_detector import BoardDetector, BoardRegion
from chesslab.fen.fen_generator import FenGenerator, InvalidPositionError
from chesslab.recognition import (
    RecognitionFlowStatus,
    RecognitionOrchestrator,
    RecognitionOrchestratorConfig,
)
from chesslab.splitting.board_splitter import BoardSplitter
from chesslab.state.state_inferrer import StateInferenceConfig, StateInferrer


class LiveFrameStatus(enum.Enum):
    """High-level outcome of processing a live frame."""

    READY = "ready"
    BOARD_NOT_FOUND = "board_not_found"
    THEME_UNRECOGNIZED = "theme_unrecognized"
    THEME_UNSUPPORTED = "theme_unsupported"
    POSITION_UNCERTAIN = "position_uncertain"
    INVALID_POSITION = "invalid_position"


@dataclass
class LiveProcessorConfig:
    """Configuration for the live frame processor."""

    theme: str = "green"
    confidence_threshold: float = 0.60
    white_perspective: bool = True
    active_color: str = "w"
    allow_uncertain: bool = False
    en_passant: str = "-"
    halfmove_clock: int = 0
    fullmove_number: int = 1

    def __post_init__(self) -> None:
        if not self.theme:
            raise ValueError("theme must not be empty.")
        if self.active_color not in {"w", "b"}:
            raise ValueError(
                f"active_color must be 'w' or 'b', got '{self.active_color}'."
            )
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be in (0, 1], "
                f"got {self.confidence_threshold}."
            )
        if self.halfmove_clock < 0:
            raise ValueError(
                f"halfmove_clock must be >= 0, got {self.halfmove_clock}."
            )
        if self.fullmove_number < 1:
            raise ValueError(
                f"fullmove_number must be >= 1, got {self.fullmove_number}."
            )


@dataclass(frozen=True)
class LiveFrameResult:
    """Structured result returned for each processed frame."""

    status: LiveFrameStatus
    message: str
    changed: bool
    total_elapsed_ms: float
    detection_elapsed_ms: float
    split_elapsed_ms: float | None
    recognition_elapsed_ms: float | None
    state_elapsed_ms: float | None
    fen_elapsed_ms: float | None
    board: BoardRegion | None
    selected_theme: str | None
    recognition_status: str | None
    uncertain_squares: tuple[str, ...]
    fen: str | None
    piece_placement: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""
        board = None if self.board is None else asdict(self.board)
        return {
            "status": self.status.value,
            "message": self.message,
            "changed": self.changed,
            "total_elapsed_ms": round(self.total_elapsed_ms, 3),
            "detection_elapsed_ms": round(self.detection_elapsed_ms, 3),
            "split_elapsed_ms": (
                None if self.split_elapsed_ms is None else round(self.split_elapsed_ms, 3)
            ),
            "recognition_elapsed_ms": (
                None
                if self.recognition_elapsed_ms is None
                else round(self.recognition_elapsed_ms, 3)
            ),
            "state_elapsed_ms": (
                None if self.state_elapsed_ms is None else round(self.state_elapsed_ms, 3)
            ),
            "fen_elapsed_ms": None if self.fen_elapsed_ms is None else round(self.fen_elapsed_ms, 3),
            "board": board,
            "selected_theme": self.selected_theme,
            "recognition_status": self.recognition_status,
            "uncertain_squares": list(self.uncertain_squares),
            "fen": self.fen,
            "piece_placement": self.piece_placement,
        }


class LivePositionProcessor:
    """Run the existing pipeline on in-memory frames for live testing."""

    def __init__(
        self,
        config: LiveProcessorConfig | None = None,
        *,
        detector: BoardDetector | None = None,
        splitter: BoardSplitter | None = None,
        orchestrator: RecognitionOrchestrator | None = None,
        state_inferrer: StateInferrer | None = None,
        fen_generator: FenGenerator | None = None,
    ) -> None:
        self._config = config or LiveProcessorConfig()
        self._detector = detector or BoardDetector()
        self._splitter = splitter or BoardSplitter()
        self._orchestrator = orchestrator or RecognitionOrchestrator(
            RecognitionOrchestratorConfig(
                confidence_threshold=self._config.confidence_threshold,
                persist_manual_theme=False,
                use_saved_theme_preference=False,
            )
        )
        self._state_inferrer = state_inferrer or StateInferrer(
            StateInferenceConfig(
                active_color_override=self._config.active_color,
                en_passant=self._config.en_passant,
                halfmove_clock=self._config.halfmove_clock,
                fullmove_number=self._config.fullmove_number,
            )
        )
        self._fen_generator = fen_generator or FenGenerator()
        self._last_fen: str | None = None

    @property
    def config(self) -> LiveProcessorConfig:
        return self._config

    def process_frame(self, frame: NDArray) -> LiveFrameResult:
        """Run detection, splitting, recognition and FEN generation."""
        started = time.perf_counter()

        detection = self._detector.detect(frame)
        if not detection.detected or detection.board is None:
            return self._result(
                status=LiveFrameStatus.BOARD_NOT_FOUND,
                message=detection.warning or "No chessboard detected in frame.",
                started=started,
                detection_elapsed_ms=detection.elapsed_ms,
            )

        split = self._splitter.split(
            frame,
            detection.board,
            white_perspective=self._config.white_perspective,
        )

        flow = self._orchestrator.recognize_position(
            split.squares,
            manual_theme=self._config.theme,
        )

        if flow.recognition is None:
            status = {
                RecognitionFlowStatus.THEME_UNRECOGNIZED: LiveFrameStatus.THEME_UNRECOGNIZED,
                RecognitionFlowStatus.THEME_UNSUPPORTED: LiveFrameStatus.THEME_UNSUPPORTED,
            }.get(flow.status, LiveFrameStatus.THEME_UNSUPPORTED)
            return self._result(
                status=status,
                message=flow.message,
                started=started,
                detection_elapsed_ms=detection.elapsed_ms,
                split_elapsed_ms=split.elapsed_ms,
                board=detection.board,
                selected_theme=flow.selected_theme,
                recognition_status=flow.status.value,
            )

        recognition = flow.recognition
        if flow.status is RecognitionFlowStatus.POSITION_UNCERTAIN and not self._config.allow_uncertain:
            return self._result(
                status=LiveFrameStatus.POSITION_UNCERTAIN,
                message=flow.message,
                started=started,
                detection_elapsed_ms=detection.elapsed_ms,
                split_elapsed_ms=split.elapsed_ms,
                recognition_elapsed_ms=recognition.elapsed_ms,
                board=detection.board,
                selected_theme=flow.selected_theme,
                recognition_status=flow.status.value,
                uncertain_squares=recognition.uncertain_squares,
            )

        state = self._state_inferrer.infer(recognition.classifications)

        try:
            fen_result = self._fen_generator.generate(
                recognition.classifications,
                config=state.fen_config,
            )
        except InvalidPositionError as exc:
            return self._result(
                status=LiveFrameStatus.INVALID_POSITION,
                message=str(exc),
                started=started,
                detection_elapsed_ms=detection.elapsed_ms,
                split_elapsed_ms=split.elapsed_ms,
                recognition_elapsed_ms=recognition.elapsed_ms,
                state_elapsed_ms=state.elapsed_ms,
                board=detection.board,
                selected_theme=flow.selected_theme,
                recognition_status=flow.status.value,
                uncertain_squares=recognition.uncertain_squares,
            )

        changed = fen_result.fen != self._last_fen
        self._last_fen = fen_result.fen

        return self._result(
            status=LiveFrameStatus.READY,
            message="Position recognized successfully.",
            started=started,
            changed=changed,
            detection_elapsed_ms=detection.elapsed_ms,
            split_elapsed_ms=split.elapsed_ms,
            recognition_elapsed_ms=recognition.elapsed_ms,
            state_elapsed_ms=state.elapsed_ms,
            fen_elapsed_ms=fen_result.elapsed_ms,
            board=detection.board,
            selected_theme=flow.selected_theme,
            recognition_status=flow.status.value,
            uncertain_squares=recognition.uncertain_squares,
            fen=fen_result.fen,
            piece_placement=fen_result.piece_placement,
        )

    def _result(
        self,
        *,
        status: LiveFrameStatus,
        message: str,
        started: float,
        changed: bool = False,
        detection_elapsed_ms: float = 0.0,
        split_elapsed_ms: float | None = None,
        recognition_elapsed_ms: float | None = None,
        state_elapsed_ms: float | None = None,
        fen_elapsed_ms: float | None = None,
        board: BoardRegion | None = None,
        selected_theme: str | None = None,
        recognition_status: str | None = None,
        uncertain_squares: tuple[str, ...] = (),
        fen: str | None = None,
        piece_placement: str | None = None,
    ) -> LiveFrameResult:
        total_elapsed_ms = (time.perf_counter() - started) * 1000.0
        return LiveFrameResult(
            status=status,
            message=message,
            changed=changed,
            total_elapsed_ms=total_elapsed_ms,
            detection_elapsed_ms=detection_elapsed_ms,
            split_elapsed_ms=split_elapsed_ms,
            recognition_elapsed_ms=recognition_elapsed_ms,
            state_elapsed_ms=state_elapsed_ms,
            fen_elapsed_ms=fen_elapsed_ms,
            board=board,
            selected_theme=selected_theme,
            recognition_status=recognition_status,
            uncertain_squares=uncertain_squares,
            fen=fen,
            piece_placement=piece_placement,
        )
