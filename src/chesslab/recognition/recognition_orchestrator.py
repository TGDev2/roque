"""Recognition orchestration with theme fallback and uncertainty gating."""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from pathlib import Path

from chesslab.recognition.piece_recognizer import (
    PieceRecognizer,
    RecognitionConfig,
    RecognitionResult,
    ThemeNotFoundError,
)
from chesslab.recognition.theme_detector import ThemeDetectionResult, ThemeDetector
from chesslab.splitting.board_splitter import SquareImage

_TEMPLATES_ROOT = Path(__file__).parent / "templates"
logger = logging.getLogger(__name__)


class RecognitionFlowStatus(enum.Enum):
    """High-level outcome of the recognition flow."""

    READY = "ready"
    THEME_UNRECOGNIZED = "theme_unrecognized"
    THEME_UNSUPPORTED = "theme_unsupported"
    POSITION_UNCERTAIN = "position_uncertain"


@dataclass
class RecognitionOrchestratorConfig:
    """Config for the recognition flow orchestration."""

    confidence_threshold: float = 0.60
    templates_dir: Path | None = None
    persist_manual_theme: bool = True
    use_saved_theme_preference: bool = True
    config_dir: Path | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0, 1], got {self.confidence_threshold}."
            )


@dataclass(frozen=True)
class RecognitionFlowResult:
    """Result of the orchestrated theme+piece recognition flow."""

    status: RecognitionFlowStatus
    suspended: bool
    message: str
    selected_theme: str | None
    available_themes: tuple[str, ...]
    theme_detection: ThemeDetectionResult | None
    recognition: RecognitionResult | None


class RecognitionOrchestrator:
    """Coordinates theme detection, piece recognition, and flow control."""

    def __init__(
        self,
        config: RecognitionOrchestratorConfig | None = None,
        theme_detector: ThemeDetector | None = None,
    ) -> None:
        self._config = config or RecognitionOrchestratorConfig()
        self._theme_detector = theme_detector or ThemeDetector()
        self._recognizers: dict[str, PieceRecognizer] = {}

    def recognize_position(
        self,
        squares: dict[str, SquareImage],
        *,
        manual_theme: str | None = None,
    ) -> RecognitionFlowResult:
        """Recognize pieces and apply fallback/uncertainty flow rules."""
        if len(squares) != 64:
            raise ValueError(f"Expected 64 squares, got {len(squares)}.")

        theme_result = self._theme_detector.detect(squares)
        selected_theme = self._select_theme(theme_result, manual_theme=manual_theme)

        if selected_theme is None:
            return RecognitionFlowResult(
                status=RecognitionFlowStatus.THEME_UNRECOGNIZED,
                suspended=True,
                message=(
                    "Theme not recognized. Select a theme manually from "
                    f"{', '.join(theme_result.available_themes)}."
                ),
                selected_theme=None,
                available_themes=theme_result.available_themes,
                theme_detection=theme_result,
                recognition=None,
            )

        recognizer = self._get_recognizer(selected_theme)
        if recognizer is None:
            available = tuple(self.list_supported_templates())
            return RecognitionFlowResult(
                status=RecognitionFlowStatus.THEME_UNSUPPORTED,
                suspended=True,
                message=(
                    f"Theme '{selected_theme}' is not available in template library. "
                    "Select one of: "
                    f"{', '.join(available) if available else '(none)'}."
                ),
                selected_theme=selected_theme,
                available_themes=available,
                theme_detection=theme_result,
                recognition=None,
            )

        recognition = recognizer.recognize(squares)

        if manual_theme and self._config.persist_manual_theme:
            try:
                ThemeDetector.save_preference(
                    selected_theme,
                    config_dir=self._config.config_dir,
                )
            except OSError as exc:
                logger.warning(
                    "Failed to persist manual theme preference '%s': %s",
                    selected_theme,
                    exc,
                )

        if recognition.uncertain:
            return RecognitionFlowResult(
                status=RecognitionFlowStatus.POSITION_UNCERTAIN,
                suspended=True,
                message=(
                    "Recognition confidence is too low. "
                    "Analysis is suspended until a better image is available."
                ),
                selected_theme=selected_theme,
                available_themes=(),
                theme_detection=theme_result,
                recognition=recognition,
            )

        return RecognitionFlowResult(
            status=RecognitionFlowStatus.READY,
            suspended=False,
            message="Position recognized successfully.",
            selected_theme=selected_theme,
            available_themes=(),
            theme_detection=theme_result,
            recognition=recognition,
        )

    def list_supported_templates(self) -> list[str]:
        """List available local template theme directories."""
        templates_dir = self._config.templates_dir or _TEMPLATES_ROOT
        if not templates_dir.is_dir():
            return []
        return sorted(
            d.name
            for d in templates_dir.iterdir()
            if d.is_dir() and (d / "theme.json").is_file()
        )

    def _get_recognizer(self, theme: str) -> PieceRecognizer | None:
        if theme in self._recognizers:
            return self._recognizers[theme]

        try:
            recognizer = PieceRecognizer(
                RecognitionConfig(
                    theme=theme,
                    confidence_threshold=self._config.confidence_threshold,
                    templates_dir=self._config.templates_dir,
                )
            )
        except ThemeNotFoundError:
            return None

        self._recognizers[theme] = recognizer
        return recognizer

    def _select_theme(
        self,
        theme_result: ThemeDetectionResult,
        *,
        manual_theme: str | None,
    ) -> str | None:
        if manual_theme:
            return self._normalize_theme_id(manual_theme)

        if theme_result.matched and theme_result.theme is not None:
            return self._normalize_theme_id(theme_result.theme.qualified_name)

        if self._config.use_saved_theme_preference:
            saved = ThemeDetector.load_preference(config_dir=self._config.config_dir)
            if saved:
                return self._normalize_theme_id(saved)

        return None

    @staticmethod
    def _normalize_theme_id(theme_id: str) -> str:
        """Normalize user/theme-detector IDs to a template directory name."""
        normalized = theme_id.replace("/", "-").strip()
        return normalized
