"""Tests for orchestrated theme + piece recognition flow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chesslab.recognition.recognition_orchestrator import (
    RecognitionFlowStatus,
    RecognitionOrchestrator,
    RecognitionOrchestratorConfig,
)
from chesslab.recognition.theme_detector import (
    ThemeDetectionConfig,
    ThemeDetector,
)
from chesslab.splitting.board_splitter import SquareImage
from tests.recognition.test_piece_recognizer import (
    SIZE,
    _STARTING_POSITION,
    _make_full_board,
    _make_synthetic_theme,
)


@pytest.fixture
def templates_dir(tmp_path: Path) -> Path:
    return _make_synthetic_theme(tmp_path)


def _make_unmatched_detector(tmp_path: Path) -> ThemeDetector:
    themes_file = tmp_path / "themes_unmatched.json"
    themes_file.write_text(
        json.dumps(
            {
                "themes": [
                    {
                        "name": "dummy",
                        "source": "test",
                        "light_bgr": [0, 0, 0],
                        "dark_bgr": [0, 0, 0],
                    }
                ]
            }
        )
    )
    return ThemeDetector(
        ThemeDetectionConfig(
            themes_file=themes_file,
            max_distance=5.0,
        )
    )


def test_unknown_theme_requires_manual_selection(
    templates_dir: Path,
    tmp_path: Path,
) -> None:
    detector = _make_unmatched_detector(tmp_path)
    orchestrator = RecognitionOrchestrator(
        RecognitionOrchestratorConfig(
            templates_dir=templates_dir,
            use_saved_theme_preference=False,
        ),
        theme_detector=detector,
    )
    squares = _make_full_board(_STARTING_POSITION)

    result = orchestrator.recognize_position(squares)

    assert result.status == RecognitionFlowStatus.THEME_UNRECOGNIZED
    assert result.suspended is True
    assert result.recognition is None
    assert len(result.available_themes) > 0


def test_manual_theme_allows_recognition(
    templates_dir: Path,
    tmp_path: Path,
) -> None:
    detector = _make_unmatched_detector(tmp_path)
    orchestrator = RecognitionOrchestrator(
        RecognitionOrchestratorConfig(templates_dir=templates_dir),
        theme_detector=detector,
    )
    squares = _make_full_board(_STARTING_POSITION)

    result = orchestrator.recognize_position(squares, manual_theme="test_theme")

    assert result.status == RecognitionFlowStatus.READY
    assert result.suspended is False
    assert result.recognition is not None
    assert result.selected_theme == "test_theme"


def test_saved_theme_used_when_detection_unknown(
    templates_dir: Path,
    tmp_path: Path,
) -> None:
    ThemeDetector.save_preference("test_theme", config_dir=tmp_path)
    detector = _make_unmatched_detector(tmp_path)
    orchestrator = RecognitionOrchestrator(
        RecognitionOrchestratorConfig(
            templates_dir=templates_dir,
            config_dir=tmp_path,
            use_saved_theme_preference=True,
        ),
        theme_detector=detector,
    )
    squares = _make_full_board(_STARTING_POSITION)

    result = orchestrator.recognize_position(squares)

    assert result.status == RecognitionFlowStatus.READY
    assert result.selected_theme == "test_theme"
    assert result.recognition is not None


def test_uncertain_position_suspends_analysis(templates_dir: Path) -> None:
    orchestrator = RecognitionOrchestrator(
        RecognitionOrchestratorConfig(templates_dir=templates_dir),
    )
    squares = _make_full_board({})
    rng = np.random.RandomState(42)
    squares["e4"] = SquareImage(
        name="e4",
        image=rng.randint(0, 256, (SIZE, SIZE, 3), dtype=np.uint8),
        file=4,
        rank=3,
    )

    result = orchestrator.recognize_position(squares, manual_theme="test_theme")

    assert result.status == RecognitionFlowStatus.POSITION_UNCERTAIN
    assert result.suspended is True
    assert result.recognition is not None
    assert result.recognition.uncertain is True


def test_manual_theme_is_persisted(templates_dir: Path, tmp_path: Path) -> None:
    orchestrator = RecognitionOrchestrator(
        RecognitionOrchestratorConfig(
            templates_dir=templates_dir,
            config_dir=tmp_path,
            persist_manual_theme=True,
        ),
    )
    squares = _make_full_board(_STARTING_POSITION)

    result = orchestrator.recognize_position(squares, manual_theme="test_theme")

    assert result.status == RecognitionFlowStatus.READY
    assert ThemeDetector.load_preference(config_dir=tmp_path) == "test_theme"


def test_unsupported_theme_reports_supported_list(templates_dir: Path) -> None:
    orchestrator = RecognitionOrchestrator(
        RecognitionOrchestratorConfig(templates_dir=templates_dir),
    )
    squares = _make_full_board(_STARTING_POSITION)

    result = orchestrator.recognize_position(squares, manual_theme="ocean")

    assert result.status == RecognitionFlowStatus.THEME_UNSUPPORTED
    assert result.suspended is True
    assert "test_theme" in result.available_themes
