"""Integration tests for multi-theme recognition.

Verifies:
- Theme detection accuracy >= 90% across multiple themes with noise
- Piece recognition per theme at multiple resolutions
- Cross-theme detection does not confuse themes
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from chesslab.recognition._generate_placeholders import generate_theme, generate_template
from chesslab.recognition.piece_recognizer import (
    PieceRecognizer,
    PieceType,
    RecognitionConfig,
)
from chesslab.recognition.theme_detector import (
    ThemeDetectionConfig,
    ThemeDetector,
)
from chesslab.splitting.board_splitter import SquareImage
from chesslab.splitting.square_color import square_background

_FILES = "abcdefgh"

# Theme BGR values from themes.json.
THEMES_UNDER_TEST = [
    ("chess.com-green", (208, 236, 235), (82, 149, 115)),
    ("chess.com-brown", (181, 217, 240), (99, 136, 181)),
    ("lichess-brown", (178, 217, 240), (107, 153, 181)),
]

RESOLUTIONS = [80, 40]

_KEY_TO_PIECETYPE: dict[str, PieceType] = {
    "wK": PieceType.WHITE_KING,
    "wQ": PieceType.WHITE_QUEEN,
    "wR": PieceType.WHITE_ROOK,
    "wB": PieceType.WHITE_BISHOP,
    "wN": PieceType.WHITE_KNIGHT,
    "wP": PieceType.WHITE_PAWN,
    "bK": PieceType.BLACK_KING,
    "bQ": PieceType.BLACK_QUEEN,
    "bR": PieceType.BLACK_ROOK,
    "bB": PieceType.BLACK_BISHOP,
    "bN": PieceType.BLACK_KNIGHT,
    "bP": PieceType.BLACK_PAWN,
}

_STARTING_POSITION: dict[str, str] = {
    "a1": "wR", "b1": "wN", "c1": "wB", "d1": "wQ",
    "e1": "wK", "f1": "wB", "g1": "wN", "h1": "wR",
    "a2": "wP", "b2": "wP", "c2": "wP", "d2": "wP",
    "e2": "wP", "f2": "wP", "g2": "wP", "h2": "wP",
    "a7": "bP", "b7": "bP", "c7": "bP", "d7": "bP",
    "e7": "bP", "f7": "bP", "g7": "bP", "h7": "bP",
    "a8": "bR", "b8": "bN", "c8": "bB", "d8": "bQ",
    "e8": "bK", "f8": "bB", "g8": "bN", "h8": "bR",
}


def _make_full_board(
    piece_map: dict[str, str],
    *,
    light_bgr: tuple[int, int, int],
    dark_bgr: tuple[int, int, int],
    size: int = 80,
) -> dict[str, SquareImage]:
    """Build 64 SquareImage entries with specified theme colours."""
    squares: dict[str, SquareImage] = {}
    for f_idx in range(8):
        for r_idx in range(8):
            name = f"{_FILES[f_idx]}{r_idx + 1}"
            bg = square_background(f_idx, r_idx)
            bg_colour = light_bgr if bg == "light" else dark_bgr
            piece_key = piece_map.get(name)
            if piece_key:
                img = generate_template(
                    piece_key, bg,
                    light_bgr=light_bgr, dark_bgr=dark_bgr, size=size,
                )
            else:
                img = np.full((size, size, 3), bg_colour, dtype=np.uint8)
            squares[name] = SquareImage(
                name=name, image=img, file=f_idx, rank=r_idx,
            )
    return squares


def _add_noise(
    squares: dict[str, SquareImage],
    sigma: float,
    rng: np.random.RandomState,
) -> dict[str, SquareImage]:
    """Add Gaussian noise to all square images."""
    noisy: dict[str, SquareImage] = {}
    for name, sq in squares.items():
        noise = rng.normal(0, sigma, sq.image.shape)
        img = np.clip(sq.image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        noisy[name] = SquareImage(name=name, image=img, file=sq.file, rank=sq.rank)
    return noisy


def _make_themes_json(
    tmp_path: Path,
    themes: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]],
) -> Path:
    """Create a themes.json for the given theme tuples."""
    entries = []
    for name, light, dark in themes:
        source, tname = name.split("-", 1)
        entries.append({
            "name": tname,
            "source": source,
            "light_bgr": list(light),
            "dark_bgr": list(dark),
        })
    path = tmp_path / "themes.json"
    with path.open("w") as fh:
        json.dump({"themes": entries}, fh)
    return path


@pytest.fixture
def themes_json(tmp_path: Path) -> Path:
    return _make_themes_json(tmp_path, THEMES_UNDER_TEST)


@pytest.fixture
def templates_root(tmp_path: Path) -> Path:
    """Generate synthetic templates for all themes under test."""
    root = tmp_path / "templates"
    for name, light, dark in THEMES_UNDER_TEST:
        generate_theme(
            root / name,
            theme_name=name,
            light_bgr=light,
            dark_bgr=dark,
            size=80,
        )
    return root


# ---------------------------------------------------------------------------
# Theme detection accuracy
# ---------------------------------------------------------------------------


class TestThemeDetectionAccuracy:
    """AC: >= 90% detection accuracy with noisy boards (10 per theme)."""

    @pytest.mark.parametrize(
        "theme_name, light_bgr, dark_bgr",
        THEMES_UNDER_TEST,
        ids=[t[0] for t in THEMES_UNDER_TEST],
    )
    def test_detection_accuracy_with_noise(
        self,
        themes_json: Path,
        theme_name: str,
        light_bgr: tuple[int, int, int],
        dark_bgr: tuple[int, int, int],
    ) -> None:
        """10 noisy boards, >= 9 correctly detected."""
        config = ThemeDetectionConfig(themes_file=themes_json)
        detector = ThemeDetector(config)

        correct = 0
        for seed in range(10):
            rng = np.random.RandomState(seed)
            sigma = rng.uniform(2.0, 5.0)
            squares = _make_full_board(
                {}, light_bgr=light_bgr, dark_bgr=dark_bgr,
            )
            noisy_squares = _add_noise(squares, sigma, rng)
            result = detector.detect(noisy_squares)
            if (
                result.matched
                and result.theme is not None
                and f"{result.theme.source}-{result.theme.name}" == theme_name
            ):
                correct += 1

        assert correct >= 9, (
            f"Theme {theme_name}: {correct}/10 correct (need >= 9)"
        )


# ---------------------------------------------------------------------------
# Piece recognition per theme
# ---------------------------------------------------------------------------


class TestPieceRecognitionPerTheme:
    """AC: starting position recognized correctly per theme and resolution."""

    @pytest.mark.parametrize(
        "theme_name, light_bgr, dark_bgr",
        THEMES_UNDER_TEST,
        ids=[t[0] for t in THEMES_UNDER_TEST],
    )
    @pytest.mark.parametrize("size", RESOLUTIONS, ids=[f"{s}px" for s in RESOLUTIONS])
    def test_starting_position(
        self,
        tmp_path: Path,
        theme_name: str,
        light_bgr: tuple[int, int, int],
        dark_bgr: tuple[int, int, int],
        size: int,
    ) -> None:
        """Synthetic templates at *size* recognise the starting position."""
        root = tmp_path / "templates"
        generate_theme(
            root / theme_name,
            theme_name=theme_name,
            light_bgr=light_bgr,
            dark_bgr=dark_bgr,
            size=size,
        )
        config = RecognitionConfig(
            theme=theme_name,
            templates_dir=root,
        )
        recognizer = PieceRecognizer(config)
        squares = _make_full_board(
            _STARTING_POSITION,
            light_bgr=light_bgr,
            dark_bgr=dark_bgr,
            size=size,
        )
        result = recognizer.recognize(squares)

        correct = 0
        for name, cls in result.classifications.items():
            expected_key = _STARTING_POSITION.get(name)
            if expected_key is None:
                if cls.piece == PieceType.EMPTY:
                    correct += 1
            elif cls.piece == _KEY_TO_PIECETYPE[expected_key]:
                correct += 1

        accuracy = correct / 64
        assert accuracy >= 0.95, (
            f"Theme {theme_name} @ {size}px: accuracy {accuracy:.1%} < 95%"
        )


# ---------------------------------------------------------------------------
# Cross-theme detection
# ---------------------------------------------------------------------------


class TestCrossThemeDetection:
    """AC: board of theme A is not misidentified as theme B."""

    def test_no_cross_confusion(self, themes_json: Path) -> None:
        """Each theme detects as itself, not as another."""
        config = ThemeDetectionConfig(themes_file=themes_json)
        detector = ThemeDetector(config)

        for theme_name, light_bgr, dark_bgr in THEMES_UNDER_TEST:
            squares = _make_full_board(
                {}, light_bgr=light_bgr, dark_bgr=dark_bgr,
            )
            result = detector.detect(squares)
            assert result.matched, f"Theme {theme_name} not matched"
            assert result.theme is not None
            detected = f"{result.theme.source}-{result.theme.name}"
            assert detected == theme_name, (
                f"Expected {theme_name}, detected {detected}"
            )
