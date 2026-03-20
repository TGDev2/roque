"""Tests for the theme detection module.

Covers acceptance criteria:
- Automatic detection: correct theme identified from board colours, < 2 seconds
- Unknown theme fallback: available themes listed for manual selection
- Persistence: preference saved and loaded across sessions
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from chesslab.recognition.theme_detector import (
    ThemeDetectionConfig,
    ThemeConfigError,
    ThemeDetectionResult,
    ThemeDetector,
    ThemeSignature,
)
from chesslab.splitting.board_splitter import SquareImage
from chesslab.splitting.square_color import square_background

# ---------------------------------------------------------------------------
# Constants — chess.com Green (BGR)
# ---------------------------------------------------------------------------

_FILES = "abcdefgh"
SIZE = 80

LIGHT_BG = (208, 236, 235)
DARK_BG = (82, 149, 115)

WHITE_PIECE = (255, 255, 255)
BLACK_PIECE = (30, 30, 30)


# ---------------------------------------------------------------------------
# Drawing helpers — mirror _generate_placeholders patterns
# ---------------------------------------------------------------------------


def _bg_colour(bg: str) -> tuple[int, int, int]:
    return LIGHT_BG if bg == "light" else DARK_BG


def _piece_colour(key: str) -> tuple[int, int, int]:
    return WHITE_PIECE if key[0] == "w" else BLACK_PIECE


def _draw_king(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    c = SIZE // 2
    t = max(SIZE // 10, 2)
    cv2.line(img, (c, SIZE // 5), (c, 4 * SIZE // 5), colour, t)
    cv2.line(img, (SIZE // 5, c), (4 * SIZE // 5, c), colour, t)


def _draw_pawn(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    c = SIZE // 2
    r = SIZE // 4
    cv2.circle(img, (c, c), r, colour, -1)


def _bg_for_square(file_idx: int, rank_idx: int) -> str:
    return square_background(file_idx, rank_idx)


def _make_piece_square(
    piece_key: str, bg: str, size: int = SIZE,
) -> NDArray[np.uint8]:
    img = np.full((size, size, 3), _bg_colour(bg), dtype=np.uint8)
    if piece_key[1] == "K":
        _draw_king(img, _piece_colour(piece_key))
    else:
        _draw_pawn(img, _piece_colour(piece_key))
    return img


def _make_empty_square(bg: str, size: int = SIZE) -> NDArray[np.uint8]:
    return np.full((size, size, 3), _bg_colour(bg), dtype=np.uint8)


def _make_full_board(
    piece_map: dict[str, str] | None = None,
    size: int = SIZE,
    light_bgr: tuple[int, int, int] = LIGHT_BG,
    dark_bgr: tuple[int, int, int] = DARK_BG,
) -> dict[str, SquareImage]:
    """Build 64 SquareImage entries with configurable background colours."""
    piece_map = piece_map or {}
    squares: dict[str, SquareImage] = {}
    for f_idx in range(8):
        for r_idx in range(8):
            name = f"{_FILES[f_idx]}{r_idx + 1}"
            bg = _bg_for_square(f_idx, r_idx)
            bg_colour = light_bgr if bg == "light" else dark_bgr
            piece_key = piece_map.get(name)
            if piece_key:
                img = np.full((size, size, 3), bg_colour, dtype=np.uint8)
                if piece_key[1] == "K":
                    _draw_king(img, _piece_colour(piece_key))
                else:
                    _draw_pawn(img, _piece_colour(piece_key))
            else:
                img = np.full((size, size, 3), bg_colour, dtype=np.uint8)
            squares[name] = SquareImage(
                name=name, image=img, file=f_idx, rank=r_idx,
            )
    return squares


def _make_themes_file(tmp_path: Path, themes: list[dict[str, object]]) -> Path:
    """Create a custom themes.json in *tmp_path*."""
    themes_file = tmp_path / "themes.json"
    with themes_file.open("w") as fh:
        json.dump({"themes": themes}, fh)
    return themes_file


# ---------------------------------------------------------------------------
# Scenario: Colour extraction
# ---------------------------------------------------------------------------


class TestColorExtraction:
    """Verify background colour extraction from board squares."""

    def test_solid_squares_yield_exact_color(self) -> None:
        """Empty board returns exact light/dark BGR values."""
        squares = _make_full_board()
        detector = ThemeDetector()
        light, dark = detector._extract_colours(squares)
        np.testing.assert_array_almost_equal(light, LIGHT_BG, decimal=0)
        np.testing.assert_array_almost_equal(dark, DARK_BG, decimal=0)

    def test_squares_with_pieces_yield_background_color(self) -> None:
        """Corners of piece squares still reveal the background colour."""
        piece_map = {
            "e1": "wK", "d1": "wP", "e8": "bK", "d8": "bP",
        }
        squares = _make_full_board(piece_map)
        detector = ThemeDetector()
        light, dark = detector._extract_colours(squares)
        # Median across 32 squares should be very close to the background
        np.testing.assert_array_almost_equal(light, LIGHT_BG, decimal=0)
        np.testing.assert_array_almost_equal(dark, DARK_BG, decimal=0)

    def test_corner_sampling_ignores_center(self) -> None:
        """A centre blob does not affect corner sampling."""
        img = np.full((80, 80, 3), (100, 150, 200), dtype=np.uint8)
        # Paint the centre a different colour
        img[20:60, 20:60] = (0, 0, 0)

        colour = ThemeDetector._sample_corner_colour(img, 4)
        np.testing.assert_array_almost_equal(colour, [100, 150, 200], decimal=0)

    def test_bgra_images_handled(self) -> None:
        """4-channel images are supported in corner sampling."""
        bgr = np.full((80, 80, 3), (100, 150, 200), dtype=np.uint8)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        colour = ThemeDetector._sample_corner_colour(bgra, 4)
        np.testing.assert_array_almost_equal(colour, [100, 150, 200], decimal=0)


# ---------------------------------------------------------------------------
# Scenario: Theme matching
# ---------------------------------------------------------------------------


class TestThemeMatching:
    """Verify theme signature matching logic."""

    @pytest.fixture
    def signatures(self) -> list[ThemeSignature]:
        return [
            ThemeSignature("green", "chess.com", (208, 236, 235), (82, 149, 115)),
            ThemeSignature("brown", "chess.com", (181, 217, 240), (99, 136, 181)),
        ]

    def test_exact_match(self, signatures: list[ThemeSignature]) -> None:
        light = np.array([208, 236, 235], dtype=np.float64)
        dark = np.array([82, 149, 115], dtype=np.float64)
        sig, dist = ThemeDetector._match_theme(light, dark, signatures)
        assert sig.qualified_name == "chess.com/green"
        assert dist < 1.0

    def test_close_match_within_threshold(
        self, signatures: list[ThemeSignature],
    ) -> None:
        light = np.array([218, 240, 240], dtype=np.float64)  # +10 noise
        dark = np.array([85, 155, 120], dtype=np.float64)
        sig, dist = ThemeDetector._match_theme(light, dark, signatures)
        assert sig.qualified_name == "chess.com/green"
        assert dist < 30.0

    def test_no_match_high_distance(
        self, signatures: list[ThemeSignature],
    ) -> None:
        light = np.array([0, 0, 255], dtype=np.float64)  # pure red
        dark = np.array([255, 0, 0], dtype=np.float64)  # pure blue
        sig, dist = ThemeDetector._match_theme(light, dark, signatures)
        assert dist > 30.0

    def test_distinguishes_similar_themes(
        self, signatures: list[ThemeSignature],
    ) -> None:
        # Feed exact brown values
        light = np.array([181, 217, 240], dtype=np.float64)
        dark = np.array([99, 136, 181], dtype=np.float64)
        sig, _ = ThemeDetector._match_theme(light, dark, signatures)
        assert sig.qualified_name == "chess.com/brown"


# ---------------------------------------------------------------------------
# Scenario: End-to-end detection
# ---------------------------------------------------------------------------


class TestDetectEndToEnd:
    """AC: automatic detection, < 2 seconds, fallback on unknown."""

    def test_detect_green_theme(self) -> None:
        """Board with chess.com Green colours is detected correctly."""
        squares = _make_full_board()
        detector = ThemeDetector()
        result = detector.detect(squares)
        assert result.matched is True
        assert result.theme is not None
        assert result.theme.qualified_name == "chess.com/green"

    def test_detect_performance_under_2_seconds(self) -> None:
        squares = _make_full_board()
        detector = ThemeDetector()
        start = time.perf_counter()
        detector.detect(squares)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 2000, f"Detection took {elapsed_ms:.1f} ms (limit: 2000)"

    def test_detect_no_user_action_required(self) -> None:
        """When theme is matched, the result is fully automatic."""
        squares = _make_full_board()
        detector = ThemeDetector()
        result = detector.detect(squares)
        assert result.matched is True
        assert result.theme is not None

    def test_detect_unknown_returns_fallback(self) -> None:
        """Unknown colours yield matched=False and available themes list."""
        squares = _make_full_board(
            light_bgr=(0, 0, 255),
            dark_bgr=(255, 0, 0),
        )
        detector = ThemeDetector()
        result = detector.detect(squares)
        assert result.matched is False
        assert result.theme is None
        assert len(result.available_themes) > 0

    def test_available_themes_always_populated(self) -> None:
        squares = _make_full_board()
        detector = ThemeDetector()
        result = detector.detect(squares)
        assert len(result.available_themes) == 8

    def test_observed_colours_populated(self) -> None:
        squares = _make_full_board()
        detector = ThemeDetector()
        result = detector.detect(squares)
        assert len(result.observed_light_bgr) == 3
        assert len(result.observed_dark_bgr) == 3

    def test_detect_with_custom_themes_file(self, tmp_path: Path) -> None:
        themes = [
            {"name": "custom", "source": "test", "light_bgr": [100, 100, 100], "dark_bgr": [50, 50, 50]},
        ]
        themes_file = _make_themes_file(tmp_path, themes)
        config = ThemeDetectionConfig(themes_file=themes_file)
        detector = ThemeDetector(config)

        squares = _make_full_board(light_bgr=(100, 100, 100), dark_bgr=(50, 50, 50))
        result = detector.detect(squares)
        assert result.matched is True
        assert result.theme is not None
        assert result.theme.qualified_name == "test/custom"

    def test_detect_walnut_theme(self, tmp_path: Path) -> None:
        """Scénario AC : thème Walnut détecté automatiquement."""
        walnut_light = (140, 184, 218)
        walnut_dark = (73, 116, 164)
        squares = _make_full_board(light_bgr=walnut_light, dark_bgr=walnut_dark)
        detector = ThemeDetector()
        result = detector.detect(squares)
        assert result.matched is True
        assert result.theme is not None
        assert result.theme.name == "walnut"
        assert result.theme.source == "chess.com"


# ---------------------------------------------------------------------------
# Scenario: Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    """AC: preference saved and loaded across sessions."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        ThemeDetector.save_preference("chess.com/green", config_dir=tmp_path)
        loaded = ThemeDetector.load_preference(config_dir=tmp_path)
        assert loaded == "chess.com/green"

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        loaded = ThemeDetector.load_preference(config_dir=tmp_path)
        assert loaded is None

    def test_load_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text("{invalid json")
        loaded = ThemeDetector.load_preference(config_dir=tmp_path)
        assert loaded is None

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        ThemeDetector.save_preference("lichess/blue", config_dir=nested)
        assert (nested / "config.json").is_file()
        loaded = ThemeDetector.load_preference(config_dir=nested)
        assert loaded == "lichess/blue"

    def test_save_preserves_other_keys(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text('{"other_setting": 42}')
        ThemeDetector.save_preference("chess.com/brown", config_dir=tmp_path)

        with config_file.open() as fh:
            data = json.load(fh)
        assert data["theme"] == "chess.com/brown"
        assert data["other_setting"] == 42

    def test_save_overwrites_previous_preference(self, tmp_path: Path) -> None:
        ThemeDetector.save_preference("chess.com/green", config_dir=tmp_path)
        ThemeDetector.save_preference("lichess/brown", config_dir=tmp_path)
        loaded = ThemeDetector.load_preference(config_dir=tmp_path)
        assert loaded == "lichess/brown"


# ---------------------------------------------------------------------------
# Data class checks
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_theme_signature_qualified_name(self) -> None:
        sig = ThemeSignature("green", "chess.com", (208, 236, 235), (82, 149, 115))
        assert sig.qualified_name == "chess.com/green"

    def test_theme_signature_frozen(self) -> None:
        sig = ThemeSignature("green", "chess.com", (208, 236, 235), (82, 149, 115))
        with pytest.raises(AttributeError):
            sig.name = "blue"  # type: ignore[misc]

    def test_detection_result_frozen(self) -> None:
        result = ThemeDetectionResult(
            theme=None,
            distance=0.0,
            matched=False,
            elapsed_ms=1.0,
            observed_light_bgr=(0, 0, 0),
            observed_dark_bgr=(0, 0, 0),
            available_themes=(),
        )
        with pytest.raises(AttributeError):
            result.matched = True  # type: ignore[misc]

    def test_config_defaults(self) -> None:
        config = ThemeDetectionConfig()
        assert config.max_distance == 30.0
        assert config.corner_patch_size == 4

    def test_config_invalid_max_distance(self) -> None:
        with pytest.raises(ValueError, match="max_distance"):
            ThemeDetectionConfig(max_distance=0)

    def test_config_invalid_corner_patch_size(self) -> None:
        with pytest.raises(ValueError, match="corner_patch_size"):
            ThemeDetectionConfig(corner_patch_size=0)


# ---------------------------------------------------------------------------
# List themes
# ---------------------------------------------------------------------------


class TestListThemes:
    def test_lists_all_known_themes(self) -> None:
        detector = ThemeDetector()
        themes = detector.list_themes()
        assert len(themes) == 8

    def test_custom_themes_file(self, tmp_path: Path) -> None:
        themes = [
            {"name": "a", "source": "x", "light_bgr": [0, 0, 0], "dark_bgr": [0, 0, 0]},
            {"name": "b", "source": "y", "light_bgr": [0, 0, 0], "dark_bgr": [0, 0, 0]},
        ]
        themes_file = _make_themes_file(tmp_path, themes)
        config = ThemeDetectionConfig(themes_file=themes_file)
        detector = ThemeDetector(config)
        assert len(detector.list_themes()) == 2


class TestValidation:
    def test_detect_wrong_square_count_raises(self) -> None:
        detector = ThemeDetector()
        with pytest.raises(ValueError, match="Expected 64"):
            detector.detect({})

    def test_match_theme_requires_signatures(self) -> None:
        light = np.array([0, 0, 0], dtype=np.float64)
        dark = np.array([0, 0, 0], dtype=np.float64)
        with pytest.raises(ValueError, match="At least one"):
            ThemeDetector._match_theme(light, dark, [])

    def test_init_with_empty_themes_file_raises(self, tmp_path: Path) -> None:
        themes_file = _make_themes_file(tmp_path, [])
        config = ThemeDetectionConfig(themes_file=themes_file)
        with pytest.raises(ThemeConfigError, match="No themes configured"):
            ThemeDetector(config)

    def test_init_with_invalid_themes_structure_raises(self, tmp_path: Path) -> None:
        themes_file = tmp_path / "themes.json"
        themes_file.write_text('{"bad": []}')
        config = ThemeDetectionConfig(themes_file=themes_file)
        with pytest.raises(ThemeConfigError, match="missing 'themes' list"):
            ThemeDetector(config)


# ---------------------------------------------------------------------------
# Multi-theme parametrized detection
# ---------------------------------------------------------------------------

_ALL_THEME_PARAMS = [
    ("chess.com-green", (208, 236, 235), (82, 149, 115)),
    ("chess.com-brown", (181, 217, 240), (99, 136, 181)),
    ("chess.com-blue", (230, 227, 222), (173, 162, 140)),
    ("chess.com-purple", (244, 218, 232), (203, 114, 155)),
    ("chess.com-walnut", (140, 184, 218), (73, 116, 164)),
    ("lichess-brown", (178, 217, 240), (107, 153, 181)),
    ("lichess-blue", (222, 228, 227), (187, 175, 136)),
    ("lichess-green", (221, 255, 255), (102, 166, 134)),
]


class TestMultiThemeDetection:
    """Verify each registered theme is detected correctly."""

    @pytest.mark.parametrize(
        "theme_id, light_bgr, dark_bgr",
        _ALL_THEME_PARAMS,
        ids=[t[0] for t in _ALL_THEME_PARAMS],
    )
    def test_exact_colours_detected(
        self,
        theme_id: str,
        light_bgr: tuple[int, int, int],
        dark_bgr: tuple[int, int, int],
    ) -> None:
        """Board with exact signature colours detects correctly."""
        squares = _make_full_board(
            light_bgr=light_bgr, dark_bgr=dark_bgr,
        )
        detector = ThemeDetector()
        result = detector.detect(squares)
        assert result.matched, f"{theme_id} not matched (dist={result.distance:.1f})"
        assert result.theme is not None
        detected_id = f"{result.theme.source}-{result.theme.name}"
        assert detected_id == theme_id

    @pytest.mark.parametrize(
        "theme_id, light_bgr, dark_bgr",
        _ALL_THEME_PARAMS,
        ids=[t[0] for t in _ALL_THEME_PARAMS],
    )
    def test_noisy_detection_at_least_90_percent(
        self,
        theme_id: str,
        light_bgr: tuple[int, int, int],
        dark_bgr: tuple[int, int, int],
    ) -> None:
        """10 boards with Gaussian noise, >= 9 correct."""
        detector = ThemeDetector()
        correct = 0
        for seed in range(10):
            rng = np.random.RandomState(seed)
            sigma = rng.uniform(2.0, 5.0)
            squares = _make_full_board(light_bgr=light_bgr, dark_bgr=dark_bgr)
            # Add noise to each square.
            noisy: dict[str, SquareImage] = {}
            for name, sq in squares.items():
                noise = rng.normal(0, sigma, sq.image.shape)
                img = np.clip(
                    sq.image.astype(np.float32) + noise, 0, 255,
                ).astype(np.uint8)
                noisy[name] = SquareImage(
                    name=name, image=img, file=sq.file, rank=sq.rank,
                )
            result = detector.detect(noisy)
            if (
                result.matched
                and result.theme is not None
                and f"{result.theme.source}-{result.theme.name}" == theme_id
            ):
                correct += 1
        assert correct >= 9, f"{theme_id}: {correct}/10 correct (need >= 9)"
