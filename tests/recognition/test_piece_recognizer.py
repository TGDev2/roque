"""Tests for the piece recognition module.

Covers acceptance criteria:
- Nominal recognition: correct type+colour for all 64 squares, <200 ms
- Empty squares classified as EMPTY without error
- Low-confidence squares flagged, result marked uncertain
- Unknown theme raises ThemeNotFoundError with helpful message
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from chesslab.recognition.piece_recognizer import (
    PieceRecognizer,
    PieceType,
    RecognitionConfig,
    RecognitionResult,
    SquareClassification,
    TemplateLoadError,
    ThemeNotFoundError,
    _parse_template_key,
)
from chesslab.splitting.board_splitter import SquareImage
from chesslab.splitting.square_color import square_background

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FILES = "abcdefgh"
SIZE = 80

# chess.com Green (BGR)
LIGHT_BG = (208, 236, 235)
DARK_BG = (82, 149, 115)

WHITE_PIECE = (255, 255, 255)
BLACK_PIECE = (30, 30, 30)

_PIECE_KEYS = ["wK", "wQ", "wR", "wB", "wN", "wP",
               "bK", "bQ", "bR", "bB", "bN", "bP"]

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


# ---------------------------------------------------------------------------
# Drawing helpers — mirror _generate_placeholders patterns exactly
# ---------------------------------------------------------------------------


def _draw_king(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    c = SIZE // 2
    t = max(SIZE // 10, 2)
    cv2.line(img, (c, SIZE // 5), (c, 4 * SIZE // 5), colour, t)
    cv2.line(img, (SIZE // 5, c), (4 * SIZE // 5, c), colour, t)


def _draw_queen(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    c = SIZE // 2
    r = SIZE // 3
    t = max(SIZE // 12, 2)
    for angle_deg in (0, 60, 120):
        rad = np.radians(angle_deg)
        dx, dy = int(r * np.cos(rad)), int(r * np.sin(rad))
        cv2.line(img, (c - dx, c - dy), (c + dx, c + dy), colour, t)


def _draw_rook(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    m = SIZE // 5
    cv2.rectangle(img, (m, m), (SIZE - m, SIZE - m), colour, -1)


def _draw_bishop(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    c = SIZE // 2
    r = SIZE // 3
    pts = np.array([[c, c - r], [c + r, c], [c, c + r], [c - r, c]], dtype=np.int32)
    cv2.fillPoly(img, [pts], colour)


def _draw_knight(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    m = SIZE // 5
    t = max(SIZE // 8, 2)
    cv2.line(img, (m, SIZE - m), (m, m), colour, t)
    cv2.line(img, (m, m), (SIZE // 2, m), colour, t)


def _draw_pawn(img: NDArray[np.uint8], colour: tuple[int, int, int]) -> None:
    c = SIZE // 2
    r = SIZE // 4
    cv2.circle(img, (c, c), r, colour, -1)


_DRAW_FNS: dict[str, object] = {
    "K": _draw_king,
    "Q": _draw_queen,
    "R": _draw_rook,
    "B": _draw_bishop,
    "N": _draw_knight,
    "P": _draw_pawn,
}


def _bg_colour(bg: str) -> tuple[int, int, int]:
    return LIGHT_BG if bg == "light" else DARK_BG


def _piece_colour(key: str) -> tuple[int, int, int]:
    return WHITE_PIECE if key[0] == "w" else BLACK_PIECE


def _make_piece_square(
    piece_key: str, bg: str, size: int = SIZE,
) -> NDArray[np.uint8]:
    """Create a square image matching the synthetic template pattern."""
    return _make_piece_square_with_colours(
        piece_key,
        bg,
        size=size,
    )


def _make_piece_square_with_colours(
    piece_key: str,
    bg: str,
    *,
    white_piece: tuple[int, int, int] = WHITE_PIECE,
    black_piece: tuple[int, int, int] = BLACK_PIECE,
    size: int = SIZE,
) -> NDArray[np.uint8]:
    """Create a square image using custom white/black piece colours."""
    img = np.full((size, size, 3), _bg_colour(bg), dtype=np.uint8)
    letter = piece_key[1]
    piece_colour = white_piece if piece_key[0] == "w" else black_piece
    _DRAW_FNS[letter](img, piece_colour)  # type: ignore[operator]
    return img


def _make_empty_square(bg: str, size: int = SIZE) -> NDArray[np.uint8]:
    """Create a solid-colour empty square."""
    return np.full((size, size, 3), _bg_colour(bg), dtype=np.uint8)


def _bg_for_square(file_idx: int, rank_idx: int) -> str:
    return square_background(file_idx, rank_idx)


# ---------------------------------------------------------------------------
# Synthetic theme fixture (uses on-disk green templates)
# ---------------------------------------------------------------------------


def _make_synthetic_theme(tmp_path: Path, size: int = SIZE) -> Path:
    """Create a temporary theme directory with synthetic templates."""
    return _make_theme(
        tmp_path,
        size=size,
        theme_name="test_theme",
        synthetic=True,
    )


def _make_theme(
    tmp_path: Path,
    *,
    size: int = SIZE,
    theme_name: str = "test_theme",
    synthetic: bool = True,
    white_piece: tuple[int, int, int] = WHITE_PIECE,
    black_piece: tuple[int, int, int] = BLACK_PIECE,
) -> Path:
    """Create a temporary test theme with configurable piece luminance."""
    theme_dir = tmp_path / "themes" / theme_name
    theme_dir.mkdir(parents=True)

    templates: dict[str, str] = {}
    for key in _PIECE_KEYS:
        for bg in ("light", "dark"):
            tkey = f"{key}_{bg}"
            filename = f"{tkey}.png"
            img = _make_piece_square_with_colours(
                key,
                bg,
                size=size,
                white_piece=white_piece,
                black_piece=black_piece,
            )
            cv2.imwrite(str(theme_dir / filename), img)
            templates[tkey] = filename

    manifest = {
        "name": theme_name,
        "description": "Synthetic test theme" if synthetic else "Real-like test theme",
        "templates": templates,
    }
    if synthetic:
        manifest["synthetic"] = True
    with (theme_dir / "theme.json").open("w") as fh:
        json.dump(manifest, fh)

    return tmp_path / "themes"


def _make_full_board(
    piece_map: dict[str, str],
    size: int = SIZE,
    *,
    white_piece: tuple[int, int, int] = WHITE_PIECE,
    black_piece: tuple[int, int, int] = BLACK_PIECE,
) -> dict[str, SquareImage]:
    """Build 64 SquareImage entries.

    *piece_map* maps algebraic names to piece keys (e.g. ``{"e1": "wK"}``).
    Unlisted squares are empty.
    """
    squares: dict[str, SquareImage] = {}
    for f_idx in range(8):
        for r_idx in range(8):
            name = f"{_FILES[f_idx]}{r_idx + 1}"
            bg = _bg_for_square(f_idx, r_idx)
            piece_key = piece_map.get(name)
            if piece_key:
                img = _make_piece_square_with_colours(
                    piece_key,
                    bg,
                    size=size,
                    white_piece=white_piece,
                    black_piece=black_piece,
                )
            else:
                img = _make_empty_square(bg, size)
            squares[name] = SquareImage(
                name=name, image=img, file=f_idx, rank=r_idx,
            )
    return squares


# ---------------------------------------------------------------------------
# Starting position piece map
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scenario: Nominal recognition — chess.com Green
# ---------------------------------------------------------------------------


class TestNominalRecognition:
    """AC: correct type+colour, ≥ 95% accuracy, < 200 ms."""

    @pytest.fixture
    def theme_dir(self, tmp_path: Path) -> Path:
        return _make_synthetic_theme(tmp_path)

    @pytest.fixture
    def recognizer(self, theme_dir: Path) -> PieceRecognizer:
        config = RecognitionConfig(
            theme="test_theme", templates_dir=theme_dir,
        )
        return PieceRecognizer(config)

    @pytest.fixture
    def result(self, recognizer: PieceRecognizer) -> RecognitionResult:
        squares = _make_full_board(_STARTING_POSITION)
        return recognizer.recognize(squares)

    def test_64_squares_classified(self, result: RecognitionResult) -> None:
        assert len(result.classifications) == 64

    def test_correct_piece_types(self, result: RecognitionResult) -> None:
        for name, expected_key in _STARTING_POSITION.items():
            cls = result.classifications[name]
            expected = _KEY_TO_PIECETYPE[expected_key]
            assert cls.piece == expected, (
                f"Square {name}: expected {expected}, got {cls.piece}"
            )

    def test_empty_squares_correct(self, result: RecognitionResult) -> None:
        for name, cls in result.classifications.items():
            if name not in _STARTING_POSITION:
                assert cls.piece == PieceType.EMPTY, (
                    f"Square {name} should be EMPTY, got {cls.piece}"
                )

    def test_accuracy_above_95_percent(
        self, recognizer: PieceRecognizer,
    ) -> None:
        """Classify starting position and verify ≥ 95% accuracy."""
        squares = _make_full_board(_STARTING_POSITION)
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
        assert accuracy >= 0.95, f"Accuracy {accuracy:.1%} < 95%"

    def test_latency_under_200ms(self, recognizer: PieceRecognizer) -> None:
        squares = _make_full_board(_STARTING_POSITION)
        start = time.perf_counter()
        recognizer.recognize(squares)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"Recognition took {elapsed_ms:.1f} ms (limit: 200)"


class TestThemeRelativeColorVerification:
    """Real-template colour verification adapts to theme luminance polarity."""

    def test_dark_white_pieces_are_not_flipped(self, tmp_path: Path) -> None:
        theme_dir = _make_theme(
            tmp_path,
            theme_name="inverted_real_theme",
            synthetic=False,
            white_piece=(40, 40, 40),
            black_piece=(250, 250, 250),
        )
        recognizer = PieceRecognizer(
            RecognitionConfig(
                theme="inverted_real_theme",
                templates_dir=theme_dir,
            )
        )

        piece_map = {
            "d4": "wP",
            "e5": "bP",
            "a1": "wR",
            "h8": "bR",
        }
        squares = _make_full_board(
            piece_map,
            white_piece=(40, 40, 40),
            black_piece=(250, 250, 250),
        )
        result = recognizer.recognize(squares)

        for name, expected_key in piece_map.items():
            cls = result.classifications[name]
            expected = _KEY_TO_PIECETYPE[expected_key]
            assert cls.piece == expected, (
                f"Square {name}: expected {expected}, got {cls.piece}"
            )


# ---------------------------------------------------------------------------
# Scenario: Empty squares
# ---------------------------------------------------------------------------


class TestEmptySquare:
    """AC: empty squares labelled EMPTY without error."""

    @pytest.fixture
    def recognizer(self, tmp_path: Path) -> PieceRecognizer:
        theme_dir = _make_synthetic_theme(tmp_path)
        config = RecognitionConfig(
            theme="test_theme", templates_dir=theme_dir,
        )
        return PieceRecognizer(config)

    def test_empty_square_classified_as_empty(
        self, recognizer: PieceRecognizer,
    ) -> None:
        squares = _make_full_board({})  # all empty
        result = recognizer.recognize(squares)
        for name, cls in result.classifications.items():
            assert cls.piece == PieceType.EMPTY, (
                f"Square {name} should be EMPTY, got {cls.piece}"
            )

    def test_empty_square_no_error(
        self, recognizer: PieceRecognizer,
    ) -> None:
        """Recognition of an all-empty board does not raise."""
        squares = _make_full_board({})
        result = recognizer.recognize(squares)
        assert len(result.classifications) == 64

    def test_empty_square_not_low_confidence(
        self, recognizer: PieceRecognizer,
    ) -> None:
        squares = _make_full_board({})
        result = recognizer.recognize(squares)
        for cls in result.classifications.values():
            assert not cls.low_confidence, (
                f"Square {cls.square_name} should not be low confidence"
            )


# ---------------------------------------------------------------------------
# Scenario: Low confidence / ambiguity
# ---------------------------------------------------------------------------


class TestLowConfidence:
    """AC: ambiguous classification flagged with low confidence."""

    @pytest.fixture
    def recognizer(self, tmp_path: Path) -> PieceRecognizer:
        theme_dir = _make_synthetic_theme(tmp_path)
        config = RecognitionConfig(
            theme="test_theme",
            confidence_threshold=0.60,
            templates_dir=theme_dir,
        )
        return PieceRecognizer(config)

    def test_noise_square_flagged_low_confidence(
        self, recognizer: PieceRecognizer,
    ) -> None:
        """A square filled with random noise should trigger low confidence."""
        rng = np.random.RandomState(42)
        noise = rng.randint(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)

        # Build board: 63 empty + 1 noise square
        squares = _make_full_board({})
        squares["e4"] = SquareImage(
            name="e4", image=noise, file=4, rank=3,
        )
        result = recognizer.recognize(squares)

        cls = result.classifications["e4"]
        assert cls.low_confidence, "Noisy square should be low confidence"

    def test_uncertain_squares_populated(
        self, recognizer: PieceRecognizer,
    ) -> None:
        rng = np.random.RandomState(42)
        noise = rng.randint(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)

        squares = _make_full_board({})
        squares["e4"] = SquareImage(
            name="e4", image=noise, file=4, rank=3,
        )
        result = recognizer.recognize(squares)
        assert "e4" in result.uncertain_squares

    def test_result_marked_uncertain(
        self, recognizer: PieceRecognizer,
    ) -> None:
        rng = np.random.RandomState(42)
        noise = rng.randint(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)

        squares = _make_full_board({})
        squares["e4"] = SquareImage(
            name="e4", image=noise, file=4, rank=3,
        )
        result = recognizer.recognize(squares)
        assert result.uncertain is True


# ---------------------------------------------------------------------------
# Scenario: Unknown theme
# ---------------------------------------------------------------------------


class TestUnknownTheme:
    """AC: informative error + suggests manual configuration."""

    def test_missing_theme_raises(self, tmp_path: Path) -> None:
        theme_dir = _make_synthetic_theme(tmp_path)
        config = RecognitionConfig(
            theme="ocean", templates_dir=theme_dir,
        )
        with pytest.raises(ThemeNotFoundError):
            PieceRecognizer(config)

    def test_error_lists_available_themes(self, tmp_path: Path) -> None:
        theme_dir = _make_synthetic_theme(tmp_path)
        config = RecognitionConfig(
            theme="ocean", templates_dir=theme_dir,
        )
        with pytest.raises(ThemeNotFoundError, match="test_theme"):
            PieceRecognizer(config)

    def test_error_suggests_config(self, tmp_path: Path) -> None:
        theme_dir = _make_synthetic_theme(tmp_path)
        config = RecognitionConfig(
            theme="ocean", templates_dir=theme_dir,
        )
        with pytest.raises(ThemeNotFoundError, match="RecognitionConfig"):
            PieceRecognizer(config)


# ---------------------------------------------------------------------------
# Template loading errors
# ---------------------------------------------------------------------------


class TestTemplateLoading:
    """Template load edge cases."""

    def test_missing_template_file_raises(self, tmp_path: Path) -> None:
        theme_dir = tmp_path / "themes" / "broken"
        theme_dir.mkdir(parents=True)
        manifest = {
            "name": "broken",
            "templates": {"wK_light": "wK_light.png"},
        }
        with (theme_dir / "theme.json").open("w") as fh:
            json.dump(manifest, fh)

        config = RecognitionConfig(
            theme="broken", templates_dir=tmp_path / "themes",
        )
        with pytest.raises(TemplateLoadError, match="wK_light"):
            PieceRecognizer(config)

    def test_empty_manifest_raises(self, tmp_path: Path) -> None:
        theme_dir = tmp_path / "themes" / "empty_manifest"
        theme_dir.mkdir(parents=True)
        manifest = {"name": "empty_manifest", "templates": {}}
        with (theme_dir / "theme.json").open("w") as fh:
            json.dump(manifest, fh)

        config = RecognitionConfig(
            theme="empty_manifest", templates_dir=tmp_path / "themes",
        )
        with pytest.raises(TemplateLoadError, match="No templates"):
            PieceRecognizer(config)

    def test_incomplete_manifest_raises(self, tmp_path: Path) -> None:
        theme_dir = tmp_path / "themes" / "incomplete"
        theme_dir.mkdir(parents=True)
        img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        cv2.imwrite(str(theme_dir / "wK_light.png"), img)
        manifest = {
            "name": "incomplete",
            "templates": {"wK_light": "wK_light.png"},
        }
        with (theme_dir / "theme.json").open("w") as fh:
            json.dump(manifest, fh)

        config = RecognitionConfig(
            theme="incomplete", templates_dir=tmp_path / "themes",
        )
        with pytest.raises(TemplateLoadError, match="incomplete"):
            PieceRecognizer(config)

    def test_template_path_traversal_rejected(self, tmp_path: Path) -> None:
        theme_dir = tmp_path / "themes" / "path_escape"
        theme_dir.mkdir(parents=True)
        manifest = {
            "name": "path_escape",
            "templates": {"wK_light": "../outside.png"},
        }
        with (theme_dir / "theme.json").open("w") as fh:
            json.dump(manifest, fh)

        config = RecognitionConfig(
            theme="path_escape", templates_dir=tmp_path / "themes",
        )
        with pytest.raises(TemplateLoadError, match="escapes theme directory"):
            PieceRecognizer(config)

    def test_custom_templates_dir(self, tmp_path: Path) -> None:
        theme_dir = _make_synthetic_theme(tmp_path)
        config = RecognitionConfig(
            theme="test_theme", templates_dir=theme_dir,
        )
        recognizer = PieceRecognizer(config)
        squares = _make_full_board({})
        result = recognizer.recognize(squares)
        assert len(result.classifications) == 64


# ---------------------------------------------------------------------------
# Dataclass checks
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_square_classification_frozen(self) -> None:
        cls = SquareClassification(
            square_name="a1",
            piece=PieceType.EMPTY,
            confidence=1.0,
            low_confidence=False,
        )
        with pytest.raises(AttributeError):
            cls.piece = PieceType.WHITE_KING  # type: ignore[misc]

    def test_recognition_result_frozen(self) -> None:
        result = RecognitionResult(
            classifications={},
            elapsed_ms=1.0,
            uncertain=False,
            uncertain_squares=(),
        )
        with pytest.raises(AttributeError):
            result.elapsed_ms = 2.0  # type: ignore[misc]

    def test_config_invalid_threshold_zero(self) -> None:
        with pytest.raises(ValueError, match="Confidence threshold"):
            RecognitionConfig(confidence_threshold=0.0)

    def test_config_invalid_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="Confidence threshold"):
            RecognitionConfig(confidence_threshold=1.5)

    def test_config_empty_theme(self) -> None:
        with pytest.raises(ValueError, match="Theme name"):
            RecognitionConfig(theme="")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.fixture
    def recognizer(self, tmp_path: Path) -> PieceRecognizer:
        theme_dir = _make_synthetic_theme(tmp_path)
        config = RecognitionConfig(
            theme="test_theme", templates_dir=theme_dir,
        )
        return PieceRecognizer(config)

    def test_bgra_square_images(self, recognizer: PieceRecognizer) -> None:
        """4-channel (BGRA) square images are handled correctly."""
        piece_map = {"e1": "wK"}
        squares_bgr = _make_full_board(piece_map)

        # Convert all to BGRA
        squares: dict[str, SquareImage] = {}
        for name, sq in squares_bgr.items():
            bgra = cv2.cvtColor(sq.image, cv2.COLOR_BGR2BGRA)
            squares[name] = SquareImage(
                name=name, image=bgra, file=sq.file, rank=sq.rank,
            )

        result = recognizer.recognize(squares)
        assert result.classifications["e1"].piece == PieceType.WHITE_KING

    def test_different_square_size(self, tmp_path: Path) -> None:
        """Templates resize to match square dimensions."""
        small_size = 40
        # Create templates at original SIZE — they will be resized to
        # small_size internally.  Use the same small_size for the board
        # squares so the patterns are generated at target resolution.
        theme_dir = _make_synthetic_theme(tmp_path, size=small_size)
        config = RecognitionConfig(
            theme="test_theme", templates_dir=theme_dir,
        )
        recognizer = PieceRecognizer(config)

        squares = _make_full_board({"e1": "wK"}, size=small_size)
        result = recognizer.recognize(squares)
        assert result.classifications["e1"].piece == PieceType.WHITE_KING

    def test_wrong_square_count_raises(
        self, recognizer: PieceRecognizer,
    ) -> None:
        squares = {
            "e4": SquareImage(
                name="e4",
                image=np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
                file=4,
                rank=3,
            ),
        }
        with pytest.raises(ValueError, match="Expected 64"):
            recognizer.recognize(squares)

    def test_parse_template_key_valid(self) -> None:
        piece, bg = _parse_template_key("wK_light")
        assert piece == PieceType.WHITE_KING
        assert bg == "light"

    def test_parse_template_key_invalid(self) -> None:
        with pytest.raises(ValueError):
            _parse_template_key("invalid_key")


# ---------------------------------------------------------------------------
# Built-in green theme smoke test
# ---------------------------------------------------------------------------


class TestBuiltinGreenTheme:
    """Verify the shipped green theme loads and works."""

    def test_loads_without_error(self) -> None:
        recognizer = PieceRecognizer()
        assert recognizer is not None

    def test_classifies_starting_position(self) -> None:
        recognizer = PieceRecognizer()
        squares = _make_full_board(_STARTING_POSITION)
        result = recognizer.recognize(squares)
        assert len(result.classifications) == 64
