"""Piece recognition module for classifying chess pieces on board squares.

Uses OpenCV template matching to identify piece type (king, queen, rook,
bishop, knight, pawn) and colour (white, black) on each of the 64 squares,
supporting multiple graphical themes via loadable template libraries.
"""

from __future__ import annotations

import enum
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from chesslab.splitting.board_splitter import SquareImage
from chesslab.splitting.square_color import square_background

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Piece type enumeration
# ---------------------------------------------------------------------------

_TEMPLATES_ROOT = Path(__file__).parent / "templates"


class PieceType(enum.Enum):
    """Chess piece classification using FEN notation."""

    WHITE_KING = "K"
    WHITE_QUEEN = "Q"
    WHITE_ROOK = "R"
    WHITE_BISHOP = "B"
    WHITE_KNIGHT = "N"
    WHITE_PAWN = "P"
    BLACK_KING = "k"
    BLACK_QUEEN = "q"
    BLACK_ROOK = "r"
    BLACK_BISHOP = "b"
    BLACK_KNIGHT = "n"
    BLACK_PAWN = "p"
    EMPTY = "."


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class RecognitionError(RuntimeError):
    """Base class for piece recognition failures."""


class ThemeNotFoundError(RecognitionError):
    """Raised when the requested theme directory or manifest is missing."""


class TemplateLoadError(RecognitionError):
    """Raised when a template file referenced in the manifest cannot be loaded."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RecognitionConfig:
    """Configuration for the piece recognizer."""

    theme: str = "chess.com-green"
    confidence_threshold: float = 0.60
    templates_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.theme:
            raise ValueError("Theme name must not be empty.")
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                f"Confidence threshold must be in (0, 1], got {self.confidence_threshold}."
            )


@dataclass(frozen=True)
class SquareClassification:
    """Classification result for a single board square."""

    square_name: str
    piece: PieceType
    confidence: float
    low_confidence: bool


@dataclass(frozen=True)
class RecognitionResult:
    """Result of classifying all 64 board squares."""

    classifications: dict[str, SquareClassification]
    elapsed_ms: float
    uncertain: bool
    uncertain_squares: tuple[str, ...]


# ---------------------------------------------------------------------------
# Template key helpers
# ---------------------------------------------------------------------------

_KEY_TO_PIECE: dict[str, PieceType] = {
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

_PIECE_TO_KEY: dict[PieceType, str] = {v: k for k, v in _KEY_TO_PIECE.items()}


def _parse_template_key(key: str) -> tuple[PieceType, str]:
    """Parse a manifest key like ``'wK_light'`` into ``(PieceType, background)``.

    Returns:
        Tuple of (PieceType, "light" | "dark").

    Raises:
        ValueError: If the key format is unrecognised.
    """
    parts = key.rsplit("_", 1)
    if len(parts) != 2 or parts[1] not in ("light", "dark"):
        raise ValueError(f"Invalid template key format: '{key}'.")
    piece_key, background = parts
    piece = _KEY_TO_PIECE.get(piece_key)
    if piece is None:
        raise ValueError(f"Unknown piece key '{piece_key}' in template key '{key}'.")
    return piece, background


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PieceRecognizer:
    """Classifies chess pieces on board squares using template matching.

    Templates are loaded from a theme directory and matched against each
    square image using normalised cross-correlation (equivalent to
    ``TM_CCOEFF_NORMED``) computed via vectorised numpy operations for
    performance.
    """

    _EMPTY_SPATIAL_STD_THRESHOLD: float = 3.0

    def __init__(self, config: RecognitionConfig | None = None) -> None:
        self._config = config or RecognitionConfig()
        self._templates_dir = self._config.templates_dir or _TEMPLATES_ROOT

        # {bg: {PieceType: image}} — raw templates as loaded from disk
        self._templates: dict[str, dict[PieceType, NDArray[np.uint8]]] = {
            "light": {},
            "dark": {},
        }
        self._load_theme(self._config.theme)

        # Precomputed normalised template matrices, keyed by background.
        # Each value is (piece_list, normed_matrix) where normed_matrix has
        # shape (num_templates, h*w*channels) with zero-mean unit-variance rows.
        self._normed: dict[str, tuple[list[PieceType], NDArray[np.float32]]] = {}
        self._cached_size: tuple[int, int] | None = None

        logger.info(
            "PieceRecognizer initialised (theme='%s', threshold=%.2f).",
            self._config.theme,
            self._config.confidence_threshold,
        )

    # -- Public API ---------------------------------------------------------

    def recognize(
        self, squares: dict[str, SquareImage]
    ) -> RecognitionResult:
        """Classify all 64 board squares.

        Args:
            squares: Dictionary of square images keyed by algebraic name,
                as returned by ``BoardSplitter.split().squares``.

        Returns:
            RecognitionResult with per-square classifications.

        Raises:
            ValueError: If *squares* does not contain exactly 64 entries.
        """
        if len(squares) != 64:
            raise ValueError(
                f"Expected 64 squares, got {len(squares)}."
            )

        start = time.perf_counter()

        # Ensure templates are resized to match the current square dimensions.
        first = next(iter(squares.values()))
        target_h, target_w = first.image.shape[:2]
        self._ensure_resized(target_h, target_w)

        classifications: dict[str, SquareClassification] = {}
        uncertain_names: list[str] = []

        for name, square in squares.items():
            cls = self._classify_square(square)
            classifications[name] = cls
            if cls.low_confidence:
                uncertain_names.append(name)

        elapsed = (time.perf_counter() - start) * 1000

        uncertain = len(uncertain_names) > 0
        if uncertain:
            logger.warning(
                "%d square(s) classified with low confidence: %s.",
                len(uncertain_names),
                ", ".join(uncertain_names),
            )
        logger.debug("Recognition completed for 64 squares (%.1f ms).", elapsed)

        return RecognitionResult(
            classifications=classifications,
            elapsed_ms=elapsed,
            uncertain=uncertain,
            uncertain_squares=tuple(uncertain_names),
        )

    # -- Theme loading ------------------------------------------------------

    def _load_theme(self, theme: str) -> None:
        """Load template images for *theme* from the templates directory."""
        theme_dir = self._templates_dir / theme

        if not theme_dir.is_dir():
            available = self._list_available_themes()
            available_str = ", ".join(available) if available else "(none)"
            raise ThemeNotFoundError(
                f"Theme '{theme}' not found at {theme_dir}. "
                f"Available themes: {available_str}. "
                f"Configure a valid theme via RecognitionConfig(theme='...')."
            )

        manifest_path = theme_dir / "theme.json"
        if not manifest_path.is_file():
            raise ThemeNotFoundError(
                f"Theme manifest not found: {manifest_path}. "
                f"Each theme directory must contain a theme.json file."
            )

        try:
            with manifest_path.open(encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise TemplateLoadError(
                f"Failed to read theme manifest: {manifest_path}.",
            ) from exc

        if manifest.get("synthetic", False):
            logger.warning(
                "Theme '%s' uses synthetic templates — recognition accuracy "
                "will be low. Run 'python -m chesslab.extract' with a real "
                "screenshot to generate production-quality templates.",
                theme,
            )

        templates_map = manifest.get("templates")
        if not isinstance(templates_map, dict) or not templates_map:
            raise TemplateLoadError(
                f"No templates listed in {manifest_path}."
            )

        for key, filename in templates_map.items():
            if not isinstance(filename, str) or not filename.strip():
                raise TemplateLoadError(
                    f"Invalid template filename for key '{key}' in {manifest_path}.",
                )

            try:
                piece, background = _parse_template_key(key)
            except ValueError as exc:
                raise TemplateLoadError(
                    f"Invalid template key '{key}' in {manifest_path}.",
                ) from exc

            filepath = self._resolve_template_path(theme_dir, filename)
            img = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
            if img is None:
                raise TemplateLoadError(
                    f"Failed to load template '{filepath}' "
                    f"(key='{key}')."
                )
            self._templates[background][piece] = img

        self._validate_template_coverage(theme, manifest_path)

        loaded = sum(len(v) for v in self._templates.values())
        logger.debug(
            "Loaded %d templates for theme '%s'.", loaded, theme,
        )

    @staticmethod
    def _resolve_template_path(theme_dir: Path, filename: str) -> Path:
        """Resolve a template file path and block path traversal."""
        root = theme_dir.resolve()
        resolved = (theme_dir / filename).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise TemplateLoadError(
                f"Template path '{filename}' escapes theme directory '{theme_dir}'.",
            ) from exc
        return resolved

    def _validate_template_coverage(self, theme: str, manifest_path: Path) -> None:
        """Ensure the manifest provides all piece templates for both backgrounds."""
        expected_pieces = set(_KEY_TO_PIECE.values())
        missing: list[str] = []
        for background in ("light", "dark"):
            seen = set(self._templates[background].keys())
            for piece in expected_pieces - seen:
                missing.append(f"{_PIECE_TO_KEY[piece]}_{background}")

        if missing:
            missing_str = ", ".join(sorted(missing))
            raise TemplateLoadError(
                f"Theme '{theme}' manifest is incomplete ({manifest_path}). "
                f"Missing templates: {missing_str}."
            )

    def _list_available_themes(self) -> list[str]:
        """Return names of theme directories that contain a theme.json."""
        if not self._templates_dir.is_dir():
            return []
        return sorted(
            d.name
            for d in self._templates_dir.iterdir()
            if d.is_dir() and (d / "theme.json").is_file()
        )

    # -- Template preparation ------------------------------------------------

    def _ensure_resized(self, target_h: int, target_w: int) -> None:
        """Resize templates to *(target_h, target_w)* and precompute normed vectors."""
        if self._cached_size == (target_h, target_w):
            return

        n_pixels = target_h * target_w * 3
        self._normed = {}

        for bg in ("light", "dark"):
            pieces: list[PieceType] = []
            rows: list[NDArray[np.float32]] = []

            for piece, img in self._templates[bg].items():
                interpolation = (
                    cv2.INTER_AREA
                    if img.shape[0] > target_h
                    else cv2.INTER_LINEAR
                )
                resized = cv2.resize(
                    img, (target_w, target_h), interpolation=interpolation,
                )
                vec = resized.flatten().astype(np.float32)
                vec -= vec.mean()
                std = float(np.std(vec))
                if std > 1e-6:
                    vec /= std
                pieces.append(piece)
                rows.append(vec)

            matrix = np.vstack(rows) if rows else np.empty((0, n_pixels), dtype=np.float32)
            self._normed[bg] = (pieces, matrix)

        self._cached_size = (target_h, target_w)
        logger.debug("Templates prepared for %dx%d.", target_w, target_h)

    # -- Single-square classification ---------------------------------------

    def _classify_square(self, square: SquareImage) -> SquareClassification:
        """Classify a single square image."""
        image = square.image
        # Ensure BGR (strip alpha if BGRA).
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        # Check spatial variance: if all pixels are nearly the same colour,
        # the square is empty (solid background).
        pixels = image.reshape(-1, 3).astype(np.float32)
        spatial_std = float(np.std(pixels, axis=0).max())
        if spatial_std < self._EMPTY_SPATIAL_STD_THRESHOLD:
            return SquareClassification(
                square_name=square.name,
                piece=PieceType.EMPTY,
                confidence=1.0,
                low_confidence=False,
            )

        # Flatten and normalise for cross-correlation.
        img_vec = image.flatten().astype(np.float32)
        img_vec -= img_vec.mean()
        std = float(np.std(img_vec))
        if std < 1e-6:
            img_vec[:] = 0.0
        else:
            img_vec /= std

        bg = square_background(square.file, square.rank)
        pieces, matrix = self._normed.get(bg, ([], np.empty((0,), dtype=np.float32)))

        if len(pieces) == 0:
            return SquareClassification(
                square_name=square.name,
                piece=PieceType.EMPTY,
                confidence=0.0,
                low_confidence=True,
            )

        # Normalised cross-correlation: dot product of unit-variance vectors
        # divided by the number of elements.
        scores = matrix @ img_vec / len(img_vec)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_piece = pieces[best_idx]

        # Below threshold → low confidence.  The square passed the spatial
        # uniformity check so it contains *something*; we just cannot
        # identify it confidently.
        if best_score < self._config.confidence_threshold:
            return SquareClassification(
                square_name=square.name,
                piece=best_piece,
                confidence=max(0.0, best_score),
                low_confidence=True,
            )

        return SquareClassification(
            square_name=square.name,
            piece=best_piece,
            confidence=best_score,
            low_confidence=False,
        )
