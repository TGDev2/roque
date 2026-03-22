"""Piece recognition module for classifying chess pieces on board squares.

Matches real or synthetic board squares against theme templates while
normalising square backgrounds, suppressing coordinate labels, and applying
an explicit empty-square heuristic before scoring piece candidates.
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

    _EMPTY_OCCUPANCY_THRESHOLD: float = 0.03
    _FOREGROUND_DISTANCE_THRESHOLD: float = 20.0
    _BACKGROUND_NORMALIZATION_DISTANCE_THRESHOLD: float = 28.0
    _LABEL_CLEANUP_DISTANCE_THRESHOLD: float = 10.0

    def __init__(self, config: RecognitionConfig | None = None) -> None:
        self._config = config or RecognitionConfig()
        self._templates_dir = self._config.templates_dir or _TEMPLATES_ROOT

        # {bg: {PieceType: image}} — raw templates as loaded from disk
        self._templates: dict[str, dict[PieceType, NDArray[np.uint8]]] = {
            "light": {},
            "dark": {},
        }
        self._load_theme(self._config.theme)
        self._canonical_backgrounds = self._estimate_canonical_backgrounds()

        # Precomputed feature matrices, keyed by background.
        # Each value is (piece_list, feature_matrix) where feature_matrix has
        # shape (num_templates, h*w*2) with unit-normalized rows.
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
        pending_by_background: dict[str, list[tuple[str, NDArray[np.float32]]]] = {
            "light": [],
            "dark": [],
        }

        for name, square in squares.items():
            prepared = self._prepare_square_image(square)
            background = square_background(square.file, square.rank)
            canonical_background = self._canonical_backgrounds[background]
            occupancy_ratio = self._occupancy_ratio(prepared, canonical_background)

            if occupancy_ratio <= self._EMPTY_OCCUPANCY_THRESHOLD:
                classifications[name] = SquareClassification(
                    square_name=name,
                    piece=PieceType.EMPTY,
                    confidence=1.0,
                    low_confidence=False,
                )
                continue

            pending_by_background[background].append(
                (
                    name,
                    self._build_feature_vector(
                        prepared,
                        canonical_background=canonical_background,
                    ),
                )
            )

        for background, pending in pending_by_background.items():
            if not pending:
                continue

            pieces, matrix = self._normed.get(
                background,
                ([], np.empty((0, 0), dtype=np.float32)),
            )
            if len(pieces) == 0:
                for name, _ in pending:
                    classifications[name] = SquareClassification(
                        square_name=name,
                        piece=PieceType.EMPTY,
                        confidence=0.0,
                        low_confidence=True,
                    )
                    uncertain_names.append(name)
                continue

            feature_matrix = np.vstack([vector for _, vector in pending])
            scores = feature_matrix @ matrix.T
            best_indices = np.argmax(scores, axis=1)
            best_scores = scores[np.arange(len(pending)), best_indices]

            for (name, _), best_idx, best_score in zip(
                pending,
                best_indices.tolist(),
                best_scores.tolist(),
            ):
                piece = pieces[int(best_idx)]
                score = float(best_score)
                low_confidence = score < self._config.confidence_threshold
                classifications[name] = SquareClassification(
                    square_name=name,
                    piece=piece,
                    confidence=max(0.0, score),
                    low_confidence=low_confidence,
                )
                if low_confidence:
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
        """Resize templates to *(target_h, target_w)* and precompute feature vectors."""
        if self._cached_size == (target_h, target_w):
            return

        n_pixels = target_h * target_w * 2
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
                vec = self._build_feature_vector(
                    resized,
                    canonical_background=self._canonical_backgrounds[bg],
                )
                pieces.append(piece)
                rows.append(vec)

            matrix = np.vstack(rows) if rows else np.empty((0, n_pixels), dtype=np.float32)
            self._normed[bg] = (pieces, matrix)

        self._cached_size = (target_h, target_w)
        logger.debug("Templates prepared for %dx%d.", target_w, target_h)

    # -- Single-square classification ---------------------------------------

    def _classify_square(self, square: SquareImage) -> SquareClassification:
        """Classify a single square image."""
        image = self._prepare_square_image(square)
        background = square_background(square.file, square.rank)
        canonical_background = self._canonical_backgrounds[background]

        occupancy_ratio = self._occupancy_ratio(image, canonical_background)
        if occupancy_ratio <= self._EMPTY_OCCUPANCY_THRESHOLD:
            return SquareClassification(
                square_name=square.name,
                piece=PieceType.EMPTY,
                confidence=1.0,
                low_confidence=False,
            )

        img_vec = self._build_feature_vector(
            image,
            canonical_background=canonical_background,
        )
        pieces, matrix = self._normed.get(
            background,
            ([], np.empty((0, 0), dtype=np.float32)),
        )

        if len(pieces) == 0:
            return SquareClassification(
                square_name=square.name,
                piece=PieceType.EMPTY,
                confidence=0.0,
                low_confidence=True,
            )

        scores = matrix @ img_vec

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

    def _estimate_canonical_backgrounds(self) -> dict[str, NDArray[np.uint8]]:
        """Estimate one canonical background colour per square colour."""
        colours: dict[str, NDArray[np.uint8]] = {}
        for background, templates in self._templates.items():
            samples: list[NDArray[np.float32]] = []
            for image in templates.values():
                samples.append(self._estimate_background_colour(image))
            colours[background] = np.median(np.stack(samples), axis=0).astype(np.uint8)
        return colours

    @classmethod
    def _estimate_background_colour(cls, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Estimate a square's background colour from its borders."""
        bgr = cls._ensure_bgr(image)
        h, w = bgr.shape[:2]
        border = max(4, min(h, w) // 20)
        border_pixels = np.concatenate(
            [
                bgr[:border, :, :].reshape(-1, 3),
                bgr[h - border :, :, :].reshape(-1, 3),
                bgr[:, :border, :].reshape(-1, 3),
                bgr[:, w - border :, :].reshape(-1, 3),
            ],
            axis=0,
        ).astype(np.float32)
        return np.median(border_pixels, axis=0)

    @staticmethod
    def _ensure_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Return a 3-channel BGR view of *image*."""
        if image.ndim == 3 and image.shape[2] == 4:
            return image[:, :, :3]
        return image

    def _prepare_square_image(self, square: SquareImage) -> NDArray[np.uint8]:
        """Normalize a live square before feature extraction."""
        image = self._ensure_bgr(square.image).copy()
        background = square_background(square.file, square.rank)
        target_background = self._canonical_backgrounds[background]
        source_background = self._estimate_background_colour(image).astype(np.uint8)

        self._clean_coordinate_overlays(
            image,
            square=square,
            source_background=source_background,
        )

        distance = np.linalg.norm(
            image.astype(np.float32) - source_background.astype(np.float32),
            axis=2,
        )
        background_mask = distance <= self._BACKGROUND_NORMALIZATION_DISTANCE_THRESHOLD
        image[background_mask] = target_background
        return image

    def _build_feature_vector(
        self,
        image: NDArray[np.uint8],
        *,
        canonical_background: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Build a unit-normalized shape+appearance feature vector."""
        bgr = self._ensure_bgr(image)
        background = canonical_background.astype(np.float32)
        residual = bgr.astype(np.float32) - background
        magnitude = np.linalg.norm(residual, axis=2)
        foreground = magnitude > self._FOREGROUND_DISTANCE_THRESHOLD

        gray = residual.mean(axis=2)
        mask = foreground.astype(np.float32) * 255.0
        vec = np.stack([gray, mask], axis=2).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm
        else:
            vec[:] = 0.0
        return vec

    def _occupancy_ratio(
        self,
        image: NDArray[np.uint8],
        canonical_background: NDArray[np.uint8],
    ) -> float:
        """Estimate how much of the inner square is occupied by a piece."""
        bgr = self._ensure_bgr(image)
        h, w = bgr.shape[:2]
        if min(h, w) < 64:
            inner = bgr.reshape(-1, 3)
        else:
            margin = max(4, min(h, w) // 10)
            inner = bgr[margin : h - margin, margin : w - margin].reshape(-1, 3)
        distance = np.linalg.norm(
            inner.astype(np.float32) - canonical_background.astype(np.float32),
            axis=1,
        )
        return float((distance > self._FOREGROUND_DISTANCE_THRESHOLD).mean())

    def _clean_coordinate_overlays(
        self,
        image: NDArray[np.uint8],
        *,
        square: SquareImage,
        source_background: NDArray[np.uint8],
    ) -> None:
        """Remove chess.com coordinate overlays from known board corners."""
        h, w = image.shape[:2]
        if min(h, w) < 64:
            return
        background = source_background.astype(np.float32)

        if square.file == 0:
            zone = image[: max(1, h // 4), : max(1, w // 6)]
            distance = np.linalg.norm(
                zone.astype(np.float32) - background,
                axis=2,
            )
            foreground = distance > self._LABEL_CLEANUP_DISTANCE_THRESHOLD
            if float(foreground.mean()) <= 0.20:
                zone[foreground] = source_background

        if square.rank == 0:
            zone = image[h - max(1, h // 4) :, w - max(1, w // 6) :]
            distance = np.linalg.norm(
                zone.astype(np.float32) - background,
                axis=2,
            )
            foreground = distance > self._LABEL_CLEANUP_DISTANCE_THRESHOLD
            if float(foreground.mean()) <= 0.20:
                zone[foreground] = source_background
