"""Automatic graphical theme detection for chess board screenshots.

Analyses background colours of board squares and matches them against
known theme signatures from chess.com and lichess.  Supports persistence
of the user's theme preference to ``~/.chesslab/config.json``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chesslab.splitting.board_splitter import SquareImage
from chesslab.splitting.square_color import square_background

logger = logging.getLogger(__name__)

_THEMES_FILE = Path(__file__).parent / "themes.json"
_CONFIG_DIR = Path.home() / ".chesslab"


class ThemeDetectionError(RuntimeError):
    """Base class for theme detection failures."""


class ThemeConfigError(ThemeDetectionError):
    """Raised when the theme signatures configuration is invalid."""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThemeSignature:
    """A known theme's colour signature (BGR)."""

    name: str
    source: str
    light_bgr: tuple[int, int, int]
    dark_bgr: tuple[int, int, int]

    @property
    def qualified_name(self) -> str:
        """Return ``'source/name'`` identifier, e.g. ``'chess.com/green'``."""
        return f"{self.source}/{self.name}"


@dataclass
class ThemeDetectionConfig:
    """Configuration for theme detection."""

    max_distance: float = 30.0
    corner_patch_size: int = 4
    config_dir: Path = field(default_factory=lambda: _CONFIG_DIR)
    themes_file: Path = field(default_factory=lambda: _THEMES_FILE)

    def __post_init__(self) -> None:
        if self.max_distance <= 0:
            raise ValueError(
                f"max_distance must be > 0, got {self.max_distance}.",
            )
        if self.corner_patch_size < 1:
            raise ValueError(
                f"corner_patch_size must be >= 1, got {self.corner_patch_size}.",
            )


@dataclass(frozen=True)
class ThemeDetectionResult:
    """Result of theme detection."""

    theme: ThemeSignature | None
    distance: float
    matched: bool
    elapsed_ms: float
    observed_light_bgr: tuple[int, int, int]
    observed_dark_bgr: tuple[int, int, int]
    available_themes: tuple[str, ...]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ThemeDetector:
    """Detects the graphical theme of a chess board from square images.

    Extracts the median background colour of light and dark squares by
    sampling corner pixels, then compares against known theme signatures
    using Euclidean distance in BGR space.
    """

    def __init__(self, config: ThemeDetectionConfig | None = None) -> None:
        self._config = config or ThemeDetectionConfig()
        self._signatures = self._load_signatures()
        logger.info(
            "ThemeDetector initialised (%d known themes).",
            len(self._signatures),
        )

    # -- Public API --------------------------------------------------------

    def detect(self, squares: dict[str, SquareImage]) -> ThemeDetectionResult:
        """Detect the graphical theme from board square images.

        Args:
            squares: Dictionary of square images keyed by algebraic name,
                as returned by ``BoardSplitter.split().squares``.

        Returns:
            Detection result with matched theme (or ``None``), distance,
            observed colours, and the list of available themes for fallback.
        """
        if len(squares) != 64:
            raise ValueError(f"Expected 64 squares, got {len(squares)}.")

        start = time.perf_counter()

        light_bgr, dark_bgr = self._extract_colours(squares)

        best_sig, best_dist = self._match_theme(
            light_bgr, dark_bgr, self._signatures,
        )

        matched = best_dist < self._config.max_distance
        elapsed = (time.perf_counter() - start) * 1000

        observed_light = (
            int(round(light_bgr[0])),
            int(round(light_bgr[1])),
            int(round(light_bgr[2])),
        )
        observed_dark = (
            int(round(dark_bgr[0])),
            int(round(dark_bgr[1])),
            int(round(dark_bgr[2])),
        )

        available = tuple(s.qualified_name for s in self._signatures)

        result_theme = best_sig if matched else None

        if matched:
            logger.info(
                "Theme detected: '%s' (distance=%.1f, elapsed=%.1f ms).",
                best_sig.qualified_name,
                best_dist,
                elapsed,
            )
        else:
            logger.warning(
                "No theme matched (best='%s', distance=%.1f, threshold=%.1f). "
                "Manual selection required.",
                best_sig.qualified_name,
                best_dist,
                self._config.max_distance,
            )

        return ThemeDetectionResult(
            theme=result_theme,
            distance=best_dist,
            matched=matched,
            elapsed_ms=elapsed,
            observed_light_bgr=observed_light,
            observed_dark_bgr=observed_dark,
            available_themes=available,
        )

    def list_themes(self) -> list[ThemeSignature]:
        """Return all known theme signatures."""
        return list(self._signatures)

    @staticmethod
    def save_preference(
        qualified_name: str,
        config_dir: Path | None = None,
    ) -> None:
        """Persist the theme preference to the user config file.

        Creates the config directory and file if they do not exist.
        Preserves any other keys already present in the config file.
        """
        directory = config_dir or _CONFIG_DIR
        directory.mkdir(parents=True, exist_ok=True)
        config_file = directory / "config.json"

        data: dict[str, object] = {}
        if config_file.is_file():
            try:
                with config_file.open() as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                data = {}

        data["theme"] = qualified_name
        with config_file.open("w") as fh:
            json.dump(data, fh, indent=2)

        logger.info("Theme preference saved: '%s'.", qualified_name)

    @staticmethod
    def load_preference(config_dir: Path | None = None) -> str | None:
        """Load the persisted theme preference.

        Returns:
            The qualified theme name (e.g. ``'chess.com/green'``), or
            ``None`` if no preference has been saved or the config file
            is missing / corrupt.
        """
        directory = config_dir or _CONFIG_DIR
        config_file = directory / "config.json"

        if not config_file.is_file():
            return None

        try:
            with config_file.open() as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read config file: %s.", config_file)
            return None

        theme = data.get("theme")
        if isinstance(theme, str) and theme:
            return theme
        return None

    # -- Private methods ---------------------------------------------------

    def _load_signatures(self) -> list[ThemeSignature]:
        """Load theme signatures from the themes file."""
        path = self._config.themes_file
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ThemeConfigError(
                f"Failed to read themes file: {path}.",
            ) from exc

        entries = data.get("themes")
        if not isinstance(entries, list):
            raise ThemeConfigError(
                f"Invalid themes file structure in {path}: missing 'themes' list.",
            )

        signatures: list[ThemeSignature] = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ThemeConfigError(
                    f"Invalid theme entry at index {idx} in {path}.",
                )

            name = entry.get("name")
            source = entry.get("source")
            if not isinstance(name, str) or not name.strip():
                raise ThemeConfigError(
                    f"Invalid theme name at index {idx} in {path}.",
                )
            if not isinstance(source, str) or not source.strip():
                raise ThemeConfigError(
                    f"Invalid theme source at index {idx} in {path}.",
                )

            light = self._parse_colour_triplet(
                entry.get("light_bgr"),
                field_name="light_bgr",
                idx=idx,
                path=path,
            )
            dark = self._parse_colour_triplet(
                entry.get("dark_bgr"),
                field_name="dark_bgr",
                idx=idx,
                path=path,
            )
            signatures.append(
                ThemeSignature(
                    name=name.strip(),
                    source=source.strip(),
                    light_bgr=light,
                    dark_bgr=dark,
                )
            )

        if not signatures:
            raise ThemeConfigError(f"No themes configured in {path}.")

        return signatures

    @staticmethod
    def _parse_colour_triplet(
        value: object,
        *,
        field_name: str,
        idx: int,
        path: Path,
    ) -> tuple[int, int, int]:
        if not isinstance(value, list) or len(value) != 3:
            raise ThemeConfigError(
                f"Invalid {field_name} at index {idx} in {path}: expected 3 values.",
            )

        channels: list[int] = []
        for raw in value:
            if not isinstance(raw, int) or not (0 <= raw <= 255):
                raise ThemeConfigError(
                    f"Invalid {field_name} at index {idx} in {path}: "
                    "values must be integers in [0, 255]."
                )
            channels.append(raw)
        return channels[0], channels[1], channels[2]

    def _extract_colours(
        self,
        squares: dict[str, SquareImage],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extract median background BGR for light and dark squares.

        Returns:
            Tuple of (light_bgr, dark_bgr) as float64 arrays of shape (3,).
        """
        patch = self._config.corner_patch_size
        light_colours: list[NDArray[np.float64]] = []
        dark_colours: list[NDArray[np.float64]] = []

        for square in squares.values():
            colour = self._sample_corner_colour(square.image, patch)
            if square_background(square.file, square.rank) == "light":
                light_colours.append(colour)
            else:
                dark_colours.append(colour)

        if not light_colours or not dark_colours:
            raise ValueError(
                "Unable to extract theme colours: expected both light and dark squares.",
            )

        light_median = np.median(np.stack(light_colours), axis=0)
        dark_median = np.median(np.stack(dark_colours), axis=0)
        return light_median, dark_median

    @staticmethod
    def _sample_corner_colour(
        image: NDArray[np.uint8],
        patch_size: int,
    ) -> NDArray[np.float64]:
        """Sample the background colour from the four corners of *image*.

        Returns:
            Median BGR colour as a float64 array of shape (3,).
        """
        img = image
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected image shape (H, W, 3|4), got {image.shape}.")

        h, w = img.shape[:2]
        p = min(patch_size, h // 2, w // 2)
        if p < 1:
            raise ValueError(
                f"Image too small for corner sampling: {h}x{w} (patch={patch_size}).",
            )

        corners = np.concatenate([
            img[:p, :p].reshape(-1, 3),
            img[:p, w - p:].reshape(-1, 3),
            img[h - p:, :p].reshape(-1, 3),
            img[h - p:, w - p:].reshape(-1, 3),
        ], axis=0).astype(np.float64)

        return np.median(corners, axis=0)

    @staticmethod
    def _match_theme(
        light: NDArray[np.float64],
        dark: NDArray[np.float64],
        signatures: list[ThemeSignature],
    ) -> tuple[ThemeSignature, float]:
        """Find the closest theme signature by Euclidean distance.

        Returns:
            Tuple of (best_signature, best_distance).
        """
        if not signatures:
            raise ValueError("At least one theme signature is required.")

        best_sig = signatures[0]
        best_dist = float("inf")

        for sig in signatures:
            d_light = float(np.linalg.norm(
                light - np.array(sig.light_bgr, dtype=np.float64),
            ))
            d_dark = float(np.linalg.norm(
                dark - np.array(sig.dark_bgr, dtype=np.float64),
            ))
            combined = (d_light + d_dark) / 2.0

            if combined < best_dist:
                best_dist = combined
                best_sig = sig

        return best_sig, best_dist
