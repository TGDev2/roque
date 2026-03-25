"""Embedded ECO opening detection and grouping helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable

import chess

_DATA_FILE = Path(__file__).with_name("data") / "eco_openings.json"
_UNCLASSIFIED_KEY = "unclassified"
_EXACT_KEY_PREFIX = "eco:"
_FAMILY_KEY_PREFIX = "family:"
_UNCLASSIFIED_NAME = "Unclassified"


class OpeningGrouping(Enum):
    """How opening statistics should be aggregated."""

    ECO_CODE = "eco_code"
    ECO_FAMILY = "eco_family"


@dataclass(frozen=True)
class DetectedOpening:
    """Detected opening metadata for one game."""

    eco: str | None
    name: str

    @property
    def is_classified(self) -> bool:
        """Whether the opening was matched against the embedded ECO data."""
        return self.eco is not None

    @property
    def family(self) -> str | None:
        """Return the ECO family letter for classified openings."""
        if self.eco is None:
            return None
        return self.eco[:1]

    @property
    def display_name(self) -> str:
        """Human-readable label for the opening."""
        if self.eco is None:
            return self.name
        return f"{self.eco} {self.name}"

    def to_dict(self) -> dict[str, object]:
        """Serialise to JSON-compatible data."""
        return {
            "eco": self.eco,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, payload: object) -> DetectedOpening:
        """Rebuild an opening from persisted JSON data."""
        if not isinstance(payload, dict):
            return UNCLASSIFIED_OPENING

        eco = payload.get("eco")
        name = payload.get("name")
        if eco is not None and not isinstance(eco, str):
            eco = None
        if eco == "":
            eco = None
        if not isinstance(name, str) or not name.strip():
            return UNCLASSIFIED_OPENING
        return cls(
            eco=eco,
            name=name,
        )


UNCLASSIFIED_OPENING = DetectedOpening(eco=None, name=_UNCLASSIFIED_NAME)


@dataclass(frozen=True)
class _OpeningLine:
    """One canonical ECO line and its named target position."""

    opening: DetectedOpening
    uci_moves: tuple[str, ...]
    final_position: str


def detect_opening_from_uci_moves(
    moves: Iterable[str],
    *,
    start_fen: str | None = None,
) -> DetectedOpening:
    """Detect an ECO opening from UCI moves played from the standard start."""
    normalized_start = _normalize_start_fen(start_fen)
    if normalized_start != chess.STARTING_FEN:
        return UNCLASSIFIED_OPENING
    return _opening_detector().detect(moves)


def build_opening_filter_key(
    opening: DetectedOpening,
    grouping: OpeningGrouping,
) -> str:
    """Return a stable filter key for one opening and grouping mode."""
    if not opening.is_classified:
        return _UNCLASSIFIED_KEY
    if grouping is OpeningGrouping.ECO_FAMILY:
        assert opening.family is not None
        return f"{_FAMILY_KEY_PREFIX}{opening.family}"
    assert opening.eco is not None
    return f"{_EXACT_KEY_PREFIX}{opening.eco}"


def opening_matches_filter(
    opening: DetectedOpening,
    filter_key: str | None,
) -> bool:
    """Whether one opening belongs to the selected filter bucket."""
    if filter_key is None:
        return True
    if filter_key == _UNCLASSIFIED_KEY:
        return not opening.is_classified
    if filter_key.startswith(_EXACT_KEY_PREFIX):
        return opening.eco == filter_key.removeprefix(_EXACT_KEY_PREFIX)
    if filter_key.startswith(_FAMILY_KEY_PREFIX):
        return opening.family == filter_key.removeprefix(_FAMILY_KEY_PREFIX)
    return False


def describe_opening_group(
    opening: DetectedOpening,
    grouping: OpeningGrouping,
) -> str:
    """User-facing label for an opening bucket."""
    if not opening.is_classified:
        return opening.name
    if grouping is OpeningGrouping.ECO_FAMILY:
        family = opening.family
        assert family is not None
        return f"{family}00-{family}99"
    return opening.display_name


def describe_opening_filter_key(filter_key: str | None) -> str:
    """Best-effort label for a stored opening filter key."""
    if filter_key is None:
        return "All openings"
    if filter_key == _UNCLASSIFIED_KEY:
        return _UNCLASSIFIED_NAME
    if filter_key.startswith(_FAMILY_KEY_PREFIX):
        family = filter_key.removeprefix(_FAMILY_KEY_PREFIX)
        return f"{family}00-{family}99"
    if filter_key.startswith(_EXACT_KEY_PREFIX):
        eco = filter_key.removeprefix(_EXACT_KEY_PREFIX)
        opening = _opening_lookup().get(eco)
        if opening is not None:
            return opening.display_name
        return eco
    return filter_key


def coerce_opening_filter_key(
    filter_key: str | None,
    grouping: OpeningGrouping,
) -> str | None:
    """Adapt an existing filter key when the grouping mode changes."""
    if filter_key is None or filter_key == _UNCLASSIFIED_KEY:
        return filter_key
    if grouping is OpeningGrouping.ECO_CODE:
        if filter_key.startswith(_EXACT_KEY_PREFIX):
            return filter_key
        return None
    if filter_key.startswith(_FAMILY_KEY_PREFIX):
        return filter_key
    if filter_key.startswith(_EXACT_KEY_PREFIX):
        eco = filter_key.removeprefix(_EXACT_KEY_PREFIX)
        if not eco:
            return None
        return f"{_FAMILY_KEY_PREFIX}{eco[:1]}"
    return None


class _OpeningDetector:
    """Detect openings by the deepest matched named position."""

    def __init__(self, lines: tuple[_OpeningLine, ...]) -> None:
        self._positions: dict[str, tuple[tuple[int, DetectedOpening], ...]] = {}
        for line in lines:
            bucket = self._positions.setdefault(line.final_position, ())
            self._positions[line.final_position] = tuple(sorted(
                (*bucket, (len(line.uci_moves), line.opening)),
                key=lambda item: (-item[0], item[1].display_name),
            ))

    def detect(self, moves: Iterable[str]) -> DetectedOpening:
        """Return the deepest matched opening for the played move sequence."""
        board = chess.Board()
        best_depth = 0
        best_opening = UNCLASSIFIED_OPENING

        for ply, uci in enumerate(moves, start=1):
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break
            board.push(move)
            for depth, opening in self._positions.get(board.epd(), ()):
                if depth > ply or depth <= best_depth:
                    continue
                best_depth = depth
                best_opening = opening
                break

        return best_opening


@lru_cache(maxsize=1)
def _opening_detector() -> _OpeningDetector:
    return _OpeningDetector(_load_opening_lines())


@lru_cache(maxsize=1)
def _opening_lookup() -> dict[str, DetectedOpening]:
    return {
        line.opening.eco: line.opening
        for line in _load_opening_lines()
        if line.opening.eco is not None
    }


@lru_cache(maxsize=1)
def _load_opening_lines() -> tuple[_OpeningLine, ...]:
    payload = json.loads(_DATA_FILE.read_text())
    items = payload.get("entries", [])
    if not isinstance(items, list):
        raise ValueError(f"Invalid ECO data file: {_DATA_FILE}")

    lines: list[_OpeningLine] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        eco = item.get("eco")
        name = item.get("name")
        uci = item.get("uci")
        if not isinstance(eco, str) or not isinstance(name, str):
            continue
        if not isinstance(uci, list) or not all(isinstance(move, str) for move in uci):
            continue
        opening = DetectedOpening(eco=eco, name=name)
        final_position = _final_position_for_moves(tuple(uci))
        lines.append(_OpeningLine(
            opening=opening,
            uci_moves=tuple(uci),
            final_position=final_position,
        ))
    return tuple(sorted(lines, key=lambda item: item.opening.display_name))


def _final_position_for_moves(uci_moves: tuple[str, ...]) -> str:
    board = chess.Board()
    for uci in uci_moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"Invalid ECO move sequence: {uci_moves!r}")
        board.push(move)
    return board.epd()


def _normalize_start_fen(start_fen: str | None) -> str:
    if start_fen is None:
        return chess.STARTING_FEN
    try:
        return chess.Board(start_fen).fen()
    except ValueError:
        return ""
