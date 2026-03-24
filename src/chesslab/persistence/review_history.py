"""Persistence for lightweight post-game review summaries.

The review module already computes rich :class:`GameReview` objects in memory.
This module stores a compact, extensible summary of each completed review under
``~/.chesslab/reviews/history.json`` so longitudinal progression metrics can be
computed later without re-analysing games.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from chesslab.review.game_reviewer import (
    GamePhase,
    GameReview,
    LessonTheme,
    MoveClassification,
)

from chesslab.persistence.game_storage import PersistenceError

_REVIEWS_DIR = Path.home() / ".chesslab" / "reviews"
_REVIEW_HISTORY_FILE = _REVIEWS_DIR / "history.json"
_SCHEMA_VERSION = 1
_TRACKED_CLASSIFICATIONS: tuple[MoveClassification, ...] = (
    MoveClassification.EXCELLENT,
    MoveClassification.GOOD,
    MoveClassification.INACCURACY,
    MoveClassification.MISTAKE,
    MoveClassification.BLUNDER,
)
_ERROR_CLASSIFICATIONS = frozenset({
    MoveClassification.INACCURACY,
    MoveClassification.MISTAKE,
    MoveClassification.BLUNDER,
})


@dataclass(frozen=True)
class ReviewSummary:
    """Persisted summary of a completed review."""

    game_id: str
    reviewed_at: str
    move_count: int
    avg_cp_loss: float
    classification_counts: dict[MoveClassification, int]
    error_phase_counts: dict[GamePhase, int]
    lesson_theme_counts: dict[LessonTheme, int]
    source_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialise the summary to a JSON-compatible dict."""
        return {
            "game_id": self.game_id,
            "reviewed_at": self.reviewed_at,
            "move_count": self.move_count,
            "avg_cp_loss": self.avg_cp_loss,
            "classification_counts": {
                cls.value: self.classification_counts.get(cls, 0)
                for cls in _TRACKED_CLASSIFICATIONS
            },
            "error_phase_counts": {
                phase.value: self.error_phase_counts.get(phase, 0)
                for phase in GamePhase
            },
            "lesson_theme_counts": {
                theme.value: self.lesson_theme_counts.get(theme, 0)
                for theme in LessonTheme
            },
            "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ReviewSummary:
        """Rebuild a summary from JSON payload."""
        return cls(
            game_id=str(payload["game_id"]),
            reviewed_at=str(payload["reviewed_at"]),
            move_count=int(payload.get("move_count", 0)),
            avg_cp_loss=float(payload.get("avg_cp_loss", 0.0)),
            classification_counts=_load_enum_counts(
                payload.get("classification_counts"),
                _TRACKED_CLASSIFICATIONS,
            ),
            error_phase_counts=_load_enum_counts(
                payload.get("error_phase_counts"),
                tuple(GamePhase),
            ),
            lesson_theme_counts=_load_enum_counts(
                payload.get("lesson_theme_counts"),
                tuple(LessonTheme),
            ),
            source_path=_maybe_str(payload.get("source_path")),
        )


def build_review_summary(
    review: GameReview,
    *,
    reviewed_at: datetime | None = None,
    source_path: Path | None = None,
) -> ReviewSummary:
    """Create a persisted summary from a completed :class:`GameReview`."""
    summary_time = reviewed_at or datetime.now(timezone.utc)
    normalized_source = str(source_path.resolve()) if source_path is not None else None

    classification_counts = {cls: 0 for cls in _TRACKED_CLASSIFICATIONS}
    error_phase_counts = {phase: 0 for phase in GamePhase}
    lesson_theme_counts = {theme: 0 for theme in LessonTheme}

    for move in review.moves:
        if move.classification in classification_counts:
            classification_counts[move.classification] += 1
        if move.classification in _ERROR_CLASSIFICATIONS:
            error_phase_counts[move.phase] += 1

    for lesson in review.lessons:
        lesson_theme_counts[lesson.theme] += 1

    avg_cp_loss = (
        sum(move.cp_loss for move in review.moves) / len(review.moves)
        if review.moves
        else 0.0
    )

    return ReviewSummary(
        game_id=_build_game_id(review, source_path=source_path),
        reviewed_at=summary_time.isoformat(),
        move_count=len(review.moves),
        avg_cp_loss=avg_cp_loss,
        classification_counts=classification_counts,
        error_phase_counts=error_phase_counts,
        lesson_theme_counts=lesson_theme_counts,
        source_path=normalized_source,
    )


class ReviewHistoryStorage:
    """Stores and retrieves persisted review summaries."""

    def __init__(self, history_file: Path | None = None) -> None:
        self._history_file = history_file or _REVIEW_HISTORY_FILE

    def record_review(
        self,
        review: GameReview,
        *,
        reviewed_at: datetime | None = None,
        source_path: Path | None = None,
    ) -> ReviewSummary:
        """Persist or update the summary for a completed review."""
        summary = build_review_summary(
            review,
            reviewed_at=reviewed_at,
            source_path=source_path,
        )
        by_id = {
            existing.game_id: existing
            for existing in self.list_recent_reviews(limit=None)
        }
        by_id[summary.game_id] = summary
        ordered = sorted(
            by_id.values(),
            key=lambda item: item.reviewed_at,
            reverse=True,
        )
        self._write_all(ordered)
        return summary

    def list_recent_reviews(
        self,
        limit: int | None = 20,
    ) -> list[ReviewSummary]:
        """Return saved review summaries, newest first."""
        summaries = self._read_all()
        if limit is None:
            return summaries
        return summaries[:limit]

    def _read_all(self) -> list[ReviewSummary]:
        if not self._history_file.is_file():
            return []

        try:
            with self._history_file.open() as fh:
                payload = json.load(fh)
        except Exception as exc:
            raise PersistenceError(
                f"Failed to read review history: {self._history_file}: {exc}",
            ) from exc

        if not isinstance(payload, dict):
            raise PersistenceError(
                f"Invalid review history format: {self._history_file}",
            )

        schema_version = payload.get("schema_version")
        if schema_version != _SCHEMA_VERSION:
            raise PersistenceError(
                f"Unsupported review history schema: {schema_version}",
            )

        raw_reviews = payload.get("reviews", [])
        if not isinstance(raw_reviews, list):
            raise PersistenceError(
                f"Invalid review history payload: {self._history_file}",
            )

        summaries = [ReviewSummary.from_dict(item) for item in raw_reviews]
        return sorted(summaries, key=lambda item: item.reviewed_at, reverse=True)

    def _write_all(self, summaries: list[ReviewSummary]) -> None:
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "reviews": [summary.to_dict() for summary in summaries],
        }
        try:
            with self._history_file.open("w") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            raise PersistenceError(
                f"Failed to write review history: {self._history_file}: {exc}",
            ) from exc


def _build_game_id(review: GameReview, *, source_path: Path | None) -> str:
    start_fen = review.moves[0].fen_before if review.moves else "start"
    move_string = " ".join(move.uci for move in review.moves)
    source_key = str(source_path.resolve()) if source_path is not None else ""
    payload = f"{source_key}\n{start_fen}\n{move_string}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_enum_counts(
    payload: object,
    members: tuple[object, ...],
) -> dict[object, int]:
    raw = payload if isinstance(payload, dict) else {}
    return {
        member: int(raw.get(member.value, 0))  # type: ignore[attr-defined]
        for member in members
    }


def _maybe_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
