"""Persistence for lightweight post-game review summaries.

The review module already computes rich :class:`GameReview` objects in memory.
This module stores a compact, extensible summary of each completed review under
``~/.chesslab/reviews/history.json`` so longitudinal progression metrics can be
computed later without re-analysing games.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from chesslab.review.game_reviewer import (
    ErrorMotif,
    GamePhase,
    GameReview,
    LessonTheme,
    MoveClassification,
    ReviewLesson,
)

from chesslab.persistence.game_storage import PersistenceError

_REVIEWS_DIR = Path.home() / ".chesslab" / "reviews"
_REVIEW_HISTORY_FILE = _REVIEWS_DIR / "history.json"
_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
_TRACKED_CLASSIFICATIONS: tuple[MoveClassification, ...] = (
    MoveClassification.EXCELLENT,
    MoveClassification.GOOD,
    MoveClassification.INACCURACY,
    MoveClassification.MISTAKE,
    MoveClassification.BLUNDER,
)
_TRACKED_ERROR_MOTIFS: tuple[ErrorMotif, ...] = tuple(ErrorMotif)
_ERROR_CLASSIFICATIONS = frozenset({
    MoveClassification.INACCURACY,
    MoveClassification.MISTAKE,
    MoveClassification.BLUNDER,
})
_TRAINING_TARGET_CLASSIFICATIONS = frozenset({
    MoveClassification.MISTAKE,
    MoveClassification.BLUNDER,
})
_MAX_TRAINING_POSITIONS = 8


@dataclass(frozen=True)
class TrainingPositionSummary:
    """Persisted training candidate extracted from a reviewed critical position."""

    fen: str
    ply: int
    played_san: str
    played_uci: str
    best_move_san: str
    best_move_uci: str
    best_pv_san: tuple[str, ...]
    classification: MoveClassification
    phase: GamePhase
    cp_loss: int
    lesson_themes: tuple[LessonTheme, ...] = ()
    error_motif: ErrorMotif = ErrorMotif.UNKNOWN

    def to_dict(self) -> dict[str, object]:
        """Serialise the training candidate to JSON-compatible data."""
        return {
            "fen": self.fen,
            "ply": self.ply,
            "played_san": self.played_san,
            "played_uci": self.played_uci,
            "best_move_san": self.best_move_san,
            "best_move_uci": self.best_move_uci,
            "best_pv_san": list(self.best_pv_san),
            "classification": self.classification.value,
            "phase": self.phase.value,
            "cp_loss": self.cp_loss,
            "lesson_themes": [theme.value for theme in self.lesson_themes],
            "error_motif": self.error_motif.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TrainingPositionSummary:
        """Rebuild a training candidate from JSON payload."""
        return cls(
            fen=str(payload.get("fen", "")),
            ply=int(payload.get("ply", 0)),
            played_san=str(payload.get("played_san", "")),
            played_uci=str(payload.get("played_uci", "")),
            best_move_san=str(payload.get("best_move_san", "")),
            best_move_uci=str(payload.get("best_move_uci", "")),
            best_pv_san=tuple(_load_str_list(payload.get("best_pv_san"))),
            classification=_load_enum_value(
                payload.get("classification"),
                MoveClassification,
                MoveClassification.MISTAKE,
            ),
            phase=_load_enum_value(
                payload.get("phase"),
                GamePhase,
                GamePhase.MIDDLEGAME,
            ),
            cp_loss=int(payload.get("cp_loss", 0)),
            lesson_themes=tuple(
                _load_enum_sequence(payload.get("lesson_themes"), LessonTheme),
            ),
            error_motif=_load_enum_value(
                payload.get("error_motif"),
                ErrorMotif,
                ErrorMotif.UNKNOWN,
            ),
        )


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
    error_motif_counts: dict[ErrorMotif, int] = field(default_factory=dict)
    source_path: str | None = None
    training_positions: tuple[TrainingPositionSummary, ...] = ()

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
            "error_motif_counts": {
                motif.value: self.error_motif_counts.get(motif, 0)
                for motif in _TRACKED_ERROR_MOTIFS
            },
            "source_path": self.source_path,
            "training_positions": [
                position.to_dict() for position in self.training_positions
            ],
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
            error_motif_counts=_load_enum_counts(
                payload.get("error_motif_counts"),
                _TRACKED_ERROR_MOTIFS,
            ),
            source_path=_maybe_str(payload.get("source_path")),
            training_positions=tuple(
                TrainingPositionSummary.from_dict(item)
                for item in _load_dict_list(payload.get("training_positions"))
            ),
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
    error_motif_counts = {motif: 0 for motif in _TRACKED_ERROR_MOTIFS}

    for move in review.moves:
        if move.classification in classification_counts:
            classification_counts[move.classification] += 1
        if move.classification in _ERROR_CLASSIFICATIONS:
            error_phase_counts[move.phase] += 1
            error_motif_counts[move.error_motif] += 1

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
        error_motif_counts=error_motif_counts,
        source_path=normalized_source,
        training_positions=_build_training_positions(review),
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
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
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


def _build_training_positions(
    review: GameReview,
) -> tuple[TrainingPositionSummary, ...]:
    """Extract reusable training positions from a completed review."""
    lesson_themes_by_ply = _map_lesson_themes(review.lessons, review.moves)
    candidates: list[tuple[int, TrainingPositionSummary]] = []

    for move in review.moves:
        if move.is_forced:
            continue
        if not move.best_move_uci or move.best_move_uci == move.uci:
            continue

        themes = lesson_themes_by_ply.get(move.ply, ())
        if (
            move.classification not in _TRAINING_TARGET_CLASSIFICATIONS
            and not themes
        ):
            continue

        score = _training_position_score(move, themes)
        candidates.append((
            score,
            TrainingPositionSummary(
                fen=move.fen_before,
                ply=move.ply,
                played_san=move.san,
                played_uci=move.uci,
                best_move_san=move.best_move_san,
                best_move_uci=move.best_move_uci,
                best_pv_san=move.best_pv_san,
                classification=move.classification,
                phase=move.phase,
                cp_loss=move.cp_loss,
                lesson_themes=themes,
                error_motif=move.error_motif,
            ),
        ))

    ordered = sorted(
        candidates,
        key=lambda item: (-item[0], item[1].ply),
    )
    unique: list[TrainingPositionSummary] = []
    seen: set[tuple[str, str, str]] = set()
    for _score, position in ordered:
        key = (
            position.fen,
            position.played_uci,
            position.best_move_uci,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(position)
        if len(unique) >= _MAX_TRAINING_POSITIONS:
            break

    return tuple(unique)


def _map_lesson_themes(
    lessons: tuple[ReviewLesson, ...],
    moves: tuple[object, ...],
) -> dict[int, tuple[LessonTheme, ...]]:
    themes_by_ply: dict[int, set[LessonTheme]] = {}
    move_lookup = {getattr(move, "ply"): move for move in moves}

    for lesson in lessons:
        related = set(lesson.related_plies)
        related.add(lesson.focus_ply)
        for ply in related:
            if ply in move_lookup:
                themes_by_ply.setdefault(ply, set()).add(lesson.theme)

        if lesson.phase is None or lesson.is_white is None:
            continue

        for ply, move in move_lookup.items():
            if (
                getattr(move, "phase") is lesson.phase
                and getattr(move, "is_white") is lesson.is_white
                and getattr(move, "classification") in _TRAINING_TARGET_CLASSIFICATIONS
            ):
                themes_by_ply.setdefault(ply, set()).add(lesson.theme)

    return {
        ply: tuple(sorted(themes, key=lambda theme: theme.value))
        for ply, themes in themes_by_ply.items()
    }


def _training_position_score(
    move: object,
    themes: tuple[LessonTheme, ...],
) -> int:
    severity = 0
    classification = getattr(move, "classification")
    if classification is MoveClassification.BLUNDER:
        severity = 220
    elif classification is MoveClassification.MISTAKE:
        severity = 150
    elif classification is MoveClassification.INACCURACY:
        severity = 80

    cp_loss = min(int(getattr(move, "cp_loss")), 400)
    pv_bonus = min(len(getattr(move, "best_pv_san")), 4) * 12
    theme_bonus = len(themes) * 45
    return severity + cp_loss + pv_bonus + theme_bonus


def _load_enum_counts(
    payload: object,
    members: tuple[object, ...],
) -> dict[object, int]:
    raw = payload if isinstance(payload, dict) else {}
    return {
        member: int(raw.get(member.value, 0))  # type: ignore[attr-defined]
        for member in members
    }


def _load_enum_value(
    payload: object,
    enum_type: type[object],
    default: object,
) -> object:
    if not isinstance(payload, str):
        return default

    try:
        return enum_type(payload)  # type: ignore[misc,call-arg]
    except ValueError:
        return default


def _load_enum_sequence(
    payload: object,
    enum_type: type[object],
) -> tuple[object, ...]:
    if not isinstance(payload, list):
        return ()

    values: list[object] = []
    for item in payload:
        if not isinstance(item, str):
            continue
        try:
            values.append(enum_type(item))  # type: ignore[misc,call-arg]
        except ValueError:
            continue
    return tuple(values)


def _load_dict_list(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _load_str_list(payload: object) -> list[str]:
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _maybe_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
