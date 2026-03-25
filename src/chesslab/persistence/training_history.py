"""Persistence for targeted-training attempts tied to reviewed weaknesses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from chesslab.openings import DetectedOpening, UNCLASSIFIED_OPENING
from chesslab.persistence.game_storage import PersistenceError
from chesslab.review.game_reviewer import ErrorMotif, GamePhase, LessonTheme

_TRAINING_DIR = Path.home() / ".chesslab" / "training"
_TRAINING_HISTORY_FILE = _TRAINING_DIR / "history.json"
_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})


class TrainingAttemptOutcome(Enum):
    """Persisted result bucket for a training attempt."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class TrainingAttempt:
    """One persisted attempt on a reviewed training weakness."""

    weakness_id: str
    attempted_at: str
    outcome: TrainingAttemptOutcome
    game_id: str
    reviewed_at: str
    fen: str
    ply: int
    source_path: str | None
    expected_move_san: str
    expected_move_uci: str
    original_move_san: str
    original_move_uci: str
    submitted_move_san: str
    submitted_move_uci: str
    cp_loss: int
    move_rank: int | None
    phase: GamePhase
    lesson_themes: tuple[LessonTheme, ...] = ()
    error_motif: ErrorMotif = ErrorMotif.UNKNOWN
    opening: DetectedOpening = UNCLASSIFIED_OPENING

    def to_dict(self) -> dict[str, object]:
        """Serialise the attempt to JSON-compatible data."""
        payload: dict[str, object] = {
            "weakness_id": self.weakness_id,
            "attempted_at": self.attempted_at,
            "outcome": self.outcome.value,
            "game_id": self.game_id,
            "reviewed_at": self.reviewed_at,
            "fen": self.fen,
            "ply": self.ply,
            "source_path": self.source_path,
            "expected_move_san": self.expected_move_san,
            "expected_move_uci": self.expected_move_uci,
            "original_move_san": self.original_move_san,
            "original_move_uci": self.original_move_uci,
            "submitted_move_san": self.submitted_move_san,
            "submitted_move_uci": self.submitted_move_uci,
            "cp_loss": self.cp_loss,
            "move_rank": self.move_rank,
            "phase": self.phase.value,
            "lesson_themes": [theme.value for theme in self.lesson_themes],
            "error_motif": self.error_motif.value,
            "opening": self.opening.to_dict(),
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TrainingAttempt:
        """Rebuild an attempt from JSON payload."""
        return cls(
            weakness_id=str(payload.get("weakness_id", "")),
            attempted_at=str(payload.get("attempted_at", "")),
            outcome=_load_enum_value(
                payload.get("outcome"),
                TrainingAttemptOutcome,
                TrainingAttemptOutcome.FAILED,
            ),
            game_id=str(payload.get("game_id", "")),
            reviewed_at=str(payload.get("reviewed_at", "")),
            fen=str(payload.get("fen", "")),
            ply=int(payload.get("ply", 0)),
            source_path=_maybe_str(payload.get("source_path")),
            expected_move_san=str(payload.get("expected_move_san", "")),
            expected_move_uci=str(payload.get("expected_move_uci", "")),
            original_move_san=str(payload.get("original_move_san", "")),
            original_move_uci=str(payload.get("original_move_uci", "")),
            submitted_move_san=str(payload.get("submitted_move_san", "")),
            submitted_move_uci=str(payload.get("submitted_move_uci", "")),
            cp_loss=int(payload.get("cp_loss", 0)),
            move_rank=_maybe_int(payload.get("move_rank")),
            phase=_load_enum_value(
                payload.get("phase"),
                GamePhase,
                GamePhase.MIDDLEGAME,
            ),
            lesson_themes=tuple(
                _load_enum_sequence(payload.get("lesson_themes"), LessonTheme),
            ),
            error_motif=_load_enum_value(
                payload.get("error_motif"),
                ErrorMotif,
                ErrorMotif.UNKNOWN,
            ),
            opening=DetectedOpening.from_dict(payload.get("opening")),
        )


class TrainingHistoryStorage:
    """Stores and retrieves persisted targeted-training attempts."""

    def __init__(self, history_file: Path | None = None) -> None:
        self._history_file = history_file or _TRAINING_HISTORY_FILE

    def record_attempt(
        self,
        attempt: TrainingAttempt,
    ) -> TrainingAttempt:
        """Persist one evaluated training attempt."""
        attempts = self.list_recent_attempts(limit=None)
        attempts.append(attempt)
        ordered = sorted(
            attempts,
            key=lambda item: item.attempted_at,
            reverse=True,
        )
        self._write_all(ordered)
        return attempt

    def list_recent_attempts(
        self,
        limit: int | None = 100,
    ) -> list[TrainingAttempt]:
        """Return saved training attempts, newest first."""
        attempts = self._read_all()
        if limit is None:
            return attempts
        return attempts[:limit]

    def build_attempt(
        self,
        *,
        weakness_id: str,
        outcome: TrainingAttemptOutcome,
        game_id: str,
        reviewed_at: str,
        fen: str,
        ply: int,
        source_path: str | None,
        expected_move_san: str,
        expected_move_uci: str,
        original_move_san: str,
        original_move_uci: str,
        submitted_move_san: str,
        submitted_move_uci: str,
        cp_loss: int,
        move_rank: int | None,
        phase: GamePhase,
        lesson_themes: tuple[LessonTheme, ...] = (),
        error_motif: ErrorMotif = ErrorMotif.UNKNOWN,
        opening: DetectedOpening = UNCLASSIFIED_OPENING,
        attempted_at: datetime | None = None,
    ) -> TrainingAttempt:
        """Create a timestamped attempt object ready to be stored."""
        attempt_time = attempted_at or datetime.now(timezone.utc)
        return TrainingAttempt(
            weakness_id=weakness_id,
            attempted_at=attempt_time.isoformat(),
            outcome=outcome,
            game_id=game_id,
            reviewed_at=reviewed_at,
            fen=fen,
            ply=ply,
            source_path=source_path,
            expected_move_san=expected_move_san,
            expected_move_uci=expected_move_uci,
            original_move_san=original_move_san,
            original_move_uci=original_move_uci,
            submitted_move_san=submitted_move_san,
            submitted_move_uci=submitted_move_uci,
            cp_loss=cp_loss,
            move_rank=move_rank,
            phase=phase,
            lesson_themes=lesson_themes,
            error_motif=error_motif,
            opening=opening,
        )

    def _read_all(self) -> list[TrainingAttempt]:
        if not self._history_file.is_file():
            return []

        try:
            with self._history_file.open() as fh:
                payload = json.load(fh)
        except Exception as exc:
            raise PersistenceError(
                f"Failed to read training history: {self._history_file}: {exc}",
            ) from exc

        if not isinstance(payload, dict):
            raise PersistenceError(
                f"Invalid training history format: {self._history_file}",
            )

        schema_version = payload.get("schema_version")
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise PersistenceError(
                f"Unsupported training history schema: {schema_version}",
            )

        raw_attempts = payload.get("attempts", [])
        if not isinstance(raw_attempts, list):
            raise PersistenceError(
                f"Invalid training history payload: {self._history_file}",
            )

        attempts = [TrainingAttempt.from_dict(item) for item in raw_attempts if isinstance(item, dict)]
        return sorted(attempts, key=lambda item: item.attempted_at, reverse=True)

    def _write_all(self, attempts: list[TrainingAttempt]) -> None:
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "attempts": [attempt.to_dict() for attempt in attempts],
        }
        try:
            with self._history_file.open("w") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            raise PersistenceError(
                f"Failed to write training history: {self._history_file}: {exc}",
            ) from exc


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


def _maybe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
