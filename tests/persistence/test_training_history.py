"""Tests for persisted targeted-training attempts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import chess

from chesslab.persistence.training_history import (
    TrainingAttempt,
    TrainingAttemptOutcome,
    TrainingHistoryStorage,
)
from chesslab.review.game_reviewer import ErrorMotif, GamePhase, LessonTheme


def _attempt(
    *,
    weakness_id: str = "weakness-1",
    attempted_at: str = "2026-03-24T12:00:00+00:00",
    outcome: TrainingAttemptOutcome = TrainingAttemptOutcome.SUCCESS,
) -> TrainingAttempt:
    return TrainingAttempt(
        weakness_id=weakness_id,
        attempted_at=attempted_at,
        outcome=outcome,
        game_id="game-1",
        reviewed_at="2026-03-20T12:00:00+00:00",
        fen=chess.Board().fen(),
        ply=12,
        source_path="/tmp/game-1.pgn",
        expected_move_san="d4",
        expected_move_uci="d2d4",
        original_move_san="e4",
        original_move_uci="e2e4",
        submitted_move_san="Nf3",
        submitted_move_uci="g1f3",
        cp_loss=18,
        move_rank=2,
        phase=GamePhase.MIDDLEGAME,
        lesson_themes=(LessonTheme.TURNING_POINT,),
        error_motif=ErrorMotif.MISSED_TACTIC,
    )


class TestTrainingHistoryStorage:
    def test_record_attempt_persists_context_and_result(self, tmp_path: Path) -> None:
        history_file = tmp_path / "training.json"
        storage = TrainingHistoryStorage(history_file=history_file)

        storage.record_attempt(_attempt())

        saved = storage.list_recent_attempts(limit=None)

        assert history_file.exists()
        assert len(saved) == 1
        assert saved[0].weakness_id == "weakness-1"
        assert saved[0].outcome is TrainingAttemptOutcome.SUCCESS
        assert saved[0].phase is GamePhase.MIDDLEGAME
        assert saved[0].lesson_themes == (LessonTheme.TURNING_POINT,)
        assert saved[0].submitted_move_uci == "g1f3"

    def test_reads_schema_v1_payload(self, tmp_path: Path) -> None:
        history_file = tmp_path / "training.json"
        payload = {
            "schema_version": 1,
            "attempts": [
                {
                    "weakness_id": "legacy",
                    "attempted_at": "2026-03-24T12:00:00+00:00",
                    "outcome": "partial",
                    "game_id": "game-1",
                    "reviewed_at": "2026-03-20T12:00:00+00:00",
                    "fen": chess.Board().fen(),
                    "ply": 8,
                    "source_path": None,
                    "expected_move_san": "d4",
                    "expected_move_uci": "d2d4",
                    "original_move_san": "e4",
                    "original_move_uci": "e2e4",
                    "submitted_move_san": "Nf3",
                    "submitted_move_uci": "g1f3",
                    "cp_loss": 42,
                    "move_rank": None,
                    "phase": "opening",
                    "lesson_themes": ["turning_point"],
                    "error_motif": "tactique_ratee",
                },
            ],
        }
        history_file.write_text(json.dumps(payload))

        saved = TrainingHistoryStorage(history_file=history_file).list_recent_attempts(
            limit=None,
        )

        assert len(saved) == 1
        assert saved[0].outcome is TrainingAttemptOutcome.PARTIAL
        assert saved[0].phase is GamePhase.OPENING
        assert saved[0].error_motif is ErrorMotif.MISSED_TACTIC

    def test_build_attempt_timestamps_automatically(self, tmp_path: Path) -> None:
        storage = TrainingHistoryStorage(history_file=tmp_path / "training.json")

        attempt = storage.build_attempt(
            weakness_id="w",
            outcome=TrainingAttemptOutcome.FAILED,
            game_id="g",
            reviewed_at="2026-03-20T12:00:00+00:00",
            fen=chess.Board().fen(),
            ply=4,
            source_path=None,
            expected_move_san="d4",
            expected_move_uci="d2d4",
            original_move_san="e4",
            original_move_uci="e2e4",
            submitted_move_san="f3",
            submitted_move_uci="f2f3",
            cp_loss=180,
            move_rank=None,
            phase=GamePhase.OPENING,
            lesson_themes=(LessonTheme.PHASE_PATTERN,),
            attempted_at=datetime(2026, 3, 24, 15, tzinfo=timezone.utc),
        )

        assert attempt.attempted_at == "2026-03-24T15:00:00+00:00"
        assert attempt.outcome is TrainingAttemptOutcome.FAILED
