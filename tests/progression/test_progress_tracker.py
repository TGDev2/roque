"""Tests for longitudinal progression aggregation."""

from __future__ import annotations

import pytest

from chesslab.persistence.review_history import ReviewSummary
from chesslab.progression.progress_tracker import build_progression_snapshot
from chesslab.review.game_reviewer import (
    ErrorMotif,
    GamePhase,
    LessonTheme,
    MoveClassification,
)


def _summary(
    *,
    game_id: str,
    reviewed_at: str,
    avg_cp_loss: float,
    classification_counts: dict[MoveClassification, int] | None = None,
    error_phase_counts: dict[GamePhase, int] | None = None,
    lesson_theme_counts: dict[LessonTheme, int] | None = None,
    error_motif_counts: dict[ErrorMotif, int] | None = None,
) -> ReviewSummary:
    cls_counts = {cls: 0 for cls in (
        MoveClassification.EXCELLENT,
        MoveClassification.GOOD,
        MoveClassification.INACCURACY,
        MoveClassification.MISTAKE,
        MoveClassification.BLUNDER,
    )}
    if classification_counts:
        cls_counts.update(classification_counts)

    phase_counts = {phase: 0 for phase in GamePhase}
    if error_phase_counts:
        phase_counts.update(error_phase_counts)

    theme_counts = {theme: 0 for theme in LessonTheme}
    if lesson_theme_counts:
        theme_counts.update(lesson_theme_counts)

    motif_counts = {motif: 0 for motif in ErrorMotif}
    if error_motif_counts:
        motif_counts.update(error_motif_counts)

    return ReviewSummary(
        game_id=game_id,
        reviewed_at=reviewed_at,
        move_count=sum(cls_counts.values()),
        avg_cp_loss=avg_cp_loss,
        classification_counts=cls_counts,
        error_phase_counts=phase_counts,
        lesson_theme_counts=theme_counts,
        error_motif_counts=motif_counts,
    )


class TestBuildProgressionSnapshot:
    def test_aggregates_classifications_trend_themes_and_error_phases(self) -> None:
        newest_first = [
            _summary(
                game_id="g3",
                reviewed_at="2026-03-22T12:00:00+00:00",
                avg_cp_loss=30.0,
                classification_counts={
                    MoveClassification.EXCELLENT: 1,
                    MoveClassification.GOOD: 1,
                    MoveClassification.INACCURACY: 1,
                },
                error_phase_counts={GamePhase.ENDGAME: 1},
                lesson_theme_counts={
                    LessonTheme.TURNING_POINT: 1,
                    LessonTheme.ADVANTAGE_MANAGEMENT: 1,
                },
                error_motif_counts={ErrorMotif.MISSED_TACTIC: 1},
            ),
            _summary(
                game_id="g2",
                reviewed_at="2026-03-21T12:00:00+00:00",
                avg_cp_loss=40.0,
                classification_counts={
                    MoveClassification.EXCELLENT: 1,
                    MoveClassification.MISTAKE: 1,
                },
                error_phase_counts={GamePhase.MIDDLEGAME: 1},
                lesson_theme_counts={LessonTheme.PHASE_PATTERN: 2},
                error_motif_counts={ErrorMotif.PIECE_ACTIVITY: 1},
            ),
            _summary(
                game_id="g1",
                reviewed_at="2026-03-20T12:00:00+00:00",
                avg_cp_loss=60.0,
                classification_counts={
                    MoveClassification.GOOD: 2,
                    MoveClassification.BLUNDER: 1,
                },
                error_phase_counts={GamePhase.OPENING: 1},
                lesson_theme_counts={LessonTheme.TURNING_POINT: 1},
                error_motif_counts={
                    ErrorMotif.MISSED_TACTIC: 1,
                    ErrorMotif.KING_SAFETY: 1,
                },
            ),
        ]

        snapshot = build_progression_snapshot(
            newest_first,
            sample_size=20,
            minimum_games=3,
        )

        assert snapshot.has_enough_games is True
        assert snapshot.analysed_games == 3
        assert snapshot.classification_counts[MoveClassification.EXCELLENT] == 2
        assert snapshot.classification_counts[MoveClassification.GOOD] == 3
        assert snapshot.classification_percentages[MoveClassification.EXCELLENT] == 25.0
        assert snapshot.classification_percentages[MoveClassification.GOOD] == 37.5
        assert snapshot.cp_loss_trend[0].avg_cp_loss == 60.0
        assert snapshot.cp_loss_trend[-1].avg_cp_loss == 30.0
        assert snapshot.avg_cp_loss_delta == -30.0
        assert snapshot.top_themes[0].theme is LessonTheme.PHASE_PATTERN
        assert snapshot.top_themes[0].count == 2
        assert snapshot.top_themes[1].theme is LessonTheme.TURNING_POINT
        assert snapshot.top_error_motifs[0].motif is ErrorMotif.MISSED_TACTIC
        assert snapshot.top_error_motifs[0].count == 2
        assert snapshot.top_error_motifs[0].percentage == 50.0
        assert snapshot.error_phases[0].percentage == pytest.approx(100.0 / 3.0)
        assert snapshot.error_phases[1].percentage == pytest.approx(100.0 / 3.0)
        assert snapshot.error_phases[2].percentage == pytest.approx(100.0 / 3.0)

    def test_reports_when_more_games_are_needed(self) -> None:
        snapshot = build_progression_snapshot(
            [
                _summary(
                    game_id="g1",
                    reviewed_at="2026-03-20T12:00:00+00:00",
                    avg_cp_loss=60.0,
                ),
                _summary(
                    game_id="g2",
                    reviewed_at="2026-03-21T12:00:00+00:00",
                    avg_cp_loss=40.0,
                ),
            ],
            sample_size=20,
            minimum_games=3,
        )

        assert snapshot.has_enough_games is False
        assert snapshot.games_remaining == 1
