"""Tests for longitudinal progression aggregation."""

from __future__ import annotations

import pytest

from chesslab.openings import (
    DetectedOpening,
    OpeningGrouping,
    UNCLASSIFIED_OPENING,
    build_opening_filter_key,
)
from chesslab.persistence.review_history import ReviewSummary, TrainingPositionSummary
from chesslab.persistence.training_history import TrainingAttempt, TrainingAttemptOutcome
from chesslab.progression.progress_tracker import build_progression_snapshot
from chesslab.progression.weakness_progress import WeaknessMasteryState, WeaknessTrend
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
    training_positions: tuple[TrainingPositionSummary, ...] = (),
    opening: DetectedOpening = UNCLASSIFIED_OPENING,
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
        opening=opening,
        training_positions=training_positions,
    )


def _training_position(
    *,
    played_uci: str,
    best_move_uci: str,
    best_move_san: str,
    classification: MoveClassification,
    phase: GamePhase,
    cp_loss: int,
    lesson_themes: tuple[LessonTheme, ...],
) -> TrainingPositionSummary:
    return TrainingPositionSummary(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ply=0,
        played_san="played",
        played_uci=played_uci,
        best_move_san=best_move_san,
        best_move_uci=best_move_uci,
        best_pv_san=(best_move_san, "reply"),
        classification=classification,
        phase=phase,
        cp_loss=cp_loss,
        lesson_themes=lesson_themes,
    )


def _attempt(
    *,
    position: TrainingPositionSummary,
    attempted_at: str,
    outcome: TrainingAttemptOutcome,
) -> TrainingAttempt:
    return TrainingAttempt(
        weakness_id=position.weakness_id,
        attempted_at=attempted_at,
        outcome=outcome,
        game_id="g",
        reviewed_at="2026-03-20T12:00:00+00:00",
        fen=position.fen,
        ply=position.ply,
        source_path=None,
        expected_move_san=position.best_move_san,
        expected_move_uci=position.best_move_uci,
        original_move_san=position.played_san,
        original_move_uci=position.played_uci,
        submitted_move_san="move",
        submitted_move_uci="g1f3",
        cp_loss=12,
        move_rank=1 if outcome is TrainingAttemptOutcome.SUCCESS else None,
        phase=position.phase,
        lesson_themes=position.lesson_themes,
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
        assert snapshot.tracked_weaknesses == 0
        assert snapshot.training_attempt_count == 0
        assert snapshot.urgent_weaknesses == ()
        assert snapshot.training_themes == ()
        assert snapshot.opening_stats[0].key == "unclassified"
        assert snapshot.opening_stats[0].game_count == 3
        assert snapshot.opening_grouping is OpeningGrouping.ECO_CODE
        assert snapshot.selected_opening_key is None
        assert snapshot.is_filtered_view is False

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
        assert snapshot.is_filtered_view is False
        assert snapshot.opening_stats[0].key == "unclassified"

    def test_aggregates_training_mastery_and_repeating_weaknesses(self) -> None:
        mastered = _training_position(
            played_uci="e2e4",
            best_move_uci="d2d4",
            best_move_san="d4",
            classification=MoveClassification.MISTAKE,
            phase=GamePhase.OPENING,
            cp_loss=180,
            lesson_themes=(LessonTheme.PHASE_PATTERN,),
        )
        relapsing = _training_position(
            played_uci="g1f3",
            best_move_uci="e2e4",
            best_move_san="e4",
            classification=MoveClassification.BLUNDER,
            phase=GamePhase.MIDDLEGAME,
            cp_loss=240,
            lesson_themes=(LessonTheme.TURNING_POINT,),
        )

        snapshot = build_progression_snapshot(
            [
                _summary(
                    game_id="g2",
                    reviewed_at="2026-03-21T12:00:00+00:00",
                    avg_cp_loss=35.0,
                    training_positions=(relapsing,),
                ),
                _summary(
                    game_id="g1",
                    reviewed_at="2026-03-20T12:00:00+00:00",
                    avg_cp_loss=45.0,
                    training_positions=(mastered,),
                ),
            ],
            sample_size=20,
            minimum_games=1,
            training_attempts=[
                _attempt(
                    position=mastered,
                    attempted_at="2026-03-21T09:00:00+00:00",
                    outcome=TrainingAttemptOutcome.SUCCESS,
                ),
                _attempt(
                    position=mastered,
                    attempted_at="2026-03-22T09:00:00+00:00",
                    outcome=TrainingAttemptOutcome.SUCCESS,
                ),
                _attempt(
                    position=mastered,
                    attempted_at="2026-03-23T09:00:00+00:00",
                    outcome=TrainingAttemptOutcome.SUCCESS,
                ),
                _attempt(
                    position=relapsing,
                    attempted_at="2026-03-21T10:00:00+00:00",
                    outcome=TrainingAttemptOutcome.SUCCESS,
                ),
                _attempt(
                    position=relapsing,
                    attempted_at="2026-03-22T10:00:00+00:00",
                    outcome=TrainingAttemptOutcome.SUCCESS,
                ),
                _attempt(
                    position=relapsing,
                    attempted_at="2026-03-23T10:00:00+00:00",
                    outcome=TrainingAttemptOutcome.FAILED,
                ),
                _attempt(
                    position=relapsing,
                    attempted_at="2026-03-24T10:00:00+00:00",
                    outcome=TrainingAttemptOutcome.FAILED,
                ),
            ],
        )

        assert snapshot.tracked_weaknesses == 2
        assert snapshot.trained_weaknesses == 2
        assert snapshot.mastered_weaknesses == 1
        assert snapshot.relapsing_weaknesses == 1
        assert snapshot.training_attempt_count == 7
        assert snapshot.training_success_rate == pytest.approx(5 / 7 * 100.0)
        assert snapshot.urgent_weaknesses[0].weakness_id == relapsing.weakness_id
        assert snapshot.urgent_weaknesses[0].mastery_state is WeaknessMasteryState.RELAPSING
        assert snapshot.urgent_weaknesses[0].trend is WeaknessTrend.REPEATING
        assert snapshot.training_themes[0].theme is LessonTheme.TURNING_POINT
        assert snapshot.training_themes[0].trend is WeaknessTrend.REPEATING

    def test_builds_opening_ranking_and_filters_other_metrics(self) -> None:
        najdorf = DetectedOpening(
            eco="B90",
            name="Sicilian Defense: Najdorf Variation",
        )
        ruy_lopez = DetectedOpening(
            eco="C60",
            name="Ruy Lopez",
        )

        summaries = [
            _summary(
                game_id="g3",
                reviewed_at="2026-03-22T12:00:00+00:00",
                avg_cp_loss=55.0,
                classification_counts={MoveClassification.BLUNDER: 1},
                error_phase_counts={GamePhase.MIDDLEGAME: 1},
                opening=najdorf,
            ),
            _summary(
                game_id="g2",
                reviewed_at="2026-03-21T12:00:00+00:00",
                avg_cp_loss=35.0,
                classification_counts={MoveClassification.MISTAKE: 1},
                error_phase_counts={GamePhase.OPENING: 1},
                opening=najdorf,
            ),
            _summary(
                game_id="g1",
                reviewed_at="2026-03-20T12:00:00+00:00",
                avg_cp_loss=12.0,
                classification_counts={MoveClassification.GOOD: 2},
                opening=ruy_lopez,
            ),
        ]

        snapshot = build_progression_snapshot(
            summaries,
            sample_size=20,
            minimum_games=3,
            opening_grouping=OpeningGrouping.ECO_CODE,
            opening_filter_key=build_opening_filter_key(
                najdorf,
                OpeningGrouping.ECO_CODE,
            ),
        )

        assert snapshot.is_filtered_view is True
        assert snapshot.selected_opening_key == "eco:B90"
        assert snapshot.analysed_games == 2
        assert snapshot.has_enough_games is False
        assert snapshot.classification_counts[MoveClassification.BLUNDER] == 1
        assert snapshot.classification_counts[MoveClassification.MISTAKE] == 1
        assert snapshot.classification_counts[MoveClassification.GOOD] == 0
        assert snapshot.opening_stats[0].key == "eco:B90"
        assert snapshot.opening_stats[0].avg_cp_loss == 45.0
        assert snapshot.opening_stats[0].game_count == 2
        assert snapshot.opening_stats[1].key == "eco:C60"

    def test_aggregates_opening_families(self) -> None:
        snapshot = build_progression_snapshot(
            [
                _summary(
                    game_id="g3",
                    reviewed_at="2026-03-22T12:00:00+00:00",
                    avg_cp_loss=50.0,
                    opening=DetectedOpening(
                        eco="B90",
                        name="Sicilian Defense: Najdorf Variation",
                    ),
                ),
                _summary(
                    game_id="g2",
                    reviewed_at="2026-03-21T12:00:00+00:00",
                    avg_cp_loss=30.0,
                    opening=DetectedOpening(
                        eco="B20",
                        name="Sicilian Defense",
                    ),
                ),
                _summary(
                    game_id="g1",
                    reviewed_at="2026-03-20T12:00:00+00:00",
                    avg_cp_loss=10.0,
                    opening=UNCLASSIFIED_OPENING,
                ),
            ],
            sample_size=20,
            minimum_games=1,
            opening_grouping=OpeningGrouping.ECO_FAMILY,
        )

        assert snapshot.opening_grouping is OpeningGrouping.ECO_FAMILY
        assert snapshot.opening_stats[0].key == "family:B"
        assert snapshot.opening_stats[0].label == "B00-B99"
        assert snapshot.opening_stats[0].avg_cp_loss == 40.0
        assert snapshot.opening_stats[0].game_count == 2
        assert snapshot.opening_stats[1].key == "unclassified"
