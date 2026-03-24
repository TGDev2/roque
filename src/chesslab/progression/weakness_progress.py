"""Aggregate review weaknesses and training attempts into mastery signals."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from statistics import fmean

from chesslab.persistence.review_history import ReviewSummary, TrainingPositionSummary
from chesslab.persistence.training_history import TrainingAttempt, TrainingAttemptOutcome
from chesslab.review.game_reviewer import ErrorMotif, GamePhase, LessonTheme, MoveClassification

_THEME_LABELS: dict[LessonTheme, str] = {
    LessonTheme.PHASE_PATTERN: "Phase pattern",
    LessonTheme.TURNING_POINT: "Turning point",
    LessonTheme.ADVANTAGE_MANAGEMENT: "Advantage management",
}
_PHASE_LABELS: dict[GamePhase, str] = {
    GamePhase.OPENING: "Opening",
    GamePhase.MIDDLEGAME: "Middlegame",
    GamePhase.ENDGAME: "Endgame",
}
_POSITION_PRIORITY: dict[MoveClassification, int] = {
    MoveClassification.BLUNDER: 220,
    MoveClassification.MISTAKE: 150,
    MoveClassification.INACCURACY: 80,
    MoveClassification.GOOD: 10,
    MoveClassification.EXCELLENT: 0,
    MoveClassification.FORCED: 0,
}
_OUTCOME_SCORES: dict[TrainingAttemptOutcome, float] = {
    TrainingAttemptOutcome.SUCCESS: 1.0,
    TrainingAttemptOutcome.PARTIAL: 0.55,
    TrainingAttemptOutcome.FAILED: 0.0,
}


class WeaknessMasteryState(Enum):
    """Coarse mastery state for a recurring weakness."""

    UNTRAINED = "untrained"
    ACTIVE = "active"
    IMPROVING = "improving"
    MASTERED = "mastered"
    RELAPSING = "relapsing"


class WeaknessTrend(Enum):
    """Short trend label for a weakness or area."""

    IMPROVING = "improving"
    STABLE = "stable"
    REPEATING = "repeating"


@dataclass(frozen=True)
class WeaknessProgressStat:
    """Aggregated training and mastery view for one recurring weakness."""

    weakness_id: str
    label: str
    phase: GamePhase
    lesson_themes: tuple[LessonTheme, ...]
    error_motif: ErrorMotif
    occurrence_count: int
    attempt_count: int
    success_count: int
    partial_count: int
    failed_count: int
    success_rate: float
    mastery_score: float
    recent_score: float
    success_streak: int
    struggle_streak: int
    review_priority_score: int
    urgency_score: int
    mastery_state: WeaknessMasteryState
    trend: WeaknessTrend
    summary: str

    @property
    def urgency_delta(self) -> int:
        """How much training history changes priority versus review severity."""
        return self.urgency_score - self.review_priority_score

    @property
    def needs_attention(self) -> bool:
        """Whether this weakness should remain visible in future sessions."""
        return self.mastery_state in {
            WeaknessMasteryState.UNTRAINED,
            WeaknessMasteryState.ACTIVE,
            WeaknessMasteryState.RELAPSING,
        }


@dataclass(frozen=True)
class TrainingThemeProgressStat:
    """Training follow-up for one recurring lesson theme."""

    theme: LessonTheme
    weakness_count: int
    urgent_count: int
    mastered_count: int
    attempt_count: int
    success_rate: float
    trend: WeaknessTrend


@dataclass
class _WeaknessAggregate:
    """Mutable aggregation while combining reviews and attempts."""

    position: TrainingPositionSummary
    occurrence_count: int = 0
    max_priority: int = 0
    lesson_themes: set[LessonTheme] = field(default_factory=set)
    error_motif: ErrorMotif = ErrorMotif.UNKNOWN


def build_weakness_progress_stats(
    summaries: list[ReviewSummary],
    training_attempts: list[TrainingAttempt] | None = None,
) -> tuple[WeaknessProgressStat, ...]:
    """Aggregate weakness-level mastery from reviewed positions and attempts."""
    attempts = training_attempts or []
    aggregates = _build_catalog(summaries)
    if not aggregates:
        return ()

    attempts_by_id: dict[str, list[TrainingAttempt]] = defaultdict(list)
    for attempt in attempts:
        if attempt.weakness_id in aggregates:
            attempts_by_id[attempt.weakness_id].append(attempt)

    stats: list[WeaknessProgressStat] = []
    for weakness_id, aggregate in aggregates.items():
        weakness_attempts = sorted(
            attempts_by_id.get(weakness_id, []),
            key=lambda item: item.attempted_at,
        )
        scores = [
            _OUTCOME_SCORES.get(attempt.outcome, 0.0)
            for attempt in weakness_attempts
        ]
        attempt_count = len(weakness_attempts)
        success_count = sum(
            attempt.outcome is TrainingAttemptOutcome.SUCCESS
            for attempt in weakness_attempts
        )
        partial_count = sum(
            attempt.outcome is TrainingAttemptOutcome.PARTIAL
            for attempt in weakness_attempts
        )
        failed_count = sum(
            attempt.outcome is TrainingAttemptOutcome.FAILED
            for attempt in weakness_attempts
        )
        success_rate = (
            success_count / attempt_count * 100.0
            if attempt_count
            else 0.0
        )
        mastery_score = _weighted_mastery_score(scores)
        recent_scores = scores[-3:]
        recent_score = fmean(recent_scores) if recent_scores else 0.0
        previous_scores = scores[:-2]
        previous_avg = fmean(previous_scores) if previous_scores else recent_score
        success_streak = _success_streak(weakness_attempts)
        struggle_streak = _struggle_streak(weakness_attempts)
        mastery_state, trend = _classify_mastery(
            scores=scores,
            mastery_score=mastery_score,
            recent_score=recent_score,
            previous_avg=previous_avg,
            success_streak=success_streak,
            struggle_streak=struggle_streak,
        )
        review_priority = aggregate.max_priority + max(0, aggregate.occurrence_count - 1) * 36
        urgency_score = max(0, review_priority + _urgency_delta(
            mastery_state=mastery_state,
            mastery_score=mastery_score,
            success_streak=success_streak,
            struggle_streak=struggle_streak,
        ))

        position = aggregate.position
        themes = tuple(sorted(aggregate.lesson_themes, key=lambda item: item.value))
        stats.append(WeaknessProgressStat(
            weakness_id=weakness_id,
            label=_weakness_label(position, themes=themes),
            phase=position.phase,
            lesson_themes=themes,
            error_motif=aggregate.error_motif,
            occurrence_count=aggregate.occurrence_count,
            attempt_count=attempt_count,
            success_count=success_count,
            partial_count=partial_count,
            failed_count=failed_count,
            success_rate=success_rate,
            mastery_score=mastery_score,
            recent_score=recent_score,
            success_streak=success_streak,
            struggle_streak=struggle_streak,
            review_priority_score=review_priority,
            urgency_score=urgency_score,
            mastery_state=mastery_state,
            trend=trend,
            summary=_weakness_summary(
                attempt_count=attempt_count,
                success_rate=success_rate,
                mastery_state=mastery_state,
                trend=trend,
            ),
        ))

    return tuple(sorted(
        stats,
        key=lambda item: (
            -item.urgency_score,
            -item.occurrence_count,
            item.label,
        ),
    ))


def build_theme_progress_stats(
    weakness_stats: tuple[WeaknessProgressStat, ...],
) -> tuple[TrainingThemeProgressStat, ...]:
    """Aggregate weakness mastery by lesson theme."""
    by_theme: dict[LessonTheme, list[WeaknessProgressStat]] = defaultdict(list)
    for stat in weakness_stats:
        for theme in stat.lesson_themes:
            by_theme[theme].append(stat)

    theme_stats: list[TrainingThemeProgressStat] = []
    for theme, items in by_theme.items():
        attempt_count = sum(item.attempt_count for item in items)
        success_count = sum(item.success_count for item in items)
        urgent_count = sum(item.needs_attention for item in items)
        mastered_count = sum(
            item.mastery_state is WeaknessMasteryState.MASTERED
            for item in items
        )
        success_rate = (
            success_count / attempt_count * 100.0
            if attempt_count
            else 0.0
        )
        theme_stats.append(TrainingThemeProgressStat(
            theme=theme,
            weakness_count=len(items),
            urgent_count=urgent_count,
            mastered_count=mastered_count,
            attempt_count=attempt_count,
            success_rate=success_rate,
            trend=_aggregate_trend(items),
        ))

    return tuple(sorted(
        theme_stats,
        key=lambda item: (
            -item.urgent_count,
            -item.weakness_count,
            item.theme.value,
        ),
    ))


def _build_catalog(
    summaries: list[ReviewSummary],
) -> dict[str, _WeaknessAggregate]:
    aggregates: dict[str, _WeaknessAggregate] = {}
    for summary_index, summary in enumerate(summaries):
        recency_bonus = max(0, 36 - summary_index * 6)
        for position in summary.training_positions:
            weakness_id = position.weakness_id
            priority = _position_priority(position) + recency_bonus
            existing = aggregates.get(weakness_id)
            if existing is None:
                existing = _WeaknessAggregate(
                    position=position,
                    error_motif=position.error_motif,
                )
                aggregates[weakness_id] = existing
            if priority >= existing.max_priority:
                existing.position = position
                existing.error_motif = position.error_motif
            existing.occurrence_count += 1
            existing.max_priority = max(existing.max_priority, priority)
            existing.lesson_themes.update(position.lesson_themes)
    return aggregates


def _position_priority(position: TrainingPositionSummary) -> int:
    severity = _POSITION_PRIORITY.get(position.classification, 0)
    cp_loss = min(position.cp_loss, 350)
    pv_bonus = min(len(position.best_pv_san), 4) * 12
    theme_bonus = len(position.lesson_themes) * 20
    return severity + cp_loss + pv_bonus + theme_bonus


def _weighted_mastery_score(scores: list[float]) -> float:
    if not scores:
        return 0.0
    weights = list(range(1, len(scores) + 1))
    weighted_total = sum(score * weight for score, weight in zip(scores, weights))
    return weighted_total / sum(weights)


def _success_streak(attempts: list[TrainingAttempt]) -> int:
    streak = 0
    for attempt in reversed(attempts):
        if attempt.outcome is not TrainingAttemptOutcome.SUCCESS:
            break
        streak += 1
    return streak


def _struggle_streak(attempts: list[TrainingAttempt]) -> int:
    streak = 0
    for attempt in reversed(attempts):
        if attempt.outcome is TrainingAttemptOutcome.SUCCESS:
            break
        streak += 1
    return streak


def _classify_mastery(
    *,
    scores: list[float],
    mastery_score: float,
    recent_score: float,
    previous_avg: float,
    success_streak: int,
    struggle_streak: int,
) -> tuple[WeaknessMasteryState, WeaknessTrend]:
    attempt_count = len(scores)
    if attempt_count == 0:
        return WeaknessMasteryState.UNTRAINED, WeaknessTrend.STABLE

    recent_window = scores[-2:]
    recent_window_avg = fmean(recent_window) if recent_window else recent_score
    older_scores = scores[:-2]
    older_avg = fmean(older_scores) if older_scores else previous_avg

    if older_scores and older_avg >= 0.75 and recent_window_avg <= 0.35:
        return WeaknessMasteryState.RELAPSING, WeaknessTrend.REPEATING

    if (
        attempt_count >= 3
        and success_streak >= 2
        and recent_score >= 0.85
        and mastery_score >= 0.75
    ):
        if older_scores and recent_score >= older_avg + 0.1:
            return WeaknessMasteryState.MASTERED, WeaknessTrend.IMPROVING
        return WeaknessMasteryState.MASTERED, WeaknessTrend.STABLE

    if (
        attempt_count >= 2
        and recent_score >= 0.65
        and (
            success_streak >= 1
            or recent_score >= previous_avg + 0.15
        )
    ):
        return WeaknessMasteryState.IMPROVING, WeaknessTrend.IMPROVING

    if struggle_streak >= 2 or (attempt_count >= 3 and recent_score < 0.4):
        return WeaknessMasteryState.ACTIVE, WeaknessTrend.REPEATING

    return WeaknessMasteryState.ACTIVE, WeaknessTrend.STABLE


def _urgency_delta(
    *,
    mastery_state: WeaknessMasteryState,
    mastery_score: float,
    success_streak: int,
    struggle_streak: int,
) -> int:
    state_bonus = {
        WeaknessMasteryState.UNTRAINED: 120,
        WeaknessMasteryState.ACTIVE: 70,
        WeaknessMasteryState.IMPROVING: 20,
        WeaknessMasteryState.MASTERED: -190,
        WeaknessMasteryState.RELAPSING: 160,
    }[mastery_state]
    state_bonus += min(struggle_streak, 3) * 24
    state_bonus -= min(success_streak, 3) * 28
    state_bonus -= int(mastery_score * 40)
    return state_bonus


def _weakness_label(
    position: TrainingPositionSummary,
    *,
    themes: tuple[LessonTheme, ...],
) -> str:
    parts: list[str] = []
    if themes:
        parts.append(_THEME_LABELS[themes[0]])
    parts.append(_PHASE_LABELS[position.phase])
    if position.best_move_san:
        parts.append(f"best {position.best_move_san}")
    return " | ".join(parts)


def _weakness_summary(
    *,
    attempt_count: int,
    success_rate: float,
    mastery_state: WeaknessMasteryState,
    trend: WeaknessTrend,
) -> str:
    if attempt_count == 0:
        return "Not trained yet."

    if mastery_state is WeaknessMasteryState.MASTERED:
        return (
            f"{attempt_count} attempt{'s' if attempt_count != 1 else ''}, "
            f"{success_rate:.0f}% good answers, mastered for now."
        )
    if mastery_state is WeaknessMasteryState.RELAPSING:
        return (
            f"{attempt_count} attempt{'s' if attempt_count != 1 else ''}, "
            f"{success_rate:.0f}% good answers, repeating lately."
        )
    if trend is WeaknessTrend.IMPROVING:
        return (
            f"{attempt_count} attempt{'s' if attempt_count != 1 else ''}, "
            f"{success_rate:.0f}% good answers, improving."
        )
    return (
        f"{attempt_count} attempt{'s' if attempt_count != 1 else ''}, "
        f"{success_rate:.0f}% good answers, still fragile."
    )


def _aggregate_trend(
    items: list[WeaknessProgressStat],
) -> WeaknessTrend:
    counts = Counter(item.trend for item in items)
    if counts[WeaknessTrend.REPEATING] > 0 and (
        counts[WeaknessTrend.REPEATING] >= counts[WeaknessTrend.IMPROVING]
    ):
        return WeaknessTrend.REPEATING
    if counts[WeaknessTrend.IMPROVING] > 0:
        return WeaknessTrend.IMPROVING
    return WeaknessTrend.STABLE
