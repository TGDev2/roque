"""Aggregates persisted review summaries into a learning progression snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from chesslab.persistence.review_history import ReviewHistoryStorage, ReviewSummary
from chesslab.review.game_reviewer import GamePhase, LessonTheme, MoveClassification

_TRACKED_CLASSIFICATIONS: tuple[MoveClassification, ...] = (
    MoveClassification.EXCELLENT,
    MoveClassification.GOOD,
    MoveClassification.INACCURACY,
    MoveClassification.MISTAKE,
    MoveClassification.BLUNDER,
)


@dataclass(frozen=True)
class ProgressionTrendPoint:
    """One per-game point in the centipawn-loss trend."""

    reviewed_at: str
    avg_cp_loss: float


@dataclass(frozen=True)
class ProgressionThemeStat:
    """Frequency of a recurring lesson theme."""

    theme: LessonTheme
    count: int


@dataclass(frozen=True)
class ProgressionPhaseStat:
    """Frequency of errors in one game phase."""

    phase: GamePhase
    count: int
    percentage: float


@dataclass(frozen=True)
class ProgressionSnapshot:
    """Aggregated view over the N most recent analysed games."""

    sample_size: int
    analysed_games: int
    minimum_games: int
    classification_counts: dict[MoveClassification, int]
    classification_percentages: dict[MoveClassification, float]
    cp_loss_trend: tuple[ProgressionTrendPoint, ...]
    avg_cp_loss_delta: float
    sample_avg_cp_loss: float
    top_themes: tuple[ProgressionThemeStat, ...]
    error_phases: tuple[ProgressionPhaseStat, ...]

    @property
    def has_enough_games(self) -> bool:
        """Whether the snapshot has enough history for longitudinal insights."""
        return self.analysed_games >= self.minimum_games

    @property
    def games_remaining(self) -> int:
        """How many more analysed games are needed for a full dashboard."""
        return max(0, self.minimum_games - self.analysed_games)


class ProgressTracker:
    """Builds a learning-progress snapshot from stored review summaries."""

    def __init__(
        self,
        review_history: ReviewHistoryStorage | None = None,
        *,
        minimum_games: int = 3,
    ) -> None:
        self._review_history = review_history or ReviewHistoryStorage()
        self._minimum_games = minimum_games

    def build_snapshot(self, limit: int = 20) -> ProgressionSnapshot:
        """Aggregate the most recent review summaries."""
        summaries = self._review_history.list_recent_reviews(limit=limit)
        return build_progression_snapshot(
            summaries,
            sample_size=limit,
            minimum_games=self._minimum_games,
        )


def build_progression_snapshot(
    summaries: list[ReviewSummary],
    *,
    sample_size: int,
    minimum_games: int = 3,
) -> ProgressionSnapshot:
    """Aggregate already-persisted review summaries."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    classification_counts = {cls: 0 for cls in _TRACKED_CLASSIFICATIONS}
    theme_counts = {theme: 0 for theme in LessonTheme}
    error_phase_counts = {phase: 0 for phase in GamePhase}

    for summary in summaries:
        for cls in _TRACKED_CLASSIFICATIONS:
            classification_counts[cls] += summary.classification_counts.get(cls, 0)
        for theme in LessonTheme:
            theme_counts[theme] += summary.lesson_theme_counts.get(theme, 0)
        for phase in GamePhase:
            error_phase_counts[phase] += summary.error_phase_counts.get(phase, 0)

    total_classifications = sum(classification_counts.values())
    classification_percentages = {
        cls: (
            classification_counts[cls] / total_classifications * 100.0
            if total_classifications
            else 0.0
        )
        for cls in _TRACKED_CLASSIFICATIONS
    }

    chronological = list(reversed(summaries))
    cp_loss_trend = tuple(
        ProgressionTrendPoint(
            reviewed_at=summary.reviewed_at,
            avg_cp_loss=summary.avg_cp_loss,
        )
        for summary in chronological
    )
    avg_cp_loss_delta = (
        cp_loss_trend[-1].avg_cp_loss - cp_loss_trend[0].avg_cp_loss
        if len(cp_loss_trend) >= 2
        else 0.0
    )
    sample_avg_cp_loss = (
        sum(point.avg_cp_loss for point in cp_loss_trend) / len(cp_loss_trend)
        if cp_loss_trend
        else 0.0
    )

    top_themes = tuple(
        ProgressionThemeStat(theme=theme, count=count)
        for theme, count in sorted(
            theme_counts.items(),
            key=lambda item: (-item[1], item[0].value),
        )
        if count > 0
    )[:3]

    total_errors = sum(error_phase_counts.values())
    error_phases = tuple(
        ProgressionPhaseStat(
            phase=phase,
            count=error_phase_counts[phase],
            percentage=(
                error_phase_counts[phase] / total_errors * 100.0
                if total_errors
                else 0.0
            ),
        )
        for phase in GamePhase
    )

    return ProgressionSnapshot(
        sample_size=sample_size,
        analysed_games=len(summaries),
        minimum_games=minimum_games,
        classification_counts=classification_counts,
        classification_percentages=classification_percentages,
        cp_loss_trend=cp_loss_trend,
        avg_cp_loss_delta=avg_cp_loss_delta,
        sample_avg_cp_loss=sample_avg_cp_loss,
        top_themes=top_themes,
        error_phases=error_phases,
    )
