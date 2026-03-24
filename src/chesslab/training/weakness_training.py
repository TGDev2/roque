"""Turn persisted review weaknesses into targeted training exercises."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

import chess

from chesslab.analysis import AnalysisConfig, AnalysisResult, Evaluation, PositionAnalyser
from chesslab.persistence.review_history import ReviewSummary, TrainingPositionSummary
from chesslab.persistence.training_history import TrainingAttempt, TrainingAttemptOutcome
from chesslab.progression import build_progression_snapshot
from chesslab.progression.weakness_progress import (
    WeaknessMasteryState,
    WeaknessProgressStat,
    WeaknessTrend,
    build_weakness_progress_stats,
)
from chesslab.review.game_reviewer import (
    GamePhase,
    GameReview,
    LessonTheme,
    MoveClassification,
    ReviewConfig,
)

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
_FEEDBACK_MESSAGES: dict[str, str] = {
    "best": "Bon coup: c'etait le meilleur choix.",
    "top3": "Bon coup: cette idee faisait partie des 3 meilleurs choix.",
    "better": "Il y avait mieux: l'idee etait jouable, mais moins precise.",
    "significant": "Erreur significative: la position demandait une reponse plus precise.",
}
_SEVERITY_SCORES: dict[MoveClassification, int] = {
    MoveClassification.BLUNDER: 260,
    MoveClassification.MISTAKE: 180,
    MoveClassification.INACCURACY: 90,
    MoveClassification.GOOD: 10,
    MoveClassification.EXCELLENT: 0,
    MoveClassification.FORCED: 0,
}
_PHASE_PRIORITY_SCORES = (130, 85, 45)
_THEME_PRIORITY_SCORES = (160, 110, 70)
_MATE_CP = 100_000


class TrainingFeedbackLabel(Enum):
    """User-facing quality bucket for a played training move."""

    GOOD = "good"
    BETTER_AVAILABLE = "better_available"
    SIGNIFICANT_ERROR = "significant_error"


@dataclass(frozen=True)
class TrainingExercise:
    """One exercise selected from the user's persisted reviews."""

    game_id: str
    reviewed_at: str
    source_path: str | None
    position: TrainingPositionSummary
    relevance_score: int
    focus_reason: str
    focus_theme: LessonTheme | None = None
    focus_phase: GamePhase | None = None
    attempt_count: int = 0
    success_rate: float = 0.0
    mastery_state: WeaknessMasteryState = WeaknessMasteryState.UNTRAINED
    weakness_trend: WeaknessTrend = WeaknessTrend.STABLE
    mastery_summary: str = "Not trained yet."


@dataclass(frozen=True)
class TrainingMoveFeedback:
    """Immediate engine-backed feedback for a move played in training mode."""

    label: TrainingFeedbackLabel
    played_san: str
    played_uci: str
    cp_loss: int
    move_rank: int | None
    best_move_san: str
    best_move_uci: str
    best_line_san: tuple[str, ...]
    top_move_sans: tuple[str, ...]
    message: str


class WeaknessTrainingService:
    """Builds training sessions and evaluates candidate moves."""

    def __init__(
        self,
        review_config: ReviewConfig | None = None,
        *,
        analyser_factory: Callable[[AnalysisConfig], PositionAnalyser] | None = None,
    ) -> None:
        self._review_config = review_config or ReviewConfig()
        self._analyser_factory = analyser_factory or PositionAnalyser

    def build_session(
        self,
        summaries: list[ReviewSummary],
        *,
        sample_size: int = 20,
        focus_review: GameReview | None = None,
        session_size: int = 5,
        training_attempts: list[TrainingAttempt] | None = None,
    ) -> tuple[TrainingExercise, ...]:
        """Select the most relevant training exercises from persisted reviews."""
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if session_size <= 0:
            raise ValueError("session_size must be positive")

        relevant_summaries = summaries[:sample_size]
        if not relevant_summaries:
            return ()

        snapshot = build_progression_snapshot(
            relevant_summaries,
            sample_size=sample_size,
            minimum_games=1,
            training_attempts=training_attempts,
        )
        weakness_stats = {
            stat.weakness_id: stat
            for stat in build_weakness_progress_stats(
                relevant_summaries,
                training_attempts=training_attempts,
            )
        }
        theme_priority = _merge_priorities(
            _focus_review_theme_priority(focus_review),
            [stat.theme for stat in snapshot.top_themes if stat.count > 0],
        )
        phase_priority = _merge_priorities(
            _focus_review_phase_priority(focus_review),
            [
                stat.phase
                for stat in sorted(
                    snapshot.error_phases,
                    key=lambda item: (-item.count, item.phase.value),
                )
                if stat.count > 0
            ],
        )

        candidates: list[TrainingExercise] = []
        for summary_index, summary in enumerate(relevant_summaries):
            for position in summary.training_positions:
                weakness_stat = weakness_stats.get(position.weakness_id)
                score, focus_theme, focus_phase = _score_position(
                    position,
                    theme_priority=theme_priority,
                    phase_priority=phase_priority,
                    summary_index=summary_index,
                    weakness_stat=weakness_stat,
                )
                candidates.append(TrainingExercise(
                    game_id=summary.game_id,
                    reviewed_at=summary.reviewed_at,
                    source_path=summary.source_path,
                    position=position,
                    relevance_score=score,
                    focus_reason=_build_focus_reason(
                        position,
                        focus_theme=focus_theme,
                        focus_phase=focus_phase,
                    ),
                    focus_theme=focus_theme,
                    focus_phase=focus_phase,
                    attempt_count=(
                        weakness_stat.attempt_count
                        if weakness_stat is not None
                        else 0
                    ),
                    success_rate=(
                        weakness_stat.success_rate
                        if weakness_stat is not None
                        else 0.0
                    ),
                    mastery_state=(
                        weakness_stat.mastery_state
                        if weakness_stat is not None
                        else WeaknessMasteryState.UNTRAINED
                    ),
                    weakness_trend=(
                        weakness_stat.trend
                        if weakness_stat is not None
                        else WeaknessTrend.STABLE
                    ),
                    mastery_summary=(
                        weakness_stat.summary
                        if weakness_stat is not None
                        else "Not trained yet."
                    ),
                ))

        ordered = sorted(
            candidates,
            key=lambda item: (-item.relevance_score, item.position.ply),
        )
        selected: list[TrainingExercise] = []
        seen: set[tuple[str, str, str]] = set()
        for exercise in ordered:
            key = (
                exercise.position.fen,
                exercise.position.played_uci,
                exercise.position.best_move_uci,
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(exercise)
            if len(selected) >= session_size:
                break
        return tuple(selected)

    def evaluate_move(
        self,
        fen: str,
        move_uci: str,
    ) -> TrainingMoveFeedback:
        """Evaluate a move against Stockfish's top 3 choices for the position."""
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal training move: {move_uci}")

        played_san = board.san(move)
        is_white = board.turn == chess.WHITE

        analyser = self._analyser_factory(self._analysis_config())
        analyser.start()
        try:
            before = analyser.analyse(fen)
            board.push(move)
            after = analyser.analyse(board.fen())
        finally:
            analyser.stop()

        top_variations = before.variations[:3]
        move_rank = next(
            (
                index + 1
                for index, variation in enumerate(top_variations)
                if variation.move_uci == move_uci
            ),
            None,
        )
        cp_loss = self._compute_cp_loss(before, after, is_white=is_white)

        if move_rank is not None:
            label = TrainingFeedbackLabel.GOOD
            message_key = "best" if move_rank == 1 else "top3"
        elif (
            cp_loss >= self._review_config.mistake_threshold
            or self._lost_mate(before, after)
        ):
            label = TrainingFeedbackLabel.SIGNIFICANT_ERROR
            message_key = "significant"
        else:
            label = TrainingFeedbackLabel.BETTER_AVAILABLE
            message_key = "better"

        return TrainingMoveFeedback(
            label=label,
            played_san=played_san,
            played_uci=move_uci,
            cp_loss=cp_loss,
            move_rank=move_rank,
            best_move_san=before.best_move_san,
            best_move_uci=before.best_move_uci,
            best_line_san=top_variations[0].pv_san if top_variations else (),
            top_move_sans=tuple(variation.move_san for variation in top_variations),
            message=_feedback_message(
                message_key,
                best_move_san=before.best_move_san,
                cp_loss=cp_loss,
            ),
        )

    def _analysis_config(self) -> AnalysisConfig:
        return AnalysisConfig(
            stockfish_path=self._review_config.stockfish_path,
            depth=self._review_config.depth,
            time_limit=self._review_config.time_limit,
            multipv=3,
        )

    @staticmethod
    def _eval_to_cp(evaluation: Evaluation, *, from_white: bool) -> int:
        if evaluation.is_mate:
            assert evaluation.mate_in is not None
            cp = (
                _MATE_CP - abs(evaluation.mate_in)
                if evaluation.mate_in > 0
                else -(_MATE_CP - abs(evaluation.mate_in))
            )
        else:
            assert evaluation.centipawns is not None
            cp = evaluation.centipawns
        return cp if from_white else -cp

    def _compute_cp_loss(
        self,
        result_before: AnalysisResult,
        result_after: AnalysisResult,
        *,
        is_white: bool,
    ) -> int:
        white_before = self._eval_to_cp(
            result_before.evaluation,
            from_white=is_white,
        )
        white_after = self._eval_to_cp(
            result_after.evaluation,
            from_white=not is_white,
        )
        loss = (
            white_before - white_after
            if is_white
            else white_after - white_before
        )
        return max(0, loss)

    @staticmethod
    def _lost_mate(
        result_before: AnalysisResult,
        result_after: AnalysisResult,
    ) -> bool:
        before = result_before.evaluation
        after = result_after.evaluation
        if not before.is_mate or before.mate_in is None or before.mate_in <= 0:
            return False
        if not after.is_mate:
            return True
        return after.mate_in is not None and after.mate_in > 0


def build_training_attempt(
    exercise: TrainingExercise,
    feedback: TrainingMoveFeedback,
    *,
    attempted_at: datetime | None = None,
) -> TrainingAttempt:
    """Build a persisted attempt record from one evaluated exercise."""
    attempt_time = attempted_at or datetime.now(timezone.utc)
    return TrainingAttempt(
        weakness_id=exercise.position.weakness_id,
        attempted_at=attempt_time.isoformat(),
        outcome=_feedback_outcome(feedback.label),
        game_id=exercise.game_id,
        reviewed_at=exercise.reviewed_at,
        fen=exercise.position.fen,
        ply=exercise.position.ply,
        source_path=exercise.source_path,
        expected_move_san=exercise.position.best_move_san,
        expected_move_uci=exercise.position.best_move_uci,
        original_move_san=exercise.position.played_san,
        original_move_uci=exercise.position.played_uci,
        submitted_move_san=feedback.played_san,
        submitted_move_uci=feedback.played_uci,
        cp_loss=feedback.cp_loss,
        move_rank=feedback.move_rank,
        phase=exercise.position.phase,
        lesson_themes=exercise.position.lesson_themes,
        error_motif=exercise.position.error_motif,
    )


def _feedback_outcome(label: TrainingFeedbackLabel) -> TrainingAttemptOutcome:
    if label is TrainingFeedbackLabel.GOOD:
        return TrainingAttemptOutcome.SUCCESS
    if label is TrainingFeedbackLabel.BETTER_AVAILABLE:
        return TrainingAttemptOutcome.PARTIAL
    return TrainingAttemptOutcome.FAILED


def _merge_priorities(*groups: list[object]) -> tuple[object, ...]:
    ordered: list[object] = []
    seen: set[object] = set()
    for group in groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _focus_review_theme_priority(
    review: GameReview | None,
) -> list[LessonTheme]:
    if review is None:
        return []

    counts = Counter(lesson.theme for lesson in review.lessons)
    return [
        theme
        for theme, _count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0].value),
        )
    ]


def _focus_review_phase_priority(
    review: GameReview | None,
) -> list[GamePhase]:
    if review is None:
        return []

    counts = Counter(
        move.phase
        for move in review.moves
        if move.classification in {
            MoveClassification.INACCURACY,
            MoveClassification.MISTAKE,
            MoveClassification.BLUNDER,
        }
    )
    return [
        phase
        for phase, _count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0].value),
        )
    ]


def _score_position(
    position: TrainingPositionSummary,
    *,
    theme_priority: tuple[object, ...],
    phase_priority: tuple[object, ...],
    summary_index: int,
    weakness_stat: WeaknessProgressStat | None = None,
) -> tuple[int, LessonTheme | None, GamePhase | None]:
    score = _SEVERITY_SCORES.get(position.classification, 0)
    score += min(position.cp_loss, 350)
    score += min(len(position.best_pv_san), 4) * 12
    score += max(0, 36 - (summary_index * 6))

    focus_theme = _match_theme(position.lesson_themes, theme_priority)
    focus_phase = _match_phase(position.phase, phase_priority)

    if focus_theme is not None:
        score += _priority_bonus(focus_theme, theme_priority, _THEME_PRIORITY_SCORES)
    elif position.lesson_themes:
        score += 35

    if focus_phase is not None:
        score += _priority_bonus(focus_phase, phase_priority, _PHASE_PRIORITY_SCORES)

    if weakness_stat is not None:
        score += weakness_stat.urgency_delta

    return score, focus_theme, focus_phase


def _match_theme(
    themes: tuple[LessonTheme, ...],
    theme_priority: tuple[object, ...],
) -> LessonTheme | None:
    for priority in theme_priority:
        if priority in themes:
            return priority  # type: ignore[return-value]
    return themes[0] if themes else None


def _match_phase(
    phase: GamePhase,
    phase_priority: tuple[object, ...],
) -> GamePhase | None:
    return phase if phase in phase_priority else None


def _priority_bonus(
    item: object,
    priority: tuple[object, ...],
    values: tuple[int, ...],
) -> int:
    for index, candidate in enumerate(priority):
        if candidate == item:
            if index < len(values):
                return values[index]
            return values[-1]
    return 0


def _build_focus_reason(
    position: TrainingPositionSummary,
    *,
    focus_theme: LessonTheme | None,
    focus_phase: GamePhase | None,
) -> str:
    if focus_theme is not None and focus_phase is not None:
        return (
            f"{_THEME_LABELS[focus_theme]} in your "
            f"{_PHASE_LABELS[focus_phase].lower()} trouble spots"
        )
    if focus_theme is not None:
        return f"Recurring theme: {_THEME_LABELS[focus_theme]}"
    if focus_phase is not None:
        return f"High-error phase: {_PHASE_LABELS[focus_phase]}"
    return f"Critical {_PHASE_LABELS[position.phase].lower()} decision"


def _feedback_message(
    key: str,
    *,
    best_move_san: str,
    cp_loss: int,
) -> str:
    base = _FEEDBACK_MESSAGES[key]
    if key in {"better", "significant"} and best_move_san:
        return f"{base} Stockfish preferait {best_move_san} ({cp_loss}cp de mieux)."
    return base
