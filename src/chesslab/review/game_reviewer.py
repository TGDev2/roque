"""Post-game review with move classification and lesson extraction.

Replays a completed game position by position, analyses each with Stockfish,
classifies every move as excellent / good / inaccuracy / mistake / blunder
based on the evaluation drop, and extracts a few prioritised lessons from the
most actionable patterns.  A :class:`GameReview` summary aggregates statistics
per phase, highlights critical moments, and surfaces concrete takeaways.

Usage::

    from chesslab.review import GameReviewer, ReviewConfig

    reviewer = GameReviewer(ReviewConfig(depth=16))
    review = reviewer.review(board, progress_callback=on_progress)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import chess

from chesslab.analysis import (
    AnalysisConfig,
    AnalysisResult,
    Evaluation,
    PositionAnalyser,
    VariationInfo,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class ReviewError(RuntimeError):
    """Base class for review module failures."""


class ReviewCancelledError(ReviewError):
    """Raised when a review is cancelled via *cancel_event*."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MoveClassification(Enum):
    """Quality label assigned to a move based on centipawn loss."""

    EXCELLENT = "excellent"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"
    FORCED = "forced"


class GamePhase(Enum):
    """Approximate phase of the game."""

    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"


class LessonTheme(Enum):
    """High-level type of lesson extracted from a review."""

    PHASE_PATTERN = "phase_pattern"
    TURNING_POINT = "turning_point"
    ADVANTAGE_MANAGEMENT = "advantage_management"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Standard piece values (no king) used for material counting.
_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# Large centipawn stand-in for mate scores.
_MATE_CP = 100_000

_LESSON_MIN_PLIES = 6
_LESSON_MAX_COUNT = 3
_LESSON_PHASE_MIN_NOTABLE_MOVES = 2
_LESSON_PHASE_MIN_TOTAL_LOSS = 120
_LESSON_TURNING_POINT_MIN_LOSS = 120
_LESSON_ADVANTAGE_THRESHOLD = 120
_LESSON_EQUALITY_BAND = 60
_SIDE_NAMES: dict[bool, str] = {
    True: "White",
    False: "Black",
}
_NOTABLE_CLASSIFICATIONS = frozenset({
    MoveClassification.INACCURACY,
    MoveClassification.MISTAKE,
    MoveClassification.BLUNDER,
})
_SEVERITY_SCORES: dict[MoveClassification, int] = {
    MoveClassification.EXCELLENT: 0,
    MoveClassification.GOOD: 0,
    MoveClassification.INACCURACY: 20,
    MoveClassification.MISTAKE: 80,
    MoveClassification.BLUNDER: 140,
    MoveClassification.FORCED: 0,
}


@dataclass
class ReviewConfig:
    """Configurable thresholds for move classification and analysis depth.

    All threshold values are in centipawns.  A move whose centipawn loss is
    strictly less than *excellent_threshold* is classified ``EXCELLENT``, etc.
    """

    depth: int = 16
    time_limit: float = 1.5
    multipv: int = 1
    stockfish_path: str = "stockfish"

    # Classification thresholds (upper bound exclusive for each tier).
    excellent_threshold: int = 10
    good_threshold: int = 30
    inaccuracy_threshold: int = 80
    mistake_threshold: int = 200

    # Phase detection parameters.
    opening_end_ply: int = 30
    endgame_material_threshold: int = 13


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewedMove:
    """Analysis and classification of a single half-move."""

    ply: int
    san: str
    uci: str
    fen_before: str
    fen_after: str
    classification: MoveClassification
    eval_before: Evaluation
    eval_after: Evaluation
    best_move_san: str
    best_move_uci: str
    best_eval: Evaluation
    cp_loss: int
    best_pv_san: tuple[str, ...]
    phase: GamePhase
    is_white: bool
    is_forced: bool


@dataclass(frozen=True)
class PhaseSummary:
    """Aggregated statistics for one game phase."""

    phase: GamePhase
    move_count: int
    avg_cp_loss: float
    classification_counts: dict[MoveClassification, int]


@dataclass(frozen=True)
class ReviewLesson:
    """Prioritised lesson extracted from a completed review."""

    priority: int
    theme: LessonTheme
    title: str
    finding: str
    takeaway: str
    focus_ply: int
    related_plies: tuple[int, ...] = field(default_factory=tuple)
    phase: GamePhase | None = None
    is_white: bool | None = None


@dataclass(frozen=True)
class GameReview:
    """Complete review of a game with per-move classifications and summary."""

    moves: tuple[ReviewedMove, ...]
    phase_summaries: tuple[PhaseSummary, ...]
    white_classification_counts: dict[MoveClassification, int]
    black_classification_counts: dict[MoveClassification, int]
    white_avg_cp_loss: float
    black_avg_cp_loss: float
    critical_moments: tuple[int, ...]
    lessons: tuple[ReviewLesson, ...]
    total_elapsed_ms: float


# ---------------------------------------------------------------------------
# Callback type
# ---------------------------------------------------------------------------

ReviewProgressCallback = Callable[[int, int], None]
"""``callback(current_ply, total_plies)`` — called after each position."""


# ---------------------------------------------------------------------------
# Reviewer
# ---------------------------------------------------------------------------


class GameReviewer:
    """Analyses a completed game and classifies every move.

    The reviewer creates its own :class:`PositionAnalyser` internally so it
    does not interfere with any live analysis running in the coach.
    """

    def __init__(self, config: ReviewConfig | None = None) -> None:
        self._config = config or ReviewConfig()

    # -- public API ----------------------------------------------------------

    def review(
        self,
        board: chess.Board,
        *,
        progress_callback: ReviewProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> GameReview:
        """Analyse every position in *board*'s move stack and classify moves.

        Parameters
        ----------
        board:
            A :class:`chess.Board` whose *move_stack* contains the game to
            review.  The board is **not** mutated.
        progress_callback:
            Optional callback invoked after each position is analysed.
        cancel_event:
            If set before analysis completes, :class:`ReviewCancelledError`
            is raised.

        Returns
        -------
        GameReview

        Raises
        ------
        ReviewCancelledError
            If *cancel_event* is set during analysis.
        ReviewError
            On engine or analysis failures.
        """
        moves_played = list(board.move_stack)
        total_plies = len(moves_played)

        if total_plies == 0:
            return self._empty_review()

        # Determine the root position (board state before any moves).
        root = board.copy()
        while root.move_stack:
            root.pop()
        self._root_fen = root.fen()

        t0 = time.perf_counter()

        # Build analysis config from review settings.
        analysis_cfg = AnalysisConfig(
            stockfish_path=self._config.stockfish_path,
            depth=self._config.depth,
            time_limit=self._config.time_limit,
            multipv=self._config.multipv,
        )

        try:
            analyser = PositionAnalyser(analysis_cfg)
            analyser.start()
        except Exception as exc:
            raise ReviewError(f"Failed to start engine: {exc}") from exc

        try:
            reviewed = self._analyse_game(
                analyser,
                moves_played,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        finally:
            analyser.stop()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._build_review(reviewed, elapsed_ms)

    # -- internal analysis loop ----------------------------------------------

    def _analyse_game(
        self,
        analyser: PositionAnalyser,
        moves: list[chess.Move],
        *,
        progress_callback: ReviewProgressCallback | None,
        cancel_event: threading.Event | None,
    ) -> list[ReviewedMove]:
        """Walk through the game, analyse each position, classify each move."""
        total = len(moves)
        replay = chess.Board(self._root_fen)

        # We need evaluations at N+1 positions (before each move + after last).
        # Evaluate the starting position first.
        evals: list[AnalysisResult] = [analyser.analyse(replay.fen())]

        reviewed: list[ReviewedMove] = []

        for ply, move in enumerate(moves):
            if cancel_event is not None and cancel_event.is_set():
                raise ReviewCancelledError("Review cancelled by user.")

            fen_before = replay.fen()
            san = replay.san(move)
            uci = move.uci()
            is_white = replay.turn == chess.WHITE
            is_forced = len(list(replay.legal_moves)) == 1
            phase = self._detect_phase(ply, replay)

            replay.push(move)
            fen_after = replay.fen()

            result_after = analyser.analyse(fen_after)
            evals.append(result_after)

            result_before = evals[ply]

            cp_loss = self._compute_cp_loss(
                result_before, result_after, is_white,
            )

            classification = self._classify(
                cp_loss, result_before, result_after, is_white, is_forced,
            )

            best_move_san = result_before.best_move_san
            best_move_uci = result_before.best_move_uci
            best_eval = result_before.evaluation
            best_pv_san = (
                result_before.variations[0].pv_san
                if result_before.variations
                else ()
            )

            reviewed.append(ReviewedMove(
                ply=ply,
                san=san,
                uci=uci,
                fen_before=fen_before,
                fen_after=fen_after,
                classification=classification,
                eval_before=result_before.evaluation,
                eval_after=result_after.evaluation,
                best_move_san=best_move_san,
                best_move_uci=best_move_uci,
                best_eval=best_eval,
                cp_loss=cp_loss,
                best_pv_san=best_pv_san,
                phase=phase,
                is_white=is_white,
                is_forced=is_forced,
            ))

            if progress_callback is not None:
                progress_callback(ply + 1, total)

        return reviewed

    # -- classification logic ------------------------------------------------

    @staticmethod
    def _eval_to_cp(evaluation: Evaluation, from_white: bool) -> int:
        """Convert an Evaluation to centipawns from White's perspective.

        Mate scores are mapped to large positive/negative values.  The
        *from_white* flag indicates whether the evaluation is already from
        White's perspective (``True`` when White is to move in the position
        that produced this evaluation).
        """
        if evaluation.is_mate:
            assert evaluation.mate_in is not None
            cp = _MATE_CP - abs(evaluation.mate_in) if evaluation.mate_in > 0 else -(_MATE_CP - abs(evaluation.mate_in))
        else:
            assert evaluation.centipawns is not None
            cp = evaluation.centipawns

        return cp if from_white else -cp

    def _compute_cp_loss(
        self,
        result_before: AnalysisResult,
        result_after: AnalysisResult,
        is_white: bool,
    ) -> int:
        """Compute centipawn loss for the move that transitioned between positions.

        The loss is always >= 0 (clamped).  It represents how much worse the
        mover's position became compared to the engine's best option.
        """
        # eval_before is from the mover's perspective (side-to-move).
        # eval_after is from the opponent's perspective (other side to move).
        # Convert both to White-relative, then compute mover's loss.
        white_before = self._eval_to_cp(result_before.evaluation, from_white=is_white)
        white_after = self._eval_to_cp(result_after.evaluation, from_white=not is_white)

        if is_white:
            loss = white_before - white_after
        else:
            loss = (-white_before) - (-white_after)
            # Simplifies to white_after - white_before, which is the loss
            # from Black's perspective when both are in White-relative cp.
            loss = white_after - white_before

        return max(0, loss)

    def _classify(
        self,
        cp_loss: int,
        result_before: AnalysisResult,
        result_after: AnalysisResult,
        is_white: bool,
        is_forced: bool,
    ) -> MoveClassification:
        """Assign a classification label to a move."""
        if is_forced:
            return MoveClassification.FORCED

        # Detect lost mate: mover had a forced mate before but not after.
        if self._lost_mate(result_before, result_after, is_white):
            return MoveClassification.BLUNDER

        cfg = self._config
        if cp_loss < cfg.excellent_threshold:
            return MoveClassification.EXCELLENT
        if cp_loss < cfg.good_threshold:
            return MoveClassification.GOOD
        if cp_loss < cfg.inaccuracy_threshold:
            return MoveClassification.INACCURACY
        if cp_loss < cfg.mistake_threshold:
            return MoveClassification.MISTAKE
        return MoveClassification.BLUNDER

    @staticmethod
    def _lost_mate(
        result_before: AnalysisResult,
        result_after: AnalysisResult,
        is_white: bool,
    ) -> bool:
        """Return True if the mover had a forced mate but lost it."""
        eval_before = result_before.evaluation
        eval_after = result_after.evaluation

        if not eval_before.is_mate or eval_before.mate_in is None:
            return False

        # Mover had mate?  For White: mate_in > 0 in a white-to-move position.
        # For Black: mate_in > 0 in a black-to-move position (positive means
        # the side-to-move has mate).
        mover_had_mate = eval_before.mate_in > 0

        if not mover_had_mate:
            return False

        # After the move, it's the opponent's turn.  If the opponent does NOT
        # face a forced mate against them (i.e. eval_after.mate_in is None or
        # eval_after.mate_in > 0 meaning opponent now has mate), then the
        # mover lost their advantage.
        if not eval_after.is_mate:
            return True
        if eval_after.mate_in is not None and eval_after.mate_in > 0:
            # Opponent now has a forced mate — mover definitely lost theirs.
            return True

        return False

    # -- phase detection -----------------------------------------------------

    def _detect_phase(self, ply: int, board: chess.Board) -> GamePhase:
        """Approximate game phase from ply count and material."""
        if ply < self._config.opening_end_ply:
            return GamePhase.OPENING

        material = self._count_material(board)
        if material <= self._config.endgame_material_threshold:
            return GamePhase.ENDGAME

        return GamePhase.MIDDLEGAME

    @staticmethod
    def _count_material(board: chess.Board) -> int:
        """Sum material value for both sides (excluding kings)."""
        total = 0
        for piece_type, value in _PIECE_VALUES.items():
            total += len(board.pieces(piece_type, chess.WHITE)) * value
            total += len(board.pieces(piece_type, chess.BLACK)) * value
        return total

    # -- summary building ----------------------------------------------------

    def _build_review(
        self,
        reviewed: list[ReviewedMove],
        elapsed_ms: float,
    ) -> GameReview:
        """Aggregate per-move data into a GameReview."""
        white_moves = [m for m in reviewed if m.is_white]
        black_moves = [m for m in reviewed if not m.is_white]

        white_counts = self._count_classifications(white_moves)
        black_counts = self._count_classifications(black_moves)

        white_avg = self._avg_cp_loss(white_moves)
        black_avg = self._avg_cp_loss(black_moves)

        phase_summaries = self._build_phase_summaries(reviewed)
        lessons = self._build_lessons(reviewed)

        critical = tuple(
            m.ply for m in reviewed
            if m.classification in (MoveClassification.MISTAKE, MoveClassification.BLUNDER)
        )

        return GameReview(
            moves=tuple(reviewed),
            phase_summaries=phase_summaries,
            white_classification_counts=white_counts,
            black_classification_counts=black_counts,
            white_avg_cp_loss=white_avg,
            black_avg_cp_loss=black_avg,
            critical_moments=critical,
            lessons=lessons,
            total_elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _count_classifications(
        moves: list[ReviewedMove],
    ) -> dict[MoveClassification, int]:
        counter: Counter[MoveClassification] = Counter()
        for cls in MoveClassification:
            counter[cls] = 0
        for m in moves:
            counter[m.classification] += 1
        return dict(counter)

    @staticmethod
    def _avg_cp_loss(moves: list[ReviewedMove]) -> float:
        if not moves:
            return 0.0
        return sum(m.cp_loss for m in moves) / len(moves)

    def _build_phase_summaries(
        self,
        reviewed: list[ReviewedMove],
    ) -> tuple[PhaseSummary, ...]:
        by_phase: dict[GamePhase, list[ReviewedMove]] = {}
        for m in reviewed:
            by_phase.setdefault(m.phase, []).append(m)

        summaries = []
        for phase in GamePhase:
            moves = by_phase.get(phase, [])
            if not moves:
                continue
            summaries.append(PhaseSummary(
                phase=phase,
                move_count=len(moves),
                avg_cp_loss=self._avg_cp_loss(moves),
                classification_counts=self._count_classifications(moves),
            ))
        return tuple(summaries)

    def _build_lessons(
        self,
        reviewed: list[ReviewedMove],
    ) -> tuple[ReviewLesson, ...]:
        """Extract a few concrete lessons from repeated or decisive errors."""
        if len(reviewed) < _LESSON_MIN_PLIES:
            return ()

        candidates: list[tuple[int, ReviewLesson]] = []
        candidates.extend(self._build_phase_pattern_candidates(reviewed))

        turning_point = self._build_turning_point_candidate(reviewed)
        if turning_point is not None:
            candidates.append(turning_point)

        advantage_management = self._build_advantage_management_candidate(reviewed)
        if advantage_management is not None:
            candidates.append(advantage_management)

        if not candidates:
            return ()

        selected: list[ReviewLesson] = []
        used_focus_plies: set[int] = set()
        for _, lesson in sorted(candidates, key=lambda item: item[0], reverse=True):
            if lesson.focus_ply in used_focus_plies:
                continue
            selected.append(ReviewLesson(
                priority=len(selected) + 1,
                theme=lesson.theme,
                title=lesson.title,
                finding=lesson.finding,
                takeaway=lesson.takeaway,
                focus_ply=lesson.focus_ply,
                related_plies=lesson.related_plies,
                phase=lesson.phase,
                is_white=lesson.is_white,
            ))
            used_focus_plies.add(lesson.focus_ply)
            if len(selected) >= _LESSON_MAX_COUNT:
                break

        return tuple(selected)

    def _build_phase_pattern_candidates(
        self,
        reviewed: list[ReviewedMove],
    ) -> list[tuple[int, ReviewLesson]]:
        grouped: dict[tuple[bool, GamePhase], list[ReviewedMove]] = {}
        for move in reviewed:
            if move.classification not in _NOTABLE_CLASSIFICATIONS:
                continue
            grouped.setdefault((move.is_white, move.phase), []).append(move)

        candidates: list[tuple[int, ReviewLesson]] = []
        for (is_white, phase), moves in grouped.items():
            if len(moves) < _LESSON_PHASE_MIN_NOTABLE_MOVES:
                continue

            total_loss = sum(move.cp_loss for move in moves)
            if total_loss < _LESSON_PHASE_MIN_TOTAL_LOSS:
                continue

            focus = max(
                moves,
                key=lambda move: (move.cp_loss, _SEVERITY_SCORES[move.classification]),
            )
            side = _SIDE_NAMES[is_white]
            score = (
                total_loss
                + sum(_SEVERITY_SCORES[move.classification] for move in moves)
                + (len(moves) * 20)
                + 40
            )
            candidates.append((
                score,
                ReviewLesson(
                    priority=0,
                    theme=LessonTheme.PHASE_PATTERN,
                    title=f"{phase.value.capitalize()}: {side} leaked the most equity",
                    finding=(
                        f"{side} had {len(moves)} moves rated inaccuracy or worse in the "
                        f"{phase.value} ({self._format_move_refs(moves)}), for roughly "
                        f"{total_loss}cp lost."
                    ),
                    takeaway=self._phase_takeaway(phase, focus),
                    focus_ply=focus.ply,
                    related_plies=tuple(move.ply for move in moves),
                    phase=phase,
                    is_white=is_white,
                ),
            ))

        return candidates

    def _build_turning_point_candidate(
        self,
        reviewed: list[ReviewedMove],
    ) -> tuple[int, ReviewLesson] | None:
        focus = max(
            reviewed,
            key=lambda move: (move.cp_loss, _SEVERITY_SCORES[move.classification]),
        )
        if focus.cp_loss < _LESSON_TURNING_POINT_MIN_LOSS:
            return None

        side = _SIDE_NAMES[focus.is_white]
        before_cp = self._mover_cp_before(focus)
        after_cp = self._mover_cp_after(focus)
        best_move = self._best_move_label(focus)
        score = focus.cp_loss + _SEVERITY_SCORES[focus.classification] + 20

        return (
            score,
            ReviewLesson(
                priority=0,
                theme=LessonTheme.TURNING_POINT,
                title=f"Turning point: {self._move_ref(focus)}",
                finding=(
                    f"{self._move_ref(focus)} cost about {focus.cp_loss}cp for {side} "
                    f"and {self._describe_turning_point(side, before_cp, after_cp)}. "
                    f"Stockfish preferred {best_move}."
                ),
                takeaway=self._turning_point_takeaway(focus, before_cp),
                focus_ply=focus.ply,
                related_plies=(focus.ply,),
                phase=focus.phase,
                is_white=focus.is_white,
            ),
        )

    def _build_advantage_management_candidate(
        self,
        reviewed: list[ReviewedMove],
    ) -> tuple[int, ReviewLesson] | None:
        candidates = [
            move for move in reviewed
            if (
                self._mover_cp_before(move) >= _LESSON_ADVANTAGE_THRESHOLD
                and move.cp_loss >= self._config.inaccuracy_threshold
                and move.classification in _NOTABLE_CLASSIFICATIONS
            )
        ]
        if not candidates:
            return None

        focus = max(
            candidates,
            key=lambda move: (move.cp_loss, _SEVERITY_SCORES[move.classification]),
        )
        side = _SIDE_NAMES[focus.is_white]
        best_move = self._best_move_label(focus)
        score = focus.cp_loss + _SEVERITY_SCORES[focus.classification] + 30

        return (
            score,
            ReviewLesson(
                priority=0,
                theme=LessonTheme.ADVANTAGE_MANAGEMENT,
                title=f"Advantage management: {side} let the edge slip",
                finding=(
                    f"Before {self._move_ref(focus)}, {side} was clearly better, "
                    f"but the move gave away about {focus.cp_loss}cp. "
                    f"Stockfish kept more control with {best_move}."
                ),
                takeaway=(
                    "When better, first shortlist the move that cuts counterplay "
                    f"before looking for something sharper, especially candidates like "
                    f"{best_move}."
                ),
                focus_ply=focus.ply,
                related_plies=(focus.ply,),
                phase=focus.phase,
                is_white=focus.is_white,
            ),
        )

    def _phase_takeaway(self, phase: GamePhase, focus: ReviewedMove) -> str:
        best_move = self._best_move_label(focus)
        if phase is GamePhase.OPENING:
            return (
                "In similar openings, compare natural development with forcing "
                f"alternatives such as {best_move} before committing."
            )
        if phase is GamePhase.ENDGAME:
            return (
                "In similar endgames, slow the position down and compare the "
                f"cleanest technical continuation, such as {best_move}."
            )
        return (
            "In similar middlegames, spend a quick scan on checks, captures, and "
            f"threats before choosing a plan, especially against lines like {best_move}."
        )

    def _turning_point_takeaway(
        self,
        focus: ReviewedMove,
        before_cp: int,
    ) -> str:
        best_move = self._best_move_label(focus)
        if before_cp >= _LESSON_ADVANTAGE_THRESHOLD:
            return (
                "When a side is better, prefer the move that keeps control before "
                f"looking for something sharper, and compare it with {best_move}."
            )
        if -_LESSON_EQUALITY_BAND <= before_cp <= _LESSON_EQUALITY_BAND:
            return (
                "In balanced positions, compare a candidate move with the engine's "
                f"forcing alternative, such as {best_move}, before committing."
            )
        return (
            "When the position can swing this fast, double-check forcing replies "
            f"and the engine alternative {best_move} first."
        )

    def _describe_turning_point(
        self,
        side: str,
        before_cp: int,
        after_cp: int,
    ) -> str:
        if (
            before_cp >= _LESSON_ADVANTAGE_THRESHOLD
            and after_cp <= -_LESSON_EQUALITY_BAND
        ):
            return f"turned a clearly better position into a worse one for {side}"
        if before_cp >= _LESSON_ADVANTAGE_THRESHOLD:
            return f"gave away most of {side}'s advantage"
        if (
            -_LESSON_EQUALITY_BAND <= before_cp <= _LESSON_EQUALITY_BAND
            and after_cp <= -_LESSON_ADVANTAGE_THRESHOLD
        ):
            return f"turned a roughly equal position into a clear disadvantage for {side}"
        if before_cp <= -_LESSON_ADVANTAGE_THRESHOLD:
            return "made an already difficult defense even harder"
        return "created the sharpest evaluation swing in the game"

    def _mover_cp_before(self, move: ReviewedMove) -> int:
        white_cp = self._eval_to_cp(move.eval_before, from_white=move.is_white)
        return white_cp if move.is_white else -white_cp

    def _mover_cp_after(self, move: ReviewedMove) -> int:
        white_cp = self._eval_to_cp(move.eval_after, from_white=not move.is_white)
        return white_cp if move.is_white else -white_cp

    @staticmethod
    def _best_move_label(move: ReviewedMove) -> str:
        if move.best_move_san:
            return move.best_move_san
        if move.best_move_uci:
            return move.best_move_uci
        return "the engine line"

    def _format_move_refs(
        self,
        moves: list[ReviewedMove],
        *,
        max_items: int = 3,
    ) -> str:
        refs = [self._move_ref(move) for move in moves[:max_items]]
        if len(moves) > max_items:
            refs.append("...")
        return ", ".join(refs)

    @staticmethod
    def _move_ref(move: ReviewedMove) -> str:
        move_num = move.ply // 2 + 1
        dot = "." if move.is_white else "..."
        return f"{move_num}{dot} {move.san}"

    @staticmethod
    def _empty_review() -> GameReview:
        return GameReview(
            moves=(),
            phase_summaries=(),
            white_classification_counts={cls: 0 for cls in MoveClassification},
            black_classification_counts={cls: 0 for cls in MoveClassification},
            white_avg_cp_loss=0.0,
            black_avg_cp_loss=0.0,
            critical_moments=(),
            lessons=(),
            total_elapsed_ms=0.0,
        )
