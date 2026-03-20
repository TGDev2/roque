"""Post-game review with move classification by centipawn loss.

Replays a completed game position by position, analyses each with Stockfish,
and classifies every move as excellent / good / inaccuracy / mistake / blunder
based on the evaluation drop.  A :class:`GameReview` summary aggregates
statistics per phase and highlights critical moments.

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
class GameReview:
    """Complete review of a game with per-move classifications and summary."""

    moves: tuple[ReviewedMove, ...]
    phase_summaries: tuple[PhaseSummary, ...]
    white_classification_counts: dict[MoveClassification, int]
    black_classification_counts: dict[MoveClassification, int]
    white_avg_cp_loss: float
    black_avg_cp_loss: float
    critical_moments: tuple[int, ...]
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
            total_elapsed_ms=0.0,
        )
