"""Stockfish position analysis via python-chess UCI interface.

Wraps a Stockfish subprocess to provide position evaluation with multi-PV
support.  The engine process is reused across calls and managed through an
explicit start/stop lifecycle or a context manager.

Requirements
------------
* Stockfish 16+ binary accessible on the filesystem (default: ``"stockfish"``
  resolved via ``PATH``).
* ``python-chess >= 1.10`` (already a project dependency).

Limitations
-----------
* The time limit is enforced cooperatively by Stockfish itself — there is no
  hard external watchdog.  In practice Stockfish respects ``movetime``
  reliably.
* Positions that are already checkmate or stalemate are detected before
  calling the engine and return a synthetic result with no variations.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import chess
import chess.engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class AnalysisError(RuntimeError):
    """Base class for analysis module failures."""


class EngineNotFoundError(AnalysisError):
    """Raised when the Stockfish binary cannot be found or started."""


class EngineClosedError(AnalysisError):
    """Raised when analyse() is called on an analyser whose engine has been stopped."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    """Configuration for Stockfish position analysis.

    Parameters
    ----------
    stockfish_path:
        Filesystem path to the Stockfish binary.  Defaults to ``"stockfish"``
        (relies on PATH resolution).
    depth:
        Maximum search depth.  Default 18.
    time_limit:
        Maximum wall-clock seconds for analysis.  Default 2.0.
    multipv:
        Number of principal variations (top N moves) to return.  Default 3.
    threads:
        Number of Stockfish threads.  ``None`` means auto-detect
        (``os.cpu_count() - 1``, minimum 1).
    hash_mb:
        Hash table size in megabytes for Stockfish.  Default 128.
    """

    stockfish_path: str = "stockfish"
    depth: int = 18
    time_limit: float = 2.0
    multipv: int = 3
    threads: int | None = None
    hash_mb: int = 128

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}.")
        if self.time_limit <= 0:
            raise ValueError(f"time_limit must be > 0, got {self.time_limit}.")
        if self.multipv < 1:
            raise ValueError(f"multipv must be >= 1, got {self.multipv}.")
        if self.threads is not None and self.threads < 1:
            raise ValueError(f"threads must be >= 1 or None, got {self.threads}.")
        if self.hash_mb < 1:
            raise ValueError(f"hash_mb must be >= 1, got {self.hash_mb}.")


@dataclass(frozen=True)
class Evaluation:
    """A position evaluation from Stockfish.

    Exactly one of ``centipawns`` or ``mate_in`` is not ``None``.

    Attributes
    ----------
    centipawns:
        Score in centipawns relative to the side to move (positive = better).
        ``None`` when the position is a forced mate.
    mate_in:
        Mate in *N* moves (positive = side to move delivers mate, negative =
        side to move gets mated, 0 = already checkmate).  ``None`` for
        non-mate evaluations.
    """

    centipawns: int | None
    mate_in: int | None

    @property
    def is_mate(self) -> bool:
        """``True`` if this evaluation represents a forced mate sequence."""
        return self.mate_in is not None

    def __str__(self) -> str:
        if self.mate_in is not None:
            return f"#{self.mate_in:+d}" if self.mate_in != 0 else "#0"
        assert self.centipawns is not None
        score = self.centipawns / 100.0
        return f"{score:+.2f}"


@dataclass(frozen=True)
class VariationInfo:
    """A single principal variation line from multi-PV analysis.

    Attributes
    ----------
    rank:
        1-based rank (1 = best move).
    move_uci:
        Best move in UCI notation (e.g. ``"e2e4"``).
    move_san:
        Best move in Standard Algebraic Notation (e.g. ``"e4"``).
    evaluation:
        The evaluation for this line.
    pv_uci:
        Full principal variation as UCI move strings.
    pv_san:
        Full principal variation as SAN move strings.
    depth:
        Search depth reached for this line.
    """

    rank: int
    move_uci: str
    move_san: str
    evaluation: Evaluation
    pv_uci: tuple[str, ...]
    pv_san: tuple[str, ...]
    depth: int


@dataclass(frozen=True)
class AnalysisResult:
    """Outcome of a position analysis.

    Attributes
    ----------
    fen:
        The FEN string that was analysed.
    best_move_uci:
        The best move in UCI notation (empty string for terminal positions).
    best_move_san:
        The best move in SAN notation (empty string for terminal positions).
    evaluation:
        The top-line evaluation.
    variations:
        Principal variations sorted by rank (empty for terminal positions).
    depth_reached:
        The actual depth reached (may be < configured depth on timeout).
    is_partial:
        ``True`` if the analysis was interrupted before reaching target depth.
    elapsed_ms:
        Wall-clock time for analysis in milliseconds.
    """

    fen: str
    best_move_uci: str
    best_move_san: str
    evaluation: Evaluation
    variations: tuple[VariationInfo, ...]
    depth_reached: int
    is_partial: bool
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PositionAnalyser:
    """Stockfish-based position analyser.

    Manages the Stockfish subprocess lifecycle and provides position
    evaluation with multi-PV support.

    Usage::

        with PositionAnalyser() as analyser:
            result = analyser.analyse("rnbqkbnr/.../RNBQKBNR w KQkq - 0 1")
            print(result.evaluation, result.best_move_uci)
    """

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self._config = config or AnalysisConfig()
        self._engine: chess.engine.SimpleEngine | None = None

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the Stockfish engine subprocess.

        Raises
        ------
        EngineNotFoundError
            If the Stockfish binary cannot be found or fails to start.
        """
        if self._engine is not None:
            logger.debug("Engine already running — skipping start.")
            return

        threads = self._resolve_threads()
        logger.info(
            "Starting Stockfish: path=%s threads=%d hash=%dMB",
            self._config.stockfish_path,
            threads,
            self._config.hash_mb,
        )

        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(
                self._config.stockfish_path,
            )
        except (FileNotFoundError, chess.engine.EngineTerminatedError) as exc:
            raise EngineNotFoundError(
                f"Cannot start Stockfish at '{self._config.stockfish_path}': {exc}",
            ) from exc

        self._engine.configure({"Threads": threads, "Hash": self._config.hash_mb})
        logger.info("Stockfish engine started successfully.")

    def stop(self) -> None:
        """Stop the Stockfish engine subprocess."""
        if self._engine is None:
            return
        try:
            self._engine.quit()
        except chess.engine.EngineTerminatedError:
            logger.debug("Engine already terminated on stop().")
        finally:
            self._engine = None
            logger.info("Stockfish engine stopped.")

    @property
    def is_running(self) -> bool:
        """``True`` if the engine subprocess is alive."""
        return self._engine is not None

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> PositionAnalyser:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- Analysis ------------------------------------------------------------

    def analyse(self, fen: str) -> AnalysisResult:
        """Analyse a position and return evaluation with principal variations.

        Parameters
        ----------
        fen:
            A valid FEN string.

        Returns
        -------
        AnalysisResult
            The analysis outcome including evaluation, best move, and PV lines.

        Raises
        ------
        EngineClosedError
            If the engine has not been started or has been stopped.
        AnalysisError
            If the FEN is invalid.
        """
        if self._engine is None:
            raise EngineClosedError("Engine is not running. Call start() first.")

        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise AnalysisError(f"Invalid FEN: {exc}") from exc

        t0 = time.perf_counter()

        # Terminal positions — no engine call needed.
        if board.is_checkmate():
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.info("Position is checkmate — skipping engine analysis.")
            return AnalysisResult(
                fen=fen,
                best_move_uci="",
                best_move_san="",
                evaluation=Evaluation(centipawns=None, mate_in=0),
                variations=(),
                depth_reached=0,
                is_partial=False,
                elapsed_ms=elapsed,
            )

        if board.is_stalemate():
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.info("Position is stalemate — skipping engine analysis.")
            return AnalysisResult(
                fen=fen,
                best_move_uci="",
                best_move_san="",
                evaluation=Evaluation(centipawns=0, mate_in=None),
                variations=(),
                depth_reached=0,
                is_partial=False,
                elapsed_ms=elapsed,
            )

        # Engine analysis with depth + time limit.
        limit = chess.engine.Limit(
            depth=self._config.depth,
            time=self._config.time_limit,
        )
        logger.debug(
            "Analysing FEN=%s depth=%d time=%.1fs multipv=%d",
            fen,
            self._config.depth,
            self._config.time_limit,
            self._config.multipv,
        )

        infos: list[chess.engine.InfoDict] = self._engine.analyse(
            board,
            limit,
            multipv=self._config.multipv,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        # Build variations from engine output.
        variations: list[VariationInfo] = []
        for rank, info in enumerate(infos, start=1):
            variations.append(self._to_variation_info(info, board, rank))

        depth_reached = variations[0].depth if variations else 0
        is_partial = depth_reached < self._config.depth

        best_move_uci = variations[0].move_uci if variations else ""
        best_move_san = variations[0].move_san if variations else ""
        evaluation = variations[0].evaluation if variations else Evaluation(
            centipawns=0, mate_in=None,
        )

        logger.info(
            "Analysis complete: eval=%s best=%s depth=%d partial=%s elapsed=%.1fms",
            evaluation,
            best_move_uci,
            depth_reached,
            is_partial,
            elapsed,
        )

        return AnalysisResult(
            fen=fen,
            best_move_uci=best_move_uci,
            best_move_san=best_move_san,
            evaluation=evaluation,
            variations=tuple(variations),
            depth_reached=depth_reached,
            is_partial=is_partial,
            elapsed_ms=elapsed,
        )

    # -- Private helpers -----------------------------------------------------

    def _resolve_threads(self) -> int:
        """Resolve the number of Stockfish threads to use."""
        if self._config.threads is not None:
            return self._config.threads
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count - 1)

    @staticmethod
    def _to_evaluation(pov_score: chess.engine.PovScore) -> Evaluation:
        """Convert a python-chess ``PovScore`` (relative) to an Evaluation."""
        score = pov_score.relative
        if score.is_mate():
            return Evaluation(centipawns=None, mate_in=score.mate())
        return Evaluation(centipawns=score.score(), mate_in=None)

    def _to_variation_info(
        self,
        info: chess.engine.InfoDict,
        board: chess.Board,
        rank: int,
    ) -> VariationInfo:
        """Convert a single engine ``InfoDict`` to a ``VariationInfo``."""
        pv_moves: list[chess.Move] = info.get("pv", [])
        pv_uci = tuple(m.uci() for m in pv_moves)

        # Replay moves on a board copy to get SAN notation.
        san_board = board.copy()
        pv_san_list: list[str] = []
        for move in pv_moves:
            pv_san_list.append(san_board.san(move))
            san_board.push(move)

        best_move = pv_moves[0] if pv_moves else None
        return VariationInfo(
            rank=rank,
            move_uci=best_move.uci() if best_move else "",
            move_san=board.san(best_move) if best_move else "",
            evaluation=self._to_evaluation(info["score"]),
            pv_uci=pv_uci,
            pv_san=tuple(pv_san_list),
            depth=info.get("depth", 0),
        )
