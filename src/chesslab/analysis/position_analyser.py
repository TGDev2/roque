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

import hashlib
import json
import logging
import math
import os
import shutil
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, ClassVar

import chess
import chess.engine

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".chesslab"
_STOCKFISH_ENV_VAR = "CHESSLAB_STOCKFISH_PATH"
_STOCKFISH_FALLBACK_CANDIDATES: tuple[str, ...] = (
    "/usr/games/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
)


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
        (resolved from ``CHESSLAB_STOCKFISH_PATH`` first, then ``PATH``).
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
    engine_elo:
        Optional Stockfish limited-strength Elo.  When set, ChessLab enables
        ``UCI_LimitStrength`` with ``UCI_Elo=<value>`` so suggestions reflect
        that approximate skill level instead of full-strength engine play.
    """

    MIN_USER_DEPTH: ClassVar[int] = 10
    MAX_USER_DEPTH: ClassVar[int] = 25
    MIN_ENGINE_ELO: ClassVar[int] = 1320
    MAX_ENGINE_ELO: ClassVar[int] = 3190
    ENGINE_ELO_STEP: ClassVar[int] = 100
    MIN_SELECTABLE_ENGINE_ELO: ClassVar[int] = (
        ((MIN_ENGINE_ELO + ENGINE_ELO_STEP - 1) // ENGINE_ELO_STEP) * ENGINE_ELO_STEP
    )
    MAX_SELECTABLE_ENGINE_ELO: ClassVar[int] = (
        (MAX_ENGINE_ELO // ENGINE_ELO_STEP) * ENGINE_ELO_STEP
    )
    DEFAULT_ENGINE_ELO: ClassVar[int] = 1800

    stockfish_path: str = "stockfish"
    depth: int = 18
    time_limit: float = 2.0
    multipv: int = 3
    threads: int | None = None
    hash_mb: int = 128
    engine_elo: int | None = None

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
        if self.engine_elo is not None and not (
            self.MIN_ENGINE_ELO <= self.engine_elo <= self.MAX_ENGINE_ELO
        ):
            raise ValueError(
                f"engine_elo must be between {self.MIN_ENGINE_ELO} and "
                f"{self.MAX_ENGINE_ELO}, got {self.engine_elo}.",
            )

    @classmethod
    def snap_engine_elo(cls, elo: int) -> int:
        """Snap *elo* to the nearest selectable 100-point bucket."""
        bounded = max(
            cls.MIN_SELECTABLE_ENGINE_ELO,
            min(cls.MAX_SELECTABLE_ENGINE_ELO, elo),
        )
        relative = bounded - cls.MIN_SELECTABLE_ENGINE_ELO
        snapped_steps = int(round(relative / cls.ENGINE_ELO_STEP))
        return cls.MIN_SELECTABLE_ENGINE_ELO + (snapped_steps * cls.ENGINE_ELO_STEP)

    @classmethod
    def selectable_engine_levels(cls) -> int:
        """Return how many selectable Elo buckets exist."""
        return (
            (cls.MAX_SELECTABLE_ENGINE_ELO - cls.MIN_SELECTABLE_ENGINE_ELO)
            // cls.ENGINE_ELO_STEP
        ) + 1

    @classmethod
    def engine_level_from_elo(cls, elo: int) -> int:
        """Return a 1-based level index that matches selectable Elo buckets."""
        snapped = cls.snap_engine_elo(elo)
        return (
            (snapped - cls.MIN_SELECTABLE_ENGINE_ELO) // cls.ENGINE_ELO_STEP
        ) + 1

    # -- Persistence ---------------------------------------------------------

    @staticmethod
    def save_preference(
        depth: int,
        time_limit: float,
        config_dir: Path | None = None,
    ) -> None:
        """Persist analysis preferences to the user config file.

        Parameters
        ----------
        depth:
            Search depth.  Must be in ``[MIN_USER_DEPTH, MAX_USER_DEPTH]``.
        time_limit:
            Maximum analysis time in seconds.  Must be > 0.
        config_dir:
            Override for the config directory (default ``~/.chesslab``).

        Raises
        ------
        ValueError
            If *depth* or *time_limit* is outside the allowed range.
        """
        if not (AnalysisConfig.MIN_USER_DEPTH <= depth <= AnalysisConfig.MAX_USER_DEPTH):
            raise ValueError(
                f"depth must be between {AnalysisConfig.MIN_USER_DEPTH} and "
                f"{AnalysisConfig.MAX_USER_DEPTH}, got {depth}.",
            )
        if time_limit <= 0:
            raise ValueError(f"time_limit must be > 0, got {time_limit}.")

        directory = config_dir or _CONFIG_DIR
        directory.mkdir(parents=True, exist_ok=True)
        config_file = directory / "config.json"

        data: dict[str, object] = {}
        if config_file.is_file():
            try:
                with config_file.open() as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                data = {}

        data["analysis"] = {"depth": depth, "time_limit": time_limit}
        with config_file.open("w") as fh:
            json.dump(data, fh, indent=2)

        logger.info(
            "Analysis preference saved: depth=%d time_limit=%.1f.",
            depth,
            time_limit,
        )

    @staticmethod
    def load_preference(config_dir: Path | None = None) -> dict[str, object] | None:
        """Load persisted analysis preferences.

        Returns
        -------
        dict or None
            A dict with ``"depth"`` (int) and ``"time_limit"`` (float) keys,
            or ``None`` if no preference has been saved or the config file is
            missing / corrupt.
        """
        directory = config_dir or _CONFIG_DIR
        config_file = directory / "config.json"

        if not config_file.is_file():
            return None

        try:
            with config_file.open() as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read config file: %s.", config_file)
            return None

        analysis = data.get("analysis")
        if not isinstance(analysis, dict):
            return None

        depth = analysis.get("depth")
        time_limit = analysis.get("time_limit")
        if isinstance(depth, int) and isinstance(time_limit, (int, float)):
            return {"depth": depth, "time_limit": float(time_limit)}
        return None

    @staticmethod
    def estimate_analysis_time(
        depth: int,
        threads: int | None = None,
    ) -> float:
        """Estimate wall-clock analysis time for a given depth.

        Uses a simple exponential heuristic.  The estimate is indicative and
        depends heavily on the position complexity and hardware.

        Parameters
        ----------
        depth:
            Search depth.
        threads:
            Number of Stockfish threads.  ``None`` means auto-detect.

        Returns
        -------
        float
            Estimated analysis time in seconds, rounded to one decimal place.
        """
        base_time = 0.1  # seconds at depth 10
        branching_factor = 1.35

        raw = base_time * branching_factor ** (depth - 10)

        effective_threads = threads if threads is not None else max(1, (os.cpu_count() or 1) - 1)
        raw /= math.sqrt(effective_threads)

        clamped = max(0.1, min(raw, 300.0))
        return round(clamped, 1)


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


@dataclass(frozen=True)
class ContinuousUpdate:
    """An incremental analysis update emitted during continuous analysis.

    Attributes
    ----------
    fen:
        The FEN string being analysed.
    variations:
        Principal variations at the current depth, sorted by rank.
    evaluation:
        The top-line evaluation at the current depth.
    best_move_uci:
        Best move in UCI notation.
    best_move_san:
        Best move in Standard Algebraic Notation.
    depth_reached:
        Search depth reached for this update.
    elapsed_ms:
        Wall-clock time since analysis started, in milliseconds.
    """

    fen: str
    variations: tuple[VariationInfo, ...]
    evaluation: Evaluation
    best_move_uci: str
    best_move_san: str
    depth_reached: int
    elapsed_ms: float


AnalysisCallback = Callable[[ContinuousUpdate], None]


class ContinuousAnalysisHandle:
    """Handle for a running continuous analysis.

    Returned by :meth:`PositionAnalyser.analyse_continuous`.  Call
    :meth:`stop` to cancel the analysis early.
    """

    def __init__(
        self,
        stop_event: threading.Event,
        thread: threading.Thread,
    ) -> None:
        self._stop_event = stop_event
        self._thread = thread

    def stop(self) -> None:
        """Signal the analysis to stop."""
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        """``True`` if the analysis thread is still alive."""
        return self._thread.is_alive()

    def wait(self, timeout: float | None = None) -> None:
        """Block until the analysis thread finishes."""
        self._thread.join(timeout=timeout)


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
        engine_command = self._resolve_stockfish_command()
        logger.info(
            "Starting Stockfish: path=%s resolved=%s threads=%d hash=%dMB elo=%s",
            self._config.stockfish_path,
            engine_command,
            threads,
            self._config.hash_mb,
            self._config.engine_elo if self._config.engine_elo is not None else "max",
        )

        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(
                engine_command,
            )
        except (FileNotFoundError, chess.engine.EngineTerminatedError) as exc:
            raise EngineNotFoundError(
                self._format_engine_not_found_error(engine_command, exc),
            ) from exc

        engine_options: dict[str, int | bool] = {
            "Threads": threads,
            "Hash": self._config.hash_mb,
        }
        if self._config.engine_elo is not None:
            engine_options["UCI_LimitStrength"] = True
            engine_options["UCI_Elo"] = self._config.engine_elo
            engine_options["Skill Level"] = self._elo_to_skill_level(
                self._config.engine_elo,
            )

        self._engine.configure(engine_options)
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
        limit = self._build_search_limit()
        target_depth = limit.depth if limit.depth is not None else self._config.depth
        target_time = limit.time if limit.time is not None else self._config.time_limit
        effective_multipv = self._effective_multipv()
        logger.debug(
            "Analysing FEN=%s depth=%d time=%.3fs multipv=%d/%d elo=%s",
            fen,
            target_depth,
            target_time,
            effective_multipv,
            self._config.multipv,
            self._config.engine_elo if self._config.engine_elo is not None else "max",
        )

        infos: list[chess.engine.InfoDict] = self._engine.analyse(
            board,
            limit,
            multipv=effective_multipv,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        # Build variations from engine output.
        variations: list[VariationInfo] = []
        for rank, info in enumerate(infos, start=1):
            variations.append(self._to_variation_info(info, board, rank))
        variations = list(self._apply_strength_profile(fen, tuple(variations)))
        variations = variations[:self._config.multipv]

        depth_reached = variations[0].depth if variations else 0
        is_partial = depth_reached < target_depth

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

    # -- Continuous analysis --------------------------------------------------

    def analyse_continuous(
        self,
        fen: str,
        callback: AnalysisCallback,
        *,
        stop_event: threading.Event | None = None,
    ) -> ContinuousAnalysisHandle:
        """Start continuous analysis, calling *callback* as depth increases.

        The analysis runs on a daemon thread and emits a
        :class:`ContinuousUpdate` each time a new depth is fully reported
        across all principal variations.  The analysis stops when the
        configured depth or time limit is reached, or when the returned
        handle's :meth:`~ContinuousAnalysisHandle.stop` is called.

        Parameters
        ----------
        fen:
            A valid FEN string.
        callback:
            Called with a :class:`ContinuousUpdate` each time depth increases.
        stop_event:
            Optional external cancellation event.  If ``None`` a new one is
            created internally.

        Returns
        -------
        ContinuousAnalysisHandle
            A handle to stop or wait for the analysis.

        Raises
        ------
        EngineClosedError
            If the engine has not been started.
        AnalysisError
            If the FEN is invalid.
        """
        if self._engine is None:
            raise EngineClosedError("Engine is not running. Call start() first.")

        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise AnalysisError(f"Invalid FEN: {exc}") from exc

        event = stop_event or threading.Event()
        thread = threading.Thread(
            target=self._continuous_worker,
            args=(board, fen, callback, event),
            daemon=True,
            name="continuous-analysis",
        )
        thread.start()
        return ContinuousAnalysisHandle(event, thread)

    def _continuous_worker(
        self,
        board: chess.Board,
        fen: str,
        callback: AnalysisCallback,
        stop_event: threading.Event,
    ) -> None:
        """Background worker for continuous analysis."""
        t0 = time.perf_counter()

        # Terminal positions — emit single update and return.
        if board.is_checkmate():
            elapsed = (time.perf_counter() - t0) * 1000.0
            callback(ContinuousUpdate(
                fen=fen,
                variations=(),
                evaluation=Evaluation(centipawns=None, mate_in=0),
                best_move_uci="",
                best_move_san="",
                depth_reached=0,
                elapsed_ms=elapsed,
            ))
            return

        if board.is_stalemate():
            elapsed = (time.perf_counter() - t0) * 1000.0
            callback(ContinuousUpdate(
                fen=fen,
                variations=(),
                evaluation=Evaluation(centipawns=0, mate_in=None),
                best_move_uci="",
                best_move_san="",
                depth_reached=0,
                elapsed_ms=elapsed,
            ))
            return

        limit = self._build_search_limit()
        effective_multipv = self._effective_multipv()

        try:
            with self._engine.analysis(
                board, limit, multipv=effective_multipv,
            ) as stream:
                previous_depth = 0
                for _info in stream:
                    if stop_event.is_set():
                        stream.stop()
                        break

                    # Read grouped multi-PV state.
                    multipv_infos = stream.multipv
                    if not multipv_infos:
                        continue

                    current_depth = multipv_infos[0].get("depth", 0)
                    if current_depth <= previous_depth:
                        continue

                    previous_depth = current_depth
                    elapsed = (time.perf_counter() - t0) * 1000.0

                    variations: list[VariationInfo] = []
                    for rank, info in enumerate(multipv_infos, start=1):
                        variations.append(
                            self._to_variation_info(info, board, rank),
                        )
                    variations = list(self._apply_strength_profile(fen, tuple(variations)))
                    variations = variations[:self._config.multipv]

                    best_uci = variations[0].move_uci if variations else ""
                    best_san = variations[0].move_san if variations else ""
                    evaluation = (
                        variations[0].evaluation
                        if variations
                        else Evaluation(centipawns=0, mate_in=None)
                    )

                    callback(ContinuousUpdate(
                        fen=fen,
                        variations=tuple(variations),
                        evaluation=evaluation,
                        best_move_uci=best_uci,
                        best_move_san=best_san,
                        depth_reached=current_depth,
                        elapsed_ms=elapsed,
                    ))
        except chess.engine.EngineTerminatedError:
            logger.debug("Engine terminated during continuous analysis.")
        except Exception:
            logger.exception("Continuous analysis worker failed.")

    # -- Private helpers -----------------------------------------------------

    def _build_search_limit(self) -> chess.engine.Limit:
        """Build an effective search limit, scaled by selected Elo if set."""
        if self._config.engine_elo is None:
            return chess.engine.Limit(
                depth=self._config.depth,
                time=self._config.time_limit,
            )

        elo_ratio = self._elo_ratio(self._config.engine_elo)

        # Strongly cap search at low Elo so move quality drops noticeably.
        min_depth = min(4, self._config.depth)
        effective_depth = int(round(
            min_depth + elo_ratio * (self._config.depth - min_depth),
        ))

        min_time = min(0.03, self._config.time_limit)
        effective_time = min_time + elo_ratio * (self._config.time_limit - min_time)

        return chess.engine.Limit(
            depth=max(1, effective_depth),
            time=max(0.01, round(effective_time, 3)),
        )

    @staticmethod
    def _elo_ratio(elo: int) -> float:
        """Return normalised Elo ratio in [0.0, 1.0]."""
        span = AnalysisConfig.MAX_ENGINE_ELO - AnalysisConfig.MIN_ENGINE_ELO
        if span <= 0:
            return 1.0
        ratio = (elo - AnalysisConfig.MIN_ENGINE_ELO) / span
        return max(0.0, min(1.0, ratio))

    def _elo_to_skill_level(self, elo: int) -> int:
        """Map Elo to Stockfish Skill Level [0, 20] with progressive buckets."""
        if elo <= AnalysisConfig.MIN_ENGINE_ELO:
            return 0
        if elo >= AnalysisConfig.MAX_ENGINE_ELO:
            return 20
        return AnalysisConfig.engine_level_from_elo(elo)

    def _effective_multipv(self) -> int:
        """Increase candidate pool at low Elo so weaker moves can be selected."""
        if self._config.engine_elo is None:
            return self._config.multipv

        elo_ratio = self._elo_ratio(self._config.engine_elo)
        extra_candidates = int(round((1.0 - elo_ratio) * 9.0))
        return self._config.multipv + extra_candidates

    def _apply_strength_profile(
        self,
        fen: str,
        variations: tuple[VariationInfo, ...],
    ) -> tuple[VariationInfo, ...]:
        """Apply Elo-dependent move choice to avoid always returning top line."""
        if self._config.engine_elo is None or len(variations) <= 1:
            return variations

        elo_ratio = self._elo_ratio(self._config.engine_elo)
        top_cp = self._score_to_centipawns(variations[0].evaluation)
        tolerance_cp = 220.0 - (180.0 * elo_ratio)
        index_alpha = 0.25 + (1.75 * elo_ratio)

        weights: list[float] = []
        for idx, variation in enumerate(variations):
            cp_gap = max(
                0,
                top_cp - self._score_to_centipawns(variation.evaluation),
            )
            quality_weight = math.exp(-cp_gap / tolerance_cp) if cp_gap > 0 else 1.0
            index_weight = 1.0 / ((idx + 1) ** index_alpha)
            weights.append(quality_weight * index_weight)

        # At very low Elo, explicitly lower the top-line dominance.
        if weights and elo_ratio < 0.35:
            top_scale = 0.55 + elo_ratio
            weights[0] *= top_scale

        selected_index = self._pick_weighted_index(fen, weights)
        if selected_index == 0:
            return variations

        ordered = [variations[selected_index]]
        ordered.extend(
            variation
            for idx, variation in enumerate(variations)
            if idx != selected_index
        )
        return tuple(
            replace(variation, rank=index + 1)
            for index, variation in enumerate(ordered)
        )

    def _pick_weighted_index(self, fen: str, weights: list[float]) -> int:
        """Pick a stable weighted index from FEN + Elo."""
        total = sum(weights)
        if total <= 0:
            return 0

        elo = self._config.engine_elo
        assert elo is not None

        digest = hashlib.sha256(f"{fen}|{elo}".encode("utf-8")).digest()
        roll = int.from_bytes(digest[:8], "big") / float(1 << 64)
        target = roll * total

        cumulative = 0.0
        for idx, weight in enumerate(weights):
            cumulative += weight
            if target <= cumulative:
                return idx
        return len(weights) - 1

    @staticmethod
    def _score_to_centipawns(evaluation: Evaluation) -> int:
        """Convert evaluation (including mate scores) to comparable centipawns."""
        if evaluation.centipawns is not None:
            return evaluation.centipawns
        assert evaluation.mate_in is not None
        if evaluation.mate_in > 0:
            return 100_000
        return -100_000

    def _resolve_threads(self) -> int:
        """Resolve the number of Stockfish threads to use."""
        if self._config.threads is not None:
            return self._config.threads
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count - 1)

    def _resolve_stockfish_command(self) -> str:
        """Resolve the Stockfish command from config, env, PATH, and fallbacks."""
        configured_path = str(Path(self._config.stockfish_path).expanduser())
        if self._config.stockfish_path != "stockfish":
            return configured_path

        env_path = os.environ.get(_STOCKFISH_ENV_VAR)
        if env_path:
            return str(Path(env_path).expanduser())

        path_match = shutil.which("stockfish")
        if path_match:
            return path_match

        for candidate in _STOCKFISH_FALLBACK_CANDIDATES:
            if Path(candidate).is_file() and os.access(candidate, os.X_OK):
                return candidate

        return configured_path

    def _format_engine_not_found_error(
        self,
        resolved_path: str,
        exc: Exception,
    ) -> str:
        """Build a user-actionable error message when Stockfish cannot start."""
        configured_path = self._config.stockfish_path
        location = f"'{configured_path}'"
        if resolved_path != configured_path:
            location += f" (resolved to '{resolved_path}')"

        message = f"Cannot start Stockfish at {location}: {exc}."
        if configured_path == "stockfish":
            return (
                f"{message} Install Stockfish (Ubuntu: sudo apt install stockfish), "
                f"or set {_STOCKFISH_ENV_VAR} to the engine binary path."
            )
        return f"{message} Verify that the configured path points to an executable binary."

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
