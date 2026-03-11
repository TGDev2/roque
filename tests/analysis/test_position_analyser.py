"""Tests for the position analyser

Covers:
- Config validation (depth, time_limit, multipv, threads, hash_mb)
- Evaluation display and properties (centipawns, mate, stalemate)
- Nominal analysis with multi-PV (3 best moves, UCI + SAN)
- Mate / stalemate detection (terminal positions)
- Timeout / partial results (depth not reached)
- Engine lifecycle (start, stop, context manager)
- Engine not found / closed errors
- Thread auto-detection
- 10 reference positions with expected results
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import chess
import chess.engine
import pytest

from chesslab.analysis.position_analyser import (
    AnalysisConfig,
    AnalysisError,
    AnalysisResult,
    EngineClosedError,
    EngineNotFoundError,
    Evaluation,
    PositionAnalyser,
    VariationInfo,
)

# ---------------------------------------------------------------------------
# Constants — reference positions
# ---------------------------------------------------------------------------

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
ITALIAN_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
SCHOLAR_THREAT_FEN = (
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
)
MATE_IN_1_FEN = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
STALEMATE_FEN = "8/8/8/8/8/kq6/8/K7 w - - 0 1"
QUEEN_VS_PAWN_FEN = "K7/1P6/8/8/8/8/7q/5k2 w - - 0 1"
RUY_LOPEZ_FEN = (
    "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4"
)
LUCENA_FEN = "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1"

ALL_REFERENCE_FENS = [
    STARTING_FEN,
    AFTER_E4_FEN,
    ITALIAN_FEN,
    SCHOLAR_THREAT_FEN,
    MATE_IN_1_FEN,
    CHECKMATE_FEN,
    STALEMATE_FEN,
    QUEEN_VS_PAWN_FEN,
    RUY_LOPEZ_FEN,
    LUCENA_FEN,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pov_score_cp(cp: int) -> chess.engine.PovScore:
    """Create a centipawn PovScore relative to the side to move."""
    return chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)


def _make_pov_score_mate(moves: int) -> chess.engine.PovScore:
    """Create a mate-in-N PovScore relative to the side to move."""
    return chess.engine.PovScore(chess.engine.Mate(moves), chess.WHITE)


def _make_info(
    *,
    score: chess.engine.PovScore,
    pv: list[chess.Move],
    depth: int = 18,
) -> chess.engine.InfoDict:
    """Build a minimal InfoDict as returned by engine.analyse()."""
    return {"score": score, "pv": pv, "depth": depth}


def _make_analyser_with_mock(
    config: AnalysisConfig | None = None,
) -> tuple[PositionAnalyser, MagicMock]:
    """Create a PositionAnalyser with a mocked engine, already started."""
    analyser = PositionAnalyser(config)
    mock_engine = MagicMock(spec=chess.engine.SimpleEngine)
    with patch(
        "chesslab.analysis.position_analyser.chess.engine.SimpleEngine.popen_uci",
        return_value=mock_engine,
    ):
        analyser.start()
    return analyser, mock_engine


def _setup_mock_analysis(
    mock_engine: MagicMock,
    infos: list[chess.engine.InfoDict],
) -> None:
    """Configure the mock engine to return the given InfoDict list."""
    mock_engine.analyse.return_value = infos


# ---------------------------------------------------------------------------
# AC: Config validation
# ---------------------------------------------------------------------------


class TestAnalysisConfig:
    """AnalysisConfig validates its fields on construction."""

    def test_defaults(self) -> None:
        config = AnalysisConfig()
        assert config.stockfish_path == "stockfish"
        assert config.depth == 18
        assert config.time_limit == 2.0
        assert config.multipv == 3
        assert config.threads is None
        assert config.hash_mb == 128

    def test_custom_values(self) -> None:
        config = AnalysisConfig(
            stockfish_path="/usr/bin/stockfish",
            depth=20,
            time_limit=5.0,
            multipv=5,
            threads=4,
            hash_mb=256,
        )
        assert config.depth == 20
        assert config.threads == 4

    def test_depth_below_minimum_rejected(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            AnalysisConfig(depth=0)

    def test_time_limit_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="time_limit"):
            AnalysisConfig(time_limit=0)

    def test_time_limit_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="time_limit"):
            AnalysisConfig(time_limit=-1.0)

    def test_multipv_below_minimum_rejected(self) -> None:
        with pytest.raises(ValueError, match="multipv"):
            AnalysisConfig(multipv=0)

    def test_threads_below_minimum_rejected(self) -> None:
        with pytest.raises(ValueError, match="threads"):
            AnalysisConfig(threads=0)

    def test_threads_none_accepted(self) -> None:
        config = AnalysisConfig(threads=None)
        assert config.threads is None

    def test_hash_mb_below_minimum_rejected(self) -> None:
        with pytest.raises(ValueError, match="hash_mb"):
            AnalysisConfig(hash_mb=0)


# ---------------------------------------------------------------------------
# AC: Evaluation display and properties
# ---------------------------------------------------------------------------


class TestEvaluation:
    """Evaluation correctly formats scores and exposes is_mate."""

    def test_centipawns_positive(self) -> None:
        ev = Evaluation(centipawns=134, mate_in=None)
        assert str(ev) == "+1.34"
        assert not ev.is_mate

    def test_centipawns_negative(self) -> None:
        ev = Evaluation(centipawns=-50, mate_in=None)
        assert str(ev) == "-0.50"

    def test_centipawns_zero(self) -> None:
        ev = Evaluation(centipawns=0, mate_in=None)
        assert str(ev) == "+0.00"

    def test_mate_positive(self) -> None:
        ev = Evaluation(centipawns=None, mate_in=3)
        assert str(ev) == "#+3"
        assert ev.is_mate

    def test_mate_negative(self) -> None:
        ev = Evaluation(centipawns=None, mate_in=-2)
        assert str(ev) == "#-2"
        assert ev.is_mate

    def test_mate_zero(self) -> None:
        ev = Evaluation(centipawns=None, mate_in=0)
        assert str(ev) == "#0"
        assert ev.is_mate


# ---------------------------------------------------------------------------
# AC: Engine lifecycle
# ---------------------------------------------------------------------------


class TestEngineLifecycle:
    """PositionAnalyser manages the Stockfish process lifecycle."""

    def test_start_and_stop(self) -> None:
        analyser, mock_engine = _make_analyser_with_mock()
        assert analyser.is_running
        analyser.stop()
        assert not analyser.is_running
        mock_engine.quit.assert_called_once()

    def test_context_manager(self) -> None:
        with patch(
            "chesslab.analysis.position_analyser.chess.engine.SimpleEngine.popen_uci",
            return_value=MagicMock(spec=chess.engine.SimpleEngine),
        ):
            with PositionAnalyser() as analyser:
                assert analyser.is_running
            assert not analyser.is_running

    def test_start_idempotent(self) -> None:
        analyser, _ = _make_analyser_with_mock()
        analyser.start()  # second call should be a no-op
        assert analyser.is_running

    def test_stop_idempotent(self) -> None:
        analyser, _ = _make_analyser_with_mock()
        analyser.stop()
        analyser.stop()  # second stop should not raise
        assert not analyser.is_running

    def test_engine_not_found(self) -> None:
        with patch(
            "chesslab.analysis.position_analyser.chess.engine.SimpleEngine.popen_uci",
            side_effect=FileNotFoundError("stockfish not found"),
        ):
            analyser = PositionAnalyser()
            with pytest.raises(EngineNotFoundError, match="Cannot start Stockfish"):
                analyser.start()

    def test_analyse_when_stopped(self) -> None:
        analyser = PositionAnalyser()
        with pytest.raises(EngineClosedError, match="not running"):
            analyser.analyse(STARTING_FEN)

    def test_analyse_after_stop(self) -> None:
        analyser, _ = _make_analyser_with_mock()
        analyser.stop()
        with pytest.raises(EngineClosedError, match="not running"):
            analyser.analyse(STARTING_FEN)

    def test_thread_auto_detection(self) -> None:
        with patch.object(os, "cpu_count", return_value=8):
            analyser = PositionAnalyser()
            assert analyser._resolve_threads() == 7

    def test_thread_auto_detection_single_cpu(self) -> None:
        with patch.object(os, "cpu_count", return_value=1):
            analyser = PositionAnalyser()
            assert analyser._resolve_threads() == 1

    def test_thread_auto_detection_none_cpu(self) -> None:
        with patch.object(os, "cpu_count", return_value=None):
            analyser = PositionAnalyser()
            assert analyser._resolve_threads() == 1

    def test_thread_explicit(self) -> None:
        analyser = PositionAnalyser(AnalysisConfig(threads=4))
        assert analyser._resolve_threads() == 4


# ---------------------------------------------------------------------------
# AC: Invalid FEN
# ---------------------------------------------------------------------------


class TestInvalidFen:
    """analyse() rejects invalid FEN strings."""

    def test_invalid_fen_raises(self) -> None:
        analyser, _ = _make_analyser_with_mock()
        with pytest.raises(AnalysisError, match="Invalid FEN"):
            analyser.analyse("not a fen")

    def test_empty_fen_raises(self) -> None:
        analyser, _ = _make_analyser_with_mock()
        with pytest.raises(AnalysisError, match="Invalid FEN"):
            analyser.analyse("")


# ---------------------------------------------------------------------------
# AC: Terminal positions (checkmate / stalemate)
# ---------------------------------------------------------------------------


class TestTerminalPositions:
    """Checkmate and stalemate positions return synthetic results."""

    def test_checkmate_detected(self) -> None:
        analyser, mock_engine = _make_analyser_with_mock()
        result = analyser.analyse(CHECKMATE_FEN)

        assert result.evaluation.is_mate
        assert result.evaluation.mate_in == 0
        assert result.best_move_uci == ""
        assert result.best_move_san == ""
        assert result.variations == ()
        assert result.depth_reached == 0
        assert not result.is_partial
        # Engine should NOT be called for terminal positions.
        mock_engine.analyse.assert_not_called()

    def test_stalemate_detected(self) -> None:
        analyser, mock_engine = _make_analyser_with_mock()
        result = analyser.analyse(STALEMATE_FEN)

        assert not result.evaluation.is_mate
        assert result.evaluation.centipawns == 0
        assert str(result.evaluation) == "+0.00"
        assert result.best_move_uci == ""
        assert result.variations == ()
        mock_engine.analyse.assert_not_called()


# ---------------------------------------------------------------------------
# AC: Nominal analysis — 10 reference positions
# ---------------------------------------------------------------------------


def _build_starting_infos() -> list[chess.engine.InfoDict]:
    """Build 3 mock PV lines for the starting position."""
    board = chess.Board(STARTING_FEN)
    e2e4 = board.parse_uci("e2e4")
    d2d4 = board.parse_uci("d2d4")
    g1f3 = board.parse_uci("g1f3")
    return [
        _make_info(score=_make_pov_score_cp(30), pv=[e2e4], depth=18),
        _make_info(score=_make_pov_score_cp(25), pv=[d2d4], depth=18),
        _make_info(score=_make_pov_score_cp(15), pv=[g1f3], depth=18),
    ]


class TestNominalAnalysis:
    """Nominal multi-PV analysis returns evaluation and 3 best moves."""

    def test_starting_position(self) -> None:
        """Reference #1: Starting position — 3 PV lines with centipawn eval."""
        analyser, mock_engine = _make_analyser_with_mock()
        _setup_mock_analysis(mock_engine, _build_starting_infos())

        result = analyser.analyse(STARTING_FEN)

        assert result.fen == STARTING_FEN
        assert len(result.variations) == 3
        assert result.best_move_uci == "e2e4"
        assert result.best_move_san == "e4"
        assert not result.evaluation.is_mate
        assert result.evaluation.centipawns == 30
        assert result.depth_reached == 18
        assert not result.is_partial
        assert result.elapsed_ms >= 0

    def test_after_e4(self) -> None:
        """Reference #2: After 1.e4 — black to move."""
        board = chess.Board(AFTER_E4_FEN)
        e7e5 = board.parse_uci("e7e5")
        c7c5 = board.parse_uci("c7c5")
        e7e6 = board.parse_uci("e7e6")
        infos = [
            _make_info(score=_make_pov_score_cp(-20), pv=[e7e5], depth=18),
            _make_info(score=_make_pov_score_cp(-25), pv=[c7c5], depth=18),
            _make_info(score=_make_pov_score_cp(-30), pv=[e7e6], depth=18),
        ]
        analyser, mock_engine = _make_analyser_with_mock()
        _setup_mock_analysis(mock_engine, infos)
        result = analyser.analyse(AFTER_E4_FEN)

        assert len(result.variations) == 3
        assert result.best_move_uci == "e7e5"
        assert not result.evaluation.is_mate

    def test_italian_game(self) -> None:
        """Reference #3: Italian Game — balanced position."""
        board = chess.Board(ITALIAN_FEN)
        g8f6 = board.parse_uci("g8f6")
        f8c5 = board.parse_uci("f8c5")
        d7d6 = board.parse_uci("d7d6")
        infos = [
            _make_info(score=_make_pov_score_cp(-5), pv=[g8f6], depth=18),
            _make_info(score=_make_pov_score_cp(-10), pv=[f8c5], depth=18),
            _make_info(score=_make_pov_score_cp(-15), pv=[d7d6], depth=18),
        ]
        analyser, mock_engine = _make_analyser_with_mock()
        _setup_mock_analysis(mock_engine, infos)
        result = analyser.analyse(ITALIAN_FEN)

        assert len(result.variations) == 3
        assert result.variations[0].rank == 1
        assert result.variations[1].rank == 2
        assert result.variations[2].rank == 3

    def test_scholar_threat(self) -> None:
        """Reference #4: Scholar's mate threat — white advantage."""
        board = chess.Board(SCHOLAR_THREAT_FEN)
        h5f7 = board.parse_uci("h5f7")
        d2d3 = board.parse_uci("d2d3")
        b1c3 = board.parse_uci("b1c3")
        infos = [
            _make_info(score=_make_pov_score_cp(600), pv=[h5f7], depth=18),
            _make_info(score=_make_pov_score_cp(150), pv=[d2d3], depth=18),
            _make_info(score=_make_pov_score_cp(100), pv=[b1c3], depth=18),
        ]
        analyser, mock_engine = _make_analyser_with_mock()
        _setup_mock_analysis(mock_engine, infos)
        result = analyser.analyse(SCHOLAR_THREAT_FEN)

        assert result.evaluation.centipawns == 600
        assert result.best_move_uci == "h5f7"

    def test_mate_in_one(self) -> None:
        """Reference #5: Mate in 1 — Re8#."""
        board = chess.Board(MATE_IN_1_FEN)
        e1e8 = board.parse_uci("e1e8")
        infos = [
            _make_info(score=_make_pov_score_mate(1), pv=[e1e8], depth=18),
        ]
        analyser, mock_engine = _make_analyser_with_mock()
        _setup_mock_analysis(mock_engine, infos)
        result = analyser.analyse(MATE_IN_1_FEN)

        assert result.evaluation.is_mate
        assert result.evaluation.mate_in == 1
        assert result.best_move_uci == "e1e8"
        assert "#+1" in str(result.evaluation)

    def test_checkmate_position(self) -> None:
        """Reference #6: Already checkmate — no engine call."""
        analyser, mock_engine = _make_analyser_with_mock()
        result = analyser.analyse(CHECKMATE_FEN)

        assert result.evaluation.mate_in == 0
        assert result.variations == ()
        mock_engine.analyse.assert_not_called()

    def test_stalemate_position(self) -> None:
        """Reference #7: Stalemate — eval 0, no moves."""
        analyser, mock_engine = _make_analyser_with_mock()
        result = analyser.analyse(STALEMATE_FEN)

        assert result.evaluation.centipawns == 0
        assert not result.evaluation.is_mate
        mock_engine.analyse.assert_not_called()

    def test_queen_vs_pawn(self) -> None:
        """Reference #8: Queen vs pawn endgame."""
        board = chess.Board(QUEEN_VS_PAWN_FEN)
        b7b8q = board.parse_uci("b7b8q")
        b7b8r = board.parse_uci("b7b8r")
        a8a7 = board.parse_uci("a8a7")
        infos = [
            _make_info(score=_make_pov_score_cp(200), pv=[b7b8q], depth=18),
            _make_info(score=_make_pov_score_cp(150), pv=[b7b8r], depth=18),
            _make_info(score=_make_pov_score_cp(-100), pv=[a8a7], depth=18),
        ]
        analyser, mock_engine = _make_analyser_with_mock()
        _setup_mock_analysis(mock_engine, infos)
        result = analyser.analyse(QUEEN_VS_PAWN_FEN)

        assert len(result.variations) == 3
        assert result.best_move_uci == "b7b8q"

    def test_ruy_lopez(self) -> None:
        """Reference #9: Ruy Lopez — roughly equal."""
        board = chess.Board(RUY_LOPEZ_FEN)
        a6a5 = board.parse_uci("a6a5")
        g8f6 = board.parse_uci("g8f6")
        f8c5 = board.parse_uci("f8c5")
        infos = [
            _make_info(score=_make_pov_score_cp(-10), pv=[a6a5], depth=18),
            _make_info(score=_make_pov_score_cp(-15), pv=[g8f6], depth=18),
            _make_info(score=_make_pov_score_cp(-20), pv=[f8c5], depth=18),
        ]
        analyser, mock_engine = _make_analyser_with_mock()
        _setup_mock_analysis(mock_engine, infos)
        result = analyser.analyse(RUY_LOPEZ_FEN)

        assert len(result.variations) == 3

    def test_lucena_position(self) -> None:
        """Reference #10: Lucena — white winning."""
        board = chess.Board(LUCENA_FEN)
        c1c8 = board.parse_uci("c1c8")
        c1a1 = board.parse_uci("c1a1")
        c1d1 = board.parse_uci("c1d1")
        infos = [
            _make_info(score=_make_pov_score_mate(5), pv=[c1c8], depth=18),
            _make_info(score=_make_pov_score_mate(8), pv=[c1a1], depth=18),
            _make_info(score=_make_pov_score_cp(900), pv=[c1d1], depth=18),
        ]
        analyser, mock_engine = _make_analyser_with_mock()
        _setup_mock_analysis(mock_engine, infos)
        result = analyser.analyse(LUCENA_FEN)

        assert result.evaluation.is_mate
        assert result.evaluation.mate_in == 5
        assert result.best_move_uci == "c1c8"


# ---------------------------------------------------------------------------
# AC: Partial results / timeout
# ---------------------------------------------------------------------------


class TestPartialResults:
    """Analysis interrupted by time limit returns partial results."""

    def test_partial_when_depth_not_reached(self) -> None:
        board = chess.Board(STARTING_FEN)
        e2e4 = board.parse_uci("e2e4")
        infos = [
            _make_info(score=_make_pov_score_cp(20), pv=[e2e4], depth=12),
        ]
        analyser, mock_engine = _make_analyser_with_mock()
        _setup_mock_analysis(mock_engine, infos)
        result = analyser.analyse(STARTING_FEN)

        assert result.is_partial
        assert result.depth_reached == 12

    def test_not_partial_when_depth_reached(self) -> None:
        board = chess.Board(STARTING_FEN)
        e2e4 = board.parse_uci("e2e4")
        infos = [
            _make_info(score=_make_pov_score_cp(20), pv=[e2e4], depth=18),
        ]
        analyser, mock_engine = _make_analyser_with_mock()
        _setup_mock_analysis(mock_engine, infos)
        result = analyser.analyse(STARTING_FEN)

        assert not result.is_partial
        assert result.depth_reached == 18


# ---------------------------------------------------------------------------
# AC: Variation details (UCI + SAN, PV lines)
# ---------------------------------------------------------------------------


class TestVariationDetails:
    """Variations contain correct UCI and SAN move notations."""

    def test_pv_uci_and_san(self) -> None:
        board = chess.Board(STARTING_FEN)
        e2e4 = board.parse_uci("e2e4")
        # Multi-move PV: 1.e4 e5 2.Nf3
        board_after_e4 = board.copy()
        board_after_e4.push(e2e4)
        e7e5 = board_after_e4.parse_uci("e7e5")
        board_after_e5 = board_after_e4.copy()
        board_after_e5.push(e7e5)
        g1f3 = board_after_e5.parse_uci("g1f3")

        infos = [
            _make_info(
                score=_make_pov_score_cp(25),
                pv=[e2e4, e7e5, g1f3],
                depth=18,
            ),
        ]
        analyser, mock_engine = _make_analyser_with_mock()
        _setup_mock_analysis(mock_engine, infos)
        result = analyser.analyse(STARTING_FEN)

        var = result.variations[0]
        assert var.rank == 1
        assert var.move_uci == "e2e4"
        assert var.move_san == "e4"
        assert var.pv_uci == ("e2e4", "e7e5", "g1f3")
        assert var.pv_san == ("e4", "e5", "Nf3")
        assert var.depth == 18

    def test_engine_called_with_correct_limits(self) -> None:
        config = AnalysisConfig(depth=20, time_limit=3.0, multipv=5)
        analyser, mock_engine = _make_analyser_with_mock(config)
        board = chess.Board(STARTING_FEN)
        e2e4 = board.parse_uci("e2e4")
        _setup_mock_analysis(
            mock_engine,
            [_make_info(score=_make_pov_score_cp(20), pv=[e2e4], depth=20)],
        )
        analyser.analyse(STARTING_FEN)

        call_args = mock_engine.analyse.call_args
        limit = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["limit"]
        assert limit.depth == 20
        assert limit.time == 3.0
        assert call_args[1].get("multipv", call_args[0][2] if len(call_args[0]) > 2 else None) == 5

    def test_engine_configured_with_threads_and_hash(self) -> None:
        config = AnalysisConfig(threads=6, hash_mb=256)
        with patch(
            "chesslab.analysis.position_analyser.chess.engine.SimpleEngine.popen_uci",
            return_value=MagicMock(spec=chess.engine.SimpleEngine),
        ) as _:
            analyser = PositionAnalyser(config)
            analyser.start()
            engine = analyser._engine
            engine.configure.assert_called_once_with({"Threads": 6, "Hash": 256})
            analyser.stop()


# ---------------------------------------------------------------------------
# AC: Result dataclass properties
# ---------------------------------------------------------------------------


class TestAnalysisResult:
    """AnalysisResult is frozen and exposes expected attributes."""

    def test_result_is_frozen(self) -> None:
        result = AnalysisResult(
            fen=STARTING_FEN,
            best_move_uci="e2e4",
            best_move_san="e4",
            evaluation=Evaluation(centipawns=30, mate_in=None),
            variations=(),
            depth_reached=18,
            is_partial=False,
            elapsed_ms=100.0,
        )
        with pytest.raises(AttributeError):
            result.fen = "other"  # type: ignore[misc]

    def test_variation_info_is_frozen(self) -> None:
        var = VariationInfo(
            rank=1,
            move_uci="e2e4",
            move_san="e4",
            evaluation=Evaluation(centipawns=30, mate_in=None),
            pv_uci=("e2e4",),
            pv_san=("e4",),
            depth=18,
        )
        with pytest.raises(AttributeError):
            var.rank = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Analysis preferences persistence
# ---------------------------------------------------------------------------


class TestAnalysisPreferences:
    """Saving and loading analysis depth/time_limit preferences."""

    def test_save_and_load_roundtrip(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path))
        AnalysisConfig.save_preference(depth=20, time_limit=3.0, config_dir=d)
        result = AnalysisConfig.load_preference(config_dir=d)
        assert result is not None
        assert result["depth"] == 20
        assert result["time_limit"] == 3.0

    def test_save_creates_directory(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path)) / "nested" / "dir"
        AnalysisConfig.save_preference(depth=15, time_limit=1.5, config_dir=d)
        assert (d / "config.json").is_file()
        result = AnalysisConfig.load_preference(config_dir=d)
        assert result is not None
        assert result["depth"] == 15

    def test_save_preserves_other_keys(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path))
        d.mkdir(parents=True, exist_ok=True)
        config_file = d / "config.json"
        config_file.write_text(json.dumps({"theme": "chess.com/green"}))

        AnalysisConfig.save_preference(depth=18, time_limit=2.0, config_dir=d)

        with config_file.open() as fh:
            data = json.load(fh)
        assert data["theme"] == "chess.com/green"
        assert data["analysis"]["depth"] == 18

    def test_load_nonexistent_returns_none(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path))
        assert AnalysisConfig.load_preference(config_dir=d) is None

    def test_load_corrupt_json_returns_none(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path))
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("not valid json {{{")
        assert AnalysisConfig.load_preference(config_dir=d) is None

    def test_save_overwrites_previous(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path))
        AnalysisConfig.save_preference(depth=12, time_limit=1.0, config_dir=d)
        AnalysisConfig.save_preference(depth=22, time_limit=5.0, config_dir=d)
        result = AnalysisConfig.load_preference(config_dir=d)
        assert result is not None
        assert result["depth"] == 22
        assert result["time_limit"] == 5.0

    def test_save_depth_below_min_raises(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path))
        with pytest.raises(ValueError, match="depth must be between"):
            AnalysisConfig.save_preference(depth=5, time_limit=2.0, config_dir=d)

    def test_save_depth_above_max_raises(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path))
        with pytest.raises(ValueError, match="depth must be between"):
            AnalysisConfig.save_preference(depth=30, time_limit=2.0, config_dir=d)

    def test_save_depth_at_boundaries(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path))
        AnalysisConfig.save_preference(depth=10, time_limit=1.0, config_dir=d)
        result = AnalysisConfig.load_preference(config_dir=d)
        assert result is not None
        assert result["depth"] == 10

        AnalysisConfig.save_preference(depth=25, time_limit=1.0, config_dir=d)
        result = AnalysisConfig.load_preference(config_dir=d)
        assert result is not None
        assert result["depth"] == 25

    def test_save_invalid_time_limit_raises(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path))
        with pytest.raises(ValueError, match="time_limit must be > 0"):
            AnalysisConfig.save_preference(depth=18, time_limit=0, config_dir=d)
        with pytest.raises(ValueError, match="time_limit must be > 0"):
            AnalysisConfig.save_preference(depth=18, time_limit=-1.0, config_dir=d)

    def test_min_max_constants(self) -> None:
        assert AnalysisConfig.MIN_USER_DEPTH == 10
        assert AnalysisConfig.MAX_USER_DEPTH == 25


# ---------------------------------------------------------------------------
# stimated analysis time
# ---------------------------------------------------------------------------


class TestEstimateAnalysisTime:
    """Heuristic analysis time estimation."""

    def test_returns_float(self) -> None:
        result = AnalysisConfig.estimate_analysis_time(depth=18)
        assert isinstance(result, float)

    def test_increases_with_depth(self) -> None:
        t15 = AnalysisConfig.estimate_analysis_time(depth=15, threads=1)
        t20 = AnalysisConfig.estimate_analysis_time(depth=20, threads=1)
        assert t20 > t15

    def test_decreases_with_more_threads(self) -> None:
        t1 = AnalysisConfig.estimate_analysis_time(depth=20, threads=1)
        t8 = AnalysisConfig.estimate_analysis_time(depth=20, threads=8)
        assert t8 < t1

    def test_clamped_to_bounds(self) -> None:
        low = AnalysisConfig.estimate_analysis_time(depth=1, threads=1)
        assert low >= 0.1
        high = AnalysisConfig.estimate_analysis_time(depth=50, threads=1)
        assert high <= 300.0

    def test_depth_10_baseline_reasonable(self) -> None:
        t = AnalysisConfig.estimate_analysis_time(depth=10, threads=1)
        assert 0.1 <= t <= 1.0
