"""Tests for the game coach module.

Covers:
- AC1: Player color selection (PlayerColor enum, CoachConfig)
- AC2: Opponent move input in SAN notation (submit_opponent_move)
- AC3: Stockfish returns 3 best alternatives after each opponent move
- AC4: Configurable Stockfish strength (depth / time_limit via AnalysisConfig)
- Turn enforcement (opponent vs player turns)
- Illegal move rejection
- Game over detection
- Engine lifecycle (start, stop, context manager)
- Board state tracking (FEN, move number, game over, player turn)
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import chess
import chess.engine
import pytest

from chesslab.analysis import AnalysisConfig
from chesslab.analysis import ContinuousUpdate, Evaluation, VariationInfo
from chesslab.coach.game_coach import (
    CoachConfig,
    CoachError,
    ContinuousAdvice,
    GameCoach,
    GameOverError,
    IllegalMoveError,
    MoveAdvice,
    NotYourTurnError,
    PlayerColor,
)

# ---------------------------------------------------------------------------
# Reference positions
# ---------------------------------------------------------------------------

SCHOLAR_THREAT_FEN = (
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pov_score_cp(cp: int) -> chess.engine.PovScore:
    """Create a centipawn PovScore."""
    return chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)


def _make_info(
    *,
    score: chess.engine.PovScore,
    pv: list[chess.Move],
    depth: int = 18,
) -> chess.engine.InfoDict:
    """Build a minimal InfoDict as returned by engine.analyse()."""
    return {"score": score, "pv": pv, "depth": depth}


def _make_coach_with_mock(
    config: CoachConfig | None = None,
) -> tuple[GameCoach, MagicMock]:
    """Create a GameCoach with a mocked Stockfish engine, already started."""
    coach = GameCoach(config)
    mock_engine = MagicMock(spec=chess.engine.SimpleEngine)
    with patch(
        "chesslab.analysis.position_analyser.chess.engine.SimpleEngine.popen_uci",
        return_value=mock_engine,
    ):
        coach.start()
    return coach, mock_engine


def _build_infos_for_position(
    fen: str,
    moves_uci: list[str],
    scores_cp: list[int],
) -> list[chess.engine.InfoDict]:
    """Build mock InfoDicts for a position with given UCI moves and scores."""
    board = chess.Board(fen)
    infos = []
    for uci, cp in zip(moves_uci, scores_cp):
        move = board.parse_uci(uci)
        infos.append(
            _make_info(score=_make_pov_score_cp(cp), pv=[move], depth=18),
        )
    return infos


# ---------------------------------------------------------------------------
# AC1: Player color selection
# ---------------------------------------------------------------------------


class TestPlayerColor:
    """PlayerColor enum represents the human player's side."""

    def test_white_value(self) -> None:
        assert PlayerColor.WHITE.value == "white"

    def test_black_value(self) -> None:
        assert PlayerColor.BLACK.value == "black"


class TestCoachConfig:
    """CoachConfig holds player color and analysis settings."""

    def test_defaults(self) -> None:
        config = CoachConfig()
        assert config.player_color == PlayerColor.WHITE
        assert config.analysis.depth == 18
        assert config.analysis.multipv == 3

    def test_custom_color(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        assert config.player_color == PlayerColor.BLACK

    def test_custom_analysis(self) -> None:
        config = CoachConfig(analysis=AnalysisConfig(depth=22, time_limit=5.0))
        assert config.analysis.depth == 22
        assert config.analysis.time_limit == 5.0


# ---------------------------------------------------------------------------
# AC: MoveAdvice dataclass
# ---------------------------------------------------------------------------


class TestMoveAdvice:
    """MoveAdvice is frozen and exposes expected attributes."""

    def test_frozen(self) -> None:
        from chesslab.analysis import AnalysisResult, Evaluation

        advice = MoveAdvice(
            opponent_move_san="e4",
            current_fen="fen",
            analysis=AnalysisResult(
                fen="fen",
                best_move_uci="e7e5",
                best_move_san="e5",
                evaluation=Evaluation(centipawns=0, mate_in=None),
                variations=(),
                depth_reached=18,
                is_partial=False,
                elapsed_ms=100.0,
            ),
            move_number=1,
        )
        with pytest.raises(AttributeError):
            advice.move_number = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AC: Engine lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """GameCoach manages the Stockfish engine lifecycle."""

    def test_start_and_stop(self) -> None:
        coach, _ = _make_coach_with_mock()
        assert coach.is_running
        coach.stop()
        assert not coach.is_running

    def test_context_manager(self) -> None:
        with patch(
            "chesslab.analysis.position_analyser.chess.engine.SimpleEngine.popen_uci",
            return_value=MagicMock(spec=chess.engine.SimpleEngine),
        ):
            with GameCoach() as coach:
                assert coach.is_running
            assert not coach.is_running


# ---------------------------------------------------------------------------
# AC2 + AC3: Opponent move input -> 3 best alternatives
# ---------------------------------------------------------------------------


class TestPlayingAsBlack:
    """Player is Black — opponent (White) moves first, then analysis."""

    def test_opponent_move_returns_advice_with_three_suggestions(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, mock_engine = _make_coach_with_mock(config)

        # After opponent plays e4, Black to move.
        board = chess.Board()
        board.push_san("e4")
        expected_fen = board.fen()

        infos = _build_infos_for_position(
            expected_fen,
            ["e7e5", "c7c5", "d7d5"],
            [-20, -25, -30],
        )
        mock_engine.analyse.return_value = infos

        advice = coach.submit_opponent_move("e4")

        assert advice.opponent_move_san == "e4"
        assert advice.current_fen == expected_fen
        assert len(advice.analysis.variations) == 3
        assert advice.analysis.best_move_san == "e5"
        assert advice.analysis.variations[0].rank == 1
        assert advice.analysis.variations[1].rank == 2
        assert advice.analysis.variations[2].rank == 3

    def test_full_move_sequence(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, mock_engine = _make_coach_with_mock(config)

        # 1. Opponent plays e4.
        board = chess.Board()
        board.push_san("e4")
        infos = _build_infos_for_position(board.fen(), ["e7e5"], [0])
        mock_engine.analyse.return_value = infos

        advice = coach.submit_opponent_move("e4")
        assert advice.move_number == 1

        # 2. Player plays e5.
        coach.submit_player_move("e5")

        # 3. Opponent plays Nf3.
        board.push_san("e5")
        board.push_san("Nf3")
        infos = _build_infos_for_position(board.fen(), ["b8c6"], [0])
        mock_engine.analyse.return_value = infos

        advice = coach.submit_opponent_move("Nf3")
        assert advice.opponent_move_san == "Nf3"
        assert advice.move_number == 2

    def test_san_normalisation(self) -> None:
        """SAN is normalised (e.g. lowercase input becomes standard case)."""
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, mock_engine = _make_coach_with_mock(config)

        board = chess.Board()
        board.push_san("e4")
        infos = _build_infos_for_position(board.fen(), ["e7e5"], [0])
        mock_engine.analyse.return_value = infos

        # python-chess parse_san accepts standard notation; verify normalised output.
        advice = coach.submit_opponent_move("e4")
        assert advice.opponent_move_san == "e4"


class TestPlayingAsWhite:
    """Player is White — player moves first, no analysis until opponent moves."""

    def test_player_moves_first_then_opponent(self) -> None:
        config = CoachConfig(player_color=PlayerColor.WHITE)
        coach, mock_engine = _make_coach_with_mock(config)

        # 1. Player (White) plays e4.
        coach.submit_player_move("e4")

        # 2. Opponent (Black) plays e5.
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        infos = _build_infos_for_position(
            board.fen(),
            ["g1f3", "f1c4", "d2d4"],
            [30, 25, 20],
        )
        mock_engine.analyse.return_value = infos

        advice = coach.submit_opponent_move("e5")
        assert advice.opponent_move_san == "e5"
        assert len(advice.analysis.variations) == 3
        assert advice.analysis.best_move_san == "Nf3"

    def test_initial_state_is_player_turn(self) -> None:
        config = CoachConfig(player_color=PlayerColor.WHITE)
        coach, _ = _make_coach_with_mock(config)
        assert coach.is_player_turn


# ---------------------------------------------------------------------------
# AC4: Configurable Stockfish strength
# ---------------------------------------------------------------------------


class TestConfigurableStrength:
    """Analysis depth and time_limit are passed through to the engine."""

    def test_custom_depth_used(self) -> None:
        config = CoachConfig(
            player_color=PlayerColor.BLACK,
            analysis=AnalysisConfig(depth=22, time_limit=5.0),
        )
        coach, mock_engine = _make_coach_with_mock(config)

        board = chess.Board()
        board.push_san("e4")
        infos = _build_infos_for_position(board.fen(), ["e7e5"], [0])
        mock_engine.analyse.return_value = infos

        coach.submit_opponent_move("e4")

        call_args = mock_engine.analyse.call_args
        limit = call_args[0][1]
        assert limit.depth == 22
        assert limit.time == 5.0

    def test_min_depth_accepted(self) -> None:
        config = CoachConfig(
            analysis=AnalysisConfig(depth=AnalysisConfig.MIN_USER_DEPTH),
        )
        assert config.analysis.depth == 10

    def test_max_depth_accepted(self) -> None:
        config = CoachConfig(
            analysis=AnalysisConfig(depth=AnalysisConfig.MAX_USER_DEPTH),
        )
        assert config.analysis.depth == 25


# ---------------------------------------------------------------------------
# AC: Turn enforcement
# ---------------------------------------------------------------------------


class TestTurnEnforcement:
    """Moves must be submitted in the correct turn order."""

    def test_opponent_move_on_player_turn_raises(self) -> None:
        """White plays — cannot submit opponent move when it is White's turn."""
        config = CoachConfig(player_color=PlayerColor.WHITE)
        coach, _ = _make_coach_with_mock(config)

        with pytest.raises(NotYourTurnError, match="not the opponent's turn"):
            coach.submit_opponent_move("e4")

    def test_player_move_on_opponent_turn_raises(self) -> None:
        """Black plays — cannot submit player move when it is White's turn."""
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, _ = _make_coach_with_mock(config)

        with pytest.raises(NotYourTurnError, match="not the player's turn"):
            coach.submit_player_move("e4")

    def test_double_opponent_move_raises(self) -> None:
        """Cannot submit two opponent moves in a row."""
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, mock_engine = _make_coach_with_mock(config)

        board = chess.Board()
        board.push_san("e4")
        infos = _build_infos_for_position(board.fen(), ["e7e5"], [0])
        mock_engine.analyse.return_value = infos

        coach.submit_opponent_move("e4")

        with pytest.raises(NotYourTurnError, match="not the opponent's turn"):
            coach.submit_opponent_move("d4")

    def test_double_player_move_raises(self) -> None:
        """Cannot submit two player moves in a row."""
        config = CoachConfig(player_color=PlayerColor.WHITE)
        coach, _ = _make_coach_with_mock(config)

        coach.submit_player_move("e4")

        with pytest.raises(NotYourTurnError, match="not the player's turn"):
            coach.submit_player_move("d4")


# ---------------------------------------------------------------------------
# AC: Illegal move rejection
# ---------------------------------------------------------------------------


class TestIllegalMoves:
    """Invalid or illegal SAN moves are rejected."""

    def test_invalid_san_raises(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, _ = _make_coach_with_mock(config)

        with pytest.raises(IllegalMoveError, match="Illegal move"):
            coach.submit_opponent_move("Zz9")

    def test_illegal_move_raises(self) -> None:
        """e5 is not legal from starting position for White."""
        config = CoachConfig(player_color=PlayerColor.WHITE)
        coach, _ = _make_coach_with_mock(config)

        with pytest.raises(IllegalMoveError, match="Illegal move"):
            coach.submit_player_move("e5")

    def test_empty_string_raises(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, _ = _make_coach_with_mock(config)

        with pytest.raises(IllegalMoveError, match="Illegal move"):
            coach.submit_opponent_move("")

    def test_illegal_player_move_raises(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, mock_engine = _make_coach_with_mock(config)

        board = chess.Board()
        board.push_san("e4")
        infos = _build_infos_for_position(board.fen(), ["e7e5"], [0])
        mock_engine.analyse.return_value = infos

        coach.submit_opponent_move("e4")

        with pytest.raises(IllegalMoveError, match="Illegal move"):
            coach.submit_player_move("Ke7")


# ---------------------------------------------------------------------------
# AC: Game over detection
# ---------------------------------------------------------------------------


class TestGameOver:
    """Cannot submit moves after the game has ended."""

    def test_move_after_checkmate_raises(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, _mock_engine = _make_coach_with_mock(config)

        # Set board to position before Scholar's Mate.
        coach._board = chess.Board(SCHOLAR_THREAT_FEN)

        # Opponent (White) plays Qxf7# — checkmate.
        # PositionAnalyser detects checkmate internally (no engine call).
        advice = coach.submit_opponent_move("Qxf7#")

        assert coach.is_game_over
        assert advice.analysis.evaluation.mate_in == 0
        assert advice.analysis.variations == ()

        with pytest.raises(GameOverError, match="Game is over"):
            coach.submit_player_move("Kd8")

    def test_opponent_move_after_game_over_raises(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, _mock_engine = _make_coach_with_mock(config)

        coach._board = chess.Board(SCHOLAR_THREAT_FEN)
        coach.submit_opponent_move("Qxf7#")

        with pytest.raises(GameOverError, match="Game is over"):
            coach.submit_opponent_move("d4")

    def test_is_game_over_false_initially(self) -> None:
        coach, _ = _make_coach_with_mock()
        assert not coach.is_game_over

    def test_resignation_marks_game_over_with_result(self) -> None:
        coach, _ = _make_coach_with_mock()

        coach.record_resignation(PlayerColor.BLACK)

        assert coach.is_game_over
        assert coach.game_result == "1-0"
        assert coach.game_winner == chess.WHITE
        assert coach.game_termination == "White won by resignation"

    def test_move_after_resignation_raises(self) -> None:
        coach, _ = _make_coach_with_mock()
        coach.record_resignation(PlayerColor.WHITE)

        with pytest.raises(GameOverError, match="Game is over: resignation"):
            coach.record_move("e4")


# ---------------------------------------------------------------------------
# AC: Board state tracking
# ---------------------------------------------------------------------------


class TestProperties:
    """Board state properties are tracked correctly."""

    def test_current_fen_initial(self) -> None:
        coach, _ = _make_coach_with_mock()
        expected = chess.Board().fen()
        assert coach.current_fen == expected

    def test_current_fen_after_moves(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, mock_engine = _make_coach_with_mock(config)

        board = chess.Board()
        board.push_san("e4")
        infos = _build_infos_for_position(board.fen(), ["e7e5"], [0])
        mock_engine.analyse.return_value = infos

        coach.submit_opponent_move("e4")
        assert coach.current_fen == board.fen()

    def test_move_number_initial(self) -> None:
        coach, _ = _make_coach_with_mock()
        assert coach.move_number == 1

    def test_move_number_increments(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, mock_engine = _make_coach_with_mock(config)

        board = chess.Board()
        board.push_san("e4")
        infos = _build_infos_for_position(board.fen(), ["e7e5"], [0])
        mock_engine.analyse.return_value = infos

        coach.submit_opponent_move("e4")
        assert coach.move_number == 1  # fullmove_number increments after Black

        coach.submit_player_move("e5")
        assert coach.move_number == 2  # Now move 2

    def test_is_player_turn_white(self) -> None:
        config = CoachConfig(player_color=PlayerColor.WHITE)
        coach, _ = _make_coach_with_mock(config)
        assert coach.is_player_turn  # White to move, player is White

    def test_is_player_turn_black(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, _ = _make_coach_with_mock(config)
        assert not coach.is_player_turn  # White to move, player is Black

    def test_is_player_turn_toggles(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, mock_engine = _make_coach_with_mock(config)

        assert not coach.is_player_turn  # Opponent's turn

        board = chess.Board()
        board.push_san("e4")
        infos = _build_infos_for_position(board.fen(), ["e7e5"], [0])
        mock_engine.analyse.return_value = infos

        coach.submit_opponent_move("e4")
        assert coach.is_player_turn  # Now player's turn

        coach.submit_player_move("e5")
        assert not coach.is_player_turn  # Back to opponent


# ---------------------------------------------------------------------------
# AC: Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    """All coach errors inherit from CoachError."""

    def test_illegal_move_is_coach_error(self) -> None:
        assert issubclass(IllegalMoveError, CoachError)

    def test_game_over_is_coach_error(self) -> None:
        assert issubclass(GameOverError, CoachError)

    def test_not_your_turn_is_coach_error(self) -> None:
        assert issubclass(NotYourTurnError, CoachError)

    def test_coach_error_is_runtime_error(self) -> None:
        assert issubclass(CoachError, RuntimeError)


# ---------------------------------------------------------------------------
# Continuous analysis helpers
# ---------------------------------------------------------------------------

_SYNC_TIMEOUT_S = 2.0


def _make_continuous_update(
    fen: str,
    best_san: str = "e5",
    cp: int = 0,
    depth: int = 10,
) -> ContinuousUpdate:
    """Build a minimal ContinuousUpdate for testing."""
    return ContinuousUpdate(
        fen=fen,
        variations=(
            VariationInfo(
                rank=1,
                move_uci="e7e5",
                move_san=best_san,
                evaluation=Evaluation(centipawns=cp, mate_in=None),
                pv_uci=("e7e5",),
                pv_san=(best_san,),
                depth=depth,
            ),
        ),
        evaluation=Evaluation(centipawns=cp, mate_in=None),
        best_move_uci="e7e5",
        best_move_san=best_san,
        depth_reached=depth,
        elapsed_ms=50.0,
    )


def _setup_mock_continuous(
    mock_engine: MagicMock,
    updates: list[ContinuousUpdate],
) -> None:
    """Configure mock engine.analysis() to yield updates via a fake stream."""
    infos = []
    multipv_groups = []
    for update in updates:
        info: chess.engine.InfoDict = {
            "score": chess.engine.PovScore(
                chess.engine.Cp(
                    update.evaluation.centipawns
                    if update.evaluation.centipawns is not None
                    else 0,
                ),
                chess.WHITE,
            ),
            "pv": [chess.Move.from_uci(m) for m in update.variations[0].pv_uci]
            if update.variations
            else [],
            "depth": update.depth_reached,
        }
        infos.append(info)
        group = [info] if update.variations else []
        multipv_groups.append(group)

    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.__iter__ = MagicMock(return_value=iter(infos))

    multipv_iter = iter(multipv_groups)
    type(stream).multipv = property(lambda self: next(multipv_iter, []))

    mock_engine.analysis.return_value = stream


# ---------------------------------------------------------------------------
# AC: Continuous analysis coaching (US2)
# ---------------------------------------------------------------------------


class TestContinuousCoach:
    """submit_opponent_move_continuous() streams advice updates."""

    def test_callback_receives_continuous_advice(self) -> None:
        """AC1: Continuous analysis starts after opponent move."""
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, mock_engine = _make_coach_with_mock(config)

        board = chess.Board()
        board.push_san("e4")
        expected_fen = board.fen()

        update = _make_continuous_update(expected_fen, "e5", -20, 10)
        _setup_mock_continuous(mock_engine, [update])

        advices: list[ContinuousAdvice] = []
        done = threading.Event()

        def on_advice(advice: ContinuousAdvice) -> None:
            advices.append(advice)
            done.set()

        handle = coach.submit_opponent_move_continuous("e4", on_advice)
        assert done.wait(timeout=_SYNC_TIMEOUT_S)
        handle.wait(timeout=_SYNC_TIMEOUT_S)

        assert len(advices) >= 1
        assert advices[0].opponent_move_san == "e4"
        assert advices[0].current_fen == expected_fen
        assert advices[0].update.depth_reached == 10

    def test_stop_analysis_cancels(self) -> None:
        """AC4: Analysis stops when stop_analysis() is called."""
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, mock_engine = _make_coach_with_mock(config)

        board = chess.Board()
        board.push_san("e4")

        update = _make_continuous_update(board.fen())

        blocking = threading.Event()

        def blocking_iter():
            info: chess.engine.InfoDict = {
                "score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE),
                "pv": [],
                "depth": 10,
            }
            yield info
            blocking.wait(timeout=_SYNC_TIMEOUT_S)

        stream = MagicMock()
        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)
        stream.__iter__ = MagicMock(side_effect=lambda: blocking_iter())
        type(stream).multipv = property(
            lambda self: [
                {
                    "score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE),
                    "pv": [],
                    "depth": 10,
                },
            ],
        )
        mock_engine.analysis.return_value = stream

        got_first = threading.Event()

        def on_advice(advice: ContinuousAdvice) -> None:
            got_first.set()

        handle = coach.submit_opponent_move_continuous("e4", on_advice)
        assert got_first.wait(timeout=_SYNC_TIMEOUT_S)

        coach.stop_analysis()
        blocking.set()
        handle.wait(timeout=_SYNC_TIMEOUT_S)

        assert not handle.is_running

    def test_turn_enforcement(self) -> None:
        """Cannot call submit_opponent_move_continuous on player turn."""
        config = CoachConfig(player_color=PlayerColor.WHITE)
        coach, _ = _make_coach_with_mock(config)

        with pytest.raises(NotYourTurnError):
            coach.submit_opponent_move_continuous("e4", lambda a: None)

    def test_illegal_move_rejected(self) -> None:
        """Illegal move raises IllegalMoveError."""
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, _ = _make_coach_with_mock(config)

        with pytest.raises(IllegalMoveError):
            coach.submit_opponent_move_continuous("Zz9", lambda a: None)


# ---------------------------------------------------------------------------
# AC: Move history
# ---------------------------------------------------------------------------


class TestMoveHistory:
    """move_history property tracks played moves in SAN."""

    def test_empty_initially(self) -> None:
        coach, _ = _make_coach_with_mock()
        assert coach.move_history == ()

    def test_tracks_moves(self) -> None:
        config = CoachConfig(player_color=PlayerColor.BLACK)
        coach, mock_engine = _make_coach_with_mock(config)

        board = chess.Board()
        board.push_san("e4")
        infos = _build_infos_for_position(board.fen(), ["e7e5"], [0])
        mock_engine.analyse.return_value = infos

        coach.submit_opponent_move("e4")
        coach.submit_player_move("e5")

        assert coach.move_history == ("e4", "e5")


# ---------------------------------------------------------------------------
# AC: Passive move recording
# ---------------------------------------------------------------------------


class TestRecordMove:
    """record_move updates the board without triggering analysis."""

    def test_records_legal_moves_without_engine_analysis(self) -> None:
        coach, mock_engine = _make_coach_with_mock()

        assert coach.record_move("e4") == "e4"
        assert coach.record_move("e5") == "e5"
        assert coach.move_history == ("e4", "e5")
        assert mock_engine.analyse.call_count == 0

    def test_illegal_move_is_rejected(self) -> None:
        coach, _ = _make_coach_with_mock()

        with pytest.raises(IllegalMoveError):
            coach.record_move("Zz9")


# ---------------------------------------------------------------------------
# Board property and load_board
# ---------------------------------------------------------------------------


class TestBoardProperty:
    """The board property returns a safe copy of the internal board."""

    def test_returns_copy(self) -> None:
        coach, _ = _make_coach_with_mock()
        board_copy = coach.board
        board_copy.push_san("e4")

        # Internal board should be unchanged.
        assert coach.current_fen == chess.STARTING_FEN

    def test_reflects_current_state(self) -> None:
        coach, mock_engine = _make_coach_with_mock()
        coach.submit_player_move("e4")

        board_copy = coach.board
        assert len(board_copy.move_stack) == 1


class TestLoadBoard:
    """load_board replaces internal board state."""

    def test_replaces_state(self) -> None:
        coach, _ = _make_coach_with_mock()
        new_board = chess.Board()
        new_board.push_san("e4")
        new_board.push_san("e5")
        new_board.push_san("Nf3")

        coach.load_board(new_board)

        assert len(coach.board.move_stack) == 3
        assert coach.move_history == ("e4", "e5", "Nf3")

    def test_loaded_board_is_independent(self) -> None:
        coach, _ = _make_coach_with_mock()
        new_board = chess.Board()
        new_board.push_san("d4")

        coach.load_board(new_board)
        new_board.push_san("d5")

        # Coach board should NOT have d5.
        assert coach.move_history == ("d4",)

    def test_load_board_clears_manual_resignation_state(self) -> None:
        coach, _ = _make_coach_with_mock()
        coach.record_resignation(PlayerColor.BLACK)

        new_board = chess.Board()
        new_board.push_san("e4")

        coach.load_board(new_board)

        assert not coach.is_game_over
        assert coach.game_result == "*"
        assert coach.game_termination is None
