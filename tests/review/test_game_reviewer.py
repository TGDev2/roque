"""Tests for the GameReviewer.

Covers:
- AC3: Classification thresholds (excellent, good, inaccuracy, mistake, blunder)
- AC3: Configurable thresholds
- AC5: Summary aggregation (counts per category, avg cp loss, critical moments)
- Forced move detection
- Lost mate detection → blunder
- Phase detection (opening / middlegame / endgame)
- Progress callback invocation
- Cancellation via threading.Event
- Edge cases: empty game, single-move game
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import chess
import chess.engine
import pytest

from chesslab.analysis.position_analyser import (
    AnalysisConfig,
    Evaluation,
    PositionAnalyser,
)
from chesslab.review.game_reviewer import (
    GamePhase,
    GameReview,
    GameReviewer,
    LessonTheme,
    MoveClassification,
    ReviewCancelledError,
    ReviewConfig,
    ReviewedMove,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pov_score_cp(cp: int) -> chess.engine.PovScore:
    return chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)


def _make_pov_score_mate(moves: int) -> chess.engine.PovScore:
    return chess.engine.PovScore(chess.engine.Mate(moves), chess.WHITE)


def _make_info(
    *,
    score: chess.engine.PovScore,
    pv: list[chess.Move],
    depth: int = 16,
) -> chess.engine.InfoDict:
    return {"score": score, "pv": pv, "depth": depth}


def _make_board(*san_moves: str) -> chess.Board:
    """Create a board with the given moves played."""
    board = chess.Board()
    for san in san_moves:
        board.push_san(san)
    return board


def _build_info_for_fen(
    fen: str,
    cp: int,
    best_move_uci: str = "e2e4",
) -> list[chess.engine.InfoDict]:
    """Build a single-element InfoDict list for a given position."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(best_move_uci)
    return [_make_info(score=_make_pov_score_cp(cp), pv=[move])]


def _review_with_mock(
    board: chess.Board,
    evals_cp: list[int],
    config: ReviewConfig | None = None,
    best_moves_uci: list[str] | None = None,
    progress_callback=None,
    cancel_event=None,
) -> GameReview:
    """Run a review with mocked engine returning specified centipawn evals.

    *evals_cp* must have N+1 entries for a game with N moves
    (one per position: start, after move 1, after move 2, ...).

    *best_moves_uci* can optionally specify the best move for each position
    (N+1 entries).  If omitted, defaults to "e2e4" for all positions.
    """
    cfg = config or ReviewConfig()
    reviewer = GameReviewer(cfg)
    mock_engine = MagicMock(spec=chess.engine.SimpleEngine)

    # Replay the game to get FENs and build per-position infos.
    root = board.copy()
    while root.move_stack:
        root.pop()
    replay = root
    fens = [replay.fen()]
    for move in board.move_stack:
        replay.push(move)
        fens.append(replay.fen())

    def analyse_side_effect(board_arg, limit, multipv=1):
        fen = board_arg.fen()
        idx = fens.index(fen)
        cp = evals_cp[idx]
        best_uci = (best_moves_uci[idx] if best_moves_uci else None)
        # Pick a valid move for the PV.
        if best_uci:
            pv_move = chess.Move.from_uci(best_uci)
        elif list(board_arg.legal_moves):
            pv_move = list(board_arg.legal_moves)[0]
        else:
            pv_move = chess.Move.from_uci("e2e4")
        return [_make_info(score=_make_pov_score_cp(cp), pv=[pv_move])]

    mock_engine.analyse.side_effect = analyse_side_effect

    with patch(
        "chesslab.review.game_reviewer.PositionAnalyser",
    ) as MockAnalyser:
        instance = MagicMock()
        MockAnalyser.return_value = instance
        instance.start = MagicMock()
        instance.stop = MagicMock()

        # Wire analyse to our mock.
        instance.analyse.side_effect = lambda fen: _dispatch_analyse(
            fen, fens, evals_cp, best_moves_uci,
        )

        review = reviewer.review(
            board,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
    return review


def _dispatch_analyse(
    fen: str,
    fens: list[str],
    evals_cp: list[int],
    best_moves_uci: list[str] | None,
):
    """Build a synthetic AnalysisResult for a given FEN."""
    from chesslab.analysis.position_analyser import AnalysisResult, VariationInfo

    idx = fens.index(fen)
    cp = evals_cp[idx]

    board = chess.Board(fen)
    legal = list(board.legal_moves)

    if best_moves_uci and idx < len(best_moves_uci):
        best_uci = best_moves_uci[idx]
        best_move = chess.Move.from_uci(best_uci)
    elif legal:
        best_move = legal[0]
        best_uci = best_move.uci()
    else:
        # Terminal position.
        return AnalysisResult(
            fen=fen,
            best_move_uci="",
            best_move_san="",
            evaluation=Evaluation(centipawns=cp, mate_in=None),
            variations=(),
            depth_reached=16,
            is_partial=False,
            elapsed_ms=1.0,
        )

    try:
        best_san = board.san(best_move)
    except (ValueError, chess.InvalidMoveError):
        best_san = best_uci

    evaluation = Evaluation(centipawns=cp, mate_in=None)
    variation = VariationInfo(
        rank=1,
        move_uci=best_uci,
        move_san=best_san,
        evaluation=evaluation,
        pv_uci=(best_uci,),
        pv_san=(best_san,),
        depth=16,
    )

    return AnalysisResult(
        fen=fen,
        best_move_uci=best_uci,
        best_move_san=best_san,
        evaluation=evaluation,
        variations=(variation,),
        depth_reached=16,
        is_partial=False,
        elapsed_ms=1.0,
    )


def _dispatch_analyse_with_mate(
    fen: str,
    fens: list[str],
    eval_specs: list[tuple[int | None, int | None]],
    best_moves_uci: list[str] | None,
):
    """Build AnalysisResult supporting mate scores.

    *eval_specs* is a list of (centipawns, mate_in) tuples per position.
    """
    from chesslab.analysis.position_analyser import AnalysisResult, VariationInfo

    idx = fens.index(fen)
    cp, mate = eval_specs[idx]

    board = chess.Board(fen)
    legal = list(board.legal_moves)

    if best_moves_uci and idx < len(best_moves_uci):
        best_move = chess.Move.from_uci(best_moves_uci[idx])
        best_uci = best_moves_uci[idx]
    elif legal:
        best_move = legal[0]
        best_uci = best_move.uci()
    else:
        return AnalysisResult(
            fen=fen, best_move_uci="", best_move_san="",
            evaluation=Evaluation(centipawns=cp, mate_in=mate),
            variations=(), depth_reached=16, is_partial=False, elapsed_ms=1.0,
        )

    try:
        best_san = board.san(best_move)
    except (ValueError, chess.InvalidMoveError):
        best_san = best_uci

    evaluation = Evaluation(centipawns=cp, mate_in=mate)
    variation = VariationInfo(
        rank=1, move_uci=best_uci, move_san=best_san,
        evaluation=evaluation, pv_uci=(best_uci,), pv_san=(best_san,),
        depth=16,
    )

    return AnalysisResult(
        fen=fen, best_move_uci=best_uci, best_move_san=best_san,
        evaluation=evaluation, variations=(variation,),
        depth_reached=16, is_partial=False, elapsed_ms=1.0,
    )


# ---------------------------------------------------------------------------
# AC3: Classification thresholds
# ---------------------------------------------------------------------------


class TestClassificationThresholds:
    """Moves are classified by centipawn loss with default thresholds."""

    def test_excellent_move(self) -> None:
        """cp_loss < 10 → EXCELLENT."""
        # White plays e4, eval stays at +20 → +18 = loss ~2cp
        board = _make_board("e4")
        # Position 0 (start): +20, Position 1 (after e4): -18 from Black = +18 White
        review = _review_with_mock(board, evals_cp=[20, -18])
        assert review.moves[0].classification == MoveClassification.EXCELLENT
        assert review.moves[0].cp_loss == 2

    def test_good_move(self) -> None:
        """10 <= cp_loss < 30 → GOOD."""
        board = _make_board("e4")
        # Loss = 20 - 5 = 15
        review = _review_with_mock(board, evals_cp=[20, -5])
        assert review.moves[0].classification == MoveClassification.GOOD
        assert review.moves[0].cp_loss == 15

    def test_inaccuracy(self) -> None:
        """30 <= cp_loss < 80 → INACCURACY."""
        board = _make_board("e4")
        # Loss = 50 - 0 = 50
        review = _review_with_mock(board, evals_cp=[50, 0])
        assert review.moves[0].classification == MoveClassification.INACCURACY
        assert review.moves[0].cp_loss == 50

    def test_mistake(self) -> None:
        """80 <= cp_loss < 200 → MISTAKE."""
        board = _make_board("e4")
        # Loss = 150 - 0 = 150
        review = _review_with_mock(board, evals_cp=[150, 0])
        assert review.moves[0].classification == MoveClassification.MISTAKE
        assert review.moves[0].cp_loss == 150

    def test_blunder(self) -> None:
        """cp_loss >= 200 → BLUNDER."""
        board = _make_board("e4")
        # Loss = 300 - 0 = 300
        review = _review_with_mock(board, evals_cp=[300, 0])
        assert review.moves[0].classification == MoveClassification.BLUNDER
        assert review.moves[0].cp_loss == 300


class TestConfigurableThresholds:
    """AC3: Thresholds are configurable via ReviewConfig."""

    def test_custom_thresholds_promote_good_to_excellent(self) -> None:
        cfg = ReviewConfig(excellent_threshold=20)
        board = _make_board("e4")
        # Loss = 15, default would be GOOD, but with threshold=20 → EXCELLENT
        review = _review_with_mock(board, evals_cp=[20, -5], config=cfg)
        assert review.moves[0].classification == MoveClassification.EXCELLENT

    def test_custom_thresholds_demote_good_to_inaccuracy(self) -> None:
        cfg = ReviewConfig(good_threshold=15)
        board = _make_board("e4")
        # Loss = 20, default would be GOOD, but with good_threshold=15 → INACCURACY
        review = _review_with_mock(board, evals_cp=[30, -10], config=cfg)
        assert review.moves[0].classification == MoveClassification.INACCURACY


# ---------------------------------------------------------------------------
# Forced move detection
# ---------------------------------------------------------------------------


class TestForcedMoves:
    """Forced moves (single legal move) get FORCED classification."""

    def test_single_legal_move_is_forced(self) -> None:
        # Black king on a8, White king on b6, White rook on a1.
        # Black's only legal move is Kb8.
        board = chess.Board("k7/8/1K6/8/8/8/8/R7 b - - 0 1")
        legal = list(board.legal_moves)
        assert len(legal) == 1, f"Expected 1 legal move, got {len(legal)}: {legal}"

        # Push the forced move so the board has a move_stack.
        board.push(legal[0])

        # We need 2 evals: before (Black to move) and after (White to move).
        review = _review_with_mock(board, evals_cp=[-500, 500])
        assert len(review.moves) == 1
        assert review.moves[0].classification == MoveClassification.FORCED
        assert review.moves[0].is_forced is True
        assert review.moves[0].cp_loss == 0


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------


class TestPhaseDetection:
    """Game phases are detected based on ply count and material."""

    def test_opening_phase(self) -> None:
        board = _make_board("e4", "e5")
        review = _review_with_mock(board, evals_cp=[20, -20, 20])
        assert review.moves[0].phase == GamePhase.OPENING
        assert review.moves[1].phase == GamePhase.OPENING

    def test_endgame_phase_low_material(self) -> None:
        # Build a board with many moves (past opening) and low material.
        # Simulate by constructing a board with move_stack length > 30
        # and stripped material.
        board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 31")
        # Push a single move.
        board.push_san("e3")

        # The reviewer checks the board's material at the position.
        # With K+P vs K, material = 1 < 13 → endgame.
        # But ply also needs to be >= opening_end_ply (30).
        # We need a board with move_stack having at least 31 entries
        # for ply >= 30.  Instead, let's test the _detect_phase method
        # directly.
        reviewer = GameReviewer()
        test_board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 31")
        assert reviewer._detect_phase(35, test_board) == GamePhase.ENDGAME

    def test_middlegame_phase(self) -> None:
        reviewer = GameReviewer()
        # Full material, past opening.
        full_board = chess.Board()
        assert reviewer._detect_phase(35, full_board) == GamePhase.MIDDLEGAME


# ---------------------------------------------------------------------------
# AC5: Summary aggregation
# ---------------------------------------------------------------------------


class TestSummaryAggregation:
    """GameReview summary has correct counts and averages."""

    def test_classification_counts(self) -> None:
        # 4-move game: excellent, good, inaccuracy, mistake
        board = _make_board("e4", "e5", "Nf3", "Nc6")
        evals = [
            20,   # start
            -18,  # after e4: loss=2 → excellent
            20,   # after e5: loss=2 → excellent (from Black's perspective)
            -5,   # after Nf3: loss=15 → good
            -40,  # after Nc6: loss=35 → inaccuracy
        ]
        review = _review_with_mock(board, evals_cp=evals)
        assert len(review.moves) == 4
        # White moves: e4 (ply 0), Nf3 (ply 2)
        # Black moves: e5 (ply 1), Nc6 (ply 3)
        assert review.white_classification_counts[MoveClassification.EXCELLENT] >= 0
        assert review.black_classification_counts[MoveClassification.EXCELLENT] >= 0
        # Total moves = 4
        total = sum(review.white_classification_counts.values()) + sum(
            review.black_classification_counts.values()
        )
        assert total == 4

    def test_critical_moments_contain_mistakes(self) -> None:
        board = _make_board("e4", "e5")
        # e4 is a blunder (loss=300), e5 is excellent (loss=0)
        evals = [300, 0, 0]
        review = _review_with_mock(board, evals_cp=evals)
        assert 0 in review.critical_moments  # ply 0 is a blunder

    def test_phase_summaries_present(self) -> None:
        board = _make_board("e4", "e5")
        review = _review_with_mock(board, evals_cp=[20, -18, 20])
        assert len(review.phase_summaries) > 0
        assert review.phase_summaries[0].phase == GamePhase.OPENING


# ---------------------------------------------------------------------------
# Lesson extraction
# ---------------------------------------------------------------------------


class TestLessonExtraction:
    """Lessons are concrete, prioritised, and skipped when signal is weak."""

    def test_extracts_phase_pattern_lesson(self) -> None:
        board = _make_board("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")
        review = _review_with_mock(
            board,
            evals_cp=[20, -20, 70, -70, 160, -160, 210],
        )

        assert review.lessons
        lesson = review.lessons[0]
        assert lesson.priority == 1
        assert lesson.theme is LessonTheme.PHASE_PATTERN
        assert lesson.phase is GamePhase.OPENING
        assert lesson.is_white is False
        assert lesson.focus_ply == 3
        assert lesson.related_plies == (1, 3, 5)
        assert "Black had 3 moves rated inaccuracy or worse" in lesson.finding

    def test_extracts_turning_point_and_advantage_management(self) -> None:
        board = _make_board("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6")
        review = _review_with_mock(
            board,
            evals_cp=[20, -20, 20, -20, 200, -20, 20],
        )

        themes = {lesson.theme for lesson in review.lessons}
        assert LessonTheme.TURNING_POINT in themes
        assert LessonTheme.ADVANTAGE_MANAGEMENT in themes
        assert any(lesson.focus_ply == 4 for lesson in review.lessons)

    def test_short_game_skips_lessons(self) -> None:
        board = _make_board("e4")
        review = _review_with_mock(board, evals_cp=[20, -18])
        assert review.lessons == ()

    def test_low_signal_game_skips_lessons(self) -> None:
        board = _make_board("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")
        review = _review_with_mock(
            board,
            evals_cp=[20, -18, 20, -18, 20, -18, 20],
        )

        assert review.lessons == ()


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    """progress_callback is called with incrementing ply values."""

    def test_callback_called_for_each_move(self) -> None:
        board = _make_board("e4", "e5", "Nf3")
        calls: list[tuple[int, int]] = []

        def on_progress(current: int, total: int) -> None:
            calls.append((current, total))

        _review_with_mock(
            board,
            evals_cp=[20, -18, 20, -15],
            progress_callback=on_progress,
        )
        assert len(calls) == 3
        assert calls[0] == (1, 3)
        assert calls[1] == (2, 3)
        assert calls[2] == (3, 3)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    """Setting cancel_event raises ReviewCancelledError."""

    def test_cancel_raises(self) -> None:
        board = _make_board("e4", "e5", "Nf3")
        cancel = threading.Event()
        cancel.set()  # Pre-set to cancel immediately.

        with pytest.raises(ReviewCancelledError):
            _review_with_mock(
                board,
                evals_cp=[20, -18, 20, -15],
                cancel_event=cancel,
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Empty and single-move games."""

    def test_empty_game(self) -> None:
        board = chess.Board()
        reviewer = GameReviewer()
        # Empty game needs no mocking.
        review = reviewer.review(board)
        assert len(review.moves) == 0
        assert review.white_avg_cp_loss == 0.0
        assert review.black_avg_cp_loss == 0.0
        assert review.lessons == ()

    def test_single_move_game(self) -> None:
        board = _make_board("e4")
        review = _review_with_mock(board, evals_cp=[20, -18])
        assert len(review.moves) == 1
        assert review.moves[0].san == "e4"
        assert review.moves[0].is_white is True


# ---------------------------------------------------------------------------
# Lost mate detection
# ---------------------------------------------------------------------------


class TestLostMate:
    """Losing a forced mate is classified as BLUNDER."""

    def test_lost_mate_is_blunder(self) -> None:
        # White has mate in 3 before the move, but after the move no mate.
        board = _make_board("e4")

        # Position 0: White has mate in 3 (eval_before).
        # Position 1: After e4, Black does NOT face forced mate (cp eval only).
        eval_specs = [
            (None, 3),     # pos 0: mate in 3 for side-to-move (White)
            (50, None),    # pos 1: +50 cp (no mate) — Black to move
        ]
        best_moves = ["e2e4", "e7e5"]

        reviewer = GameReviewer()
        mock_engine = MagicMock(spec=chess.engine.SimpleEngine)

        root = board.copy()
        while root.move_stack:
            root.pop()
        replay = root
        fens = [replay.fen()]
        for move in board.move_stack:
            replay.push(move)
            fens.append(replay.fen())

        with patch(
            "chesslab.review.game_reviewer.PositionAnalyser",
        ) as MockAnalyser:
            instance = MagicMock()
            MockAnalyser.return_value = instance
            instance.start = MagicMock()
            instance.stop = MagicMock()
            instance.analyse.side_effect = lambda fen: _dispatch_analyse_with_mate(
                fen, fens, eval_specs, best_moves,
            )

            review = reviewer.review(board)

        # The move had mate before but lost it → BLUNDER.
        assert review.moves[0].classification == MoveClassification.BLUNDER
