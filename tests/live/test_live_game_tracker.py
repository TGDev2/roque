"""Tests for the LiveGameTracker."""

from __future__ import annotations

import chess
import pytest

from chesslab.live.live_game_tracker import (
    GameTrackerConfig,
    GameTrackingEventType,
    LiveGameTracker,
)


def _placement_after(board: chess.Board, san: str) -> str:
    """Push *san* on a copy and return the piece-placement field."""
    b = board.copy()
    b.push_san(san)
    return b.fen().split(" ", 1)[0]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInitialisation:
    """First-frame behaviour."""

    def test_first_frame_standard_start(self) -> None:
        tracker = LiveGameTracker()
        placement = chess.Board().fen().split(" ", 1)[0]

        event = tracker.update(placement)

        assert event.type is GameTrackingEventType.GAME_STARTED
        assert event.is_new_game
        assert event.active_color == "w"
        assert tracker.is_tracking

    def test_first_frame_midgame_position(self) -> None:
        tracker = LiveGameTracker()
        # A position that is NOT the starting position.
        placement = "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR"

        event = tracker.update(placement)

        assert event.type is GameTrackingEventType.GAME_STARTED
        assert not event.is_new_game  # Not standard start
        assert tracker.is_tracking

    def test_initial_fen_override(self) -> None:
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        config = GameTrackerConfig(initial_fen=fen)
        tracker = LiveGameTracker(config)

        assert tracker.is_tracking
        assert tracker.board is not None
        assert tracker.board.fen() == fen

    def test_invalid_confirmation_frames_rejected(self) -> None:
        with pytest.raises(ValueError, match="confirmation_frames"):
            GameTrackerConfig(confirmation_frames=0)


# ---------------------------------------------------------------------------
# Consecutive moves
# ---------------------------------------------------------------------------


class TestConsecutiveMoves:
    """AC: track 5+ consecutive moves correctly."""

    def test_five_move_sequence(self) -> None:
        tracker = LiveGameTracker()
        board = chess.Board()

        # Initialise with starting position.
        start_placement = board.fen().split(" ", 1)[0]
        event = tracker.update(start_placement)
        assert event.type is GameTrackingEventType.GAME_STARTED

        moves = ["e4", "e5", "Nf3", "Nc6", "Bb5"]
        for san in moves:
            placement = _placement_after(board, san)
            board.push_san(san)

            event = tracker.update(placement)
            assert event.type is GameTrackingEventType.MOVE, (
                f"Expected MOVE for {san}, got {event.type}"
            )
            assert event.move_san == san
            assert event.move is not None

        assert len(tracker.move_stack) == 5
        assert tracker.board is not None
        assert tracker.board.fen() == board.fen()

    def test_move_number_and_active_color_tracked(self) -> None:
        tracker = LiveGameTracker()
        board = chess.Board()

        tracker.update(board.fen().split(" ", 1)[0])

        # 1. e4
        placement = _placement_after(board, "e4")
        board.push_san("e4")
        event = tracker.update(placement)
        assert event.active_color == "b"
        assert event.move_number == 1

        # 1... e5
        placement = _placement_after(board, "e5")
        board.push_san("e5")
        event = tracker.update(placement)
        assert event.active_color == "w"
        assert event.move_number == 2

    def test_midgame_initial_sync_recovers_unknown_turn(self) -> None:
        tracker = LiveGameTracker()
        board = chess.Board()
        board.push_san("e4")

        # First seen frame is midgame: side-to-move is ambiguous.
        start_event = tracker.update(board.fen().split(" ", 1)[0])
        assert start_event.type is GameTrackingEventType.GAME_STARTED

        # Next frame is a legal black move.
        placement_e5 = _placement_after(board, "e5")
        board.push_san("e5")
        event = tracker.update(placement_e5)

        assert event.type is GameTrackingEventType.MOVE
        assert event.move_san == "e5"
        assert tracker.board is not None
        assert tracker.board.fen() == board.fen()


# ---------------------------------------------------------------------------
# No-change (same position twice)
# ---------------------------------------------------------------------------


class TestNoChange:
    """AC: same position twice produces NO_CHANGE."""

    def test_duplicate_placement_returns_no_change(self) -> None:
        tracker = LiveGameTracker()
        board = chess.Board()
        placement = board.fen().split(" ", 1)[0]

        # First call initialises.
        tracker.update(placement)

        # Second call with same placement.
        event = tracker.update(placement)
        assert event.type is GameTrackingEventType.NO_CHANGE


# ---------------------------------------------------------------------------
# New game detection
# ---------------------------------------------------------------------------


class TestNewGameDetection:
    """AC: starting position after moves triggers NEW_GAME_DETECTED."""

    def test_new_game_after_moves_with_confirmation(self) -> None:
        tracker = LiveGameTracker(GameTrackerConfig(require_confirmation=True))
        board = chess.Board()

        # Start and play one move.
        tracker.update(board.fen().split(" ", 1)[0])
        placement_e4 = _placement_after(board, "e4")
        board.push_san("e4")
        tracker.update(placement_e4)

        # Feed starting position — first time is pending.
        start = chess.Board().fen().split(" ", 1)[0]
        event = tracker.update(start)
        assert event.type is GameTrackingEventType.NO_CHANGE

        # Second time — confirmed.
        event = tracker.update(start)
        assert event.type is GameTrackingEventType.NEW_GAME_DETECTED
        assert event.is_new_game
        assert len(tracker.move_stack) == 0

    def test_new_game_without_confirmation(self) -> None:
        tracker = LiveGameTracker(GameTrackerConfig(require_confirmation=False))
        board = chess.Board()

        tracker.update(board.fen().split(" ", 1)[0])
        placement_e4 = _placement_after(board, "e4")
        board.push_san("e4")
        tracker.update(placement_e4)

        # Feed starting position — immediate.
        start = chess.Board().fen().split(" ", 1)[0]
        event = tracker.update(start)
        assert event.type is GameTrackingEventType.NEW_GAME_DETECTED


# ---------------------------------------------------------------------------
# Position reset (non-starting discontinuity)
# ---------------------------------------------------------------------------


class TestPositionReset:
    """AC: impossible position triggers POSITION_RESET."""

    def test_discontinuity_with_confirmation(self) -> None:
        tracker = LiveGameTracker(GameTrackerConfig(require_confirmation=True))
        board = chess.Board()

        tracker.update(board.fen().split(" ", 1)[0])
        placement_e4 = _placement_after(board, "e4")
        board.push_san("e4")
        tracker.update(placement_e4)

        # Feed an impossible position.
        impossible = "8/8/8/4k3/8/8/8/4K3"

        # First occurrence — pending.
        event = tracker.update(impossible)
        assert event.type is GameTrackingEventType.NO_CHANGE

        # Second — confirmed reset.
        event = tracker.update(impossible)
        assert event.type is GameTrackingEventType.POSITION_RESET
        assert not event.is_new_game
        assert len(tracker.move_stack) == 0

    def test_discontinuity_without_confirmation(self) -> None:
        tracker = LiveGameTracker(GameTrackerConfig(require_confirmation=False))
        board = chess.Board()

        tracker.update(board.fen().split(" ", 1)[0])
        placement_e4 = _placement_after(board, "e4")
        board.push_san("e4")
        tracker.update(placement_e4)

        impossible = "8/8/8/4k3/8/8/8/4K3"
        event = tracker.update(impossible)
        assert event.type is GameTrackingEventType.POSITION_RESET


# ---------------------------------------------------------------------------
# Single-frame confirmation absorbs transient errors
# ---------------------------------------------------------------------------


class TestConfirmationAbsorbsErrors:
    """AC: one bad frame then correction does not reset."""

    def test_transient_error_absorbed(self) -> None:
        tracker = LiveGameTracker(GameTrackerConfig(require_confirmation=True))
        board = chess.Board()

        tracker.update(board.fen().split(" ", 1)[0])
        placement_e4 = _placement_after(board, "e4")
        board.push_san("e4")
        tracker.update(placement_e4)

        # Bad frame (impossible position).
        impossible = "8/8/8/4k3/8/8/8/4K3"
        event = tracker.update(impossible)
        assert event.type is GameTrackingEventType.NO_CHANGE

        # Correct next frame (actual move e5).
        placement_e5 = _placement_after(board, "e5")
        event = tracker.update(placement_e5)
        assert event.type is GameTrackingEventType.MOVE
        assert event.move_san == "e5"


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


class TestPromotion:
    """AC: promotion moves are tracked correctly."""

    def test_promotion_to_queen(self) -> None:
        fen = "8/P3k3/8/8/8/8/8/4K3 w - - 0 1"
        config = GameTrackerConfig(initial_fen=fen)
        tracker = LiveGameTracker(config)

        board = chess.Board(fen)
        placement = _placement_after(board, "a8=Q")

        event = tracker.update(placement)
        assert event.type is GameTrackingEventType.MOVE
        assert event.move is not None
        assert event.move.promotion == chess.QUEEN
        assert event.move_san == "a8=Q"

    def test_promotion_to_knight(self) -> None:
        fen = "8/P3k3/8/8/8/8/8/4K3 w - - 0 1"
        config = GameTrackerConfig(initial_fen=fen)
        tracker = LiveGameTracker(config)

        board = chess.Board(fen)
        placement = _placement_after(board, "a8=N")

        event = tracker.update(placement)
        assert event.type is GameTrackingEventType.MOVE
        assert event.move is not None
        assert event.move.promotion == chess.KNIGHT


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Verify explicit reset clears state."""

    def test_reset_clears_board(self) -> None:
        tracker = LiveGameTracker()
        tracker.update(chess.Board().fen().split(" ", 1)[0])
        assert tracker.is_tracking

        tracker.reset()
        assert not tracker.is_tracking
        assert tracker.board is None
        assert tracker.move_stack == ()


# ---------------------------------------------------------------------------
# Board property
# ---------------------------------------------------------------------------


class TestBoardProperty:
    """Verify the board property returns a copy."""

    def test_board_is_copy(self) -> None:
        tracker = LiveGameTracker()
        tracker.update(chess.Board().fen().split(" ", 1)[0])

        board = tracker.board
        assert board is not None
        board.push_san("e4")

        # Internal board should not be affected.
        assert tracker.board is not None
        assert len(tracker.board.move_stack) == 0
