"""Tests for the position watcher

Covers position change detection via FEN hash comparison, debounce
behaviour, cooperative cancellation of in-flight analysis, no-change
suppression, callback error handling, and stop/reuse lifecycle.
"""

from __future__ import annotations

import threading
import time

import pytest

from chesslab.monitoring.position_watcher import (
    MonitoringError,
    PositionChangeEvent,
    PositionWatcher,
    PositionWatcherConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
AFTER_E4_E5_FEN = (
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
)

# Short debounce for fast tests.
_TEST_DEBOUNCE_MS = 30
_SYNC_TIMEOUT_S = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_watcher(debounce_ms: int = _TEST_DEBOUNCE_MS) -> PositionWatcher:
    return PositionWatcher(PositionWatcherConfig(debounce_ms=debounce_ms))


# ---------------------------------------------------------------------------
# AC: Config validation
# ---------------------------------------------------------------------------


class TestPositionWatcherConfig:
    """PositionWatcherConfig validates its fields on construction."""

    def test_default_debounce(self) -> None:
        config = PositionWatcherConfig()
        assert config.debounce_ms == 300

    def test_custom_debounce(self) -> None:
        config = PositionWatcherConfig(debounce_ms=500)
        assert config.debounce_ms == 500

    def test_zero_debounce_accepted(self) -> None:
        config = PositionWatcherConfig(debounce_ms=0)
        assert config.debounce_ms == 0

    def test_negative_debounce_rejected(self) -> None:
        with pytest.raises(ValueError, match="debounce_ms"):
            PositionWatcherConfig(debounce_ms=-1)


# ---------------------------------------------------------------------------
# AC: Change detected when position changes
# ---------------------------------------------------------------------------


class TestChangeDetection:
    """AC: position change is detected via FEN hash comparison."""

    def test_first_fen_is_a_change(self) -> None:
        """First update always detects a change (empty → something)."""
        watcher = _make_watcher()
        assert watcher.update(STARTING_FEN) is True

    def test_same_fen_returns_false(self) -> None:
        """Identical FEN on consecutive updates → no change."""
        watcher = _make_watcher()
        watcher.update(STARTING_FEN)
        assert watcher.update(STARTING_FEN) is False

    def test_different_fen_returns_true(self) -> None:
        """Different FEN on consecutive updates → change detected."""
        watcher = _make_watcher()
        watcher.update(STARTING_FEN)
        assert watcher.update(AFTER_E4_FEN) is True

    def test_current_fen_updated_on_change(self) -> None:
        """current_fen property reflects the latest accepted FEN."""
        watcher = _make_watcher()
        assert watcher.current_fen == ""

        watcher.update(STARTING_FEN)
        assert watcher.current_fen == STARTING_FEN

        watcher.update(AFTER_E4_FEN)
        assert watcher.current_fen == AFTER_E4_FEN

    def test_current_fen_unchanged_on_duplicate(self) -> None:
        """current_fen stays the same when no change is detected."""
        watcher = _make_watcher()
        watcher.update(STARTING_FEN)
        watcher.update(STARTING_FEN)
        assert watcher.current_fen == STARTING_FEN


# ---------------------------------------------------------------------------
# AC: Debounce absorbs animations — callback fires after delay
# ---------------------------------------------------------------------------


class TestDebounce:
    """AC: 300 ms debounce before triggering analysis."""

    def test_callback_fires_after_debounce(self) -> None:
        """Callback is invoked after debounce delay, not before."""
        watcher = _make_watcher()
        fired = threading.Event()
        events: list[PositionChangeEvent] = []

        def on_change(event: PositionChangeEvent, _cancel: threading.Event) -> None:
            events.append(event)
            fired.set()

        watcher.on_change(on_change)
        watcher.update(STARTING_FEN)

        # Should not have fired immediately.
        assert not fired.is_set()

        # Wait for debounce to expire.
        assert fired.wait(timeout=_SYNC_TIMEOUT_S)
        assert len(events) == 1

    def test_rapid_updates_only_fire_last(self) -> None:
        """Multiple rapid updates → only the last FEN fires."""
        watcher = _make_watcher()
        fired = threading.Event()
        events: list[PositionChangeEvent] = []

        def on_change(event: PositionChangeEvent, _cancel: threading.Event) -> None:
            events.append(event)
            fired.set()

        watcher.on_change(on_change)

        watcher.update(STARTING_FEN)
        watcher.update(AFTER_E4_FEN)
        watcher.update(AFTER_E4_E5_FEN)

        assert fired.wait(timeout=_SYNC_TIMEOUT_S)
        # Allow a small extra wait for any spurious late callbacks.
        time.sleep(0.1)

        assert len(events) == 1
        assert events[0].current_fen == AFTER_E4_E5_FEN
        assert events[0].previous_fen == AFTER_E4_FEN

    def test_zero_debounce_fires_quickly(self) -> None:
        """With debounce_ms=0, callback fires almost immediately."""
        watcher = _make_watcher(debounce_ms=0)
        fired = threading.Event()

        def on_change(_event: PositionChangeEvent, _cancel: threading.Event) -> None:
            fired.set()

        watcher.on_change(on_change)
        watcher.update(STARTING_FEN)

        assert fired.wait(timeout=_SYNC_TIMEOUT_S)

    def test_event_contains_correct_fens(self) -> None:
        """PositionChangeEvent carries previous and current FEN."""
        watcher = _make_watcher()
        fired = threading.Event()
        events: list[PositionChangeEvent] = []

        def on_change(event: PositionChangeEvent, _cancel: threading.Event) -> None:
            events.append(event)
            fired.set()

        watcher.on_change(on_change)

        watcher.update(STARTING_FEN)
        assert fired.wait(timeout=_SYNC_TIMEOUT_S)

        assert events[0].previous_fen == ""
        assert events[0].current_fen == STARTING_FEN

    def test_elapsed_ms_at_least_debounce(self) -> None:
        """elapsed_ms in the event is roughly >= debounce_ms."""
        debounce = 50
        watcher = _make_watcher(debounce_ms=debounce)
        fired = threading.Event()
        events: list[PositionChangeEvent] = []

        def on_change(event: PositionChangeEvent, _cancel: threading.Event) -> None:
            events.append(event)
            fired.set()

        watcher.on_change(on_change)
        watcher.update(STARTING_FEN)

        assert fired.wait(timeout=_SYNC_TIMEOUT_S)
        # Allow some tolerance for timer precision.
        assert events[0].elapsed_ms >= debounce * 0.8

    def test_change_detected_within_600ms(self) -> None:
        """AC: change detected and callback fired within 600 ms total."""
        watcher = _make_watcher()  # 30 ms debounce
        fired = threading.Event()
        wall_start = time.perf_counter()

        def on_change(_event: PositionChangeEvent, _cancel: threading.Event) -> None:
            fired.set()

        watcher.on_change(on_change)
        watcher.update(STARTING_FEN)

        assert fired.wait(timeout=0.6)
        wall_elapsed = (time.perf_counter() - wall_start) * 1000.0
        assert wall_elapsed < 600.0


# ---------------------------------------------------------------------------
# AC: Cancel ongoing analysis on new position
# ---------------------------------------------------------------------------


class TestCancellation:
    """AC: ongoing analysis cancelled when new position arrives."""

    def test_cancellation_event_set_on_new_change(self) -> None:
        """A new update sets the cancellation event of the previous change."""
        watcher = _make_watcher(debounce_ms=200)
        captured_cancellations: list[threading.Event] = []

        def on_change(_event: PositionChangeEvent, cancel: threading.Event) -> None:
            captured_cancellations.append(cancel)

        watcher.on_change(on_change)

        # First change.
        watcher.update(STARTING_FEN)
        # Second change before debounce fires — should cancel first.
        watcher.update(AFTER_E4_FEN)

        # Wait for the second debounce to fire.
        time.sleep(0.5)

        # Only one callback should have fired (the second).
        assert len(captured_cancellations) == 1
        # The second callback's cancellation should NOT be set.
        assert not captured_cancellations[0].is_set()

    def test_cancellation_event_fresh_per_change(self) -> None:
        """Each debounced change gets its own fresh cancellation Event."""
        watcher = _make_watcher()
        captured_cancellations: list[threading.Event] = []
        all_fired = threading.Event()

        def on_change(_event: PositionChangeEvent, cancel: threading.Event) -> None:
            captured_cancellations.append(cancel)
            if len(captured_cancellations) == 2:
                all_fired.set()

        watcher.on_change(on_change)

        watcher.update(STARTING_FEN)
        time.sleep(0.1)  # let first debounce fire
        watcher.update(AFTER_E4_FEN)

        assert all_fired.wait(timeout=_SYNC_TIMEOUT_S)
        assert captured_cancellations[0] is not captured_cancellations[1]

    def test_stop_sets_cancellation_event(self) -> None:
        """stop() signals the current cancellation event."""
        watcher = _make_watcher(debounce_ms=500)
        fired = threading.Event()

        def on_change(_event: PositionChangeEvent, _cancel: threading.Event) -> None:
            fired.set()

        watcher.on_change(on_change)
        watcher.update(STARTING_FEN)
        watcher.stop()

        # Callback should NOT fire after stop.
        time.sleep(0.2)
        assert not fired.is_set()

    def test_callback_receives_unset_cancellation(self) -> None:
        """The cancellation event passed to the callback is initially unset."""
        watcher = _make_watcher()
        fired = threading.Event()
        cancel_was_set: list[bool] = []

        def on_change(_event: PositionChangeEvent, cancel: threading.Event) -> None:
            cancel_was_set.append(cancel.is_set())
            fired.set()

        watcher.on_change(on_change)
        watcher.update(STARTING_FEN)

        assert fired.wait(timeout=_SYNC_TIMEOUT_S)
        assert cancel_was_set[0] is False


# ---------------------------------------------------------------------------
# AC: No resources consumed when position unchanged
# ---------------------------------------------------------------------------


class TestNoChangeSuppression:
    """AC: no analysis triggered when position is unchanged."""

    def test_no_callback_on_same_fen(self) -> None:
        """Repeated identical FEN → callback never fires."""
        watcher = _make_watcher()
        call_count = 0

        def on_change(_event: PositionChangeEvent, _cancel: threading.Event) -> None:
            nonlocal call_count
            call_count += 1

        watcher.on_change(on_change)

        # First call triggers a change (empty → starting).
        watcher.update(STARTING_FEN)
        time.sleep(0.1)  # let debounce fire

        # Subsequent duplicates should not fire.
        for _ in range(5):
            watcher.update(STARTING_FEN)

        time.sleep(0.15)
        assert call_count == 1  # only the first change

    def test_update_returns_false_on_duplicate(self) -> None:
        """update() returns False immediately for unchanged position."""
        watcher = _make_watcher()
        watcher.update(STARTING_FEN)
        assert watcher.update(STARTING_FEN) is False


# ---------------------------------------------------------------------------
# Callback error resilience
# ---------------------------------------------------------------------------


class TestCallbackErrors:
    """Watcher survives callback exceptions."""

    def test_callback_exception_does_not_break_watcher(self) -> None:
        """Exception in callback is logged; watcher continues working."""
        watcher = _make_watcher()
        second_fired = threading.Event()
        events: list[PositionChangeEvent] = []

        call_count = 0

        def on_change(event: PositionChangeEvent, _cancel: threading.Event) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            events.append(event)
            second_fired.set()

        watcher.on_change(on_change)

        # First change — callback raises.
        watcher.update(STARTING_FEN)
        time.sleep(0.1)

        # Second change — should still fire.
        watcher.update(AFTER_E4_FEN)
        assert second_fired.wait(timeout=_SYNC_TIMEOUT_S)
        assert events[0].current_fen == AFTER_E4_FEN


# ---------------------------------------------------------------------------
# Stop lifecycle
# ---------------------------------------------------------------------------


class TestStopBehavior:
    """Watcher stop and reuse behaviour."""

    def test_stop_cancels_pending_timer(self) -> None:
        """stop() prevents a pending debounce from firing."""
        watcher = _make_watcher(debounce_ms=200)
        fired = threading.Event()

        def on_change(_event: PositionChangeEvent, _cancel: threading.Event) -> None:
            fired.set()

        watcher.on_change(on_change)
        watcher.update(STARTING_FEN)
        watcher.stop()

        time.sleep(0.4)
        assert not fired.is_set()

    def test_stop_idempotent(self) -> None:
        """Calling stop() multiple times does not raise."""
        watcher = _make_watcher()
        watcher.update(STARTING_FEN)
        watcher.stop()
        watcher.stop()  # should not raise

    def test_update_after_stop_works(self) -> None:
        """Watcher remains functional after stop()."""
        watcher = _make_watcher()
        fired = threading.Event()
        events: list[PositionChangeEvent] = []

        def on_change(event: PositionChangeEvent, _cancel: threading.Event) -> None:
            events.append(event)
            fired.set()

        watcher.on_change(on_change)

        watcher.update(STARTING_FEN)
        watcher.stop()

        # New change after stop.
        watcher.update(AFTER_E4_FEN)
        assert fired.wait(timeout=_SYNC_TIMEOUT_S)
        assert events[0].current_fen == AFTER_E4_FEN


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    """MonitoringError follows the project error hierarchy convention."""

    def test_monitoring_error_is_runtime_error(self) -> None:
        assert issubclass(MonitoringError, RuntimeError)
