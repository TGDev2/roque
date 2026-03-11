"""Position change detection with debounce and cooperative cancellation.

Monitors FEN strings produced by the recognition pipeline and detects when
a move has been played.  A configurable debounce delay absorbs intermediate
states caused by piece-animation on chess GUIs before firing a change event.

Each change event carries a :class:`threading.Event` cancellation token so
that long-running downstream work (e.g. Stockfish analysis) can be aborted
cooperatively when a newer position supersedes the current one.

Limitations
-----------
* The debounce delay adds latency between the actual move and the change
  notification.  A 300 ms default strikes a balance between animation
  absorption and responsiveness.
* Cancellation is cooperative — the callback must periodically check the
  event to honour abort requests.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class MonitoringError(RuntimeError):
    """Base class for monitoring module failures."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PositionWatcherConfig:
    """Configuration for position change detection.

    Parameters
    ----------
    debounce_ms:
        Delay in milliseconds before firing a change event.  Absorbs
        intermediate states caused by piece animation on chess GUIs.
    """

    debounce_ms: int = 300

    def __post_init__(self) -> None:
        if self.debounce_ms < 0:
            raise ValueError(
                f"debounce_ms must be >= 0, got {self.debounce_ms}."
            )


@dataclass(frozen=True)
class PositionChangeEvent:
    """Describes a detected position change after debounce.

    Attributes
    ----------
    previous_fen:
        The FEN string before the change (empty string for the very first
        position received).
    current_fen:
        The new FEN string that triggered the event.
    elapsed_ms:
        Wall-clock time from the ``update()`` call that detected the raw
        change to the moment the debounce timer fires the event.
    """

    previous_fen: str
    current_fen: str
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ChangeCallback = Callable[[PositionChangeEvent, threading.Event], None]


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------


class PositionWatcher:
    """Detects position changes via FEN hashing and debounces notifications.

    Designed to sit between the FEN generation pipeline and a downstream
    analysis module.  The caller feeds FEN strings via :meth:`update`; when
    a change is detected, the registered callback fires after a configurable
    debounce delay.

    A :class:`threading.Event` (cancellation token) is passed to the callback
    so that long-running analysis can be aborted if a newer position arrives
    before analysis completes.
    """

    def __init__(self, config: PositionWatcherConfig | None = None) -> None:
        self._config = config or PositionWatcherConfig()
        self._previous_fen: str = ""
        self._previous_hash: str = ""
        self._callback: ChangeCallback | None = None
        self._lock = threading.Lock()
        self._debounce_timer: threading.Timer | None = None
        self._cancellation_event: threading.Event | None = None
        self._change_detected_at: float = 0.0

    @property
    def config(self) -> PositionWatcherConfig:
        """Return the current configuration."""
        return self._config

    @property
    def current_fen(self) -> str:
        """Return the last FEN that was accepted by :meth:`update`."""
        with self._lock:
            return self._previous_fen

    def on_change(self, callback: ChangeCallback) -> None:
        """Register a callback invoked when a debounced position change fires.

        The callback receives a :class:`PositionChangeEvent` and a
        :class:`threading.Event` cancellation token.
        """
        self._callback = callback

    def update(self, fen: str) -> bool:
        """Feed a new FEN string from the pipeline.

        Compares the MD5 hash of *fen* with the previously stored hash.
        If a change is detected, any pending debounce timer is cancelled,
        any ongoing analysis is signalled to abort, and a new debounce
        timer is started.

        Parameters
        ----------
        fen:
            The current FEN string produced by the pipeline.

        Returns
        -------
        bool
            ``True`` if a raw change was detected (before debounce),
            ``False`` if the position is unchanged.
        """
        new_hash = hashlib.md5(
            fen.encode(), usedforsecurity=False
        ).hexdigest()

        with self._lock:
            if new_hash == self._previous_hash:
                return False

            logger.debug("Position change detected (hash %s).", new_hash[:8])

            # Cancel pending debounce timer.
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
                self._debounce_timer = None

            # Signal any in-flight analysis to abort.
            if self._cancellation_event is not None:
                self._cancellation_event.set()

            # Capture closure variables before updating state.
            previous_fen = self._previous_fen
            cancellation = threading.Event()
            detected_at = time.perf_counter()

            self._previous_fen = fen
            self._previous_hash = new_hash
            self._cancellation_event = cancellation
            self._change_detected_at = detected_at

            # Start debounce timer.
            timer = threading.Timer(
                self._config.debounce_ms / 1000.0,
                self._fire_change,
                args=(previous_fen, fen, cancellation, detected_at),
            )
            timer.daemon = True
            self._debounce_timer = timer
            timer.start()

        return True

    def stop(self) -> None:
        """Cancel any pending debounce timer and signal ongoing analysis."""
        with self._lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
                self._debounce_timer = None
            if self._cancellation_event is not None:
                self._cancellation_event.set()
                self._cancellation_event = None
        logger.info("PositionWatcher stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire_change(
        self,
        previous_fen: str,
        current_fen: str,
        cancellation: threading.Event,
        detected_at: float,
    ) -> None:
        """Invoke the registered callback after debounce expires.

        Called by the :class:`threading.Timer` in its daemon thread.
        """
        if cancellation.is_set():
            return

        elapsed = (time.perf_counter() - detected_at) * 1000.0
        event = PositionChangeEvent(
            previous_fen=previous_fen,
            current_fen=current_fen,
            elapsed_ms=elapsed,
        )

        logger.info(
            "Position change fired after %.1f ms: %s",
            elapsed,
            current_fen[:40],
        )

        if self._callback is not None:
            try:
                self._callback(event, cancellation)
            except Exception:
                logger.exception("Change callback raised an error.")
