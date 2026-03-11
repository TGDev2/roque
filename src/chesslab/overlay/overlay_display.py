"""Analysis overlay display with evaluation bar and click-through.

Provides a semi-transparent, always-on-top tkinter window that shows the best
move, numeric evaluation, a visual evaluation bar, and up to three principal
variations.  The overlay is click-through so the user can interact normally
with the chess GUI underneath.

Thread safety
-------------
The :meth:`OverlayDisplay.update` and :meth:`OverlayDisplay.set_loading`
methods are safe to call from any thread.  They enqueue data onto an internal
:class:`queue.Queue` that the tkinter main loop polls via ``root.after()``.

Platform notes
--------------
* Click-through uses X11 + XFixes on Linux.  Wayland and macOS are not
  supported for click-through (a warning is logged and the overlay remains
  interactive).
* Windows click-through is not implemented in this MVP.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import math
import queue
import sys
import tkinter as tk
from dataclasses import dataclass
from typing import ClassVar

from chesslab.analysis import AnalysisResult, Evaluation, VariationInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class OverlayError(RuntimeError):
    """Base class for overlay module failures."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class OverlayConfig:
    """Configuration for the analysis overlay window.

    Parameters
    ----------
    alpha:
        Window opacity (0.1 = nearly transparent, 1.0 = fully opaque).
    position_x:
        Initial X screen coordinate.
    position_y:
        Initial Y screen coordinate.
    width:
        Window width in pixels.
    bar_height:
        Evaluation bar height in pixels.
    font_family:
        Font family for text rendering.
    font_size:
        Base font size in points.
    refresh_interval_ms:
        How often the queue is polled for new results (milliseconds).
        Must be <= 500 to satisfy the update latency acceptance criterion.
    """

    MIN_ALPHA: ClassVar[float] = 0.1
    MAX_ALPHA: ClassVar[float] = 1.0
    MIN_REFRESH_MS: ClassVar[int] = 50
    MAX_REFRESH_MS: ClassVar[int] = 500
    MIN_WIDTH: ClassVar[int] = 150

    alpha: float = 0.85
    position_x: int = 50
    position_y: int = 50
    width: int = 280
    bar_height: int = 200
    font_family: str = "monospace"
    font_size: int = 14
    refresh_interval_ms: int = 100

    def __post_init__(self) -> None:
        if not (self.MIN_ALPHA <= self.alpha <= self.MAX_ALPHA):
            raise ValueError(
                f"alpha must be between {self.MIN_ALPHA} and {self.MAX_ALPHA}, "
                f"got {self.alpha}.",
            )
        if not (self.MIN_REFRESH_MS <= self.refresh_interval_ms <= self.MAX_REFRESH_MS):
            raise ValueError(
                f"refresh_interval_ms must be between {self.MIN_REFRESH_MS} and "
                f"{self.MAX_REFRESH_MS}, got {self.refresh_interval_ms}.",
            )
        if self.width < self.MIN_WIDTH:
            raise ValueError(
                f"width must be >= {self.MIN_WIDTH}, got {self.width}.",
            )


@dataclass(frozen=True)
class VariationDisplay:
    """A formatted variation line ready for display.

    Attributes
    ----------
    rank:
        1-based rank (1 = best move).
    move_san:
        The move in Standard Algebraic Notation.
    evaluation_text:
        Formatted evaluation string (e.g. ``"+1.20"`` or ``"#3"``).
    pv_text:
        Truncated principal variation text.
    """

    rank: int
    move_san: str
    evaluation_text: str
    pv_text: str


@dataclass(frozen=True)
class OverlayState:
    """Pure display state decoupled from tkinter widgets.

    Separating the display state from the widget tree makes the formatting
    logic fully unit-testable without a display server.

    Attributes
    ----------
    evaluation_text:
        Formatted evaluation (e.g. ``"+2.50"`` or ``"#-3"``).
    best_move_text:
        Best move in SAN or a placeholder when loading.
    variations:
        Up to 3 formatted variation lines.
    bar_fraction:
        Evaluation bar position (0.0 = black winning, 1.0 = white winning).
    is_loading:
        ``True`` while an analysis is in progress.
    depth_reached:
        Search depth reached (0 when loading with no prior result).
    is_partial:
        ``True`` if the result was interrupted before target depth.
    """

    evaluation_text: str
    best_move_text: str
    variations: tuple[VariationDisplay, ...]
    bar_fraction: float
    is_loading: bool
    depth_reached: int
    is_partial: bool


# ---------------------------------------------------------------------------
# Pure formatting functions (testable without display)
# ---------------------------------------------------------------------------

# Sigmoid steepness: 250 cp ≈ 2.5 pawns maps to ~0.73.
_SIGMOID_SCALE = 250.0

# Clamp bounds so the bar always shows a sliver of each colour.
_BAR_MIN = 0.02
_BAR_MAX = 0.98


def eval_to_bar_fraction(evaluation: Evaluation) -> float:
    """Convert an :class:`Evaluation` to a 0.0–1.0 bar fraction.

    * ``0.0`` means black is totally winning.
    * ``1.0`` means white is totally winning.
    * ``0.5`` is an equal position.

    Uses a sigmoid curve for centipawn values and hard clamp for mates.
    The result is always in ``[0.02, 0.98]`` so both colours remain visible.
    """
    if evaluation.mate_in is not None:
        return _BAR_MAX if evaluation.mate_in > 0 else _BAR_MIN

    cp = evaluation.centipawns if evaluation.centipawns is not None else 0
    raw = 1.0 / (1.0 + math.exp(-cp / _SIGMOID_SCALE))
    return max(_BAR_MIN, min(raw, _BAR_MAX))


def format_pv(pv_san: tuple[str, ...], max_moves: int = 5) -> str:
    """Format a principal variation as a truncated move list.

    Parameters
    ----------
    pv_san:
        SAN move strings.
    max_moves:
        Maximum moves to include before appending ``"..."``.
    """
    if not pv_san:
        return ""
    truncated = pv_san[:max_moves]
    text = " ".join(truncated)
    if len(pv_san) > max_moves:
        text += " ..."
    return text


def build_overlay_state(
    result: AnalysisResult | None,
    is_loading: bool,
) -> OverlayState:
    """Build a pure :class:`OverlayState` from an analysis result.

    Parameters
    ----------
    result:
        The latest analysis result, or ``None`` if no result is available yet.
    is_loading:
        ``True`` when an analysis is currently in progress.
    """
    if result is None:
        return OverlayState(
            evaluation_text="--",
            best_move_text="Analysing..." if is_loading else "--",
            variations=(),
            bar_fraction=0.5,
            is_loading=is_loading,
            depth_reached=0,
            is_partial=False,
        )

    variations: list[VariationDisplay] = []
    for v in result.variations:
        variations.append(
            VariationDisplay(
                rank=v.rank,
                move_san=v.move_san,
                evaluation_text=str(v.evaluation),
                pv_text=format_pv(v.pv_san),
            ),
        )

    return OverlayState(
        evaluation_text=str(result.evaluation),
        best_move_text=result.best_move_san if result.best_move_san else "--",
        variations=tuple(variations),
        bar_fraction=eval_to_bar_fraction(result.evaluation),
        is_loading=is_loading,
        depth_reached=result.depth_reached,
        is_partial=result.is_partial,
    )


# ---------------------------------------------------------------------------
# Overlay display
# ---------------------------------------------------------------------------

# Colour palette.
_BG_COLOUR = "#1e1e1e"
_FG_COLOUR = "#e0e0e0"
_FG_DIM = "#888888"
_BAR_WHITE = "#f0f0f0"
_BAR_BLACK = "#2a2a2a"
_ACCENT = "#4fc3f7"
_TITLE_BG = "#2d2d2d"
_LOADING_COLOUR = "#ffa726"


class OverlayDisplay:
    """Tkinter overlay window for displaying analysis results.

    The window is always-on-top, borderless, semi-transparent, and
    click-through (on supported platforms).  Analysis results are fed via the
    thread-safe :meth:`update` and :meth:`set_loading` methods.

    Usage::

        overlay = OverlayDisplay(OverlayConfig(alpha=0.85))
        overlay.start()

        # From an analysis thread:
        overlay.set_loading()
        result = analyser.analyse(fen)
        overlay.update(result)

        # In the main thread (blocking):
        overlay.mainloop()
    """

    def __init__(self, config: OverlayConfig | None = None) -> None:
        self._config = config or OverlayConfig()
        self._queue: queue.Queue[AnalysisResult | None] = queue.Queue()
        self._root: tk.Tk | None = None
        self._last_result: AnalysisResult | None = None

        # Widget references (populated by start()).
        self._canvas: tk.Canvas | None = None
        self._best_move_label: tk.Label | None = None
        self._variation_labels: list[tk.Label] = []
        self._status_label: tk.Label | None = None
        self._eval_text_id: int | None = None
        self._white_rect_id: int | None = None
        self._black_rect_id: int | None = None

    # -- Public API (thread-safe) --------------------------------------------

    def update(self, result: AnalysisResult) -> None:
        """Enqueue a new analysis result for display.

        Safe to call from any thread.
        """
        self._queue.put(result)

    def set_loading(self) -> None:
        """Signal that an analysis is in progress.

        Safe to call from any thread.  The overlay shows a loading indicator
        while keeping the last result visible.
        """
        self._queue.put(None)

    # -- Lifecycle (main thread only) ----------------------------------------

    def start(self) -> None:
        """Create the tkinter window and start the queue polling loop.

        Must be called from the thread that will run :meth:`mainloop`.
        """
        root = tk.Tk()
        root.title("ChessLab")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", self._config.alpha)
        root.configure(bg=_BG_COLOUR)
        root.geometry(
            f"{self._config.width}x{self._config.bar_height + 160}"
            f"+{self._config.position_x}+{self._config.position_y}"
        )

        self._root = root
        self._build_widgets(root)
        self._apply_click_through()

        # Render initial empty state.
        self._render_state(build_overlay_state(None, is_loading=False))

        # Start polling.
        root.after(self._config.refresh_interval_ms, self._poll_queue)
        logger.info("Overlay started at (%d, %d).", self._config.position_x, self._config.position_y)

    def stop(self) -> None:
        """Destroy the overlay window."""
        if self._root is not None:
            try:
                self._root.destroy()
            except tk.TclError:
                pass
            self._root = None
            logger.info("Overlay stopped.")

    def mainloop(self) -> None:
        """Run the tkinter event loop (blocking).

        Call :meth:`start` before this method.

        Raises
        ------
        OverlayError
            If :meth:`start` has not been called.
        """
        if self._root is None:
            raise OverlayError("Overlay not started. Call start() first.")
        self._root.mainloop()

    # -- Widget construction -------------------------------------------------

    def _build_widgets(self, root: tk.Tk) -> None:
        """Build the overlay widget tree."""
        cfg = self._config
        width = cfg.width

        # Title bar with drag support.
        title_frame = tk.Frame(root, bg=_TITLE_BG, height=24)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="ChessLab",
            bg=_TITLE_BG,
            fg=_ACCENT,
            font=(cfg.font_family, cfg.font_size - 2, "bold"),
            anchor="w",
            padx=8,
        )
        title_label.pack(fill="both", expand=True)

        # Drag bindings.
        for widget in (title_frame, title_label):
            widget.bind("<Button-1>", self._on_drag_start)
            widget.bind("<B1-Motion>", self._on_drag_motion)

        # Evaluation bar (Canvas).
        canvas = tk.Canvas(
            root,
            width=width,
            height=cfg.bar_height,
            bg=_BAR_BLACK,
            highlightthickness=0,
        )
        canvas.pack(fill="x")

        self._black_rect_id = canvas.create_rectangle(
            0, 0, width, cfg.bar_height, fill=_BAR_BLACK, outline="",
        )
        self._white_rect_id = canvas.create_rectangle(
            0, cfg.bar_height // 2, width, cfg.bar_height, fill=_BAR_WHITE, outline="",
        )
        self._eval_text_id = canvas.create_text(
            width // 2,
            cfg.bar_height // 2,
            text="--",
            fill=_FG_COLOUR,
            font=(cfg.font_family, cfg.font_size + 2, "bold"),
        )
        self._canvas = canvas

        # Best move section.
        self._best_move_label = tk.Label(
            root,
            text="--",
            bg=_BG_COLOUR,
            fg=_FG_COLOUR,
            font=(cfg.font_family, cfg.font_size + 4, "bold"),
            anchor="w",
            padx=8,
            pady=4,
        )
        self._best_move_label.pack(fill="x")

        # Variation lines (3 slots).
        self._variation_labels = []
        for _ in range(3):
            lbl = tk.Label(
                root,
                text="",
                bg=_BG_COLOUR,
                fg=_FG_DIM,
                font=(cfg.font_family, cfg.font_size - 2),
                anchor="w",
                padx=8,
            )
            lbl.pack(fill="x")
            self._variation_labels.append(lbl)

        # Status bar.
        self._status_label = tk.Label(
            root,
            text="",
            bg=_BG_COLOUR,
            fg=_FG_DIM,
            font=(cfg.font_family, cfg.font_size - 3),
            anchor="w",
            padx=8,
            pady=2,
        )
        self._status_label.pack(fill="x")

    # -- Click-through -------------------------------------------------------

    def _apply_click_through(self) -> None:
        """Apply platform-specific click-through to the overlay window."""
        if sys.platform == "linux":
            self._apply_click_through_x11()
        else:
            logger.warning(
                "Click-through not supported on %s — overlay will capture clicks.",
                sys.platform,
            )

    def _apply_click_through_x11(self) -> None:
        """Use X11 + XFixes to make the window transparent to input events."""
        try:
            xlib_name = ctypes.util.find_library("X11")
            xfixes_name = ctypes.util.find_library("Xfixes")
            if xlib_name is None or xfixes_name is None:
                logger.warning("X11/XFixes libraries not found — click-through disabled.")
                return

            xlib = ctypes.cdll.LoadLibrary(xlib_name)
            xfixes = ctypes.cdll.LoadLibrary(xfixes_name)

            assert self._root is not None
            self._root.update_idletasks()
            display = xlib.XOpenDisplay(None)
            if not display:
                logger.warning("Cannot open X display — click-through disabled.")
                return

            window_id = self._root.winfo_id()

            # ShapeInput = 2 — make the input region empty.
            region = xfixes.XFixesCreateRegion(display, None, 0)
            xfixes.XFixesSetWindowShapeRegion(display, window_id, 2, 0, 0, region)
            xfixes.XFixesDestroyRegion(display, region)
            xlib.XFlush(display)
            xlib.XCloseDisplay(display)

            logger.info("Click-through enabled via X11/XFixes.")
        except (OSError, AttributeError) as exc:
            logger.warning("Click-through setup failed: %s", exc)

    # -- Drag support --------------------------------------------------------

    def _on_drag_start(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Record the initial drag position."""
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag_motion(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Move the window following the mouse."""
        if self._root is None:
            return
        x = self._root.winfo_x() + event.x - self._drag_x
        y = self._root.winfo_y() + event.y - self._drag_y
        self._root.geometry(f"+{x}+{y}")

    # -- Queue polling -------------------------------------------------------

    def _poll_queue(self) -> None:
        """Drain the queue and render the latest state."""
        try:
            while True:
                item = self._queue.get_nowait()
                if item is None:
                    # Loading signal — keep previous result visible.
                    state = build_overlay_state(self._last_result, is_loading=True)
                else:
                    self._last_result = item
                    state = build_overlay_state(item, is_loading=False)
                self._render_state(state)
        except queue.Empty:
            pass

        if self._root is not None:
            self._root.after(self._config.refresh_interval_ms, self._poll_queue)

    # -- Rendering -----------------------------------------------------------

    def _render_state(self, state: OverlayState) -> None:
        """Update all widgets from an :class:`OverlayState`."""
        if self._root is None:
            return

        cfg = self._config
        canvas = self._canvas
        assert canvas is not None

        # Evaluation bar.
        white_height = int(state.bar_fraction * cfg.bar_height)
        black_height = cfg.bar_height - white_height
        canvas.coords(
            self._black_rect_id,
            0, 0, cfg.width, black_height,
        )
        canvas.coords(
            self._white_rect_id,
            0, black_height, cfg.width, cfg.bar_height,
        )

        # Eval text on bar — use contrasting colour based on position.
        eval_text_colour = _BAR_BLACK if state.bar_fraction > 0.5 else _BAR_WHITE
        canvas.itemconfig(self._eval_text_id, text=state.evaluation_text, fill=eval_text_colour)
        canvas.coords(self._eval_text_id, cfg.width // 2, cfg.bar_height // 2)

        # Best move.
        assert self._best_move_label is not None
        if state.is_loading and state.best_move_text != "--":
            self._best_move_label.config(
                text=f"Best: {state.best_move_text}",
                fg=_FG_COLOUR,
            )
        elif state.is_loading:
            self._best_move_label.config(
                text="Analysing...",
                fg=_LOADING_COLOUR,
            )
        elif state.best_move_text == "--":
            self._best_move_label.config(text="--", fg=_FG_DIM)
        else:
            self._best_move_label.config(
                text=f"Best: {state.best_move_text}",
                fg=_FG_COLOUR,
            )

        # Variations.
        for i, lbl in enumerate(self._variation_labels):
            if i < len(state.variations):
                v = state.variations[i]
                lbl.config(
                    text=f"{v.rank}. {v.move_san}  {v.evaluation_text}  {v.pv_text}",
                )
            else:
                lbl.config(text="")

        # Status bar.
        assert self._status_label is not None
        if state.depth_reached > 0:
            status = f"depth {state.depth_reached}"
            if state.is_partial:
                status += " (partial)"
            if state.is_loading:
                status += "  \u2022 analysing..."
            self._status_label.config(text=status)
        elif state.is_loading:
            self._status_label.config(text="analysing...")
        else:
            self._status_label.config(text="")
