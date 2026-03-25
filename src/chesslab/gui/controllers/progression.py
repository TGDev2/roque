"""Cross-session progression dashboard controller."""

from __future__ import annotations

import logging
import tkinter as tk
import tkinter.ttk as ttk

from chesslab.gui import theme
from chesslab.gui.controllers.base import (
    AppMode,
    ControllerBase,
    ProgressionModeRequest,
    TrainingModeRequest,
)
from chesslab.gui.panels import ProgressionPanel
from chesslab.openings import (
    OpeningGrouping,
    coerce_opening_filter_key,
)

logger = logging.getLogger(__name__)

_GROUPING_LABELS: dict[OpeningGrouping, str] = {
    OpeningGrouping.ECO_CODE: "Exact ECO",
    OpeningGrouping.ECO_FAMILY: "ECO family",
}
_GROUPING_BY_LABEL = {
    label: grouping
    for grouping, label in _GROUPING_LABELS.items()
}
_ALL_OPENINGS_LABEL = "All openings"


class ProgressionController(ControllerBase):
    """Own the root progression dashboard mode."""

    mode = AppMode.PROGRESSION

    def __init__(self, host: object) -> None:
        super().__init__(host)
        self._container: tk.Frame | None = None
        self._panel: ProgressionPanel | None = None
        self._limit_var: tk.IntVar | None = None
        self._grouping_var: tk.StringVar | None = None
        self._opening_var: tk.StringVar | None = None
        self._grouping_combo: ttk.Combobox | None = None
        self._opening_combo: ttk.Combobox | None = None
        self._return_mode = AppMode.PLAY
        self._focus_review = None
        self._opening_grouping = OpeningGrouping.ECO_CODE
        self._opening_filter_key: str | None = None
        self._opening_options: dict[str, str | None] = {}
        self._suspend_filter_events = False

    def activate(self, request: object | None = None) -> None:
        progression_request = self._coerce_request(request)
        if progression_request is not None:
            self._return_mode = progression_request.return_mode
            self._focus_review = progression_request.focus_review
            if progression_request.sample_size is not None:
                self._host.progression_sample_size = max(3, progression_request.sample_size)

        self._build_widgets()
        assert self._container is not None
        assert self._limit_var is not None

        self._host.set_board_visible(False)
        self._container.pack(fill="both", expand=True)
        self._limit_var.set(self._host.progression_sample_size)
        self._refresh_progression_panel()
        self._host.set_status("Progression — review your recent learning trends.")

    def deactivate(self) -> None:
        if self._container is not None:
            self._container.pack_forget()
        self._host.set_board_visible(True)

    def _coerce_request(
        self,
        request: object | None,
    ) -> ProgressionModeRequest | None:
        if isinstance(request, ProgressionModeRequest):
            return request
        return None

    def _build_widgets(self) -> None:
        if self._container is not None:
            return

        self._container = tk.Frame(self.widgets.side_host, bg=theme.BG)
        self._container.pack_propagate(False)

        title_row = tk.Frame(self._container, bg=theme.BG)
        title_row.pack(fill="x", padx=16, pady=(16, 10))

        tk.Label(
            title_row,
            text="Progression",
            bg=theme.BG,
            fg=theme.FG,
            font=("sans-serif", 16, "bold"),
            anchor="w",
        ).pack(side="left")

        controls = tk.Frame(self._container, bg=theme.BG)
        controls.pack(fill="x", padx=16, pady=(0, 10))

        tk.Label(
            controls,
            text="Last",
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
        ).pack(side="left")

        self._limit_var = tk.IntVar(value=self._host.progression_sample_size)
        limit_spin = tk.Spinbox(
            controls,
            from_=3,
            to=100,
            width=5,
            textvariable=self._limit_var,
            justify="center",
            bg=theme.ENTRY_BG,
            fg=theme.ENTRY_FG,
            buttonbackground=theme.BTN_BG,
            insertbackground=theme.FG,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("monospace", 10),
        )
        limit_spin.pack(side="left", padx=(6, 6))
        limit_spin.bind("<Return>", lambda _event: self._refresh_progression_panel())
        limit_spin.bind("<FocusOut>", lambda _event: self._refresh_progression_panel())

        tk.Label(
            controls,
            text="reviewed games",
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
        ).pack(side="left")

        theme.styled_button(
            controls,
            text="Refresh",
            command=self._refresh_progression_panel,
            font=("sans-serif", 9, "bold"),
            padx=10,
            pady=4,
        ).pack(side="right")

        filter_row = tk.Frame(self._container, bg=theme.BG)
        filter_row.pack(fill="x", padx=16, pady=(0, 10))

        tk.Label(
            filter_row,
            text="Openings",
            bg=theme.BG,
            fg=theme.FG_DIM,
            font=("sans-serif", 9),
        ).pack(side="left")

        self._grouping_var = tk.StringVar(
            value=_GROUPING_LABELS[self._opening_grouping],
        )
        self._grouping_combo = ttk.Combobox(
            filter_row,
            state="readonly",
            width=12,
            textvariable=self._grouping_var,
            values=tuple(_GROUPING_BY_LABEL),
        )
        self._grouping_combo.pack(side="left", padx=(8, 6))
        self._grouping_combo.bind("<<ComboboxSelected>>", self._on_grouping_changed)

        self._opening_var = tk.StringVar(value=_ALL_OPENINGS_LABEL)
        self._opening_combo = ttk.Combobox(
            filter_row,
            state="readonly",
            width=28,
            textvariable=self._opening_var,
            values=(_ALL_OPENINGS_LABEL,),
        )
        self._opening_combo.pack(side="left", fill="x", expand=True)
        self._opening_combo.bind("<<ComboboxSelected>>", self._on_opening_changed)

        self._panel = ProgressionPanel(self._container)
        self._panel.pack(fill="both", expand=True, padx=16)

        footer = tk.Frame(self._container, bg=theme.BG)
        footer.pack(fill="x", padx=16, pady=(10, 16))
        theme.styled_button(
            footer,
            text="Train weakness",
            command=self._start_training_from_dashboard,
            font=("sans-serif", 10, "bold"),
        ).pack(side="left")
        theme.styled_button(
            footer,
            text="Back",
            command=self._exit_progression_mode,
            font=("sans-serif", 10),
        ).pack(side="right")

    def _refresh_progression_panel(self) -> None:
        if self._panel is None:
            return

        limit = self._coerce_progression_limit()
        try:
            snapshot = self.services.progress_tracker.build_snapshot(
                limit=limit,
                opening_grouping=self._opening_grouping,
                opening_filter_key=self._opening_filter_key,
            )
        except Exception as exc:
            logger.exception("Failed to build progression snapshot.")
            self._panel.show_error(f"Could not load progression data: {exc}")
            return

        self._update_opening_options(snapshot)
        self._panel.update_snapshot(snapshot)

    def _coerce_progression_limit(self) -> int:
        if self._limit_var is None:
            return self._host.progression_sample_size

        try:
            limit = int(self._limit_var.get())
        except (tk.TclError, ValueError):
            limit = self._host.progression_sample_size

        limit = max(3, min(limit, 100))
        self._host.progression_sample_size = limit
        self._limit_var.set(limit)
        return limit

    def _start_training_from_dashboard(self) -> None:
        self._host.switch_mode(
            AppMode.TRAINING,
            TrainingModeRequest(
                sample_size=self._coerce_progression_limit(),
                focus_review=self._focus_review,
                return_mode=AppMode.PROGRESSION,
                opening_filter_key=self._opening_filter_key,
            ),
        )

    def _exit_progression_mode(self) -> None:
        self._host.switch_mode(self._return_mode)

    def _on_grouping_changed(self, _event: object | None = None) -> None:
        if self._grouping_var is None:
            return
        grouping = _GROUPING_BY_LABEL.get(self._grouping_var.get())
        if grouping is None or grouping is self._opening_grouping:
            return
        self._opening_grouping = grouping
        self._opening_filter_key = coerce_opening_filter_key(
            self._opening_filter_key,
            grouping,
        )
        self._refresh_progression_panel()

    def _on_opening_changed(self, _event: object | None = None) -> None:
        if self._suspend_filter_events or self._opening_var is None:
            return
        self._opening_filter_key = self._opening_options.get(self._opening_var.get())
        self._refresh_progression_panel()

    def _update_opening_options(self, snapshot: object) -> None:
        if self._opening_combo is None or self._opening_var is None:
            return

        opening_stats = getattr(snapshot, "opening_stats", ())
        options: dict[str, str | None] = {_ALL_OPENINGS_LABEL: None}
        for stat in opening_stats:
            options[stat.label] = stat.key

        if self._opening_filter_key is not None and self._opening_filter_key not in options.values():
            self._opening_filter_key = None

        label = next(
            (
                option_label
                for option_label, option_key in options.items()
                if option_key == self._opening_filter_key
            ),
            _ALL_OPENINGS_LABEL,
        )

        self._opening_options = options
        self._suspend_filter_events = True
        self._opening_combo.config(values=tuple(options))
        self._opening_var.set(label)
        if self._grouping_var is not None:
            self._grouping_var.set(_GROUPING_LABELS[self._opening_grouping])
        self._suspend_filter_events = False
