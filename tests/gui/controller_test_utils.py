"""Shared test helpers for GUI mode controllers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from chesslab.analysis import AnalysisConfig
from chesslab.gui.controllers import AppMode


class DummyHost:
    """Small host double for isolated controller tests."""

    def __init__(self) -> None:
        self.widgets = SimpleNamespace(
            root=MagicMock(),
            board_host=MagicMock(),
            board_widget=MagicMock(),
            side_host=MagicMock(),
            status_host=MagicMock(),
            status_label=MagicMock(),
            status_icon=MagicMock(),
        )
        self.services = MagicMock()
        self.current_mode: AppMode | None = AppMode.PLAY
        self.engine_elo = AnalysisConfig.DEFAULT_ENGINE_ELO
        self.progression_sample_size = 20
        self.last_switch: tuple[AppMode, object | None] | None = None

        self.create_coach = MagicMock()
        self.show_color_dialog = MagicMock()
        self.set_status = MagicMock()
        self.set_board_visible = MagicMock()
        self.relaunch_startup = MagicMock()

        def switch_mode(mode: AppMode, request: object | None = None) -> None:
            self.current_mode = mode
            self.last_switch = (mode, request)

        self.switch_mode = MagicMock(side_effect=switch_mode)
