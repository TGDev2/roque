"""Tests for the progression mode controller."""

from __future__ import annotations

from unittest.mock import MagicMock

from chesslab.gui.controllers import AppMode, ProgressionModeRequest, TrainingModeRequest
from chesslab.gui.controllers.progression import ProgressionController

from tests.gui.controller_test_utils import DummyHost


class TestProgressionController:
    def test_activate_hides_board_and_refreshes_snapshot(self) -> None:
        host = DummyHost()
        controller = ProgressionController(host)
        controller._build_widgets = MagicMock()
        controller._container = MagicMock()
        controller._limit_var = MagicMock()
        controller._refresh_progression_panel = MagicMock()

        controller.activate(
            ProgressionModeRequest(sample_size=24, return_mode=AppMode.REVIEW)
        )

        host.set_board_visible.assert_called_once_with(False)
        controller._container.pack.assert_called_once_with(fill="both", expand=True)
        controller._refresh_progression_panel.assert_called_once()
        assert host.progression_sample_size == 24

    def test_start_training_from_dashboard_uses_current_limit_and_focus(self) -> None:
        host = DummyHost()
        controller = ProgressionController(host)
        controller._focus_review = "review"
        controller._coerce_progression_limit = MagicMock(return_value=12)

        controller._start_training_from_dashboard()

        mode, request = host.last_switch
        assert mode is AppMode.TRAINING
        assert isinstance(request, TrainingModeRequest)
        assert request.sample_size == 12
        assert request.focus_review == "review"
        assert request.return_mode is AppMode.PROGRESSION
