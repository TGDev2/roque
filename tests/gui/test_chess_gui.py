"""Smoke tests for the ChessGui shell."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from chesslab.analysis import AnalysisConfig
from chesslab.coach.game_coach import PlayerColor
from chesslab.gui.chess_gui import ChessGui, GuiConfig, StartupJourneyMode
from chesslab.gui.controllers import AppMode
from chesslab.persistence.remote_import import RemoteGameSource


class TestGuiConfig:
    def test_defaults(self) -> None:
        config = GuiConfig()
        assert config.player_color is None
        assert config.analysis.depth == 18
        assert config.square_size == 64
        assert config.startup_mode is None
        assert config.startup_review_file is None
        assert config.progression_sample_size == 20

    def test_custom_values(self) -> None:
        config = GuiConfig(
            player_color=PlayerColor.BLACK,
            analysis=AnalysisConfig(depth=22),
            square_size=72,
            progression_sample_size=30,
        )
        assert config.player_color == PlayerColor.BLACK
        assert config.analysis.depth == 22
        assert config.square_size == 72
        assert config.progression_sample_size == 30

    def test_supports_forced_startup_mode(self) -> None:
        config = GuiConfig(startup_mode=StartupJourneyMode.OBSERVE_LIVE)
        assert config.startup_mode is StartupJourneyMode.OBSERVE_LIVE

    def test_supports_progression_startup_mode(self) -> None:
        config = GuiConfig(startup_mode=StartupJourneyMode.PROGRESSION)
        assert config.startup_mode is StartupJourneyMode.PROGRESSION


def _make_shell() -> ChessGui:
    gui = ChessGui()
    gui._widgets = SimpleNamespace(
        board_host=MagicMock(),
        side_host=MagicMock(),
        board_widget=MagicMock(),
        status_host=MagicMock(),
        status_label=MagicMock(),
        status_icon=MagicMock(),
        root=MagicMock(),
    )
    return gui


class TestChessGuiShell:
    def test_switch_mode_deactivates_previous_and_activates_next(self) -> None:
        gui = _make_shell()
        play = MagicMock()
        review = MagicMock()
        gui._controllers = {AppMode.PLAY: play, AppMode.REVIEW: review}
        gui._active_mode = AppMode.PLAY

        gui.switch_mode(AppMode.REVIEW, request="payload")

        play.deactivate.assert_called_once()
        review.activate.assert_called_once_with("payload")
        assert gui.current_mode is AppMode.REVIEW

    def test_switch_mode_reuses_active_controller_for_same_mode(self) -> None:
        gui = _make_shell()
        play = MagicMock()
        gui._controllers = {AppMode.PLAY: play}
        gui._active_mode = AppMode.PLAY

        gui.switch_mode(AppMode.PLAY, request="payload")

        play.deactivate.assert_not_called()
        play.activate.assert_called_once_with("payload")

    def test_poll_routes_to_active_controller(self) -> None:
        gui = _make_shell()
        gui._root = MagicMock()
        play = MagicMock()
        gui._controllers = {AppMode.PLAY: play}
        gui._active_mode = AppMode.PLAY

        gui._poll_active_controller()

        play.poll.assert_called_once()
        gui._root.after.assert_called_once()

    def test_set_board_visible_hides_board_host(self) -> None:
        gui = _make_shell()
        gui.widgets.board_host.winfo_manager.return_value = "pack"

        gui.set_board_visible(False)

        gui.widgets.board_host.pack_forget.assert_called_once()

    def test_set_board_visible_restores_board_before_side_host(self) -> None:
        gui = _make_shell()
        gui.widgets.board_host.winfo_manager.return_value = ""

        gui.set_board_visible(True)

        gui.widgets.board_host.pack.assert_called_once_with(
            side="left",
            padx=(0, 8),
            before=gui.widgets.side_host,
        )

    @pytest.mark.parametrize(
        ("mode", "expected_app_mode"),
        [
            (StartupJourneyMode.PROGRESSION, AppMode.PROGRESSION),
            (StartupJourneyMode.REVIEW_RECENT, AppMode.REVIEW),
            (StartupJourneyMode.REVIEW_PASTE, AppMode.REVIEW),
            (StartupJourneyMode.IMPORT_LICHESS, AppMode.REVIEW),
            (StartupJourneyMode.IMPORT_CHESS_COM, AppMode.REVIEW),
        ],
    )
    def test_startup_runtime_routes_modes(
        self,
        mode: StartupJourneyMode,
        expected_app_mode: AppMode,
    ) -> None:
        gui = _make_shell()
        gui.switch_mode = MagicMock()

        if mode is StartupJourneyMode.PROGRESSION:
            gui.open_progression_startup()
        elif mode is StartupJourneyMode.REVIEW_RECENT:
            gui.open_review_recent_startup()
        elif mode is StartupJourneyMode.REVIEW_PASTE:
            gui.open_review_paste_startup()
        elif mode is StartupJourneyMode.IMPORT_LICHESS:
            gui.open_import_startup(RemoteGameSource.LICHESS)
        else:
            gui.open_import_startup(RemoteGameSource.CHESS_COM)

        assert gui.switch_mode.call_args.args[0] is expected_app_mode

    def test_apply_startup_mode_progression_opens_dashboard(self) -> None:
        gui = _make_shell()
        gui._root = MagicMock()
        gui.open_progression_startup = MagicMock()

        gui._apply_startup_mode(StartupJourneyMode.PROGRESSION)

        gui.open_progression_startup.assert_called_once()

    def test_apply_startup_mode_import_lichess_opens_dialog(self) -> None:
        gui = _make_shell()
        gui._root = MagicMock()
        gui.open_import_startup = MagicMock()

        gui._apply_startup_mode(StartupJourneyMode.IMPORT_LICHESS)

        gui.open_import_startup.assert_called_once_with(RemoteGameSource.LICHESS)

    def test_apply_startup_mode_import_chess_com_opens_dialog(self) -> None:
        gui = _make_shell()
        gui._root = MagicMock()
        gui.open_import_startup = MagicMock()

        gui._apply_startup_mode(StartupJourneyMode.IMPORT_CHESS_COM)

        gui.open_import_startup.assert_called_once_with(RemoteGameSource.CHESS_COM)
