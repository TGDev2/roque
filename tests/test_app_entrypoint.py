"""Tests for the unified ChessLab app entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from chesslab.analysis import EngineNotFoundError
from chesslab.app import main
from chesslab.gui import StartupJourneyMode


@patch("chesslab.app.ChessGui")
def test_default_keeps_startup_launcher(mock_gui_cls: MagicMock) -> None:
    mock_gui_cls.return_value.run.return_value = None

    exit_code = main([])

    assert exit_code == 0
    config = mock_gui_cls.call_args.args[0]
    assert config.startup_mode is None
    assert config.startup_review_file is None


@patch("chesslab.app.ChessGui")
def test_mode_maps_to_startup_mode(mock_gui_cls: MagicMock) -> None:
    mock_gui_cls.return_value.run.return_value = None

    exit_code = main(["--mode", "observe"])

    assert exit_code == 0
    config = mock_gui_cls.call_args.args[0]
    assert config.startup_mode is StartupJourneyMode.OBSERVE_LIVE


@patch("chesslab.app.ChessGui")
def test_progression_mode_maps_to_startup_mode(mock_gui_cls: MagicMock) -> None:
    mock_gui_cls.return_value.run.return_value = None

    exit_code = main(["--mode", "progression"])

    assert exit_code == 0
    config = mock_gui_cls.call_args.args[0]
    assert config.startup_mode is StartupJourneyMode.PROGRESSION


@patch("chesslab.app.ChessGui")
def test_review_file_forces_direct_review_mode(mock_gui_cls: MagicMock) -> None:
    mock_gui_cls.return_value.run.return_value = None

    exit_code = main(["--review-file", "game.pgn"])

    assert exit_code == 0
    config = mock_gui_cls.call_args.args[0]
    assert config.startup_mode is StartupJourneyMode.REVIEW_FILE
    assert config.startup_review_file is not None
    assert config.startup_review_file.name == "game.pgn"


@patch("chesslab.app.ChessGui")
def test_engine_not_found_returns_non_zero(mock_gui_cls: MagicMock) -> None:
    mock_gui_cls.return_value.run.side_effect = EngineNotFoundError("missing stockfish")

    exit_code = main([])

    assert exit_code == 1
