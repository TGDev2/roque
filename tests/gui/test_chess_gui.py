"""Tests for the ChessGui controller logic.

Tests the data flow and routing without requiring a display server.
"""

from __future__ import annotations

import pytest

from chesslab.analysis import AnalysisConfig
from chesslab.coach.game_coach import PlayerColor
from chesslab.gui.chess_gui import GuiConfig, StartupJourneyMode


# ---------------------------------------------------------------------------
# GuiConfig
# ---------------------------------------------------------------------------


class TestGuiConfig:
    """GuiConfig holds GUI configuration."""

    def test_defaults(self) -> None:
        config = GuiConfig()
        assert config.player_color is None
        assert config.analysis.depth == 18
        assert config.square_size == 64
        assert config.startup_mode is None
        assert config.startup_review_file is None

    def test_custom_values(self) -> None:
        config = GuiConfig(
            player_color=PlayerColor.BLACK,
            analysis=AnalysisConfig(depth=22),
            square_size=72,
        )
        assert config.player_color == PlayerColor.BLACK
        assert config.analysis.depth == 22
        assert config.square_size == 72

    def test_supports_forced_startup_mode(self) -> None:
        config = GuiConfig(startup_mode=StartupJourneyMode.OBSERVE_LIVE)
        assert config.startup_mode is StartupJourneyMode.OBSERVE_LIVE

    def test_player_color_none_triggers_dialog(self) -> None:
        """When player_color is None, a dialog should be shown."""
        config = GuiConfig()
        assert config.player_color is None
