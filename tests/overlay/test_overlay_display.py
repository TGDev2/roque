"""Tests for the analysis overlay display.

Covers:
- Evaluation-to-bar-fraction conversion (sigmoid, mate, symmetry, clamping)
- Overlay state building from AnalysisResult (normal, loading, partial, terminal)
- PV formatting and truncation
- OverlayConfig validation (alpha, refresh_interval_ms, width)
- Frozen dataclass immutability (OverlayState, VariationDisplay)
"""

from __future__ import annotations

import pytest

from chesslab.analysis import AnalysisResult, Evaluation, VariationInfo
from chesslab.overlay.overlay_display import (
    OverlayConfig,
    OverlayState,
    VariationDisplay,
    build_overlay_state,
    eval_to_bar_fraction,
    format_pv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _make_variation(
    rank: int = 1,
    move_san: str = "Nf3",
    centipawns: int = 50,
    pv_san: tuple[str, ...] = ("Nf3", "d5", "d4", "Nf6", "c4", "e6", "Nc3", "Be7"),
    depth: int = 18,
) -> VariationInfo:
    return VariationInfo(
        rank=rank,
        move_uci="g1f3",
        move_san=move_san,
        evaluation=Evaluation(centipawns=centipawns, mate_in=None),
        pv_uci=("g1f3", "d7d5", "d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8e7"),
        pv_san=pv_san,
        depth=depth,
    )


def _make_result(
    evaluation: Evaluation | None = None,
    best_move_san: str = "Nf3",
    variations: tuple[VariationInfo, ...] | None = None,
    depth_reached: int = 18,
    is_partial: bool = False,
) -> AnalysisResult:
    if evaluation is None:
        evaluation = Evaluation(centipawns=50, mate_in=None)
    if variations is None:
        variations = (
            _make_variation(rank=1, move_san="Nf3", centipawns=50),
            _make_variation(rank=2, move_san="e4", centipawns=30),
            _make_variation(rank=3, move_san="d4", centipawns=25),
        )
    return AnalysisResult(
        fen=_STARTING_FEN,
        best_move_uci="g1f3",
        best_move_san=best_move_san,
        evaluation=evaluation,
        variations=variations,
        depth_reached=depth_reached,
        is_partial=is_partial,
        elapsed_ms=1234.5,
    )


# ---------------------------------------------------------------------------
# AC: Evaluation bar fraction
# ---------------------------------------------------------------------------


class TestEvalToBarFraction:
    """AC: Evaluation displayed as visual bar (white/black advantage)."""

    def test_zero_centipawns_gives_half(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=0, mate_in=None))
        assert frac == pytest.approx(0.5, abs=0.001)

    def test_positive_centipawns_above_half(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=250, mate_in=None))
        assert 0.7 < frac < 0.8

    def test_negative_centipawns_below_half(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=-250, mate_in=None))
        assert 0.2 < frac < 0.3

    def test_large_positive_saturates(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=1000, mate_in=None))
        assert frac == pytest.approx(0.98, abs=0.01)

    def test_large_negative_saturates(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=-1000, mate_in=None))
        assert frac == pytest.approx(0.02, abs=0.01)

    def test_mate_positive_gives_max(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=None, mate_in=3))
        assert frac == 0.98

    def test_mate_negative_gives_min(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=None, mate_in=-3))
        assert frac == 0.02

    def test_mate_zero_gives_min(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=None, mate_in=0))
        assert frac == 0.02

    def test_symmetry(self) -> None:
        for cp in (50, 100, 250, 500):
            pos = eval_to_bar_fraction(Evaluation(centipawns=cp, mate_in=None))
            neg = eval_to_bar_fraction(Evaluation(centipawns=-cp, mate_in=None))
            assert pos + neg == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# AC: PV formatting
# ---------------------------------------------------------------------------


class TestFormatPv:
    """AC: Top 3 alternative moves listed with PV lines."""

    def test_short_pv_no_ellipsis(self) -> None:
        assert format_pv(("Nf3", "d5", "d4")) == "Nf3 d5 d4"

    def test_long_pv_truncated(self) -> None:
        pv = ("Nf3", "d5", "d4", "Nf6", "c4", "e6", "Nc3", "Be7")
        result = format_pv(pv)
        assert result == "Nf3 d5 d4 Nf6 c4 ..."

    def test_empty_pv(self) -> None:
        assert format_pv(()) == ""

    def test_exact_max_no_ellipsis(self) -> None:
        pv = ("e4", "e5", "Nf3", "Nc6", "Bb5")
        assert format_pv(pv) == "e4 e5 Nf3 Nc6 Bb5"

    def test_custom_max_moves(self) -> None:
        pv = ("e4", "e5", "Nf3", "Nc6")
        assert format_pv(pv, max_moves=2) == "e4 e5 ..."


# ---------------------------------------------------------------------------
# AC: Overlay state building from analysis result
# ---------------------------------------------------------------------------


class TestBuildOverlayState:
    """AC: Best move in SAN, evaluation, 3 alternatives, loading state."""

    def test_loading_state_no_result(self) -> None:
        state = build_overlay_state(None, is_loading=True)
        assert state.is_loading is True
        assert state.best_move_text == "Analysing..."
        assert state.evaluation_text == "--"
        assert state.bar_fraction == 0.5
        assert state.variations == ()
        assert state.depth_reached == 0

    def test_idle_state_no_result(self) -> None:
        state = build_overlay_state(None, is_loading=False)
        assert state.is_loading is False
        assert state.best_move_text == "--"
        assert state.evaluation_text == "--"

    def test_normal_result_with_variations(self) -> None:
        result = _make_result()
        state = build_overlay_state(result, is_loading=False)

        assert state.is_loading is False
        assert state.best_move_text == "Nf3"
        assert state.evaluation_text == "+0.50"
        assert len(state.variations) == 3
        assert state.depth_reached == 18
        assert state.is_partial is False

    def test_variation_details(self) -> None:
        result = _make_result()
        state = build_overlay_state(result, is_loading=False)

        v1 = state.variations[0]
        assert v1.rank == 1
        assert v1.move_san == "Nf3"
        assert v1.evaluation_text == "+0.50"
        assert "Nf3 d5 d4 Nf6 c4" in v1.pv_text

        v2 = state.variations[1]
        assert v2.rank == 2
        assert v2.move_san == "e4"

    def test_pv_truncation_in_variations(self) -> None:
        result = _make_result()
        state = build_overlay_state(result, is_loading=False)
        # Default PV has 8 moves, truncated to 5 + "..."
        assert state.variations[0].pv_text.endswith("...")

    def test_partial_result_flagged(self) -> None:
        result = _make_result(is_partial=True, depth_reached=14)
        state = build_overlay_state(result, is_loading=False)
        assert state.is_partial is True
        assert state.depth_reached == 14

    def test_terminal_position_no_variations(self) -> None:
        result = AnalysisResult(
            fen=_STARTING_FEN,
            best_move_uci="",
            best_move_san="",
            evaluation=Evaluation(centipawns=None, mate_in=0),
            variations=(),
            depth_reached=0,
            is_partial=False,
            elapsed_ms=0.1,
        )
        state = build_overlay_state(result, is_loading=False)
        assert state.best_move_text == "--"
        assert state.variations == ()
        assert state.bar_fraction == 0.02

    def test_mate_evaluation_bar(self) -> None:
        result = _make_result(
            evaluation=Evaluation(centipawns=None, mate_in=3),
        )
        state = build_overlay_state(result, is_loading=False)
        assert state.bar_fraction == 0.98
        assert "#" in state.evaluation_text

    def test_loading_with_previous_result(self) -> None:
        """When loading, state should carry the previous result's data."""
        result = _make_result()
        state = build_overlay_state(result, is_loading=True)
        assert state.is_loading is True
        # Result data is still available.
        assert state.best_move_text == "Nf3"
        assert state.evaluation_text == "+0.50"
        assert len(state.variations) == 3


# ---------------------------------------------------------------------------
# AC: Config validation
# ---------------------------------------------------------------------------


class TestOverlayConfig:
    """AC: Configurable opacity and refresh interval."""

    def test_defaults(self) -> None:
        config = OverlayConfig()
        assert config.alpha == 0.85
        assert config.refresh_interval_ms == 100
        assert config.width == 280
        assert config.bar_height == 200
        assert config.font_family == "monospace"
        assert config.font_size == 14
        assert config.position_x == 50
        assert config.position_y == 50

    def test_custom_values(self) -> None:
        config = OverlayConfig(alpha=0.7, width=350, refresh_interval_ms=200)
        assert config.alpha == 0.7
        assert config.width == 350
        assert config.refresh_interval_ms == 200

    def test_alpha_below_min_rejected(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            OverlayConfig(alpha=0.0)

    def test_alpha_above_max_rejected(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            OverlayConfig(alpha=1.5)

    def test_refresh_interval_below_min_rejected(self) -> None:
        with pytest.raises(ValueError, match="refresh_interval_ms"):
            OverlayConfig(refresh_interval_ms=10)

    def test_refresh_interval_above_max_rejected(self) -> None:
        with pytest.raises(ValueError, match="refresh_interval_ms"):
            OverlayConfig(refresh_interval_ms=1000)

    def test_width_below_min_rejected(self) -> None:
        with pytest.raises(ValueError, match="width"):
            OverlayConfig(width=50)

    def test_alpha_at_boundaries_accepted(self) -> None:
        config_min = OverlayConfig(alpha=0.1)
        assert config_min.alpha == 0.1
        config_max = OverlayConfig(alpha=1.0)
        assert config_max.alpha == 1.0


# ---------------------------------------------------------------------------
# AC: Frozen dataclasses
# ---------------------------------------------------------------------------


class TestVariationDisplay:
    """VariationDisplay is immutable."""

    def test_frozen(self) -> None:
        v = VariationDisplay(rank=1, move_san="Nf3", evaluation_text="+0.50", pv_text="Nf3 d5")
        with pytest.raises(AttributeError):
            v.rank = 2  # type: ignore[misc]


class TestOverlayState:
    """OverlayState is immutable."""

    def test_frozen(self) -> None:
        state = build_overlay_state(None, is_loading=False)
        with pytest.raises(AttributeError):
            state.is_loading = True  # type: ignore[misc]
