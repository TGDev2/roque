"""Tests for GUI panel formatting logic.

Tests the data transformations used by panels, not tkinter widgets.
Reuses pure functions from the overlay module.
"""

from __future__ import annotations

from chesslab.analysis import Evaluation, VariationInfo
from chesslab.overlay.overlay_display import eval_to_bar_fraction, format_pv


# ---------------------------------------------------------------------------
# eval_to_bar_fraction (reused from overlay)
# ---------------------------------------------------------------------------


class TestEvalToBarFraction:
    """Bar fraction correctly maps evaluation to 0.0–1.0 range."""

    def test_equal_position(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=0, mate_in=None))
        assert abs(frac - 0.5) < 0.01

    def test_white_winning(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=300, mate_in=None))
        assert frac > 0.5

    def test_black_winning(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=-300, mate_in=None))
        assert frac < 0.5

    def test_mate_for_white(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=None, mate_in=3))
        assert frac >= 0.95

    def test_mate_for_black(self) -> None:
        frac = eval_to_bar_fraction(Evaluation(centipawns=None, mate_in=-3))
        assert frac <= 0.05

    def test_clamped_bounds(self) -> None:
        frac_max = eval_to_bar_fraction(Evaluation(centipawns=10000, mate_in=None))
        frac_min = eval_to_bar_fraction(Evaluation(centipawns=-10000, mate_in=None))
        assert frac_max <= 0.98
        assert frac_min >= 0.02


# ---------------------------------------------------------------------------
# format_pv
# ---------------------------------------------------------------------------


class TestFormatPv:
    """Principal variation formatting."""

    def test_empty(self) -> None:
        assert format_pv(()) == ""

    def test_short_pv(self) -> None:
        result = format_pv(("e4", "e5", "Nf3"))
        assert result == "e4 e5 Nf3"

    def test_truncated_pv(self) -> None:
        moves = ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4")
        result = format_pv(moves, max_moves=5)
        assert result == "e4 e5 Nf3 Nc6 Bb5 ..."

    def test_exact_max(self) -> None:
        moves = ("e4", "e5", "Nf3", "Nc6", "Bb5")
        result = format_pv(moves, max_moves=5)
        assert result == "e4 e5 Nf3 Nc6 Bb5"
        assert "..." not in result


# ---------------------------------------------------------------------------
# Evaluation display formatting
# ---------------------------------------------------------------------------


class TestEvaluationDisplay:
    """Evaluation __str__ formatting used by panels."""

    def test_positive_centipawns(self) -> None:
        assert str(Evaluation(centipawns=134, mate_in=None)) == "+1.34"

    def test_negative_centipawns(self) -> None:
        assert str(Evaluation(centipawns=-50, mate_in=None)) == "-0.50"

    def test_mate_positive(self) -> None:
        assert str(Evaluation(centipawns=None, mate_in=3)) == "#+3"

    def test_mate_negative(self) -> None:
        assert str(Evaluation(centipawns=None, mate_in=-2)) == "#-2"

    def test_checkmate(self) -> None:
        assert str(Evaluation(centipawns=None, mate_in=0)) == "#0"


# ---------------------------------------------------------------------------
# Move history formatting
# ---------------------------------------------------------------------------


class TestMoveHistoryFormat:
    """Move history is paired into full-move lines."""

    def test_paired_moves(self) -> None:
        """Moves should be paired: 1. e4 e5, 2. Nf3 Nc6."""
        moves = ("e4", "e5", "Nf3", "Nc6")
        lines: list[str] = []
        for i in range(0, len(moves), 2):
            move_num = i // 2 + 1
            white = moves[i]
            black = moves[i + 1] if i + 1 < len(moves) else ""
            line = f"{move_num}. {white}  {black}".rstrip()
            lines.append(line)
        assert lines[0] == "1. e4  e5"
        assert lines[1] == "2. Nf3  Nc6"

    def test_odd_number_of_moves(self) -> None:
        """Last move unpaired when odd count."""
        moves = ("e4", "e5", "Nf3")
        lines: list[str] = []
        for i in range(0, len(moves), 2):
            move_num = i // 2 + 1
            white = moves[i]
            black = moves[i + 1] if i + 1 < len(moves) else ""
            if black:
                lines.append(f"{move_num}. {white}  {black}")
            else:
                lines.append(f"{move_num}. {white}")
        assert lines[-1] == "2. Nf3"
