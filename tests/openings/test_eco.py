"""Tests for embedded ECO opening detection."""

from __future__ import annotations

import chess

from chesslab.openings import (
    OpeningGrouping,
    UNCLASSIFIED_OPENING,
    build_opening_filter_key,
    coerce_opening_filter_key,
    describe_opening_group,
    detect_opening_from_uci_moves,
    opening_matches_filter,
)


class TestDetectOpeningFromUciMoves:
    def test_detects_exact_eco_code_for_najdorf(self) -> None:
        opening = detect_opening_from_uci_moves((
            "e2e4",
            "c7c5",
            "g1f3",
            "d7d6",
            "d2d4",
            "c5d4",
            "f3d4",
            "g8f6",
            "b1c3",
            "a7a6",
        ))

        assert opening.eco == "B90"
        assert opening.name == "Sicilian Defense: Najdorf Variation"

    def test_prefers_deepest_named_match(self) -> None:
        opening = detect_opening_from_uci_moves((
            "e2e4",
            "c7c5",
            "g1f3",
            "d7d6",
            "d2d4",
            "c5d4",
            "f3d4",
            "g8f6",
            "b1c3",
            "a7a6",
        ))

        assert opening.eco == "B90"
        assert opening.eco != "B20"

    def test_recognizes_transposition_by_position(self) -> None:
        opening = detect_opening_from_uci_moves((
            "d2d4",
            "e7e6",
            "c2c4",
            "d7d5",
        ))

        assert opening.eco == "D30"
        assert opening.name == "Queen's Gambit Declined"

    def test_returns_unclassified_for_nonstandard_start(self) -> None:
        board = chess.Board()
        board.push_uci("e2e4")

        opening = detect_opening_from_uci_moves(
            ("e7e5", "g1f3"),
            start_fen=board.fen(),
        )

        assert opening == UNCLASSIFIED_OPENING

    def test_returns_unclassified_for_empty_sequence(self) -> None:
        opening = detect_opening_from_uci_moves(())

        assert opening == UNCLASSIFIED_OPENING


class TestOpeningFilters:
    def test_builds_exact_and_family_keys(self) -> None:
        opening = detect_opening_from_uci_moves(("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"))

        assert build_opening_filter_key(opening, OpeningGrouping.ECO_CODE) == "eco:C60"
        assert build_opening_filter_key(opening, OpeningGrouping.ECO_FAMILY) == "family:C"
        assert describe_opening_group(opening, OpeningGrouping.ECO_FAMILY) == "C00-C99"

    def test_matches_and_coerces_filter_keys(self) -> None:
        opening = detect_opening_from_uci_moves(("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"))

        assert opening_matches_filter(opening, "eco:C60") is True
        assert opening_matches_filter(opening, "family:C") is True
        assert opening_matches_filter(opening, "eco:B90") is False
        assert coerce_opening_filter_key("eco:C60", OpeningGrouping.ECO_FAMILY) == "family:C"
        assert coerce_opening_filter_key("family:C", OpeningGrouping.ECO_CODE) is None
