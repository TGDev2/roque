"""Embedded ECO opening detection."""

from chesslab.openings.eco import (
    DetectedOpening,
    OpeningGrouping,
    UNCLASSIFIED_OPENING,
    build_opening_filter_key,
    coerce_opening_filter_key,
    describe_opening_filter_key,
    describe_opening_group,
    detect_opening_from_uci_moves,
    opening_matches_filter,
)

__all__ = [
    "DetectedOpening",
    "OpeningGrouping",
    "UNCLASSIFIED_OPENING",
    "build_opening_filter_key",
    "coerce_opening_filter_key",
    "describe_opening_filter_key",
    "describe_opening_group",
    "detect_opening_from_uci_moves",
    "opening_matches_filter",
]
