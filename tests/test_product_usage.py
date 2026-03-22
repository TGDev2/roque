"""Tests for ChessLab product-usage framing."""

from __future__ import annotations

import pytest

from chesslab.product_usage import (
    OBSERVE_LIVE_ACKNOWLEDGEMENT,
    OBSERVE_LIVE_GUARDRAIL_BODY,
    PRODUCT_BOUNDARY,
    PRODUCT_PRIORITY_RULE,
    journey_usage,
)


def test_manual_coach_usage_mentions_self_entered_training() -> None:
    manual = journey_usage("manual_coach")

    assert "Enter moves yourself" in manual.launcher_summary
    assert "board you control" in manual.usage_summary


def test_observe_live_guardrail_mentions_passive_observation_and_rules() -> None:
    live = journey_usage("observe_live")

    assert "Passively sync" in live.launcher_summary
    assert "outside tools" in OBSERVE_LIVE_GUARDRAIL_BODY
    assert "fair-play rules" in OBSERVE_LIVE_GUARDRAIL_BODY
    assert "outside tools" in OBSERVE_LIVE_ACKNOWLEDGEMENT


def test_review_usage_mentions_post_game_analysis() -> None:
    review = journey_usage("review")

    assert "after the game" in review.launcher_summary
    assert "Post-game analysis" in review.usage_summary


def test_product_boundary_and_priority_rule_reject_live_assistance_scope() -> None:
    assert "not positioned as live move assistance" in PRODUCT_BOUNDARY
    assert "active third-party game" in PRODUCT_PRIORITY_RULE


def test_unknown_journey_raises_helpful_error() -> None:
    with pytest.raises(ValueError, match="Unknown ChessLab journey"):
        journey_usage("blitz-helper")
