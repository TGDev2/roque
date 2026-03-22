"""Central product-usage framework and trust guardrails for ChessLab."""

from __future__ import annotations

from dataclasses import dataclass


PRODUCT_SCOPE = (
    "ChessLab is for self-directed training, permitted passive observation, "
    "and post-game review."
)

PRODUCT_BOUNDARY = (
    "ChessLab is not positioned as live move assistance during active games "
    "when a platform, event, or opponent expects unaided play."
)

PRODUCT_PRIORITY_RULE = (
    "If a feature makes engine-backed guidance easier during an active "
    "third-party game, it is out of scope unless product strategy changes "
    "explicitly."
)

OBSERVE_LIVE_DIALOG_INTRO = (
    "Select a declared setup, confirm your usage context, run a live "
    "preflight, and start observation only after ChessLab returns a PASS "
    "verdict. Observe Live stays passive: board sync only, with no move "
    "entry or coaching affordances until you exit intentionally."
)

OBSERVE_LIVE_GUARDRAIL_TITLE = "Observe Live is for passive sync only"

OBSERVE_LIVE_GUARDRAIL_BODY = (
    "Use Observe Live for study boards, broadcasts, your own analysis "
    "sessions, or any context that explicitly allows outside tools. Do not "
    "use it to receive engine-backed guidance during an active game if that "
    "would break platform, event, or fair-play rules. During observation, "
    "ChessLab limits itself to passive synchronization and keeps move entry, "
    "suggestions, and coaching actions separate."
)

OBSERVE_LIVE_ACKNOWLEDGEMENT = (
    "I will only use Observe Live where passive observation and outside tools "
    "are allowed."
)


@dataclass(frozen=True)
class JourneyUsage:
    """Product framing for a top-level ChessLab journey."""

    key: str
    label: str
    launcher_summary: str
    status_prefix: str
    usage_summary: str


_JOURNEYS: dict[str, JourneyUsage] = {
    "manual_coach": JourneyUsage(
        key="manual_coach",
        label="Manual Coach",
        launcher_summary=(
            "Enter moves yourself on a board you control for training, "
            "rehearsal, or offline analysis."
        ),
        status_prefix="Manual Coach",
        usage_summary=(
            "Self-entered move training and engine study on a board you control."
        ),
    ),
    "observe_live": JourneyUsage(
        key="observe_live",
        label="Observe Live",
        launcher_summary=(
            "Passively sync a visible board only in contexts that allow "
            "outside tools and observation."
        ),
        status_prefix="Observe Live",
        usage_summary=(
            "Passive board synchronization for permitted live study or "
            "broadcast contexts."
        ),
    ),
    "review": JourneyUsage(
        key="review",
        label="Review",
        launcher_summary=(
            "Analyse saved or pasted PGNs after the game to understand "
            "mistakes and critical moments."
        ),
        status_prefix="Review",
        usage_summary="Post-game analysis and lesson extraction from completed games.",
    ),
}


def journey_usage(key: str) -> JourneyUsage:
    """Return the product framing for a named journey."""
    try:
        return _JOURNEYS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_JOURNEYS))
        raise ValueError(
            f"Unknown ChessLab journey '{key}'. Expected one of: {available}."
        ) from exc


def all_journey_usages() -> tuple[JourneyUsage, ...]:
    """Return all top-level journey framings in display order."""
    return (
        _JOURNEYS["manual_coach"],
        _JOURNEYS["observe_live"],
        _JOURNEYS["review"],
    )
