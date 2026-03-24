"""Startup journey definitions and orchestration for ChessLab."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence

from chesslab.persistence.remote_import import RemoteGameSource
from chesslab.product_usage import journey_usage

if TYPE_CHECKING:
    from chesslab.gui.live_observer import LiveObserverConfig


class StartupJourneyMode(enum.Enum):
    """Initial product flow selected at startup."""

    MANUAL_COACH = "manual_coach"
    OBSERVE_LIVE = "observe_live"
    PROGRESSION = "progression"
    REVIEW_RECENT = "review_recent"
    REVIEW_PASTE = "review_paste"
    REVIEW_FILE = "review_file"
    IMPORT_LICHESS = "import_lichess"
    IMPORT_CHESS_COM = "import_chess_com"


CLI_STARTUP_MODE_ALIASES: dict[str, StartupJourneyMode] = {
    "manual": StartupJourneyMode.MANUAL_COACH,
    "observe": StartupJourneyMode.OBSERVE_LIVE,
    "progression": StartupJourneyMode.PROGRESSION,
    "review-recent": StartupJourneyMode.REVIEW_RECENT,
    "review-paste": StartupJourneyMode.REVIEW_PASTE,
}


@dataclass(frozen=True)
class LauncherJourney:
    """User-visible launcher card for a startup journey."""

    mode: StartupJourneyMode
    label: str
    description: str


class StartupJourneyRuntime(Protocol):
    """Host interface used by the startup journey orchestrator."""

    @property
    def startup_review_file(self) -> Path | None:
        """Optional PGN file requested at startup for direct review."""

    def set_startup_status(self, text: str, *, error: bool = False) -> None:
        """Update the host status message for startup routing."""

    def open_observe_live_startup(self) -> LiveObserverConfig | None:
        """Collect observe-live setup details from the user."""

    def start_observe_live(self, config: LiveObserverConfig) -> bool:
        """Enter Observe Live with a validated configuration."""

    def open_progression_startup(self) -> None:
        """Open the progression dashboard."""

    def open_review_recent_startup(self) -> None:
        """Open the recent-games review flow."""

    def open_review_paste_startup(self) -> None:
        """Open the paste-PGN review flow."""

    def open_review_file_startup(self, file_path: Path) -> None:
        """Open a concrete PGN file directly in review."""

    def open_import_startup(self, source: RemoteGameSource) -> None:
        """Open the import flow for the selected remote source."""


class StartupJourneyHandler(Protocol):
    """Journey-specific startup orchestration responsibility."""

    def launcher_entries(self) -> tuple[LauncherJourney, ...]:
        """Return launcher cards owned by this journey."""

    def modes(self) -> frozenset[StartupJourneyMode]:
        """Return the startup modes handled by this journey."""

    def launch(
        self,
        runtime: StartupJourneyRuntime,
        mode: StartupJourneyMode,
    ) -> None:
        """Execute this journey's startup routing."""


class ManualCoachJourney:
    """Own launcher metadata for Manual Coach startup."""

    def launcher_entries(self) -> tuple[LauncherJourney, ...]:
        manual = journey_usage("manual_coach")
        return (
            LauncherJourney(
                mode=StartupJourneyMode.MANUAL_COACH,
                label=manual.label,
                description=manual.launcher_summary,
            ),
        )

    def modes(self) -> frozenset[StartupJourneyMode]:
        return frozenset({StartupJourneyMode.MANUAL_COACH})

    def launch(
        self,
        runtime: StartupJourneyRuntime,
        mode: StartupJourneyMode,
    ) -> None:
        return


class ObserveLiveJourney:
    """Own startup routing for the Observe Live journey."""

    def launcher_entries(self) -> tuple[LauncherJourney, ...]:
        observe = journey_usage("observe_live")
        return (
            LauncherJourney(
                mode=StartupJourneyMode.OBSERVE_LIVE,
                label=observe.label,
                description=observe.launcher_summary,
            ),
        )

    def modes(self) -> frozenset[StartupJourneyMode]:
        return frozenset({StartupJourneyMode.OBSERVE_LIVE})

    def launch(
        self,
        runtime: StartupJourneyRuntime,
        mode: StartupJourneyMode,
    ) -> None:
        config = runtime.open_observe_live_startup()
        if config is not None:
            runtime.start_observe_live(config)


class ProgressionJourney:
    """Own startup routing for longitudinal progression."""

    def launcher_entries(self) -> tuple[LauncherJourney, ...]:
        return (
            LauncherJourney(
                mode=StartupJourneyMode.PROGRESSION,
                label="Progression",
                description=(
                    "Open your learning profile built from recent reviews: move "
                    "quality mix, centipawn-loss trend, recurring lesson themes, "
                    "and where your errors happen most often."
                ),
            ),
        )

    def modes(self) -> frozenset[StartupJourneyMode]:
        return frozenset({StartupJourneyMode.PROGRESSION})

    def launch(
        self,
        runtime: StartupJourneyRuntime,
        mode: StartupJourneyMode,
    ) -> None:
        runtime.set_startup_status(
            "Progression — review your recent learning trends.",
        )
        runtime.open_progression_startup()


class ReviewJourney:
    """Own startup routing for the review flows."""

    def launcher_entries(self) -> tuple[LauncherJourney, ...]:
        review = journey_usage("review")
        return (
            LauncherJourney(
                mode=StartupJourneyMode.REVIEW_RECENT,
                label="Review Recent",
                description=(
                    "Open your saved PGNs, select one or more games, and run a "
                    "post-game review queue that feeds progression and training."
                ),
            ),
            LauncherJourney(
                mode=StartupJourneyMode.REVIEW_PASTE,
                label="Paste PGN",
                description=review.launcher_summary,
            ),
        )

    def modes(self) -> frozenset[StartupJourneyMode]:
        return frozenset({
            StartupJourneyMode.REVIEW_RECENT,
            StartupJourneyMode.REVIEW_PASTE,
            StartupJourneyMode.REVIEW_FILE,
        })

    def launch(
        self,
        runtime: StartupJourneyRuntime,
        mode: StartupJourneyMode,
    ) -> None:
        if mode is StartupJourneyMode.REVIEW_RECENT:
            runtime.set_startup_status(
                "Review — select a saved game for post-game analysis.",
            )
            runtime.open_review_recent_startup()
            return

        if mode is StartupJourneyMode.REVIEW_PASTE:
            runtime.set_startup_status(
                "Review — paste a PGN to launch post-game analysis.",
            )
            runtime.open_review_paste_startup()
            return

        file_path = runtime.startup_review_file
        if file_path is None:
            runtime.set_startup_status("No review file was provided.", error=True)
            return
        runtime.set_startup_status(
            f"Review — loading post-game analysis from {file_path.name}",
        )
        runtime.open_review_file_startup(file_path)


class ImportJourney:
    """Own startup routing for remote-import flows."""

    def launcher_entries(self) -> tuple[LauncherJourney, ...]:
        return (
            LauncherJourney(
                mode=StartupJourneyMode.IMPORT_LICHESS,
                label="Import Lichess",
                description=(
                    "Fetch your latest public Lichess games by username, select the "
                    "ones you want, then import and review them in one flow."
                ),
            ),
            LauncherJourney(
                mode=StartupJourneyMode.IMPORT_CHESS_COM,
                label="Import Chess.com",
                description=(
                    "Fetch your latest public Chess.com archive games by username, "
                    "select one or more, then import and review them in one flow."
                ),
            ),
        )

    def modes(self) -> frozenset[StartupJourneyMode]:
        return frozenset({
            StartupJourneyMode.IMPORT_LICHESS,
            StartupJourneyMode.IMPORT_CHESS_COM,
        })

    def launch(
        self,
        runtime: StartupJourneyRuntime,
        mode: StartupJourneyMode,
    ) -> None:
        if mode is StartupJourneyMode.IMPORT_LICHESS:
            runtime.set_startup_status(
                "Import — fetch recent public games from Lichess.",
            )
            runtime.open_import_startup(RemoteGameSource.LICHESS)
            return

        runtime.set_startup_status(
            "Import — fetch recent public games from Chess.com.",
        )
        runtime.open_import_startup(RemoteGameSource.CHESS_COM)


class StartupJourneyOrchestrator:
    """Route startup journeys without centralizing every path in the GUI."""

    def __init__(
        self,
        journeys: Sequence[StartupJourneyHandler] | None = None,
    ) -> None:
        self._journeys = tuple(
            journeys
            if journeys is not None
            else (
                ManualCoachJourney(),
                ObserveLiveJourney(),
                ProgressionJourney(),
                ReviewJourney(),
                ImportJourney(),
            )
        )
        self._journey_by_mode: dict[StartupJourneyMode, StartupJourneyHandler] = {}
        for journey in self._journeys:
            for mode in journey.modes():
                if mode in self._journey_by_mode:
                    raise ValueError(f"Duplicate startup journey registration for {mode.value}.")
                self._journey_by_mode[mode] = journey

    def launcher_entries(self) -> tuple[LauncherJourney, ...]:
        """Return launcher cards in display order."""
        entries: list[LauncherJourney] = []
        for journey in self._journeys:
            entries.extend(journey.launcher_entries())
        return tuple(entries)

    def launch(
        self,
        runtime: StartupJourneyRuntime,
        mode: StartupJourneyMode,
    ) -> None:
        """Route a startup mode to the journey that owns it."""
        try:
            journey = self._journey_by_mode[mode]
        except KeyError as exc:
            raise ValueError(f"Unsupported startup journey '{mode.value}'.") from exc
        journey.launch(runtime, mode)
