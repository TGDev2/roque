"""Tests for startup journey orchestration."""

from __future__ import annotations

from pathlib import Path

from chesslab.gui.journeys import StartupJourneyMode, StartupJourneyOrchestrator
from chesslab.gui.live_observer import LiveObserverConfig
from chesslab.persistence.remote_import import RemoteGameSource


class _FakeStartupRuntime:
    def __init__(
        self,
        *,
        startup_review_file: Path | None = None,
        observe_config: LiveObserverConfig | None = None,
    ) -> None:
        self.startup_review_file = startup_review_file
        self._observe_config = observe_config
        self.calls: list[tuple[object, ...]] = []

    def set_startup_status(self, text: str, *, error: bool = False) -> None:
        self.calls.append(("status", text, error))

    def open_observe_live_startup(self) -> LiveObserverConfig | None:
        self.calls.append(("observe_dialog",))
        return self._observe_config

    def start_observe_live(self, config: LiveObserverConfig) -> bool:
        self.calls.append(("observe_start", config))
        return True

    def open_progression_startup(self) -> None:
        self.calls.append(("progression",))

    def open_review_recent_startup(self) -> None:
        self.calls.append(("review_recent",))

    def open_review_paste_startup(self) -> None:
        self.calls.append(("review_paste",))

    def open_review_file_startup(self, file_path: Path) -> None:
        self.calls.append(("review_file", file_path))

    def open_import_startup(self, source: RemoteGameSource) -> None:
        self.calls.append(("import", source))


def test_launcher_entries_cover_major_journeys_in_display_order() -> None:
    orchestrator = StartupJourneyOrchestrator()

    entries = orchestrator.launcher_entries()

    assert [entry.mode for entry in entries] == [
        StartupJourneyMode.MANUAL_COACH,
        StartupJourneyMode.OBSERVE_LIVE,
        StartupJourneyMode.PROGRESSION,
        StartupJourneyMode.REVIEW_RECENT,
        StartupJourneyMode.REVIEW_PASTE,
        StartupJourneyMode.IMPORT_LICHESS,
        StartupJourneyMode.IMPORT_CHESS_COM,
    ]


def test_observe_live_journey_owns_dialog_and_startup_transition() -> None:
    orchestrator = StartupJourneyOrchestrator()
    config = LiveObserverConfig(monitor_index=2, theme="lichess-blue")
    runtime = _FakeStartupRuntime(observe_config=config)

    orchestrator.launch(runtime, StartupJourneyMode.OBSERVE_LIVE)

    assert runtime.calls == [
        ("observe_dialog",),
        ("observe_start", config),
    ]


def test_progression_journey_sets_status_then_opens_dashboard() -> None:
    orchestrator = StartupJourneyOrchestrator()
    runtime = _FakeStartupRuntime()

    orchestrator.launch(runtime, StartupJourneyMode.PROGRESSION)

    assert runtime.calls == [
        ("status", "Progression — review your recent learning trends.", False),
        ("progression",),
    ]


def test_review_file_journey_uses_requested_file() -> None:
    orchestrator = StartupJourneyOrchestrator()
    file_path = Path("game.pgn")
    runtime = _FakeStartupRuntime(startup_review_file=file_path)

    orchestrator.launch(runtime, StartupJourneyMode.REVIEW_FILE)

    assert runtime.calls == [
        (
            "status",
            "Review — loading post-game analysis from game.pgn",
            False,
        ),
        ("review_file", file_path),
    ]


def test_review_file_journey_surfaces_missing_file_as_error() -> None:
    orchestrator = StartupJourneyOrchestrator()
    runtime = _FakeStartupRuntime()

    orchestrator.launch(runtime, StartupJourneyMode.REVIEW_FILE)

    assert runtime.calls == [
        ("status", "No review file was provided.", True),
    ]


def test_import_journeys_route_to_their_owned_source() -> None:
    orchestrator = StartupJourneyOrchestrator()
    runtime = _FakeStartupRuntime()

    orchestrator.launch(runtime, StartupJourneyMode.IMPORT_LICHESS)
    orchestrator.launch(runtime, StartupJourneyMode.IMPORT_CHESS_COM)

    assert runtime.calls == [
        ("status", "Import — fetch recent public games from Lichess.", False),
        ("import", RemoteGameSource.LICHESS),
        ("status", "Import — fetch recent public games from Chess.com.", False),
        ("import", RemoteGameSource.CHESS_COM),
    ]
