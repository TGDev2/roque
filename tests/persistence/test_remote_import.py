"""Tests for public-platform PGN imports."""

from __future__ import annotations

from pathlib import Path

from chesslab.persistence.game_storage import GameStorage
from chesslab.persistence.remote_import import (
    RemoteGameCandidate,
    RemoteGameClient,
    RemoteGameSource,
    RemoteImportService,
)


LICHESS_PGN = """\
[Event "Rated Blitz game"]
[Site "https://lichess.org/AbCd1234"]
[Date "2026.03.24"]
[White "Alpha"]
[Black "Beta"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
"""

CHESS_COM_PGN = """\
[Event "Live Chess"]
[Site "Chess.com"]
[Date "2026.03.23"]
[White "Gamma"]
[Black "Delta"]
[Result "0-1"]

1. d4 d5 2. c4 e6 0-1
"""


class _JsonRecordingClient(RemoteGameClient):
    def __init__(self, payloads: dict[str, dict[str, object]]) -> None:
        super().__init__()
        self._payloads = payloads
        self.requested_urls: list[str] = []

    def _request_json(
        self,
        url: str,
        *,
        source_label: str,
        username: str,
    ) -> dict[str, object]:
        self.requested_urls.append(url)
        return self._payloads[url]


class _NdjsonClient(RemoteGameClient):
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        super().__init__()
        self._payloads = payloads

    def _request_ndjson(
        self,
        url: str,
        *,
        accept: str,
        source_label: str,
        username: str,
    ) -> list[dict[str, object]]:
        return self._payloads


class _StubServiceClient(RemoteGameClient):
    def __init__(self, games: list[RemoteGameCandidate], pgn_by_id: dict[str, str]) -> None:
        super().__init__()
        self._games = games
        self._pgn_by_id = pgn_by_id
        self.fetch_calls: list[str] = []

    def list_recent_games(
        self,
        source: RemoteGameSource,
        *,
        username: str,
        limit: int = 10,
    ) -> list[RemoteGameCandidate]:
        assert source is self._games[0].source
        assert username == self._games[0].username
        return self._games[:limit]

    def fetch_game_pgn(self, game: RemoteGameCandidate) -> str:
        self.fetch_calls.append(game.game_id)
        return self._pgn_by_id[game.game_id]


class TestRemoteGameClient:
    def test_lichess_listing_parses_ndjson_payload(self) -> None:
        client = _NdjsonClient([
            {
                "id": "AbCd1234",
                "winner": "white",
                "status": "mate",
                "speed": "blitz",
                "rated": True,
                "lastMoveAt": 1774358400000,
                "players": {
                    "white": {"user": {"name": "Alpha"}},
                    "black": {"user": {"name": "Beta"}},
                },
                "pgn": LICHESS_PGN,
            },
        ])

        games = client.list_recent_games(
            RemoteGameSource.LICHESS,
            username="alpha",
            limit=10,
        )

        assert len(games) == 1
        assert games[0].game_id == "AbCd1234"
        assert games[0].white == "Alpha"
        assert games[0].black == "Beta"
        assert games[0].result == "1-0"
        assert games[0].time_class == "blitz"
        assert games[0].pgn == LICHESS_PGN

    def test_chess_com_listing_paginates_archives_until_limit(self) -> None:
        archives_url = "https://api.chess.com/pub/player/gamma/games/archives"
        march_url = "https://api.chess.com/pub/player/gamma/games/2026/03"
        february_url = "https://api.chess.com/pub/player/gamma/games/2026/02"
        client = _JsonRecordingClient({
            archives_url: {"archives": [february_url, march_url]},
            march_url: {
                "games": [
                    {
                        "uuid": "march-game",
                        "url": "https://www.chess.com/game/live/1",
                        "end_time": 1774272000,
                        "time_class": "rapid",
                        "rated": True,
                        "pgn": CHESS_COM_PGN,
                        "white": {"username": "Gamma", "result": "resigned"},
                        "black": {"username": "Delta", "result": "win"},
                    },
                ],
            },
            february_url: {
                "games": [
                    {
                        "uuid": "older-game",
                        "url": "https://www.chess.com/game/live/2",
                        "end_time": 1771000000,
                        "time_class": "blitz",
                        "rated": False,
                        "pgn": CHESS_COM_PGN,
                        "white": {"username": "Gamma", "result": "win"},
                        "black": {"username": "Delta", "result": "checkmated"},
                    },
                    {
                        "uuid": "middle-game",
                        "url": "https://www.chess.com/game/live/3",
                        "end_time": 1772000000,
                        "time_class": "blitz",
                        "rated": True,
                        "pgn": CHESS_COM_PGN,
                        "white": {"username": "Gamma", "result": "agreed"},
                        "black": {"username": "Delta", "result": "agreed"},
                    },
                ],
            },
        })

        games = client.list_recent_games(
            RemoteGameSource.CHESS_COM,
            username="gamma",
            limit=2,
        )

        assert [game.game_id for game in games] == ["march-game", "middle-game"]
        assert games[0].result == "0-1"
        assert games[1].result == "1/2-1/2"
        assert client.requested_urls == [archives_url, march_url, february_url]


class TestRemoteImportService:
    def test_list_recent_games_marks_existing_duplicates(
        self,
        tmp_path: Path,
    ) -> None:
        storage = GameStorage(games_dir=tmp_path)
        storage.store_imported_game(
            LICHESS_PGN,
            source=RemoteGameSource.LICHESS.value,
            source_game_id="AbCd1234",
        )

        client = _StubServiceClient(
            games=[
                RemoteGameCandidate(
                    source=RemoteGameSource.LICHESS,
                    username="alpha",
                    game_id="AbCd1234",
                    white="Alpha",
                    black="Beta",
                    result="1-0",
                    played_at=None,
                    site_url="https://lichess.org/AbCd1234",
                    time_class="blitz",
                    rated=True,
                    pgn=LICHESS_PGN,
                ),
                RemoteGameCandidate(
                    source=RemoteGameSource.LICHESS,
                    username="alpha",
                    game_id="EfGh5678",
                    white="Alpha",
                    black="Gamma",
                    result="0-1",
                    played_at=None,
                    site_url="https://lichess.org/EfGh5678",
                    time_class="rapid",
                    rated=False,
                    pgn=LICHESS_PGN.replace("AbCd1234", "EfGh5678").replace("Beta", "Gamma"),
                ),
            ],
            pgn_by_id={},
        )
        service = RemoteImportService(storage=storage, client=client)

        games = service.list_recent_games(
            RemoteGameSource.LICHESS,
            username="alpha",
            limit=10,
        )

        assert games[0].already_imported is True
        assert games[1].already_imported is False

    def test_import_games_fetches_missing_pgn_and_reports_duplicates(
        self,
        tmp_path: Path,
    ) -> None:
        storage = GameStorage(games_dir=tmp_path)
        storage.store_imported_game(
            LICHESS_PGN,
            source=RemoteGameSource.LICHESS.value,
            source_game_id="AbCd1234",
        )

        new_game = RemoteGameCandidate(
            source=RemoteGameSource.LICHESS,
            username="alpha",
            game_id="EfGh5678",
            white="Alpha",
            black="Gamma",
            result="0-1",
            played_at=None,
            site_url="https://lichess.org/EfGh5678",
            time_class="rapid",
            rated=True,
            pgn=None,
        )
        duplicate_game = RemoteGameCandidate(
            source=RemoteGameSource.LICHESS,
            username="alpha",
            game_id="AbCd1234",
            white="Alpha",
            black="Beta",
            result="1-0",
            played_at=None,
            site_url="https://lichess.org/AbCd1234",
            time_class="blitz",
            rated=True,
            pgn=None,
        )
        client = _StubServiceClient(
            games=[new_game, duplicate_game],
            pgn_by_id={
                "EfGh5678": LICHESS_PGN.replace("AbCd1234", "EfGh5678").replace("Beta", "Gamma"),
                "AbCd1234": LICHESS_PGN,
            },
        )
        service = RemoteImportService(storage=storage, client=client)

        result = service.import_games([new_game, duplicate_game])

        assert result.imported_count == 1
        assert result.skipped_count == 1
        assert result.failed_count == 0
        assert len(result.resolved_games) == 2
        assert result.resolved_games[0].game.game_id == "EfGh5678"
        assert result.resolved_games[0].imported is True
        assert result.resolved_games[1].game.game_id == "AbCd1234"
        assert result.resolved_games[1].imported is False
        assert client.fetch_calls == ["EfGh5678", "AbCd1234"]
        assert len(list(tmp_path.glob("*.pgn"))) == 2
