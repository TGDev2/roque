"""Import recent games directly from public Lichess and Chess.com APIs."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from chesslab.persistence.game_storage import GameStorage, PersistenceError

_DEFAULT_TIMEOUT_SECONDS = 15
_DEFAULT_USER_AGENT = "ChessLab/0.1"
_CHESS_COM_DRAW_RESULTS = {
    "agreed",
    "insufficient",
    "repetition",
    "stalemate",
    "timevsinsufficient",
    "50move",
}
_CHESS_COM_LOSS_RESULTS = {
    "abandoned",
    "checkmated",
    "lose",
    "resigned",
    "timeout",
}


class RemoteImportError(RuntimeError):
    """Base class for remote-import failures."""


class RemoteImportNetworkError(RemoteImportError):
    """Raised when the remote platform cannot be reached."""


class RemoteImportUserNotFoundError(RemoteImportError):
    """Raised when the requested remote user does not exist."""


class RemoteGameSource(str, enum.Enum):
    """Supported public game sources."""

    LICHESS = "lichess"
    CHESS_COM = "chess.com"

    @property
    def label(self) -> str:
        if self is RemoteGameSource.LICHESS:
            return "Lichess"
        return "Chess.com"


@dataclass(frozen=True)
class RemoteGameCandidate:
    """Remote game that can be displayed and imported."""

    source: RemoteGameSource
    username: str
    game_id: str
    white: str
    black: str
    result: str
    played_at: datetime | None
    site_url: str | None
    time_class: str | None
    rated: bool | None
    pgn: str | None = None
    already_imported: bool = False

    @property
    def played_date(self) -> str:
        if self.played_at is None:
            return "????.??.??"
        return self.played_at.astimezone(timezone.utc).strftime("%Y.%m.%d")

    def display_label(self) -> str:
        parts = [
            self.played_date,
            f"{self.white} vs {self.black}",
            self.result,
        ]
        if self.time_class:
            parts.append(self.time_class)
        if self.already_imported:
            parts.append("already imported")
        return "  ".join(parts)


@dataclass(frozen=True)
class RemoteImportFailure:
    """Single-game import failure."""

    game: RemoteGameCandidate
    message: str


@dataclass(frozen=True)
class RemoteImportedGame:
    """Imported or resolved local PGN for a selected remote game."""

    game: RemoteGameCandidate
    file_path: Path
    imported: bool


@dataclass(frozen=True)
class RemoteImportBatchResult:
    """Aggregate result of importing several remote games."""

    imported_files: tuple[Path, ...]
    skipped_duplicates: tuple[RemoteGameCandidate, ...]
    failures: tuple[RemoteImportFailure, ...]
    resolved_games: tuple[RemoteImportedGame, ...] = ()

    @property
    def imported_count(self) -> int:
        return len(self.imported_files)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_duplicates)

    @property
    def failed_count(self) -> int:
        return len(self.failures)

    @property
    def resolved_files(self) -> tuple[Path, ...]:
        return tuple(item.file_path for item in self.resolved_games)


class RemoteGameClient:
    """HTTP client for public game archives."""

    def __init__(
        self,
        *,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        user_agent: str = _DEFAULT_USER_AGENT,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent

    def list_recent_games(
        self,
        source: RemoteGameSource,
        *,
        username: str,
        limit: int = 10,
    ) -> list[RemoteGameCandidate]:
        """Return the most recent games for *username* from *source*."""
        cleaned_username = username.strip()
        if not cleaned_username:
            raise RemoteImportError("Please enter a username.")

        if limit < 1:
            raise RemoteImportError("Please request at least one recent game.")

        if source is RemoteGameSource.LICHESS:
            return self._list_lichess_games(cleaned_username, limit)
        return self._list_chess_com_games(cleaned_username, limit)

    def fetch_game_pgn(self, game: RemoteGameCandidate) -> str:
        """Fetch the PGN for a previously listed game."""
        if game.pgn:
            return game.pgn

        if game.source is RemoteGameSource.LICHESS:
            url = f"https://lichess.org/game/export/{parse.quote(game.game_id)}"
            return self._request_text(
                url,
                accept="application/x-chess-pgn",
                source_label=game.source.label,
                username=game.username,
            )

        raise RemoteImportError(
            f"{game.source.label} did not provide a PGN for game {game.game_id}.",
        )

    def _list_lichess_games(
        self,
        username: str,
        limit: int,
    ) -> list[RemoteGameCandidate]:
        params = parse.urlencode({
            "max": str(limit),
            "moves": "false",
            "clocks": "false",
            "evals": "false",
            "opening": "false",
            "pgnInJson": "true",
            "finished": "true",
        })
        url = f"https://lichess.org/api/games/user/{parse.quote(username)}?{params}"
        payloads = self._request_ndjson(
            url,
            accept="application/x-ndjson",
            source_label=RemoteGameSource.LICHESS.label,
            username=username,
        )
        games = [
            game
            for game in (
                self._parse_lichess_game(payload, username=username)
                for payload in payloads
            )
            if game is not None
        ]
        if not games:
            raise RemoteImportError(
                f"No recent games were found for '{username}' on Lichess.",
            )
        return games[:limit]

    def _list_chess_com_games(
        self,
        username: str,
        limit: int,
    ) -> list[RemoteGameCandidate]:
        archives_url = (
            f"https://api.chess.com/pub/player/{parse.quote(username)}/games/archives"
        )
        archives_payload = self._request_json(
            archives_url,
            source_label=RemoteGameSource.CHESS_COM.label,
            username=username,
        )
        archives = archives_payload.get("archives")
        if not isinstance(archives, list) or not archives:
            raise RemoteImportError(
                f"No public game archives were found for '{username}' on Chess.com.",
            )

        games: list[RemoteGameCandidate] = []
        for archive_url in reversed(archives):
            if not isinstance(archive_url, str):
                continue
            archive_payload = self._request_json(
                archive_url,
                source_label=RemoteGameSource.CHESS_COM.label,
                username=username,
            )
            archive_games = archive_payload.get("games")
            if not isinstance(archive_games, list):
                continue
            for item in archive_games:
                if not isinstance(item, dict):
                    continue
                parsed = self._parse_chess_com_game(item, username=username)
                if parsed is not None:
                    games.append(parsed)
            if len(games) >= limit:
                break

        games.sort(
            key=lambda item: item.played_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        if not games:
            raise RemoteImportError(
                f"No recent games were found for '{username}' on Chess.com.",
            )
        return games[:limit]

    def _parse_lichess_game(
        self,
        payload: dict[str, Any],
        *,
        username: str,
    ) -> RemoteGameCandidate | None:
        game_id = payload.get("id")
        if not isinstance(game_id, str) or not game_id:
            return None

        players = payload.get("players")
        white = self._extract_lichess_player_name(players, "white")
        black = self._extract_lichess_player_name(players, "black")
        winner = payload.get("winner")
        status = payload.get("status")
        played_at = _datetime_from_millis(payload.get("lastMoveAt"))
        if played_at is None:
            played_at = _datetime_from_millis(payload.get("createdAt"))

        return RemoteGameCandidate(
            source=RemoteGameSource.LICHESS,
            username=username,
            game_id=game_id,
            white=white,
            black=black,
            result=_result_from_winner_and_status(winner, status),
            played_at=played_at,
            site_url=f"https://lichess.org/{game_id}",
            time_class=_string_or_none(payload.get("speed")),
            rated=_bool_or_none(payload.get("rated")),
            pgn=_string_or_none(payload.get("pgn")),
        )

    @staticmethod
    def _extract_lichess_player_name(
        players: Any,
        colour: str,
    ) -> str:
        if not isinstance(players, dict):
            return colour.capitalize()
        raw_player = players.get(colour)
        if not isinstance(raw_player, dict):
            return colour.capitalize()
        raw_user = raw_player.get("user")
        if isinstance(raw_user, dict):
            name = raw_user.get("name")
            if isinstance(name, str) and name:
                return name
        name = raw_player.get("name")
        if isinstance(name, str) and name:
            return name
        return colour.capitalize()

    def _parse_chess_com_game(
        self,
        payload: dict[str, Any],
        *,
        username: str,
    ) -> RemoteGameCandidate | None:
        game_id = _string_or_none(payload.get("uuid"))
        site_url = _string_or_none(payload.get("url"))
        if game_id is None:
            game_id = site_url
        if game_id is None:
            return None

        white_payload = payload.get("white") if isinstance(payload.get("white"), dict) else {}
        black_payload = payload.get("black") if isinstance(payload.get("black"), dict) else {}
        white = _string_or_default(white_payload.get("username"), "White")
        black = _string_or_default(black_payload.get("username"), "Black")
        white_result = _string_or_none(white_payload.get("result"))
        black_result = _string_or_none(black_payload.get("result"))
        played_at = _datetime_from_seconds(payload.get("end_time"))
        if played_at is None:
            played_at = _datetime_from_seconds(payload.get("start_time"))

        return RemoteGameCandidate(
            source=RemoteGameSource.CHESS_COM,
            username=username,
            game_id=game_id,
            white=white,
            black=black,
            result=_result_from_chess_com_players(white_result, black_result),
            played_at=played_at,
            site_url=site_url,
            time_class=(
                _string_or_none(payload.get("time_class"))
                or _string_or_none(payload.get("time_control"))
            ),
            rated=_bool_or_none(payload.get("rated")),
            pgn=_string_or_none(payload.get("pgn")),
        )

    def _request_json(
        self,
        url: str,
        *,
        source_label: str,
        username: str,
    ) -> dict[str, Any]:
        text = self._request_text(
            url,
            accept="application/json",
            source_label=source_label,
            username=username,
        )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RemoteImportError(
                f"{source_label} returned an invalid JSON response: {exc.msg}.",
            ) from exc
        if not isinstance(payload, dict):
            raise RemoteImportError(
                f"{source_label} returned an unexpected response format.",
            )
        return payload

    def _request_ndjson(
        self,
        url: str,
        *,
        accept: str,
        source_label: str,
        username: str,
    ) -> list[dict[str, Any]]:
        text = self._request_text(
            url,
            accept=accept,
            source_label=source_label,
            username=username,
        )
        payloads: list[dict[str, Any]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RemoteImportError(
                    f"{source_label} returned an invalid game list: {exc.msg}.",
                ) from exc
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    def _request_text(
        self,
        url: str,
        *,
        accept: str,
        source_label: str,
        username: str,
    ) -> str:
        req = request.Request(
            url,
            headers={
                "Accept": accept,
                "User-Agent": self._user_agent,
            },
        )
        try:
            with request.urlopen(req, timeout=self._timeout_seconds) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except error.HTTPError as exc:
            raise self._map_http_error(
                exc,
                source_label=source_label,
                username=username,
            ) from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise RemoteImportNetworkError(
                f"Could not reach {source_label}: {reason}.",
            ) from exc

    @staticmethod
    def _map_http_error(
        exc: error.HTTPError,
        *,
        source_label: str,
        username: str,
    ) -> RemoteImportError:
        if exc.code == 404:
            return RemoteImportUserNotFoundError(
                f"{source_label} user '{username}' was not found.",
            )
        if exc.code == 429:
            return RemoteImportNetworkError(
                f"{source_label} is rate-limiting requests right now. Please try again shortly.",
            )
        return RemoteImportNetworkError(
            f"{source_label} returned HTTP {exc.code} while loading games for '{username}'.",
        )


class RemoteImportService:
    """High-level orchestration for remote listing + local persistence."""

    def __init__(
        self,
        *,
        storage: GameStorage | None = None,
        client: RemoteGameClient | None = None,
    ) -> None:
        self._storage = storage or GameStorage()
        self._client = client or RemoteGameClient()

    def list_recent_games(
        self,
        source: RemoteGameSource,
        *,
        username: str,
        limit: int = 10,
    ) -> list[RemoteGameCandidate]:
        """List recent remote games and mark duplicates already stored locally."""
        games = self._client.list_recent_games(source, username=username, limit=limit)
        return [
            replace(
                game,
                already_imported=self._storage.is_imported_game(
                    source=game.source.value,
                    source_game_id=game.game_id,
                    pgn_text=game.pgn,
                ),
            )
            for game in games
        ]

    def import_games(
        self,
        games: list[RemoteGameCandidate],
    ) -> RemoteImportBatchResult:
        """Import selected remote games into local PGN storage."""
        imported_files: list[Path] = []
        skipped_duplicates: list[RemoteGameCandidate] = []
        failures: list[RemoteImportFailure] = []
        resolved_games: list[RemoteImportedGame] = []

        for game in games:
            try:
                pgn_text = self._client.fetch_game_pgn(game)
                result = self._storage.store_imported_game(
                    pgn_text,
                    source=game.source.value,
                    source_game_id=game.game_id,
                )
            except (RemoteImportError, PersistenceError) as exc:
                failures.append(RemoteImportFailure(game=game, message=str(exc)))
                continue

            if result.imported:
                imported_files.append(result.file_path)
                resolved_games.append(RemoteImportedGame(
                    game=game,
                    file_path=result.file_path,
                    imported=True,
                ))
            else:
                skipped_duplicates.append(replace(game, already_imported=True))
                resolved_games.append(RemoteImportedGame(
                    game=game,
                    file_path=result.file_path,
                    imported=False,
                ))

        return RemoteImportBatchResult(
            imported_files=tuple(imported_files),
            skipped_duplicates=tuple(skipped_duplicates),
            failures=tuple(failures),
            resolved_games=tuple(resolved_games),
        )


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _string_or_default(value: Any, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _datetime_from_millis(value: Any) -> datetime | None:
    if not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _datetime_from_seconds(value: Any) -> datetime | None:
    if not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _result_from_winner_and_status(
    winner: Any,
    status: Any,
) -> str:
    if winner == "white":
        return "1-0"
    if winner == "black":
        return "0-1"
    if isinstance(status, str) and status.lower() in {
        "draw",
        "stalemate",
        "repetition",
        "50move",
    }:
        return "1/2-1/2"
    return "*"


def _result_from_chess_com_players(
    white_result: str | None,
    black_result: str | None,
) -> str:
    if white_result == "win":
        return "1-0"
    if black_result == "win":
        return "0-1"
    if white_result in _CHESS_COM_DRAW_RESULTS or black_result in _CHESS_COM_DRAW_RESULTS:
        return "1/2-1/2"
    if white_result in _CHESS_COM_LOSS_RESULTS:
        return "0-1"
    if black_result in _CHESS_COM_LOSS_RESULTS:
        return "1-0"
    return "*"
