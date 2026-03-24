"""Game persistence via PGN files.

Provides saving, loading, and listing of chess games in standard PGN
format.  Auto-saved games are stored in ``~/.chesslab/games/``.

Usage::

    from chesslab.persistence import GameStorage, build_headers
    from chesslab.coach.game_coach import PlayerColor

    storage = GameStorage()
    headers = build_headers(player_color=PlayerColor.WHITE, engine_elo=1500)
    path = storage.save_game(board, headers)
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn

from chesslab.coach.game_coach import PlayerColor

logger = logging.getLogger(__name__)

_GAMES_DIR = Path.home() / ".chesslab" / "games"
_IMPORT_INDEX_FILE = ".import_index.json"


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class PersistenceError(RuntimeError):
    """Base class for persistence failures."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameSummary:
    """Lightweight summary of a saved game for listing in the UI."""

    file_path: Path
    white: str
    black: str
    result: str
    date: str
    move_count: int


@dataclass(frozen=True)
class GameRecord:
    """Full representation of a loaded PGN game."""

    headers: dict[str, str]
    moves: tuple[str, ...]
    board: chess.Board
    result: str
    file_path: Path | None


@dataclass(frozen=True)
class ImportedGameSaveResult:
    """Outcome of importing a PGN into the games directory."""

    file_path: Path
    imported: bool


@dataclass(frozen=True)
class _ImportedGameIndexEntry:
    """Deduplication metadata for previously imported PGNs."""

    file_path: Path
    source: str
    source_game_id: str | None
    pgn_sha256: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "file_path": str(self.file_path),
            "source": self.source,
            "source_game_id": self.source_game_id,
            "pgn_sha256": self.pgn_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> _ImportedGameIndexEntry:
        file_path = payload.get("file_path")
        source = payload.get("source")
        source_game_id = payload.get("source_game_id")
        pgn_sha256 = payload.get("pgn_sha256")
        if not isinstance(file_path, str):
            raise PersistenceError("Invalid imported game index entry: missing file path")
        if not isinstance(source, str):
            raise PersistenceError("Invalid imported game index entry: missing source")
        if source_game_id is not None and not isinstance(source_game_id, str):
            raise PersistenceError("Invalid imported game index entry: invalid source id")
        if not isinstance(pgn_sha256, str):
            raise PersistenceError("Invalid imported game index entry: missing PGN hash")
        return cls(
            file_path=Path(file_path),
            source=source,
            source_game_id=source_game_id,
            pgn_sha256=pgn_sha256,
        )


# ---------------------------------------------------------------------------
# Header builder
# ---------------------------------------------------------------------------


def build_headers(
    *,
    player_color: PlayerColor,
    engine_elo: int | None = None,
    result: str = "*",
) -> dict[str, str]:
    """Build standard PGN headers for a ChessLab game.

    Parameters
    ----------
    player_color:
        The color the human player controls.
    engine_elo:
        Engine strength in Elo, if configured.
    result:
        Game result (``"1-0"``, ``"0-1"``, ``"1/2-1/2"``, or ``"*"``).
    """
    is_player_white = player_color == PlayerColor.WHITE
    white = "Player" if is_player_white else "Stockfish"
    black = "Stockfish" if is_player_white else "Player"

    headers: dict[str, str] = {
        "Event": "ChessLab Game",
        "Site": "ChessLab",
        "Date": datetime.now().strftime("%Y.%m.%d"),
        "White": white,
        "Black": black,
        "Result": result,
    }

    if engine_elo is not None:
        elo_str = str(engine_elo)
        if is_player_white:
            headers["BlackElo"] = elo_str
        else:
            headers["WhiteElo"] = elo_str

    return headers


# ---------------------------------------------------------------------------
# GameStorage
# ---------------------------------------------------------------------------


class GameStorage:
    """Manages saving, loading, and listing PGN game files."""

    def __init__(self, games_dir: Path | None = None) -> None:
        self._games_dir = games_dir or _GAMES_DIR

    # -- PGN construction ----------------------------------------------------

    @staticmethod
    def build_pgn_game(
        board: chess.Board,
        headers: dict[str, str],
    ) -> chess.pgn.Game:
        """Build a ``chess.pgn.Game`` from a board and headers.

        Uses the board's move stack to reconstruct the full game.
        """
        game = chess.pgn.Game.from_board(board)

        for key, value in headers.items():
            game.headers[key] = value

        outcome = board.outcome()
        if outcome is not None:
            game.headers["Result"] = board.result()
        elif "Result" not in headers:
            game.headers["Result"] = "*"

        return game

    # -- Save / export -------------------------------------------------------

    def save_game(
        self,
        board: chess.Board,
        headers: dict[str, str],
    ) -> Path | None:
        """Auto-save a game to the games directory.

        Returns the path to the saved file, or ``None`` if the board
        has no moves.
        """
        if not board.move_stack:
            return None

        self._games_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        filename = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}.pgn"
        file_path = self._games_dir / filename

        game = self.build_pgn_game(board, headers)
        self._write_pgn(game, file_path)

        logger.info("Game saved: %s", file_path)
        return file_path

    @staticmethod
    def export_game(
        board: chess.Board,
        headers: dict[str, str],
        file_path: Path,
    ) -> None:
        """Export a game to a user-chosen path."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        game = GameStorage.build_pgn_game(board, headers)
        GameStorage._write_pgn(game, file_path)
        logger.info("Game exported: %s", file_path)

    # -- Load ----------------------------------------------------------------

    @staticmethod
    def load_game(file_path: Path) -> GameRecord:
        """Load a PGN file and return a :class:`GameRecord`.

        Raises
        ------
        PersistenceError
            If the file cannot be read or contains invalid PGN.
        """
        if not file_path.is_file():
            raise PersistenceError(f"File not found: {file_path}")

        try:
            with file_path.open() as fh:
                game = chess.pgn.read_game(fh)
        except Exception as exc:
            raise PersistenceError(
                f"Failed to read PGN: {file_path}: {exc}",
            ) from exc

        if game is None:
            raise PersistenceError(f"No game found in PGN file: {file_path}")

        board = game.board()
        san_moves: list[str] = []
        for move in game.mainline_moves():
            san_moves.append(board.san(move))
            board.push(move)

        headers = {k: v for k, v in game.headers.items()}
        result = headers.get("Result", "*")

        return GameRecord(
            headers=headers,
            moves=tuple(san_moves),
            board=board,
            result=result,
            file_path=file_path,
        )

    @staticmethod
    def load_game_from_text(pgn_text: str) -> GameRecord:
        """Load a game from PGN text and return a :class:`GameRecord`.

        This is the string-based counterpart of :meth:`load_game`.  It
        accepts raw PGN content (e.g. pasted from the clipboard) and
        parses only the first game / main line.

        Raises
        ------
        PersistenceError
            If the text is empty or contains invalid PGN.
        """
        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text))
        except Exception as exc:
            raise PersistenceError(
                f"Failed to parse PGN: {exc}",
            ) from exc

        if game is None:
            raise PersistenceError("No game found in PGN text")

        board = game.board()
        san_moves: list[str] = []
        for move in game.mainline_moves():
            san_moves.append(board.san(move))
            board.push(move)

        headers = {k: v for k, v in game.headers.items()}
        result = headers.get("Result", "*")

        return GameRecord(
            headers=headers,
            moves=tuple(san_moves),
            board=board,
            result=result,
            file_path=None,
        )

    # -- Listing -------------------------------------------------------------

    def list_recent_games(self, limit: int = 20) -> list[GameSummary]:
        """List recently saved games, newest first.

        Returns at most *limit* entries.
        """
        if not self._games_dir.is_dir():
            return []

        pgn_files = sorted(
            self._games_dir.glob("*.pgn"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        summaries: list[GameSummary] = []
        for pgn_path in pgn_files[:limit]:
            summary = self._read_summary(pgn_path)
            if summary is not None:
                summaries.append(summary)

        return summaries

    # -- Imported PGNs ------------------------------------------------------

    def is_imported_game(
        self,
        *,
        source: str,
        source_game_id: str | None = None,
        pgn_text: str | None = None,
    ) -> bool:
        """Return whether the imported game already exists locally."""
        pgn_sha256 = self._hash_pgn_text(pgn_text) if pgn_text is not None else None
        return self._find_imported_game(
            source=source,
            source_game_id=source_game_id,
            pgn_sha256=pgn_sha256,
        ) is not None

    def store_imported_game(
        self,
        pgn_text: str,
        *,
        source: str,
        source_game_id: str | None = None,
    ) -> ImportedGameSaveResult:
        """Persist an imported PGN with deduplication."""
        normalised_pgn = self._normalise_pgn_text(pgn_text)
        record = self.load_game_from_text(normalised_pgn)
        if not record.board.move_stack:
            raise PersistenceError("Imported PGN does not contain any moves")

        pgn_sha256 = self._hash_pgn_text(normalised_pgn)
        existing = self._find_imported_game(
            source=source,
            source_game_id=source_game_id,
            pgn_sha256=pgn_sha256,
        )
        if existing is not None:
            return ImportedGameSaveResult(file_path=existing.file_path, imported=False)

        self._games_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._build_import_file_path(
            source=source,
            source_game_id=source_game_id,
            pgn_sha256=pgn_sha256,
        )
        with file_path.open("w", encoding="utf-8") as fh:
            fh.write(normalised_pgn)
            fh.write("\n")

        entries = self._load_import_index()
        entries.append(_ImportedGameIndexEntry(
            file_path=file_path,
            source=source,
            source_game_id=source_game_id,
            pgn_sha256=pgn_sha256,
        ))
        self._write_import_index(entries)
        logger.info("Imported PGN saved: %s", file_path)
        return ImportedGameSaveResult(file_path=file_path, imported=True)

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _write_pgn(game: chess.pgn.Game, file_path: Path) -> None:
        """Write a PGN game to a file."""
        with file_path.open("w") as fh:
            exporter = chess.pgn.FileExporter(fh)
            game.accept(exporter)

    @staticmethod
    def _read_summary(pgn_path: Path) -> GameSummary | None:
        """Read only the headers + move count from a PGN file."""
        try:
            with pgn_path.open() as fh:
                game = chess.pgn.read_game(fh)
            if game is None:
                return None

            move_count = sum(1 for _ in game.mainline_moves())
            headers = game.headers

            return GameSummary(
                file_path=pgn_path,
                white=headers.get("White", "?"),
                black=headers.get("Black", "?"),
                result=headers.get("Result", "*"),
                date=headers.get("Date", "?"),
                move_count=move_count,
            )
        except Exception:
            logger.warning("Failed to read PGN summary: %s", pgn_path)
            return None

    @staticmethod
    def _normalise_pgn_text(pgn_text: str) -> str:
        """Normalise line endings while preserving the original PGN content."""
        return pgn_text.replace("\r\n", "\n").replace("\r", "\n").strip()

    @classmethod
    def _hash_pgn_text(cls, pgn_text: str) -> str:
        """Hash a PGN after normalising line endings."""
        normalised = cls._normalise_pgn_text(pgn_text)
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    def _build_import_file_path(
        self,
        *,
        source: str,
        source_game_id: str | None,
        pgn_sha256: str,
    ) -> Path:
        """Build a stable filename for an imported PGN."""
        now = datetime.now()
        stamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
        source_slug = re.sub(r"[^A-Za-z0-9]+", "-", source).strip("-").lower()
        game_slug_source = source_game_id or pgn_sha256[:12]
        game_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", game_slug_source).strip("-")
        game_slug = game_slug[:24] or "game"
        return self._games_dir / f"{stamp}_{source_slug}_{game_slug}.pgn"

    def _find_imported_game(
        self,
        *,
        source: str,
        source_game_id: str | None,
        pgn_sha256: str | None,
    ) -> _ImportedGameIndexEntry | None:
        """Find an existing imported PGN by source id or PGN hash."""
        entries = self._load_import_index()
        for entry in entries:
            if source_game_id and entry.source == source and entry.source_game_id == source_game_id:
                return entry
            if pgn_sha256 and entry.pgn_sha256 == pgn_sha256:
                return entry

        if pgn_sha256 is None or not self._games_dir.is_dir():
            return None

        for pgn_path in self._games_dir.glob("*.pgn"):
            try:
                existing_hash = self._hash_pgn_text(pgn_path.read_text(encoding="utf-8"))
            except OSError:
                continue
            if existing_hash == pgn_sha256:
                return _ImportedGameIndexEntry(
                    file_path=pgn_path,
                    source=source,
                    source_game_id=source_game_id,
                    pgn_sha256=pgn_sha256,
                )
        return None

    def _load_import_index(self) -> list[_ImportedGameIndexEntry]:
        """Read the imported-game index, pruning entries for missing files."""
        index_path = self._games_dir / _IMPORT_INDEX_FILE
        if not index_path.is_file():
            return []

        try:
            with index_path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            raise PersistenceError(
                f"Failed to read imported game index: {index_path}: {exc}",
            ) from exc

        if not isinstance(payload, dict):
            raise PersistenceError(f"Invalid imported game index format: {index_path}")

        raw_entries = payload.get("entries", [])
        if not isinstance(raw_entries, list):
            raise PersistenceError(f"Invalid imported game index payload: {index_path}")

        entries = [
            _ImportedGameIndexEntry.from_dict(item)
            for item in raw_entries
            if isinstance(item, dict)
        ]
        existing_entries = [entry for entry in entries if entry.file_path.is_file()]
        if len(existing_entries) != len(entries):
            self._write_import_index(existing_entries)
        return existing_entries

    def _write_import_index(self, entries: list[_ImportedGameIndexEntry]) -> None:
        """Persist the imported-game index."""
        self._games_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._games_dir / _IMPORT_INDEX_FILE
        payload = {
            "schema_version": 1,
            "entries": [entry.to_dict() for entry in entries],
        }
        with index_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
