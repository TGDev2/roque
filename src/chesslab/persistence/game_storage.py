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

import io
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn

from chesslab.coach.game_coach import PlayerColor

logger = logging.getLogger(__name__)

_GAMES_DIR = Path.home() / ".chesslab" / "games"


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
