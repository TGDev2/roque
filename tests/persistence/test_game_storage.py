"""Tests for chesslab.persistence.game_storage.

Acceptance criteria covered
---------------------------
AC-1  Auto-save to ``~/.chesslab/games/`` (via injected tmp_path).
AC-2  PGN with standard headers (Event, Date, White, Black, Result).
AC-3  Engine Elo in WhiteElo / BlackElo header.
AC-6  Recent games listing.
AC-7  Valid PGN importable by external tools.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import chess
import chess.pgn

import pytest

from chesslab.coach.game_coach import PlayerColor
from chesslab.persistence.game_storage import (
    GameRecord,
    GameStorage,
    GameSummary,
    ImportedGameSaveResult,
    PersistenceError,
    build_headers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_board(*san_moves: str) -> chess.Board:
    """Create a board with the given SAN moves played."""
    board = chess.Board()
    for san in san_moves:
        board.push_san(san)
    return board


def _make_scholar_mate_board() -> chess.Board:
    """Board after scholar's mate: 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#."""
    return _make_board("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#")


IMPORTED_PGN = """\
[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[Date "2026.03.24"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
"""


# ---------------------------------------------------------------------------
# build_headers
# ---------------------------------------------------------------------------


class TestBuildHeaders:
    """Unit tests for the build_headers helper."""

    def test_player_white(self) -> None:
        h = build_headers(player_color=PlayerColor.WHITE, engine_elo=1500)
        assert h["White"] == "Player"
        assert h["Black"] == "Stockfish"
        assert h["BlackElo"] == "1500"
        assert "WhiteElo" not in h

    def test_player_black(self) -> None:
        h = build_headers(player_color=PlayerColor.BLACK, engine_elo=2000)
        assert h["White"] == "Stockfish"
        assert h["Black"] == "Player"
        assert h["WhiteElo"] == "2000"
        assert "BlackElo" not in h

    def test_no_elo(self) -> None:
        h = build_headers(player_color=PlayerColor.WHITE)
        assert "WhiteElo" not in h
        assert "BlackElo" not in h

    def test_standard_headers_present(self) -> None:
        h = build_headers(player_color=PlayerColor.WHITE)
        assert h["Event"] == "ChessLab Game"
        assert h["Site"] == "ChessLab"
        assert re.match(r"\d{4}\.\d{2}\.\d{2}", h["Date"])
        assert h["Result"] == "*"

    def test_custom_result(self) -> None:
        h = build_headers(
            player_color=PlayerColor.WHITE, result="1-0",
        )
        assert h["Result"] == "1-0"


# ---------------------------------------------------------------------------
# build_pgn_game
# ---------------------------------------------------------------------------


class TestBuildPgnGame:
    """Tests for GameStorage.build_pgn_game."""

    def test_headers_applied(self) -> None:
        board = _make_board("e4", "e5", "Nf3")
        headers = build_headers(player_color=PlayerColor.WHITE, engine_elo=1350)
        game = GameStorage.build_pgn_game(board, headers)

        assert game.headers["White"] == "Player"
        assert game.headers["Black"] == "Stockfish"
        assert game.headers["BlackElo"] == "1350"
        assert game.headers["Event"] == "ChessLab Game"

    def test_moves_match_board(self) -> None:
        board = _make_board("e4", "e5", "Nf3", "Nc6")
        headers = build_headers(player_color=PlayerColor.WHITE)
        game = GameStorage.build_pgn_game(board, headers)

        san_moves = [
            board_tmp.san(m)
            for board_tmp in [chess.Board()]
            for m in []
        ]
        # Replay from the game mainline.
        replay = game.board()
        mainline_san: list[str] = []
        for move in game.mainline_moves():
            mainline_san.append(replay.san(move))
            replay.push(move)

        assert mainline_san == ["e4", "e5", "Nf3", "Nc6"]

    def test_result_from_outcome(self) -> None:
        board = _make_scholar_mate_board()
        headers = build_headers(player_color=PlayerColor.WHITE)
        game = GameStorage.build_pgn_game(board, headers)
        assert game.headers["Result"] == "1-0"

    def test_result_in_progress(self) -> None:
        board = _make_board("e4", "e5")
        headers = build_headers(player_color=PlayerColor.WHITE)
        game = GameStorage.build_pgn_game(board, headers)
        assert game.headers["Result"] == "*"


# ---------------------------------------------------------------------------
# save_game
# ---------------------------------------------------------------------------


class TestSaveGame:
    """Tests for GameStorage.save_game (AC-1)."""

    def test_creates_file(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path)
        board = _make_board("e4", "e5")
        headers = build_headers(player_color=PlayerColor.WHITE)

        path = storage.save_game(board, headers)

        assert path is not None
        assert path.exists()
        assert path.suffix == ".pgn"
        assert path.parent == tmp_path

    def test_filename_format(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path)
        board = _make_board("d4")
        headers = build_headers(player_color=PlayerColor.BLACK)

        path = storage.save_game(board, headers)

        assert path is not None
        assert re.match(r"\d{8}_\d{6}_\d{3}\.pgn", path.name)

    def test_valid_pgn_content(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path)
        board = _make_board("e4", "e5", "Nf3")
        headers = build_headers(player_color=PlayerColor.WHITE, engine_elo=1500)

        path = storage.save_game(board, headers)
        assert path is not None

        with path.open() as fh:
            game = chess.pgn.read_game(fh)

        assert game is not None
        assert game.headers["White"] == "Player"
        assert game.headers["BlackElo"] == "1500"

    def test_empty_board_returns_none(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path)
        board = chess.Board()
        headers = build_headers(player_color=PlayerColor.WHITE)

        result = storage.save_game(board, headers)

        assert result is None
        assert list(tmp_path.glob("*.pgn")) == []

    def test_creates_directory(self, tmp_path: Path) -> None:
        games_dir = tmp_path / "nested" / "games"
        storage = GameStorage(games_dir=games_dir)
        board = _make_board("e4")
        headers = build_headers(player_color=PlayerColor.WHITE)

        path = storage.save_game(board, headers)

        assert path is not None
        assert games_dir.is_dir()


# ---------------------------------------------------------------------------
# export_game
# ---------------------------------------------------------------------------


class TestExportGame:
    """Tests for GameStorage.export_game (AC-4)."""

    def test_writes_to_path(self, tmp_path: Path) -> None:
        board = _make_board("e4", "e5", "Nf3", "Nc6")
        headers = build_headers(player_color=PlayerColor.WHITE, engine_elo=1800)
        export_path = tmp_path / "my_game.pgn"

        GameStorage.export_game(board, headers, export_path)

        assert export_path.exists()
        with export_path.open() as fh:
            game = chess.pgn.read_game(fh)
        assert game is not None

        replay = game.board()
        moves = []
        for m in game.mainline_moves():
            moves.append(replay.san(m))
            replay.push(m)
        assert moves == ["e4", "e5", "Nf3", "Nc6"]

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        board = _make_board("d4")
        headers = build_headers(player_color=PlayerColor.BLACK)
        export_path = tmp_path / "sub" / "dir" / "game.pgn"

        GameStorage.export_game(board, headers, export_path)

        assert export_path.exists()


# ---------------------------------------------------------------------------
# load_game
# ---------------------------------------------------------------------------


class TestLoadGame:
    """Tests for GameStorage.load_game (AC-5)."""

    def test_round_trip(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path)
        board = _make_board("e4", "e5", "Nf3", "Nc6", "Bb5")
        headers = build_headers(player_color=PlayerColor.WHITE, engine_elo=1500)

        path = storage.save_game(board, headers)
        assert path is not None

        record = GameStorage.load_game(path)

        assert isinstance(record, GameRecord)
        assert record.moves == ("e4", "e5", "Nf3", "Nc6", "Bb5")
        assert record.result == "*"
        assert record.file_path == path

    def test_headers_preserved(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path)
        board = _make_board("e4", "e5")
        headers = build_headers(player_color=PlayerColor.BLACK, engine_elo=2200)

        path = storage.save_game(board, headers)
        assert path is not None

        record = GameStorage.load_game(path)

        assert record.headers["White"] == "Stockfish"
        assert record.headers["Black"] == "Player"
        assert record.headers["WhiteElo"] == "2200"
        assert record.headers["Event"] == "ChessLab Game"

    def test_board_has_move_stack(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path)
        board = _make_board("e4", "e5", "Nf3")
        headers = build_headers(player_color=PlayerColor.WHITE)

        path = storage.save_game(board, headers)
        assert path is not None

        record = GameStorage.load_game(path)
        assert len(record.board.move_stack) == 3

    def test_empty_pgn_file_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.pgn"
        bad_file.write_text("")

        with pytest.raises(PersistenceError, match="No game found"):
            GameStorage.load_game(bad_file)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PersistenceError, match="File not found"):
            GameStorage.load_game(tmp_path / "nonexistent.pgn")

    def test_game_over_result(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path)
        board = _make_scholar_mate_board()
        headers = build_headers(player_color=PlayerColor.WHITE, result="1-0")

        path = storage.save_game(board, headers)
        assert path is not None

        record = GameStorage.load_game(path)
        assert record.result == "1-0"


# ---------------------------------------------------------------------------
# list_recent_games
# ---------------------------------------------------------------------------


class TestListRecentGames:
    """Tests for GameStorage.list_recent_games (AC-6)."""

    def test_sorted_by_mtime(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path)
        headers = build_headers(player_color=PlayerColor.WHITE)

        board1 = _make_board("e4")
        path1 = storage.save_game(board1, headers)
        assert path1 is not None
        # Ensure different mtime.
        time.sleep(0.05)

        board2 = _make_board("d4", "d5")
        path2 = storage.save_game(board2, headers)
        assert path2 is not None

        games = storage.list_recent_games()

        assert len(games) == 2
        assert games[0].file_path == path2  # newest first
        assert games[1].file_path == path1

    def test_limit(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path)
        headers = build_headers(player_color=PlayerColor.WHITE)

        for i in range(5):
            board = _make_board("e4")
            # Write directly to control filenames.
            game = GameStorage.build_pgn_game(board, headers)
            path = tmp_path / f"game_{i:02d}.pgn"
            GameStorage._write_pgn(game, path)

        games = storage.list_recent_games(limit=3)
        assert len(games) == 3

    def test_empty_directory(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path)
        assert storage.list_recent_games() == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path / "nope")
        assert storage.list_recent_games() == []

    def test_summary_fields(self, tmp_path: Path) -> None:
        storage = GameStorage(games_dir=tmp_path)
        board = _make_board("e4", "e5", "Nf3")
        headers = build_headers(player_color=PlayerColor.WHITE, engine_elo=1500)

        path = storage.save_game(board, headers)
        assert path is not None

        games = storage.list_recent_games()
        assert len(games) == 1

        s = games[0]
        assert isinstance(s, GameSummary)
        assert s.white == "Player"
        assert s.black == "Stockfish"
        assert s.result == "*"
        assert s.move_count == 3
        assert s.file_path == path


# ---------------------------------------------------------------------------
# imported PGN storage
# ---------------------------------------------------------------------------


class TestImportedGameStorage:
    """Tests for imported PGN persistence and deduplication."""

    def test_store_imported_game_writes_file_and_lists_recent(
        self,
        tmp_path: Path,
    ) -> None:
        storage = GameStorage(games_dir=tmp_path)

        result = storage.store_imported_game(
            IMPORTED_PGN,
            source="lichess",
            source_game_id="abcd1234",
        )

        assert isinstance(result, ImportedGameSaveResult)
        assert result.imported is True
        assert result.file_path.exists()

        recent = storage.list_recent_games()
        assert len(recent) == 1
        assert recent[0].white == "Alice"
        assert recent[0].black == "Bob"
        assert recent[0].file_path == result.file_path

    def test_store_imported_game_deduplicates_by_source_id(
        self,
        tmp_path: Path,
    ) -> None:
        storage = GameStorage(games_dir=tmp_path)

        first = storage.store_imported_game(
            IMPORTED_PGN,
            source="lichess",
            source_game_id="abcd1234",
        )
        second = storage.store_imported_game(
            IMPORTED_PGN.replace("Alice", "Alice Updated"),
            source="lichess",
            source_game_id="abcd1234",
        )

        assert first.imported is True
        assert second.imported is False
        assert second.file_path == first.file_path
        assert len(list(tmp_path.glob("*.pgn"))) == 1

    def test_store_imported_game_deduplicates_by_hash_without_source_id(
        self,
        tmp_path: Path,
    ) -> None:
        storage = GameStorage(games_dir=tmp_path)

        first = storage.store_imported_game(IMPORTED_PGN, source="lichess")
        second = storage.store_imported_game(IMPORTED_PGN, source="chess.com")

        assert first.imported is True
        assert second.imported is False
        assert second.file_path == first.file_path

    def test_store_imported_game_rejects_headers_only_pgn(
        self,
        tmp_path: Path,
    ) -> None:
        storage = GameStorage(games_dir=tmp_path)

        with pytest.raises(PersistenceError, match="does not contain any moves"):
            storage.store_imported_game(
                '[Event "Empty"]\n[Result "*"]\n\n*',
                source="lichess",
                source_game_id="empty",
            )


# ---------------------------------------------------------------------------
# PGN format validity (AC-7)
# ---------------------------------------------------------------------------


class TestPgnValidity:
    """Verify the generated PGN is standard-compliant."""

    def test_contains_standard_tags(self, tmp_path: Path) -> None:
        board = _make_board("e4", "e5", "Nf3", "Nc6")
        headers = build_headers(player_color=PlayerColor.WHITE, engine_elo=1350)
        export_path = tmp_path / "test.pgn"

        GameStorage.export_game(board, headers, export_path)
        text = export_path.read_text()

        assert '[Event "ChessLab Game"]' in text
        assert '[Site "ChessLab"]' in text
        assert '[White "Player"]' in text
        assert '[Black "Stockfish"]' in text
        assert '[BlackElo "1350"]' in text
        assert '[Result "*"]' in text

    def test_engine_elo_in_output(self, tmp_path: Path) -> None:
        board = _make_board("e4")
        headers = build_headers(player_color=PlayerColor.BLACK, engine_elo=2100)
        export_path = tmp_path / "elo.pgn"

        GameStorage.export_game(board, headers, export_path)
        text = export_path.read_text()

        assert '[WhiteElo "2100"]' in text

    def test_movetext_present(self, tmp_path: Path) -> None:
        board = _make_board("e4", "e5", "Nf3", "Nc6")
        headers = build_headers(player_color=PlayerColor.WHITE)
        export_path = tmp_path / "moves.pgn"

        GameStorage.export_game(board, headers, export_path)
        text = export_path.read_text()

        assert "1. e4 e5 2. Nf3 Nc6" in text

    def test_game_over_result_in_movetext(self, tmp_path: Path) -> None:
        board = _make_scholar_mate_board()
        headers = build_headers(player_color=PlayerColor.WHITE, result="1-0")
        export_path = tmp_path / "mate.pgn"

        GameStorage.export_game(board, headers, export_path)
        text = export_path.read_text()

        assert "1-0" in text
        assert '[Result "1-0"]' in text
