"""Tests for GameStorage.load_game_from_text.

Validates PGN parsing from raw text (paste/clipboard) with various
formats including chess.com and lichess exports.
"""

from __future__ import annotations

import pytest

from chesslab.persistence.game_storage import (
    GameRecord,
    GameStorage,
    PersistenceError,
)


# ---------------------------------------------------------------------------
# Sample PGN strings
# ---------------------------------------------------------------------------

SIMPLE_PGN = """\
[Event "Casual Game"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O 1-0
"""

CHESS_COM_PGN = """\
[Event "Live Chess"]
[Site "Chess.com"]
[Date "2026.03.15"]
[Round "-"]
[White "player1"]
[Black "player2"]
[Result "0-1"]
[WhiteElo "1523"]
[BlackElo "1487"]
[TimeControl "600"]
[Termination "player2 won by resignation"]

1. e4 {[%clk 0:09:58.4]} 1... c5 {[%clk 0:09:56.1]} 2. Nf3 {[%clk 0:09:55.2]} 2... d6 {[%clk 0:09:53.8]} 3. d4 {[%clk 0:09:50.1]} 3... cxd4 {[%clk 0:09:49.2]} 4. Nxd4 {[%clk 0:09:48.7]} 4... Nf6 {[%clk 0:09:47.3]} 0-1
"""

LICHESS_PGN = """\
[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[Date "2026.03.10"]
[White "lichessUser1"]
[Black "lichessUser2"]
[Result "1/2-1/2"]
[WhiteElo "1800"]
[BlackElo "1756"]
[ECO "B90"]
[Opening "Sicilian Defense: Najdorf Variation"]
[TimeControl "300+3"]

1. e4 { [%eval 0.17] } 1... c5 { [%eval 0.19] } 2. Nf3 { [%eval 0.11] } 2... d6 { [%eval 0.42] } 3. d4 { [%eval 0.36] } 3... cxd4 { [%eval 0.43] } 4. Nxd4 { [%eval 0.28] } 4... Nf6 { [%eval 0.43] } 5. Nc3 { [%eval 0.17] } 5... a6 { [%eval 0.25] } 1/2-1/2
"""

MOVES_ONLY_PGN = "1. e4 e5 2. Nf3 Nc6 *"

SCHOLAR_MATE_PGN = """\
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""

PGN_WITH_VARIATIONS = """\
[Event "Annotated"]
[Result "*"]

1. e4 e5 (1... c5 2. Nf3 d6) 2. Nf3 Nc6 3. Bb5 *
"""

HEADERS_ONLY_PGN = """\
[Event "Empty"]
[White "A"]
[Black "B"]
[Result "*"]

*
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadGameFromText:
    """Tests for GameStorage.load_game_from_text."""

    def test_simple_pgn(self) -> None:
        record = GameStorage.load_game_from_text(SIMPLE_PGN)

        assert isinstance(record, GameRecord)
        assert record.moves == ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O")
        assert record.result == "1-0"
        assert record.file_path is None

    def test_headers_preserved(self) -> None:
        record = GameStorage.load_game_from_text(SIMPLE_PGN)

        assert record.headers["White"] == "Alice"
        assert record.headers["Black"] == "Bob"
        assert record.headers["Event"] == "Casual Game"
        assert record.headers["Result"] == "1-0"

    def test_board_has_move_stack(self) -> None:
        record = GameStorage.load_game_from_text(SIMPLE_PGN)
        assert len(record.board.move_stack) == 9

    def test_chess_com_pgn_with_clk_comments(self) -> None:
        record = GameStorage.load_game_from_text(CHESS_COM_PGN)

        assert record.headers["Site"] == "Chess.com"
        assert record.headers["WhiteElo"] == "1523"
        assert record.moves == ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6")
        assert record.result == "0-1"

    def test_lichess_pgn_with_eval_comments(self) -> None:
        record = GameStorage.load_game_from_text(LICHESS_PGN)

        assert "lichess.org" in record.headers["Site"]
        assert record.headers["ECO"] == "B90"
        assert record.moves == (
            "e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6",
        )
        assert record.result == "1/2-1/2"

    def test_moves_only_no_headers(self) -> None:
        record = GameStorage.load_game_from_text(MOVES_ONLY_PGN)

        assert record.moves == ("e4", "e5", "Nf3", "Nc6")
        assert len(record.board.move_stack) == 4

    def test_mainline_only_ignores_variations(self) -> None:
        record = GameStorage.load_game_from_text(PGN_WITH_VARIATIONS)

        assert record.moves == ("e4", "e5", "Nf3", "Nc6", "Bb5")

    def test_scholar_mate_result(self) -> None:
        record = GameStorage.load_game_from_text(SCHOLAR_MATE_PGN)

        assert record.result == "1-0"
        assert record.board.is_checkmate()
        assert record.moves[-1] == "Qxf7#"

    def test_headers_only_no_moves(self) -> None:
        record = GameStorage.load_game_from_text(HEADERS_ONLY_PGN)

        assert record.moves == ()
        assert len(record.board.move_stack) == 0

    def test_empty_string_raises(self) -> None:
        with pytest.raises(PersistenceError, match="No game found"):
            GameStorage.load_game_from_text("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(PersistenceError, match="No game found"):
            GameStorage.load_game_from_text("   \n\n  ")

    def test_garbage_text_returns_no_moves(self) -> None:
        """chess.pgn.read_game tolerates garbage — result has no moves."""
        record = GameStorage.load_game_from_text("this is not a PGN at all xyz")
        assert record.moves == ()
        assert len(record.board.move_stack) == 0

    def test_file_path_is_none(self) -> None:
        record = GameStorage.load_game_from_text(SIMPLE_PGN)
        assert record.file_path is None

    def test_round_trip_with_save(self, tmp_path: object) -> None:
        """Verify a text-loaded game can be re-saved."""
        record = GameStorage.load_game_from_text(SIMPLE_PGN)
        storage = GameStorage(games_dir=tmp_path)  # type: ignore[arg-type]
        path = storage.save_game(record.board, record.headers)

        assert path is not None

        reloaded = GameStorage.load_game(path)
        assert reloaded.moves == record.moves
