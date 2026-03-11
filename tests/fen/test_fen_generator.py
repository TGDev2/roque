"""Tests for the FEN generator

Covers 20 reference positions (starting, openings, middlegame, endgame),
invalid position rejection, and python-chess cross-validation.
"""

from __future__ import annotations

import chess
import pytest

from chesslab.fen.fen_generator import (
    FenConfig,
    FenGenerator,
    FenResult,
    InvalidPositionError,
)
from chesslab.recognition.piece_recognizer import PieceType, SquareClassification

_FILES = "abcdefgh"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEN_CHAR_TO_PIECE: dict[str, PieceType] = {
    "K": PieceType.WHITE_KING,
    "Q": PieceType.WHITE_QUEEN,
    "R": PieceType.WHITE_ROOK,
    "B": PieceType.WHITE_BISHOP,
    "N": PieceType.WHITE_KNIGHT,
    "P": PieceType.WHITE_PAWN,
    "k": PieceType.BLACK_KING,
    "q": PieceType.BLACK_QUEEN,
    "r": PieceType.BLACK_ROOK,
    "b": PieceType.BLACK_BISHOP,
    "n": PieceType.BLACK_KNIGHT,
    "p": PieceType.BLACK_PAWN,
}


def _make_classifications(
    placement: str,
) -> dict[str, SquareClassification]:
    """Build 64 square classifications from a FEN piece-placement string."""
    classifications: dict[str, SquareClassification] = {}
    ranks = placement.split("/")
    assert len(ranks) == 8, f"Expected 8 ranks, got {len(ranks)}"

    for rank_idx, rank_str in enumerate(ranks):
        rank_number = 8 - rank_idx  # rank 8 first
        file_idx = 0
        for ch in rank_str:
            if ch.isdigit():
                for _ in range(int(ch)):
                    square_name = f"{_FILES[file_idx]}{rank_number}"
                    classifications[square_name] = SquareClassification(
                        square_name=square_name,
                        piece=PieceType.EMPTY,
                        confidence=1.0,
                        low_confidence=False,
                    )
                    file_idx += 1
            else:
                square_name = f"{_FILES[file_idx]}{rank_number}"
                classifications[square_name] = SquareClassification(
                    square_name=square_name,
                    piece=_FEN_CHAR_TO_PIECE[ch],
                    confidence=1.0,
                    low_confidence=False,
                )
                file_idx += 1
        assert file_idx == 8, f"Rank {rank_number} has {file_idx} files"

    assert len(classifications) == 64
    return classifications


# ---------------------------------------------------------------------------
# 20 reference positions
# ---------------------------------------------------------------------------

_REFERENCE_POSITIONS: list[tuple[str, str]] = [
    # --- Starting position ---
    (
        "starting",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    ),
    # --- Openings ---
    (
        "sicilian_defense",  # 1.e4 c5
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR",
    ),
    (
        "italian_game",  # 1.e4 e5 2.Nf3 Nc6 3.Bc4
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
    ),
    (
        "queens_gambit",  # 1.d4 d5 2.c4
        "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR",
    ),
    (
        "french_defense",  # 1.e4 e6
        "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR",
    ),
    (
        "ruy_lopez",  # 1.e4 e5 2.Nf3 Nc6 3.Bb5
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R",
    ),
    (
        "caro_kann",  # 1.e4 c6
        "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR",
    ),
    # --- Middlegame ---
    (
        "middlegame_1",  # typical position with castled kings
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1",
    ),
    (
        "middlegame_2",  # open centre
        "r2qkb1r/ppp2ppp/2n1bn2/3pp3/4P3/1PN2N2/PBPP1PPP/R2QKB1R",
    ),
    (
        "middlegame_3",  # imbalanced material
        "r1b2rk1/pp3ppp/2n1pn2/q1pp4/3P4/2NBPN2/PP3PPP/R1BQ1RK1",
    ),
    (
        "middlegame_4",  # queens exchanged
        "r1b2rk1/ppp2ppp/2n2n2/3pp3/1bPP4/2N1PN2/PP3PPP/R1B1KB1R",
    ),
    (
        "middlegame_5",  # one side with bishop pair
        "r4rk1/pp2bppp/2n1pn2/3p4/3P1B2/2PB1N2/PP3PPP/R3R1K1",
    ),
    # --- Endgame ---
    (
        "endgame_kr_vs_k",  # king and rook vs king
        "8/8/8/4k3/8/8/8/4K2R",
    ),
    (
        "endgame_kp_vs_k",  # king and pawn vs king
        "8/4k3/8/8/8/8/4PK2/8",
    ),
    (
        "endgame_kbb_vs_k",  # king and 2 bishops vs king
        "8/8/8/4k3/8/4BB2/8/4K3",
    ),
    (
        "endgame_rook_ending",  # typical rook ending
        "8/5pk1/5p1p/8/3R4/6PP/5PK1/1r6",
    ),
    (
        "endgame_pawn_race",  # mutual pawn pushes
        "8/1p4k1/8/8/8/8/1P4K1/8",
    ),
    (
        "kings_only",  # bare kings
        "8/8/8/4k3/8/8/4K3/8",
    ),
    # --- Edge cases ---
    (
        "many_queens",  # promotions: extra queens
        "1q2k3/8/8/8/8/8/8/1Q2K3",
    ),
    (
        "symmetric_endgame",  # perfectly symmetric
        "r3k2r/8/8/8/8/8/8/R3K2R",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def generator() -> FenGenerator:
    return FenGenerator()


# ---------------------------------------------------------------------------
# AC: Nominal FEN generation — starting position
# ---------------------------------------------------------------------------


class TestNominalGeneration:
    """AC: starting position produces the correct FEN."""

    @pytest.fixture
    def result(self, generator: FenGenerator) -> FenResult:
        placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        classifications = _make_classifications(placement)
        return generator.generate(classifications)

    def test_fen_matches_starting_position(self, result: FenResult) -> None:
        expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert result.fen == expected

    def test_validated_by_python_chess(self, result: FenResult) -> None:
        board = chess.Board(result.fen)
        assert board.status() == chess.STATUS_VALID

    def test_piece_placement_field(self, result: FenResult) -> None:
        assert result.piece_placement == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

    def test_elapsed_ms_non_negative(self, result: FenResult) -> None:
        assert result.elapsed_ms >= 0


# ---------------------------------------------------------------------------
# AC: 20 reference positions validated by python-chess
# ---------------------------------------------------------------------------


class TestReferencePositions:
    """All 20 reference positions produce correct FEN validated by python-chess."""

    @pytest.mark.parametrize(
        "name, placement",
        _REFERENCE_POSITIONS,
        ids=[name for name, _ in _REFERENCE_POSITIONS],
    )
    def test_fen_generation(
        self, generator: FenGenerator, name: str, placement: str
    ) -> None:
        classifications = _make_classifications(placement)
        result = generator.generate(classifications)

        assert result.piece_placement == placement
        expected_fen = f"{placement} w KQkq - 0 1"
        assert result.fen == expected_fen

    @pytest.mark.parametrize(
        "name, placement",
        _REFERENCE_POSITIONS,
        ids=[name for name, _ in _REFERENCE_POSITIONS],
    )
    def test_python_chess_validation(
        self, generator: FenGenerator, name: str, placement: str
    ) -> None:
        classifications = _make_classifications(placement)
        result = generator.generate(classifications)

        board = chess.Board(result.fen)
        # Board parses without error; check it roundtrips
        assert board.board_fen() == placement


# ---------------------------------------------------------------------------
# AC: Invalid positions are rejected
# ---------------------------------------------------------------------------


class TestInvalidPositions:
    """AC: absurd positions are rejected with a clear error message."""

    def test_three_kings_rejected(self, generator: FenGenerator) -> None:
        # 3 white kings
        placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/KNBQKBNK"
        classifications = _make_classifications(placement)

        with pytest.raises(InvalidPositionError, match="white king"):
            generator.generate(classifications)

    def test_no_white_king_rejected(self, generator: FenGenerator) -> None:
        placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR"
        classifications = _make_classifications(placement)

        with pytest.raises(InvalidPositionError, match="white king"):
            generator.generate(classifications)

    def test_no_black_king_rejected(self, generator: FenGenerator) -> None:
        placement = "rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        classifications = _make_classifications(placement)

        with pytest.raises(InvalidPositionError, match="black king"):
            generator.generate(classifications)

    def test_too_many_white_pieces_rejected(self, generator: FenGenerator) -> None:
        # 17 white pieces: 8 pawns + 8 pieces + extra queen
        placement = "rnbqkbnr/pppppppp/8/8/8/Q7/PPPPPPPP/RNBQKBNR"
        classifications = _make_classifications(placement)

        with pytest.raises(InvalidPositionError, match="[Ww]hite.*16"):
            generator.generate(classifications)

    def test_too_many_black_pieces_rejected(self, generator: FenGenerator) -> None:
        placement = "rnbqkbnr/pppppppp/q7/8/8/8/PPPPPPPP/RNBQKBNR"
        classifications = _make_classifications(placement)

        with pytest.raises(InvalidPositionError, match="[Bb]lack.*16"):
            generator.generate(classifications)

    def test_pawn_on_rank_1_rejected(self, generator: FenGenerator) -> None:
        # Black pawn on a1, removed a black pawn from a7 to keep count valid
        placement = "rnbqkbnr/1ppppppp/8/8/8/8/1PPPPPPP/pNBQKBNR"
        classifications = _make_classifications(placement)

        with pytest.raises(InvalidPositionError, match="[Pp]awn.*a1"):
            generator.generate(classifications)

    def test_pawn_on_rank_8_rejected(self, generator: FenGenerator) -> None:
        # White pawn on a8, removed a white pawn from a2 to keep count valid
        placement = "Pnbqkbnr/1ppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR"
        classifications = _make_classifications(placement)

        with pytest.raises(InvalidPositionError, match="[Pp]awn.*a8"):
            generator.generate(classifications)

    def test_wrong_number_of_squares_rejected(self, generator: FenGenerator) -> None:
        placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        classifications = _make_classifications(placement)
        # Remove one square
        del classifications["a1"]

        with pytest.raises(ValueError, match="64"):
            generator.generate(classifications)


# ---------------------------------------------------------------------------
# FenConfig validation
# ---------------------------------------------------------------------------


class TestFenConfig:
    """FenConfig validates its fields on construction."""

    def test_default_config(self) -> None:
        config = FenConfig()
        assert config.active_color == "w"
        assert config.castling == "KQkq"
        assert config.en_passant == "-"
        assert config.halfmove_clock == 0
        assert config.fullmove_number == 1

    def test_custom_config_applied(self, generator: FenGenerator) -> None:
        placement = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
        classifications = _make_classifications(placement)
        config = FenConfig(active_color="b", castling="KQkq", en_passant="e3")
        result = generator.generate(classifications, config=config)

        assert "b KQkq e3" in result.fen

    def test_invalid_active_color(self) -> None:
        with pytest.raises(ValueError, match="active_color"):
            FenConfig(active_color="x")

    def test_invalid_castling(self) -> None:
        with pytest.raises(ValueError, match="castling"):
            FenConfig(castling="ZZ")

    def test_invalid_en_passant(self) -> None:
        with pytest.raises(ValueError, match="en_passant"):
            FenConfig(en_passant="z9")

    def test_negative_halfmove(self) -> None:
        with pytest.raises(ValueError, match="halfmove_clock"):
            FenConfig(halfmove_clock=-1)

    def test_zero_fullmove(self) -> None:
        with pytest.raises(ValueError, match="fullmove_number"):
            FenConfig(fullmove_number=0)

    def test_no_castling_rights(self, generator: FenGenerator) -> None:
        placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        classifications = _make_classifications(placement)
        config = FenConfig(castling="-")
        result = generator.generate(classifications, config=config)

        assert " - - 0 1" in result.fen

    def test_partial_castling(self) -> None:
        config = FenConfig(castling="Kq")
        assert config.castling == "Kq"
