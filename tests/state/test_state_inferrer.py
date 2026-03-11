"""Tests for the state inferrer

Covers castling rights inference from piece positions, active color
detection with user fallback signalling, config validation, and
end-to-end integration with FenGenerator.
"""

from __future__ import annotations

import pytest

from chesslab.fen.fen_generator import FenConfig, FenGenerator
from chesslab.recognition.piece_recognizer import PieceType, SquareClassification
from chesslab.state.state_inferrer import (
    CastlingRights,
    StateInferenceConfig,
    StateInferenceResult,
    StateInferrer,
)

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
        rank_number = 8 - rank_idx
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def inferrer() -> StateInferrer:
    return StateInferrer()


# ---------------------------------------------------------------------------
# AC: Castling rights inferred from positions
# ---------------------------------------------------------------------------


class TestCastlingInference:
    """AC: castling field reflects king/rook positions."""

    def test_starting_position_full_rights(self, inferrer: StateInferrer) -> None:
        """Both camps have all pieces on starting squares → KQkq."""
        classifications = _make_classifications(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )
        result = inferrer.infer(classifications)

        assert result.fen_config.castling == "KQkq"
        assert result.castling_rights == CastlingRights(
            white_kingside=True,
            white_queenside=True,
            black_kingside=True,
            black_queenside=True,
        )

    def test_white_king_moved(self, inferrer: StateInferrer) -> None:
        """White king off e1 → no white castling rights."""
        classifications = _make_classifications(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBK1BNR"
        )
        result = inferrer.infer(classifications)

        assert result.fen_config.castling == "kq"
        assert result.castling_rights.white_kingside is False
        assert result.castling_rights.white_queenside is False

    def test_black_king_moved(self, inferrer: StateInferrer) -> None:
        """Black king off e8 → no black castling rights."""
        classifications = _make_classifications(
            "rnbk1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )
        result = inferrer.infer(classifications)

        assert result.fen_config.castling == "KQ"
        assert result.castling_rights.black_kingside is False
        assert result.castling_rights.black_queenside is False

    def test_both_kings_moved(self, inferrer: StateInferrer) -> None:
        """Both kings off starting squares → no castling rights."""
        classifications = _make_classifications(
            "rnbk1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBK1BNR"
        )
        result = inferrer.infer(classifications)

        assert result.fen_config.castling == "-"

    def test_white_a1_rook_missing(self, inferrer: StateInferrer) -> None:
        """No white rook on a1 → remove Q right."""
        classifications = _make_classifications(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR"
        )
        result = inferrer.infer(classifications)

        assert result.fen_config.castling == "Kkq"
        assert result.castling_rights.white_queenside is False
        assert result.castling_rights.white_kingside is True

    def test_white_h1_rook_missing(self, inferrer: StateInferrer) -> None:
        """No white rook on h1 → remove K right."""
        classifications = _make_classifications(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1"
        )
        result = inferrer.infer(classifications)

        assert result.fen_config.castling == "Qkq"
        assert result.castling_rights.white_kingside is False
        assert result.castling_rights.white_queenside is True

    def test_black_a8_rook_missing(self, inferrer: StateInferrer) -> None:
        """No black rook on a8 → remove q right."""
        classifications = _make_classifications(
            "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )
        result = inferrer.infer(classifications)

        assert result.fen_config.castling == "KQk"
        assert result.castling_rights.black_queenside is False
        assert result.castling_rights.black_kingside is True

    def test_black_h8_rook_missing(self, inferrer: StateInferrer) -> None:
        """No black rook on h8 → remove k right."""
        classifications = _make_classifications(
            "rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )
        result = inferrer.infer(classifications)

        assert result.fen_config.castling == "KQq"
        assert result.castling_rights.black_kingside is False
        assert result.castling_rights.black_queenside is True

    def test_all_rooks_missing_kings_home(self, inferrer: StateInferrer) -> None:
        """Kings on starting squares but no rooks → no rights."""
        classifications = _make_classifications(
            "4k3/8/8/8/8/8/8/4K3"
        )
        result = inferrer.infer(classifications)

        assert result.fen_config.castling == "-"
        assert result.castling_rights == CastlingRights(
            white_kingside=False,
            white_queenside=False,
            black_kingside=False,
            black_queenside=False,
        )

    def test_post_castling_position(self, inferrer: StateInferrer) -> None:
        """White has castled kingside (Kg1, Rf1) → no white rights."""
        classifications = _make_classifications(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1"
        )
        result = inferrer.infer(classifications)

        assert result.castling_rights.white_kingside is False
        assert result.castling_rights.white_queenside is False
        assert result.castling_rights.black_kingside is True
        assert result.castling_rights.black_queenside is True
        assert result.fen_config.castling == "kq"


# ---------------------------------------------------------------------------
# AC: Active color detection + user fallback
# ---------------------------------------------------------------------------


class TestActiveColor:
    """AC: active color detection with uncertain flag for user fallback."""

    def test_no_override_defaults_to_white_uncertain(self) -> None:
        """No override → default white, flagged as uncertain."""
        inferrer = StateInferrer()
        classifications = _make_classifications(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )
        result = inferrer.infer(classifications)

        assert result.fen_config.active_color == "w"
        assert result.active_color_uncertain is True

    def test_override_white(self) -> None:
        """Explicit white override → not uncertain."""
        config = StateInferenceConfig(active_color_override="w")
        inferrer = StateInferrer(config)
        classifications = _make_classifications(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )
        result = inferrer.infer(classifications)

        assert result.fen_config.active_color == "w"
        assert result.active_color_uncertain is False

    def test_override_black(self) -> None:
        """Explicit black override → active_color='b', not uncertain."""
        config = StateInferenceConfig(active_color_override="b")
        inferrer = StateInferrer(config)
        classifications = _make_classifications(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
        )
        result = inferrer.infer(classifications)

        assert result.fen_config.active_color == "b"
        assert result.active_color_uncertain is False

    def test_invalid_override_rejected(self) -> None:
        """Invalid active_color_override raises ValueError."""
        with pytest.raises(ValueError, match="active_color_override"):
            StateInferenceConfig(active_color_override="x")


# ---------------------------------------------------------------------------
# CastlingRights.fen property
# ---------------------------------------------------------------------------


class TestCastlingRightsFen:
    """CastlingRights.fen returns correct FEN castling field."""

    def test_all_rights(self) -> None:
        rights = CastlingRights(True, True, True, True)
        assert rights.fen == "KQkq"

    def test_no_rights(self) -> None:
        rights = CastlingRights(False, False, False, False)
        assert rights.fen == "-"

    def test_white_kingside_only(self) -> None:
        rights = CastlingRights(True, False, False, False)
        assert rights.fen == "K"

    def test_black_queenside_only(self) -> None:
        rights = CastlingRights(False, False, False, True)
        assert rights.fen == "q"

    def test_mixed_kq(self) -> None:
        rights = CastlingRights(True, False, False, True)
        assert rights.fen == "Kq"

    def test_white_only(self) -> None:
        rights = CastlingRights(True, True, False, False)
        assert rights.fen == "KQ"

    def test_black_only(self) -> None:
        rights = CastlingRights(False, False, True, True)
        assert rights.fen == "kq"


# ---------------------------------------------------------------------------
# StateInferenceConfig validation
# ---------------------------------------------------------------------------


class TestStateInferenceConfig:
    """Config validates its fields on construction."""

    def test_default_config(self) -> None:
        config = StateInferenceConfig()
        assert config.active_color_override is None
        assert config.en_passant == "-"
        assert config.halfmove_clock == 0
        assert config.fullmove_number == 1

    def test_invalid_en_passant(self) -> None:
        with pytest.raises(ValueError, match="en_passant"):
            StateInferenceConfig(en_passant="z9")

    def test_negative_halfmove(self) -> None:
        with pytest.raises(ValueError, match="halfmove_clock"):
            StateInferenceConfig(halfmove_clock=-1)

    def test_zero_fullmove(self) -> None:
        with pytest.raises(ValueError, match="fullmove_number"):
            StateInferenceConfig(fullmove_number=0)

    def test_valid_en_passant_accepted(self) -> None:
        config = StateInferenceConfig(en_passant="e3")
        assert config.en_passant == "e3"


# ---------------------------------------------------------------------------
# StateInferenceResult contract
# ---------------------------------------------------------------------------


class TestStateInferenceResult:
    """Result dataclass contract checks."""

    def test_elapsed_ms_non_negative(self, inferrer: StateInferrer) -> None:
        classifications = _make_classifications(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )
        result = inferrer.infer(classifications)
        assert result.elapsed_ms >= 0

    def test_wrong_number_of_squares_rejected(self, inferrer: StateInferrer) -> None:
        classifications = _make_classifications(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )
        del classifications["a1"]

        with pytest.raises(ValueError, match="64"):
            inferrer.infer(classifications)


# ---------------------------------------------------------------------------
# End-to-end: StateInferrer → FenGenerator pipeline
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration: inferred state feeds FenGenerator correctly."""

    def test_starting_position_no_override(self) -> None:
        """Default inference + generation produces valid starting FEN."""
        classifications = _make_classifications(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )
        state = StateInferrer().infer(classifications)
        fen_result = FenGenerator().generate(classifications, state.fen_config)

        assert fen_result.fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_starting_position_black_override(self) -> None:
        """Override to black produces correct FEN."""
        config = StateInferenceConfig(active_color_override="b")
        classifications = _make_classifications(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
        )
        state = StateInferrer(config).infer(classifications)
        fen_result = FenGenerator().generate(classifications, state.fen_config)

        assert "b KQkq" in fen_result.fen

    def test_post_castling_reduced_rights(self) -> None:
        """Position after white castled → reduced castling in FEN."""
        classifications = _make_classifications(
            "r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQK2R"
        )
        state = StateInferrer().infer(classifications)
        fen_result = FenGenerator().generate(classifications, state.fen_config)

        # White king on e1, rook NOT on h1 → no K right; rook on a1 → Q right
        assert "Qkq" in fen_result.fen

    def test_endgame_no_castling(self) -> None:
        """Endgame with moved kings → no castling rights."""
        classifications = _make_classifications(
            "8/8/8/4k3/8/8/8/4K2R"
        )
        state = StateInferrer().infer(classifications)
        fen_result = FenGenerator().generate(classifications, state.fen_config)

        # White king on e1 + rook on h1 → K right; black king not on e8 → no black
        assert "K -" in fen_result.fen
