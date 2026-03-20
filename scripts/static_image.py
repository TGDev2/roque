"""Run the full board->FEN pipeline on a single image."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

# Allow running as `python scripts/static_image.py` without install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chesslab.detection.board_detector import BoardDetector
from chesslab.fen.fen_generator import FenConfig, FenGenerator, InvalidPositionError
from chesslab.recognition.recognition_orchestrator import (
    RecognitionFlowStatus,
    RecognitionOrchestrator,
    RecognitionOrchestratorConfig,
)
from chesslab.splitting.board_splitter import BoardSplitter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run screenshot -> detect -> split -> recognize -> FEN.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="scripts/data/test_board.png",
        help="Path to input screenshot (default: scripts/data/test_board.png).",
    )
    parser.add_argument(
        "--theme",
        default="chess.com-green",
        help="Manual theme ID (default: chess.com-green).",
    )
    parser.add_argument(
        "--black-perspective",
        action="store_true",
        help="Treat screenshot as black perspective.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Piece recognition confidence threshold (default: 0.60).",
    )
    parser.add_argument(
        "--allow-uncertain",
        action="store_true",
        help="Continue to FEN generation even if recognition is uncertain.",
    )
    parser.add_argument(
        "--expect-fen",
        default=None,
        help="Optional expected FEN to compare against.",
    )
    parser.add_argument(
        "--active-color",
        default="w",
        choices=("w", "b"),
        help="FEN active color (default: w).",
    )
    parser.add_argument(
        "--castling",
        default="KQkq",
        help="FEN castling field (default: KQkq).",
    )
    parser.add_argument(
        "--en-passant",
        default="-",
        help="FEN en-passant field (default: -).",
    )
    parser.add_argument(
        "--halfmove",
        type=int,
        default=0,
        help="FEN halfmove clock (default: 0).",
    )
    parser.add_argument(
        "--fullmove",
        type=int,
        default=1,
        help="FEN fullmove number (default: 1).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"[error] Failed to read image: {image_path}")
        return 1

    detector = BoardDetector()
    splitter = BoardSplitter()
    orchestrator = RecognitionOrchestrator(
        RecognitionOrchestratorConfig(
            confidence_threshold=args.confidence_threshold,
            persist_manual_theme=False,
            use_saved_theme_preference=False,
        )
    )
    fen_generator = FenGenerator()

    detection = detector.detect(image)
    if not detection.detected or detection.board is None:
        print("[error] Board not detected.")
        return 1

    board = detection.board
    print(
        "[detect]",
        f"board=({board.x},{board.y},{board.width},{board.height})",
        f"confidence={board.confidence:.3f}",
        f"elapsed={detection.elapsed_ms:.1f}ms",
    )

    split = splitter.split(
        image,
        board,
        white_perspective=not args.black_perspective,
    )
    print("[split]", f"squares={len(split.squares)}", f"elapsed={split.elapsed_ms:.1f}ms")

    flow = orchestrator.recognize_position(split.squares, manual_theme=args.theme)
    print("[recognition]", f"status={flow.status.value}", f"theme={flow.selected_theme}")
    if flow.theme_detection is not None:
        theme = flow.theme_detection
        matched = (
            theme.theme.qualified_name if theme.matched and theme.theme is not None else "none"
        )
        print(
            "[theme-detect]",
            f"matched={matched}",
            f"distance={theme.distance:.1f}",
            f"light={theme.observed_light_bgr}",
            f"dark={theme.observed_dark_bgr}",
        )

    if flow.recognition is None:
        print("[error] No recognition result available.")
        print("[info]", flow.message)
        return 1

    uncertain_count = len(flow.recognition.uncertain_squares)
    print("[recognition]", f"uncertain={uncertain_count}")

    if flow.status is RecognitionFlowStatus.POSITION_UNCERTAIN and not args.allow_uncertain:
        print("[error] Position uncertain; rerun with --allow-uncertain to force FEN generation.")
        return 2

    fen_config = FenConfig(
        active_color=args.active_color,
        castling=args.castling,
        en_passant=args.en_passant,
        halfmove_clock=args.halfmove,
        fullmove_number=args.fullmove,
    )
    try:
        fen_result = fen_generator.generate(flow.recognition.classifications, fen_config)
    except InvalidPositionError as exc:
        print("[error] Invalid position:", exc)
        return 2

    print("[fen]", fen_result.fen)
    if args.expect_fen is not None:
        if fen_result.fen == args.expect_fen:
            print("[check] expected FEN: OK")
        else:
            print("[check] expected FEN: MISMATCH")
            print("[check] expected:", args.expect_fen)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
