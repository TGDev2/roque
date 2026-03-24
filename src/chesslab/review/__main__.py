"""CLI entry point for post-game review.

Usage::

    python -m chesslab.review game.pgn [options]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chesslab.persistence.game_storage import GameStorage, PersistenceError
from chesslab.persistence.review_history import ReviewHistoryStorage
from chesslab.review.game_reviewer import (
    GameReview,
    GameReviewer,
    MoveClassification,
    ReviewConfig,
    ReviewError,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m chesslab.review",
        description="Analyse a PGN game and print move-by-move review.",
    )
    parser.add_argument(
        "pgn_file",
        help="Path to a PGN file.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=16,
        help="Stockfish analysis depth (default: 16).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1.5,
        help="Time limit per position in seconds (default: 1.5).",
    )
    parser.add_argument(
        "--stockfish",
        default="stockfish",
        help="Path to Stockfish binary (default: stockfish).",
    )
    return parser.parse_args()


_CLASS_LABELS = {
    MoveClassification.EXCELLENT: "!!",
    MoveClassification.GOOD: "ok",
    MoveClassification.INACCURACY: "?!",
    MoveClassification.MISTAKE: "?",
    MoveClassification.BLUNDER: "??",
    MoveClassification.FORCED: "forced",
}


def _format_review(review: GameReview, headers: dict[str, str]) -> str:
    """Format a GameReview as human-readable text."""
    lines: list[str] = []

    white = headers.get("White", "?")
    black = headers.get("Black", "?")
    result = headers.get("Result", "*")

    lines.append(f"Game: {white} vs {black}  ({result})")
    lines.append(f"Moves: {len(review.moves)}  |  Analysis depth: {review.moves[0].eval_before is not None and 'yes' or 'n/a'}")
    lines.append(f"Time: {review.total_elapsed_ms / 1000:.1f}s")
    lines.append("")

    # Stats per colour.
    for label, counts, avg in [
        ("White", review.white_classification_counts, review.white_avg_cp_loss),
        ("Black", review.black_classification_counts, review.black_avg_cp_loss),
    ]:
        parts = []
        for cls in MoveClassification:
            count = counts.get(cls, 0)
            if count:
                parts.append(f"{cls.value}: {count}")
        lines.append(f"  {label:<6} avg cp loss: {avg:5.1f}  |  {', '.join(parts)}")

    lines.append("")

    # Prioritised lessons.
    if review.lessons:
        lines.append("Lessons:")
        for lesson in review.lessons:
            lines.append(
                f"  {lesson.priority}. {lesson.title} "
                f"[{_format_lesson_anchor(review, lesson.focus_ply)}]"
            )
            lines.append(f"     Fact: {lesson.finding}")
            lines.append(f"     Lesson: {lesson.takeaway}")
        lines.append("")
    else:
        lines.append("Lessons: no reliable lesson extracted.")
        lines.append("")

    # Critical moments.
    if review.critical_moments:
        lines.append("Critical moments:")
        for ply in review.critical_moments:
            m = review.moves[ply]
            move_num = ply // 2 + 1
            dot = "." if m.is_white else "..."
            lines.append(
                f"  {move_num}{dot} {m.san:<8} "
                f"{_CLASS_LABELS.get(m.classification, '?'):<7} "
                f"cp_loss: {m.cp_loss}"
            )
        lines.append("")

    # Move-by-move.
    lines.append("Move-by-move:")
    for m in review.moves:
        move_num = m.ply // 2 + 1
        dot = "." if m.is_white else "..."
        tag = _CLASS_LABELS.get(m.classification, "?")
        eval_str = str(m.eval_after) if m.eval_after else "?"
        lines.append(
            f"  {move_num:>3}{dot:<4} {m.san:<8} {tag:<7} {eval_str}"
        )

    return "\n".join(lines)


def _format_lesson_anchor(review: GameReview, focus_ply: int) -> str:
    """Return a compact move reference for a lesson anchor."""
    if focus_ply < 0 or focus_ply >= len(review.moves):
        return "position"

    move = review.moves[focus_ply]
    move_num = move.ply // 2 + 1
    dot = "." if move.is_white else "..."
    return f"{move_num}{dot} {move.san}"


def main() -> None:
    args = _parse_args()

    pgn_path = Path(args.pgn_file)
    try:
        record = GameStorage.load_game(pgn_path)
    except PersistenceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not record.board.move_stack:
        print("Error: no moves found in PGN.", file=sys.stderr)
        sys.exit(1)

    total = len(record.board.move_stack)
    print(
        f"Reviewing {record.headers.get('White', '?')} vs "
        f"{record.headers.get('Black', '?')} ({total} moves)...",
        file=sys.stderr,
    )

    def on_progress(current: int, total: int) -> None:
        print(f"\rAnalysing: {current}/{total}", end="", file=sys.stderr)

    config = ReviewConfig(
        depth=args.depth,
        time_limit=args.time_limit,
        stockfish_path=args.stockfish,
    )

    try:
        reviewer = GameReviewer(config)
        review = reviewer.review(
            record.board,
            progress_callback=on_progress,
        )
    except ReviewError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    print(file=sys.stderr)  # newline after progress
    try:
        ReviewHistoryStorage().record_review(
            review,
            source_path=record.file_path,
        )
    except PersistenceError as exc:
        print(
            f"Warning: failed to save review summary: {exc}",
            file=sys.stderr,
        )
    print(_format_review(review, record.headers))


if __name__ == "__main__":
    main()
