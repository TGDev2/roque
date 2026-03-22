"""Unified product entry point for ChessLab."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from chesslab.analysis import EngineNotFoundError
from chesslab.gui import ChessGui, GuiConfig, StartupJourneyMode
from chesslab.product_usage import PRODUCT_BOUNDARY, PRODUCT_SCOPE


_MODE_MAP: dict[str, StartupJourneyMode] = {
    "manual": StartupJourneyMode.MANUAL_COACH,
    "observe": StartupJourneyMode.OBSERVE_LIVE,
    "review-recent": StartupJourneyMode.REVIEW_RECENT,
    "review-paste": StartupJourneyMode.REVIEW_PASTE,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chesslab",
        description=(
            f"Launch ChessLab from a single product entry point. "
            f"{PRODUCT_SCOPE} {PRODUCT_BOUNDARY}"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("launcher", *tuple(_MODE_MAP)),
        default="launcher",
        help="Startup journey (default: launcher).",
    )
    parser.add_argument(
        "--review-file",
        default=None,
        help="Optional PGN file to open directly in post-game review mode.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the unified ChessLab launcher."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = GuiConfig()
    if args.mode != "launcher":
        config.startup_mode = _MODE_MAP[args.mode]

    if args.review_file:
        config.startup_mode = StartupJourneyMode.REVIEW_FILE
        config.startup_review_file = Path(args.review_file)

    try:
        ChessGui(config).run()
    except EngineNotFoundError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
