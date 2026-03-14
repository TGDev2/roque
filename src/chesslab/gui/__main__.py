"""Entry point for ``python -m chesslab.gui``."""

from __future__ import annotations

from chesslab.analysis import EngineNotFoundError
from chesslab.gui import ChessGui

if __name__ == "__main__":
    try:
        ChessGui().run()
    except EngineNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from None
