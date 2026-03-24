"""Shared helpers for identifying recurring training weaknesses."""

from __future__ import annotations

import hashlib


def build_weakness_id(
    fen: str,
    played_uci: str,
    best_move_uci: str,
) -> str:
    """Build a stable identifier for one reviewed weakness position."""
    payload = "\n".join((fen, played_uci, best_move_uci))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
