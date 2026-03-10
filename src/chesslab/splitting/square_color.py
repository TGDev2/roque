"""Square colour helpers using algebraic board coordinates.

Convention:
- ``file_idx``: 0..7 for ``a``..``h``
- ``rank_idx``: 0..7 for ranks ``1``..``8``
- ``a1`` is dark in standard chessboards.
"""

from __future__ import annotations


def is_light_square(file_idx: int, rank_idx: int) -> bool:
    """Return True when a square is light, False when dark.

    With standard orientation, ``a1`` is dark, so squares are light when
    ``file_idx + rank_idx`` is odd.
    """
    return (file_idx + rank_idx) % 2 == 1


def square_background(file_idx: int, rank_idx: int) -> str:
    """Return ``'light'`` or ``'dark'`` for a square."""
    return "light" if is_light_square(file_idx, rank_idx) else "dark"
