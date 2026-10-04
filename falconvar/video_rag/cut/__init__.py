"""6 · cut -- the raw transcript, onto the grid.

A word belongs to the chunk containing its midpoint. Chunks with no speech are
kept with empty text. Each chunk carries `turns`: one record per contiguous
run of one voice.
"""

from __future__ import annotations

#: The public surface: entry points, errors and return types.
from .driver import apply, cut, load

__all__ = ["apply", "cut", "load"]
