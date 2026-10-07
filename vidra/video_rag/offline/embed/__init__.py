"""8 · embed -- descriptions and transcript chunks to vectors, keyed by a hash of
the text. Writes `embedded.json`, text and vectors together.
"""

from __future__ import annotations

#: The public surface: entry points, errors and return types. `readable` is bound so `embed.readable.load()` works.
from ...core.embed import readable  # noqa: F401
from .driver import embed, encode, load
from ...core.embed.units import Unit
#: The embedders live in `shared/models`; the error is re-exported here.
from vidra.shared.models.embedders import EmbedderUnavailable

__all__ = ["EmbedderUnavailable", "Unit", "embed", "encode", "load"]
