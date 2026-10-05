"""8 · embed -- descriptions and transcript chunks to vectors, keyed by a hash of
the text. Writes `embedded.json`, text and vectors together.
"""

from __future__ import annotations

#: The public surface: entry points, errors and return types. `readable` is bound so `embed.readable.load()` works.
from . import readable  # noqa: F401
from .driver import embed, encode, load
from .units import Unit
#: The embedder registry lives in `shared/models`; re-exported here.
from vidra.shared.models.embedders import EmbedderUnavailable, available

__all__ = ["EmbedderUnavailable", "Unit", "available", "embed",
           "encode", "load"]
