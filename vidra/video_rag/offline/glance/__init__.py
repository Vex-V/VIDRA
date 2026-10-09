"""7b · glance -- kept frames to vectors with a visual embedder, no model answer.
Beside describe or instead of it. Writes `glances.json`, `embedded.json`'s
shape under the visual embedder's own key.
"""

from __future__ import annotations

#: The public surface: entry points and the key a search reads.
from .driver import glance, load, look, space_of

__all__ = ["glance", "load", "look", "space_of"]
