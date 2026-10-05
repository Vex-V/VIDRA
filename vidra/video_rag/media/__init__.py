"""1 · media -- one file in: what streams it carries, and the folder its output
goes in.
"""

from __future__ import annotations

#: The public surface: entry points, errors and return types. `split` describes a file without writing anything.
from .driver import VideoIdTaken, load, media
from .split import UnusableMedia, split

__all__ = ["UnusableMedia", "VideoIdTaken", "load", "media", "split"]
