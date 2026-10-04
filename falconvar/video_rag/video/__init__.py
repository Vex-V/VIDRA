"""5 · video -- which frames each sampler keeps from one decode pass, onto a
given grid. Only frames that survive decimation are converted to pixels.
"""

from __future__ import annotations

#: The public surface: entry points, errors and return types.
from .driver import SAMPLER_SETTINGS, ingest, load, video
from ..helpers import FrameStore
from .reader import UnreadableSource

__all__ = ["SAMPLER_SETTINGS", "FrameStore",
           "UnreadableSource", "ingest", "load", "video"]
