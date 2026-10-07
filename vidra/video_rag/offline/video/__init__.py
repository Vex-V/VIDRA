"""5 · video -- which frames each sampler keeps from one decode pass, onto a
given grid. Only frames that survive decimation are converted to pixels.
"""

from __future__ import annotations

#: The public surface: entry points, errors and return types.
from ...core.frames import FrameStore
from ...core.sampling.reader import UnreadableSource
from ...core.sampling.specs import SAMPLER_SETTINGS
from .driver import ingest, load, video
from .recovery import Mismatch, MissingVideo, recreate

__all__ = ["SAMPLER_SETTINGS", "FrameStore", "Mismatch", "MissingVideo",
           "UnreadableSource", "ingest", "load", "recreate", "video"]
