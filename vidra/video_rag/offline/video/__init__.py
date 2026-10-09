"""5 · video -- which frames each sampler keeps from one decode pass, onto a
given grid. Only frames that survive decimation are converted to pixels.
`frames` is the same decode on its own: a file's frames, upright, for code of
your own.
"""

from __future__ import annotations

#: The public surface: entry points, errors and return types.
from ...core.frames import FrameStore
from ...core.sampling.reader import Frame, UnreadableSource
from .driver import frames, ingest, load, video
from .recovery import Mismatch, MissingVideo, recreate

__all__ = ["Frame", "FrameStore", "Mismatch", "MissingVideo",
           "UnreadableSource", "frames", "ingest", "load", "recreate", "video"]
