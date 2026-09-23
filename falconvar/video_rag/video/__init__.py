"""5 · ingest -- which frames are worth describing, and why.

One decode pass, feeding every sampler from it. The grid arrives as data and is
never edited: this asks `timeline.nearest(ts)` and nothing else.

**Pixels are converted only for frames that survive decimation** -- 4% of them
at 1/s from 25 fps -- which is the difference between 3 minutes and 37 on a
three-hour file. See `reader.py`.
"""

from __future__ import annotations

#: **The public surface is the entry points, the errors and the return types.**
#: `run` does the work and `load` reads the result back; anything beyond those
#: is here because a caller cannot do without it -- a second way *in* that no
#: naming collapses into `run`, an exception they have to catch by name, or a
#: type they would annotate. Everything else is machinery, and stays reachable
#: through its own module rather than advertised here. See CLAUDE.md.
#:
#: `main` is deliberately absent: it is argparse, and `__main__.py` reaches it
#: as `from .driver import main`. Nothing ever imported it from the package.
from .driver import SAMPLER_SETTINGS, ingest, load, run
from ..frames import Frames, FrameStore, MemoryFrames
from .reader import UnreadableSource

__all__ = ["SAMPLER_SETTINGS", "FrameStore", "Frames", "MemoryFrames",
           "UnreadableSource", "ingest", "load", "run"]
