"""1 · split -- one file in, two stream descriptions out.

The fork, and only the fork: it says what streams the file carries and what
each half needs to open its own decoder. No pixels, no waveform, no model.
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
#: `split` is a second way *in*, by the same test as `boundaries.evidence`: it
#: does the component's work and writes nothing, which is how a caller asks
#: what a file is before committing a directory to it. It is also the only way
#: to obtain a `Media` without `run`, and this is the one component that takes
#: a path rather than a video id, so the question has nowhere else to go.
from .driver import VideoIdTaken, load, media
from .split import UnusableMedia, split

__all__ = ["UnusableMedia", "VideoIdTaken", "load", "media", "split"]
