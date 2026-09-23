"""One model answer per (chunk, sampler).

Reads the frame store and nothing else. There is no seek-the-video fallback:
the store exists so this stage has its frames in hand, and a fallback would do
its job while leaving it broken, silently and ~40x slower.

The question asked is the sampler's `prompt`, falling back to its name. Which
keys a call's schema may fill is narrowed by the other *questions* on the same
chunk, so exactly one call answers each key and merging is a plain union.

Resume is keyed on the manifest, the describer and a hash of `prompts.py`
together: without all three, switching describers skips every pair and reports
success having done nothing.
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
from .base import DescriberUnavailable, available
from .driver import answer, load, run
from .frames import StoreUnavailable

__all__ = ["DescriberUnavailable", "StoreUnavailable", "answer",
           "available", "load", "run"]
