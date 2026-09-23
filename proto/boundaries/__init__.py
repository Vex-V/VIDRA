"""Where the chunk boundaries fall.

Everything that decides a boundary, from both modalities.

    scenes.py   the picture: decodes video, scores frame-to-frame difference,
                thresholds it into cuts
    speech.py   the soundtrack: reads a finished transcript for silences or
                speaker changes
    grid.py     cuts + a duration -> spans, with the guards applied

`speech.py` lives here rather than in `audio/` so that package never learns
chunking exists. It reads `transcript.raw.json` as a file, never by importing
`audio` -- which is what keeps the import graph acyclic while the run order
flips between policies.
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
from .driver import (EVIDENCE_SETTINGS, boundaries, detect, evidence, load,
                     retune, timeline)
from .grid import POLICIES

__all__ = ["EVIDENCE_SETTINGS", "POLICIES", "boundaries", "detect",
           "evidence", "load", "retune", "timeline"]
