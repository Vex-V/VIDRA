"""The raw transcript, onto the grid.

Cheap and repeatable: Whisper timestamps every word, so a transcript can be
re-cut to any grid without touching a model.

A word belongs to the chunk containing its midpoint -- a word straddling a
boundary belongs to whichever side holds more of it, and every word must land
in exactly one chunk or the text is duplicated or dropped.

Chunks with no speech are kept with empty text: `chunk_id` is shared with the
manifest, so dropping the quiet ones renumbers everything after them.

A chunk carries `turns`, one bound record per contiguous run of one voice, so a
window holding three speakers still says who said what.
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
from .driver import apply, load, run

__all__ = ["apply", "load", "run"]
