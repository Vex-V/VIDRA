"""8 · embed -- documents to vectors, keyed by a hash of the text.

Both modalities land here: a description from the picture and a transcript
chunk from the soundtrack are both text with a span and some bound structure,
so audio needed no index of its own and no code past `units.from_transcript`.

It writes one document, `embedded.json`, carrying the text and the vectors.
Putting them in a database is the pipeline's job -- this component needs no
server to be reachable, and a deployment that wants its vectors somewhere else
has a file to read.
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
#: Bound so `embed.readable.load()` resolves on a bare import of the package.
#: A submodule, not surface.
from . import readable  # noqa: F401
from .driver import embed, encode, load
from .units import Unit
#: Embedding is not a video_rag concern -- it takes text and returns
#: vectors, and both tiers do it. It lives in `shared/models/`; these two
#: are re-exported because they are this component's surface.
from falconvar.shared.models.embedders import EmbedderUnavailable, available

__all__ = ["EmbedderUnavailable", "Unit", "available", "embed",
           "encode", "load"]
