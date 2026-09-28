"""What was said, and by whom. No chunks.

Decode the whole waveform, transcribe, diarize, attribute each word to a
speaker. The output is `transcript.raw.json` -- words, segments and turns, with
no chunk ids at all.

Whole-file is not an optimisation. Whisper carries context across an utterance,
and speaker labels come from clustering over the entire recording, so a
windowed run produces speakers that are not misaligned but unnameable.

This package does not know chunking exists. Boundaries are applied afterwards
by `cut`, and deriving boundaries from speech is `boundaries/speech.py`'s job.
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
from .driver import audio, listen, load

__all__ = ["listen", "load", "audio"]
