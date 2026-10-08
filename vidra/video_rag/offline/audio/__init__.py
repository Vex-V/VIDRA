"""2 · audio -- what was said, and by whom. No chunks.

Decodes the whole waveform, transcribes, diarizes and attributes each word to
a speaker, into `transcript.raw.json`: words, segments and turns. The file is
processed whole.
"""

from __future__ import annotations

#: The public surface: entry points, errors and return types.
from .driver import audio, listen, load

__all__ = ["listen", "load", "audio"]
