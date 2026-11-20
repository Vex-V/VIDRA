"""2b · voices -- one voiceprint per diarized speaker, from the soundtrack.
Reads `media.json` and `transcript.raw.json`; writes `voices.json`. Prints are
matched across videos by a database's `match_voices`.
"""

from __future__ import annotations

#: The public surface: entry points.
from .driver import load, voiceprints, voices

__all__ = ["load", "voiceprints", "voices"]
