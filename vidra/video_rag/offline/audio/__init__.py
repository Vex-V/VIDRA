"""2 · audio -- what was said, and by whom. No chunks.

Decodes the whole waveform, transcribes, diarizes and attributes each word to
a speaker, into `transcript.raw.json`: words, segments and turns. The file is
processed whole.

Which models listen is the caller's: a `Transcriber` and a `Diarizer`
(`vidra.WhisperTranscriber`, `GeminiTranscriber`, `PyannoteDiarizer`,
`NoDiarizer`, or a subclass). What they answer with is published here, for a
caller writing one:

    Transcript(segments=[Segment(start, end, text, words=[Word(start, end, text)])],
               language="en")
    Diarization(turns=[Turn("SPEAKER_00", start, end)])
"""

from __future__ import annotations

#: The public surface: entry points, errors and return types.
from vidra.shared.models.speech import Diarization, Segment, Transcript, Turn, Word
from .driver import audio, listen, load

__all__ = ["Diarization", "Segment", "Transcript", "Turn", "Word", "audio", "listen",
           "load"]
