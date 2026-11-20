"""What a transcriber and a diarizer hand back, and the two that load nothing.

    Word, Segment, Transcript      what was said (a `Transcriber`)
    Turn, Diarization              who spoke when (a `Diarizer`)
    StubTranscriber                placeholder text, no model
    NoDiarizer                     no speakers: every word keeps speaker None

The ready-made models are a module each, imported on first use:

    whisper.WhisperTranscriber     faster-whisper in this process
    gemini.GeminiTranscriber       Gemini's API, the track in windows
    pyannote.PyannoteDiarizer      pyannote's pipeline in this process

Pure data and two trivial models: nothing here imports a model library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..base import Diarizer, Transcriber


# ---------------------------------------------------------------- what was said

@dataclass
class Word:
    """One word, with the span the model placed it in."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None

    @property
    def midpoint(self) -> float:
        """Where this word belongs: its midpoint, used to attribute it to a
        speaker and to a chunk."""
        return (self.start + self.end) / 2.0

    def as_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "text": self.text, "speaker": self.speaker}


@dataclass
class Segment:
    """One utterance as the transcriber grouped it."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "text": self.text, "speaker": self.speaker}


@dataclass
class Transcript:
    """A `Transcriber`'s answer: segments in order, each with its words."""

    segments: list[Segment] = field(default_factory=list)
    language: Optional[str] = None
    language_probability: Optional[float] = None

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


# ------------------------------------------------------------ who spoke when

@dataclass
class Turn:
    speaker: str
    start: float
    end: float

    def as_dict(self) -> dict[str, Any]:
        return {"speaker": self.speaker, "start": round(self.start, 3),
                "end": round(self.end, 3)}


@dataclass
class Diarization:
    """A `Diarizer`'s answer: turns, each a label and a span."""

    turns: list[Turn] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        return sorted({t.speaker for t in self.turns})

    @property
    def speech_s(self) -> float:
        return sum(t.end - t.start for t in self.turns)


# ---------------------------------------------------- the two that load nothing

class StubTranscriber(Transcriber):
    """Fixed text every `every_s` seconds, for running the pipeline with no
    model."""

    key = "stub:transcriber"

    def __init__(self, every_s: float = 5.0) -> None:
        if every_s <= 0:
            from ...reporting.errors import Refused
            raise Refused(f"every_s must be positive, not {every_s}")
        self.every_s = every_s

    def transcribe(self, samples: Any) -> Transcript:
        duration = len(samples) / self.rate
        segments: list[Segment] = []
        t = 0.0
        while t < duration:
            end = min(t + self.every_s, duration)
            text = f"[stub {t:.1f}]"
            segments.append(Segment(t, end, text, [Word(t, end, text)]))
            t += self.every_s
        return Transcript(segments, language="stub", language_probability=1.0)

    def config(self) -> dict[str, Any]:
        return {"transcriber": "stub", "every_s": self.every_s}


class NoDiarizer(Diarizer):
    """No speaker information: every word keeps `speaker=None` -- truthful,
    where labelling everything SPEAKER_00 is not."""

    key = "none:diarizer"

    def diarize(self, samples: Any) -> Diarization:
        return Diarization([])

    def config(self) -> dict[str, Any]:
        return {"diarizer": "none"}


__all__ = ["Diarization", "NoDiarizer", "Segment", "StubTranscriber", "Transcript",
           "Turn", "Word"]
