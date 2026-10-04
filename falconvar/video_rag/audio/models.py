"""The transcriber and diarizer protocols, their results, and a lazy registry
each.

`Word`, `Segment`, `Transcript` for what was said; `Turn`, `Diarization` for
who spoke. Plus a stub transcriber and a no-op diarizer that load nothing.
Real backends are imported only when asked for by name.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .source import Track
#: Shared with the local aggregates.
from falconvar.shared.reporting.errors import ModelUnavailable
from falconvar.shared.reporting.errors import UnknownOption


# ---------------------------------------------------------------- transcribe

@dataclass
class Word:
    """One word, with the span the model placed it in."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None

    @property
    def midpoint(self) -> float:
        """Where this word belongs: its midpoint, used to attribute it to a speaker and
        to a chunk.
        """
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
    segments: list[Segment] = field(default_factory=list)
    language: Optional[str] = None
    language_probability: Optional[float] = None

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


class Transcriber(Protocol):
    def transcribe(self, track: Track) -> Transcript: ...
    def config(self) -> dict[str, Any]: ...


# ------------------------------------------------------------------ diarize

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
    turns: list[Turn] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        return sorted({t.speaker for t in self.turns})

    @property
    def speech_s(self) -> float:
        return sum(t.end - t.start for t in self.turns)


class Diarizer(Protocol):
    def diarize(self, track: Track) -> Diarization: ...
    def config(self) -> dict[str, Any]: ...


# ----------------------------------------------------------------- registry

class StubTranscriber:
    """Returns fixed text without loading anything, for running the pipeline with
    no model.
    """

    name = "stub"

    def __init__(self, every_s: float = 5.0) -> None:
        self.every_s = every_s

    def transcribe(self, track: Track) -> Transcript:
        segments: list[Segment] = []
        t = 0.0
        while t < track.duration_s:
            end = min(t + self.every_s, track.duration_s)
            text = f"[stub {t:.1f}]"
            segments.append(Segment(t, end, text,
                                    [Word(t, end, text)]))
            t += self.every_s
        return Transcript(segments, language="stub", language_probability=1.0)

    def config(self) -> dict[str, Any]:
        return {"transcriber": self.name, "every_s": self.every_s}


class NoDiarizer:
    """No speaker information: every word keeps `speaker=None`."""

    name = "none"

    def diarize(self, track: Track) -> Diarization:
        return Diarization([])

    def config(self) -> dict[str, Any]:
        return {"diarizer": self.name}


#: name -> "module:Class", imported on first use.
TRANSCRIBERS: dict[str, Any] = {"stub": StubTranscriber,
                                "whisper": "backends.whisper:WhisperTranscriber"}
DIARIZERS: dict[str, Any] = {"none": NoDiarizer,
                             "pyannote": "backends.pyannote:PyannoteDiarizer"}


def _resolve(registry: dict[str, Any], name: str, kind: str):
    if name not in registry:
        raise UnknownOption(f"unknown {kind} {name!r}; known: {', '.join(registry)}")
    entry = registry[name]
    if isinstance(entry, str):
        module_name, class_name = entry.split(":")
        module = importlib.import_module(f".{module_name}", __package__)
        entry = getattr(module, class_name)
        registry[name] = entry
    return entry


def transcriber(name: str, **kwargs) -> Transcriber:
    return _resolve(TRANSCRIBERS, name, "transcriber")(**kwargs)


def diarizer(name: str, **kwargs) -> Diarizer:
    return _resolve(DIARIZERS, name, "diarizer")(**kwargs)


def settings(kind: str, name: str) -> set[str]:
    """What this backend's constructor accepts, by argument name (read off its
    signature).
    """
    registry = {"transcriber": TRANSCRIBERS, "diarizer": DIARIZERS}.get(kind)
    if registry is None:
        raise UnknownOption(f"unknown kind {kind!r}; known: transcriber, diarizer")
    import inspect
    cls = _resolve(registry, name, kind)
    if cls.__init__ is object.__init__:
        # A backend with no constructor of its own takes nothing.
        return set()
    named = (inspect.Parameter.POSITIONAL_OR_KEYWORD,
             inspect.Parameter.KEYWORD_ONLY)
    return {p.name for p in inspect.signature(cls.__init__).parameters.values()
            if p.name != "self" and p.kind in named}


def available() -> dict[str, list[str]]:
    return {"transcribers": sorted(TRANSCRIBERS), "diarizers": sorted(DIARIZERS)}


__all__ = ["ModelUnavailable", "Word", "Segment", "Transcript", "Transcriber",
           "Turn", "Diarization", "Diarizer", "StubTranscriber", "NoDiarizer",
           "TRANSCRIBERS", "DIARIZERS", "transcriber", "diarizer", "available",
           "settings"]
