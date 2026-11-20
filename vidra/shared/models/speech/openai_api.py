"""OpenAI's transcription API as a transcriber: the track in windows.

    OpenAITranscriber()                               whisper-1: real word times
    OpenAITranscriber("gpt-4o-transcribe-diarize")    segment times, words spread

Which models give what was probed against the API (2026-11-20), 30 s of
Chernobyl each:

    whisper-1                  verbose_json: words and segments, each timed
    gpt-4o-transcribe-diarize  diarized_json: segments timed, with a speaker
    gpt-4o-transcribe          text only -- `verbose_json` and `diarized_json`
    gpt-4o-mini-transcribe       refused -- so nothing could be placed on the
    gpt-transcribe               grid; refused here, with the reason

A model not in that table needs `timing=` ("words" or "segments") to say
what it answers with. `base_url=` points it at any server speaking the same
API (a local faster-whisper server, say), where no key is needed.

Each window is sent as 16 kHz WAV, under the 25 MB file limit. Talks to the
API through the `openai` SDK, a core dependency; it retries a 429 or a 5xx
itself.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from ...reporting.errors import ModelUnavailable, Refused
from ..base import ModelFailed, Transcriber
from . import Segment, Transcript, Word
from .windows import _cjk, segments_of, spoken, wav

log = logging.getLogger(__name__)

KEY_VARS = ("OPENAI_API_KEY", "OPENAI_API")

#: What each model answers with: word times, or segment times only.
TIMING = {"whisper-1": "words", "gpt-4o-transcribe-diarize": "segments"}
#: Models that answer with text and no times at all.
TEXT_ONLY = ("gpt-4o-transcribe", "gpt-4o-mini-transcribe", "gpt-transcribe")

#: 25 MB per file; 16 kHz 16-bit WAV is 32 KB/s.
MAX_WINDOW_S = 780.0
MIN_WINDOW_S = 30.0


def _word(text: str) -> str:
    """Whisper's spelling: a leading space, except in a script written without."""
    text = text.strip()
    return text if any(_cjk(c) for c in text) else " " + text


def timed_segments(answer: dict[str, Any], offset: float) -> list[Segment]:
    """`verbose_json` -- words and segments listed apart -- as segments holding
    their words, shifted by `offset`. A word goes to the segment holding its
    midpoint, else the nearest one."""
    rows = sorted((r for r in answer.get("segments") or [] if str(r.get("text") or "").strip()),
                  key=lambda r: float(r["start"]))
    out = [Segment(round(offset + float(r["start"]), 3), round(offset + float(r["end"]), 3),
                   " " + str(r["text"]).strip()) for r in rows]
    if not out:
        return []
    for w in answer.get("words") or []:
        text = str(w.get("word") or "").strip()
        if not text:
            continue
        start, end = offset + float(w["start"]), offset + float(w["end"])
        middle = (start + end) / 2
        home = min(out, key=lambda s: 0.0 if s.start <= middle < s.end
                   else min(abs(middle - s.start), abs(middle - s.end)))
        home.words.append(Word(round(start, 3), round(end, 3), _word(text)))
    return out


class OpenAITranscriber(Transcriber):
    """`model` defaults to whisper-1, the one with word times. `window_s` is
    how much audio goes in one call (30-780 s). `concurrency` windows are sent
    at once. The key is `api_key=`, else `OPENAI_API_KEY`."""

    def __init__(self, model: str = "whisper-1", api_key: Optional[str] = None,
                 language: Optional[str] = None,
                 timing: Optional[str] = None,
                 window_s: float = 600.0,
                 concurrency: int = 4,
                 base_url: Optional[str] = None,
                 client: Any = None) -> None:
        model = str(model or "").strip()
        if not model:
            raise Refused("OpenAITranscriber needs a model name, e.g. \"whisper-1\"")
        timing = timing or TIMING.get(model)
        if timing is None:
            why = ("answers with text and no times" if model in TEXT_ONLY
                   else "is not one whose answer is known")
            raise Refused(
                f"{model} {why}, and every word must be placed in time; use "
                f"whisper-1 (word times) or gpt-4o-transcribe-diarize (segment times), "
                f"or pass timing='words' or 'segments' for a model that gives them")
        if timing not in ("words", "segments"):
            raise Refused(f"timing must be 'words' or 'segments', not {timing!r}")
        if not MIN_WINDOW_S <= window_s <= MAX_WINDOW_S:
            raise Refused(f"window_s must be {MIN_WINDOW_S:g}-{MAX_WINDOW_S:g} seconds "
                          f"(a file may be at most 25 MB), not {window_s}")
        if concurrency < 1:
            raise Refused(f"concurrency must be at least 1, not {concurrency}")
        self.model_name = model
        self.language = language
        self.timing = timing
        self.window_s = float(window_s)
        self.concurrency = concurrency
        self.base_url = base_url
        self._api_key = api_key
        self._client = client
        #: What the last `transcribe` used, summed over its windows.
        self.last_usage: dict[str, float] = {}

    @property
    def key(self) -> str:
        return f"openai:{self.model_name}"

    def __repr__(self) -> str:                    # never the API key
        where = f", base_url={self.base_url!r}" if self.base_url else ""
        return f"OpenAITranscriber({self.model_name!r}{where})"

    def api_key(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        return next((os.environ[v] for v in KEY_VARS if os.environ.get(v)), None)

    def problems(self) -> list[str]:
        if self._client is not None or self.base_url or self.api_key():
            return []
        return [f"no key for {self.key}: pass api_key= or set "
                f"{' or '.join(KEY_VARS)} in the environment"]

    def config(self) -> dict[str, Any]:
        return {"transcriber": "openai", "model": self.model_name,
                "language": self.language, "window_s": self.window_s,
                "word_times": "measured" if self.timing == "words" else "interpolated",
                **({"base_url": self.base_url} if self.base_url else {})}

    def _window(self, client: Any, samples: Any, offset: float) -> tuple[list[Segment], Optional[str], dict[str, float]]:
        length = len(samples) / self.rate
        extra: dict[str, Any] = (
            {"response_format": "verbose_json", "timestamp_granularities": ["word", "segment"]}
            if self.timing == "words" else
            {"response_format": "diarized_json", "chunking_strategy": "auto"})
        if self.language:
            extra["language"] = self.language
        try:
            answer = client.audio.transcriptions.create(
                model=self.model_name, file=("audio.wav", wav(samples, self.rate), "audio/wav"),
                **extra)
        except Exception as exc:                                  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            failure = ModelUnavailable if status in (401, 403) else ModelFailed
            raise failure(f"the {self.key} call failed: {exc}") from None
        data = answer if isinstance(answer, dict) else answer.model_dump()
        found = (timed_segments(data, offset) if self.timing == "words"
                 else segments_of(data, length, offset))
        usage = {k: float(v) for k, v in (data.get("usage") or {}).items()
                 if isinstance(v, (int, float))}
        return found, data.get("language"), usage

    def transcribe(self, samples: Any) -> Transcript:
        spans = spoken(samples, self.rate, self.window_s)
        client = self._client
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key() or "not-needed", base_url=self.base_url)
        with ThreadPoolExecutor(max_workers=max(1, min(self.concurrency, len(spans) or 1))) as pool:
            answers = list(pool.map(
                lambda span: self._window(client, samples[span[0]:span[1]], span[0] / self.rate),
                spans))
        segments: list[Segment] = []
        heard: dict[str, float] = {}
        usage: dict[str, float] = {}
        for found, language, counts in answers:
            segments += found
            if language:
                heard[language] = heard.get(language, 0.0) + sum(s.end - s.start for s in found)
            for name, count in counts.items():
                usage[name] = usage.get(name, 0.0) + count
        self.last_usage = usage
        log.info("openai transcribed %d windows", len(spans), extra={"windows": len(spans)})
        return Transcript(segments=segments,
                          language=max(heard, key=heard.get) if heard else self.language)


__all__ = ["KEY_VARS", "OpenAITranscriber", "TEXT_ONLY", "TIMING", "timed_segments"]
