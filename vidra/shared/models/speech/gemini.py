"""Gemini's API as a transcriber: the track in windows, segments back.

    GeminiTranscriber("gemini-3-flash-preview")     key: api_key=, else GEMINI_API_KEY
                                                    or GOOGLE_API_KEY

Each window is sent inline as 16 kHz WAV to `generateContent`, asked for
segments with start and end times as JSON, and shifted back onto the track's
clock. Windows are cut at the quietest moment near each boundary, so a word
is rarely split, and a silent window costs no call.

**Word times are estimated, not measured.** Gemini places segments, not
words, so a segment's words share its span in proportion to their length.
Everything after `audio` reads word times -- a word goes to the chunk and to
the speaker holding its midpoint -- so a word near a chunk edge or a speaker
change can land on the wrong side. `config()` says `word_times: interpolated`.
A language written without spaces (Japanese, Chinese) gets one "word" per
segment.

There is no default model: name one your key serves. Audio costs 32 tokens
a second. Talks to the REST API with httpx, which the core already installs.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from ...reporting.errors import ModelUnavailable, Refused
from ..base import ModelFailed, Transcriber, parse_json
from . import Segment, Transcript
#: Published here too: `gemini.windows` etc. were where these lived first.
from .windows import clock, seconds, segments_of, spoken, spread, tokens, wav, windows

log = logging.getLogger(__name__)

#: Where the key is read from when `api_key=` is not given, first set wins.
KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

#: A request carries at most 20 MB inline; base64 WAV at 16 kHz is ~43 KB/s.
MAX_WINDOW_S = 420.0
MIN_WINDOW_S = 30.0

RETRIES = 2
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

PROMPT = """Transcribe the speech in this audio clip verbatim, in the language it is spoken in. Do not translate, summarise or correct it.
Return the speech as segments in time order: one sentence or short phrase each, at most about 10 seconds long, together covering all the speech in the clip.
"start" and "end" are where each segment begins and ends in this clip, as MM:SS.s (for example 01:07.5). The clip is {length} long; no time may be later than that.
Leave out music, noises and silence. If nobody speaks, return no segments.
"language" is the ISO 639-1 code of the main language spoken, or "" if nobody speaks."""

#: Gemini's response schema (its OpenAPI subset).
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "language": {"type": "STRING"},
        "segments": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {"start": {"type": "STRING"}, "end": {"type": "STRING"},
                           "text": {"type": "STRING"}},
            "required": ["start", "end", "text"],
            "propertyOrdering": ["start", "end", "text"]}},
    },
    "required": ["language", "segments"],
    "propertyOrdering": ["language", "segments"],
}


# -------------------------------------------------------------------- model

class GeminiTranscriber(Transcriber):
    """`model` is required: Gemini's ids change, so name one your key serves.
    `window_s` is how much audio goes in one call (30-420 s; Gemini's own
    timestamps drift over long audio, so shorter is safer). `concurrency`
    windows are sent at once. `generation` is merged into the request's
    `generationConfig` -- e.g. `{"thinkingConfig": {"thinkingLevel": "low"}}` --
    and recorded, since it changes the answers."""

    def __init__(self, model: str, api_key: Optional[str] = None,
                 language: Optional[str] = None,
                 window_s: float = 300.0,
                 concurrency: int = 4,
                 max_output_tokens: int = 32768,
                 generation: Optional[dict[str, Any]] = None,
                 base_url: str = BASE_URL,
                 client: Any = None) -> None:
        if not model or not str(model).strip():
            raise Refused("GeminiTranscriber needs a model name, e.g. "
                          "GeminiTranscriber(\"gemini-3-flash-preview\")")
        if not MIN_WINDOW_S <= window_s <= MAX_WINDOW_S:
            raise Refused(f"window_s must be {MIN_WINDOW_S:g}-{MAX_WINDOW_S:g} seconds "
                          f"(a request carries at most 20 MB inline), not {window_s}")
        if concurrency < 1:
            raise Refused(f"concurrency must be at least 1, not {concurrency}")
        if max_output_tokens < 1:
            raise Refused(f"max_output_tokens must be at least 1, not {max_output_tokens}")
        self.model_name = str(model).strip()
        self.language = language
        self.window_s = float(window_s)
        self.concurrency = concurrency
        self.max_output_tokens = max_output_tokens
        self.generation = dict(generation or {})
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client
        #: Tokens the last `transcribe` used, summed over its windows.
        self.last_usage: dict[str, int] = {}

    @property
    def key(self) -> str:
        return f"gemini:{self.model_name}"

    def __repr__(self) -> str:                    # never the API key
        return f"GeminiTranscriber({self.model_name!r})"

    def api_key(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        return next((os.environ[v] for v in KEY_VARS if os.environ.get(v)), None)

    def problems(self) -> list[str]:
        if self._client is not None or self.api_key():
            return []
        return [f"no key for {self.key}: pass api_key= or set "
                f"{' or '.join(KEY_VARS)} in the environment"]

    def config(self) -> dict[str, Any]:
        return {"transcriber": "gemini", "model": self.model_name,
                "language": self.language, "window_s": self.window_s,
                "max_output_tokens": self.max_output_tokens,
                "generation": self.generation, "word_times": "interpolated"}

    # ----------------------------------------------------------- one window

    def _request(self, audio: bytes, length: float) -> dict[str, Any]:
        prompt = PROMPT.format(length=clock(length))
        if self.language:
            prompt += f"\nThe speech is in {self.language}."
        return {
            "contents": [{"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "audio/wav",
                                 "data": base64.b64encode(audio).decode("ascii")}}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseSchema": SCHEMA,
                                 "maxOutputTokens": self.max_output_tokens,
                                 **self.generation},
        }

    def _post(self, http: Any, body: dict[str, Any]) -> Any:
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        headers = {"content-type": "application/json"}
        key = self.api_key()
        if key:
            headers["x-goog-api-key"] = key
        for attempt in range(RETRIES + 1):
            try:
                response = http.post(url, json=body, headers=headers, timeout=600)
            except Exception as exc:                      # noqa: BLE001
                raise ModelFailed(f"the {self.key} call failed: {exc}") from None
            if response.status_code not in RETRY_STATUS or attempt == RETRIES:
                return response
            try:
                wait = min(60.0, float(response.headers.get("retry-after", "")))
            except (TypeError, ValueError):
                wait = float(2 ** attempt)
            time.sleep(wait)
        return response

    def _window(self, http: Any, samples: Any, offset: float) -> tuple[list[Segment], Optional[str], dict[str, int]]:
        length = len(samples) / self.rate
        response = self._post(http, self._request(wav(samples, self.rate), length))
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", {}).get("message") or response.text
            except ValueError:
                detail = response.text
            failure = ModelUnavailable if response.status_code in (401, 403) else ModelFailed
            raise failure(f"the {self.key} call failed: {response.status_code}: {detail}")
        data = response.json()
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        if blocked:
            raise ModelFailed(f"{self.key} refused the audio at {clock(offset)}: {blocked}")
        candidates = data.get("candidates") or []
        if not candidates:
            raise ModelFailed(f"{self.key} returned no answer for the audio at {clock(offset)}")
        first = candidates[0]
        if first.get("finishReason") == "MAX_TOKENS":
            raise ModelFailed(f"{self.key} was cut short at {clock(offset)}; raise "
                              f"max_output_tokens above {self.max_output_tokens} or "
                              f"lower window_s")
        text = "".join(p.get("text", "") for p in (first.get("content") or {}).get("parts", [])
                       if not p.get("thought"))
        if not text.strip():
            raise ModelFailed(f"{self.key} returned nothing for the audio at {clock(offset)}")
        answer = parse_json(text)
        usage = data.get("usageMetadata") or {}
        counts = {k: int(usage.get(k) or 0) for k in
                  ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount")}
        language = str(answer.get("language") or "").strip().lower() or None
        return segments_of(answer, length, offset), language, counts

    # ----------------------------------------------------------- the track

    def transcribe(self, samples: Any) -> Transcript:
        spans = spoken(samples, self.rate, self.window_s)

        def one(span: tuple[int, int]) -> Any:
            lo, hi = span
            return self._window(http, samples[lo:hi], lo / self.rate)

        own = self._client is None
        if own:
            import httpx
            http = httpx.Client(timeout=600)
        else:
            http = self._client
        try:
            with ThreadPoolExecutor(max_workers=max(1, min(self.concurrency,
                                                           len(spans) or 1))) as pool:
                answers = list(pool.map(one, spans))
        finally:
            if own:
                http.close()

        segments: list[Segment] = []
        heard: dict[str, float] = {}
        usage: dict[str, int] = {}
        for found, language, counts in answers:
            segments += found
            if language:
                heard[language] = heard.get(language, 0.0) + sum(
                    s.end - s.start for s in found)
            for name, count in counts.items():
                usage[name] = usage.get(name, 0) + count
        self.last_usage = usage
        log.info("gemini transcribed %d windows", len(spans),
                 extra={"windows": len(spans), **usage})
        return Transcript(segments=segments,
                          language=max(heard, key=heard.get) if heard else self.language)


__all__ = ["BASE_URL", "GeminiTranscriber", "KEY_VARS", "clock", "seconds", "segments_of",
           "spread", "tokens", "wav", "windows"]
