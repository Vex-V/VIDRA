"""What a remote transcriber needs to send a long track: windows cut at quiet
moments, WAV bytes, and -- for a model that times segments, not words -- the
words spread over each segment.

Shared by `GeminiTranscriber` and `OpenAITranscriber`. No model library here.
"""

from __future__ import annotations

import io
import wave
from typing import Any, Optional

from . import Segment, Word

#: How far either side of a nominal boundary to look for the quietest moment.
SEARCH_S = 10.0
#: The frame the quietest moment is measured over.
FRAME_S = 0.1
#: Below this RMS a window is silent and is not sent.
SILENCE_RMS = 1e-3


def spoken(samples: Any, rate: int, window_s: float) -> list[tuple[int, int]]:
    """`windows`, less the silent ones: no call for those."""
    import numpy as np

    out = []
    for lo, hi in windows(samples, rate, window_s):
        part = np.asarray(samples[lo:hi], dtype=np.float32)
        if part.size and float(np.sqrt(np.mean(part ** 2))) >= SILENCE_RMS:
            out.append((lo, hi))
    return out


def clock(seconds: float) -> str:
    """`seconds` as MM:SS.s, the form the model is asked to answer in."""
    minutes, rest = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes):02d}:{rest:04.1f}"


def seconds(value: Any) -> Optional[float]:
    """A time the model wrote -- `67.5`, `"67.5s"`, `"01:07.5"`, `"0:01:07"` --
    in seconds, or None when it is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().rstrip("sS").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) > 3:
        return None
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return None
    total = 0.0
    for number in numbers:
        if number < 0:
            return None
        total = total * 60.0 + number
    return total


def windows(samples: Any, rate: int, window_s: float) -> list[tuple[int, int]]:
    """`(start, end)` sample spans covering the track, each about `window_s`
    long, every inner boundary moved to the quietest `FRAME_S` within
    `SEARCH_S` of where it would fall."""
    import numpy as np

    total = len(samples)
    step = int(window_s * rate)
    if total <= step + int(SEARCH_S * rate):
        return [(0, total)] if total else []
    frame = max(1, int(FRAME_S * rate))
    reach = int(SEARCH_S * rate)
    cuts = [0]
    while total - cuts[-1] > step + reach:
        nominal = cuts[-1] + step
        lo, hi = max(cuts[-1] + frame, nominal - reach), min(total - frame, nominal + reach)
        span = np.asarray(samples[lo:hi], dtype=np.float32)
        usable = (len(span) // frame) * frame
        if usable:
            energy = np.sqrt(np.mean(span[:usable].reshape(-1, frame) ** 2, axis=1))
            cut = lo + int(np.argmin(energy)) * frame + frame // 2
        else:
            cut = nominal
        cuts.append(cut)
    cuts.append(total)
    return list(zip(cuts[:-1], cuts[1:]))


def wav(samples: Any, rate: int) -> bytes:
    """Mono float32 samples as 16-bit PCM WAV bytes."""
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0
           ).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


def _cjk(char: str) -> bool:
    """Han, kana and the CJK punctuation block: scripts written without spaces."""
    point = ord(char)
    return (0x3000 <= point <= 0x30FF or 0x3400 <= point <= 0x4DBF
            or 0x4E00 <= point <= 0x9FFF or 0xF900 <= point <= 0xFAFF
            or 0xFF00 <= point <= 0xFFEF)


def tokens(text: str) -> list[str]:
    """The words of `text` as Whisper writes them: a leading space before each
    space-separated word; Chinese and Japanese one character each with no
    space, punctuation kept with the character before it."""
    out: list[str] = []
    for part in text.split():
        if not any(_cjk(c) for c in part):
            out.append(" " + part)
            continue
        run, latin = "", False           # `latin`: the run is Latin letters/digits
        for char in part:
            if not char.isalnum():                     # punctuation stays with the run
                run += char
                continue
            joins = latin and not _cjk(char)           # Latin continues Latin
            if run and not joins:
                out.append(run)
                run = ""
            run += char
            latin = not _cjk(char)
        if run:
            out.append(run)
    return out


def spread(text: str, start: float, end: float) -> list[Word]:
    """A segment's words over its span, each in proportion to its length:
    Gemini places segments, not words."""
    found = tokens(text)
    if not found:
        return []
    weights = [len(t.strip()) + 1 for t in found]
    scale = (end - start) / sum(weights)
    words, at = [], start
    for token, weight in zip(found, weights):
        nxt = at + weight * scale
        words.append(Word(round(at, 3), round(nxt, 3), token))
        at = nxt
    words[-1].end = round(end, 3)
    return words


def segments_of(answer: dict[str, Any], length: float, offset: float) -> list[Segment]:
    """The model's segments, made usable and shifted by `offset`: times it could
    not write are taken from the previous segment, everything is clamped to
    the window, and a segment with no span gets a short one."""
    rows = answer.get("segments") if isinstance(answer, dict) else None
    out: list[tuple[float, float, str]] = []
    previous = 0.0
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        start, end = seconds(row.get("start")), seconds(row.get("end"))
        start = min(max(previous if start is None else start, 0.0), length)
        if end is None or end <= start:
            end = start + 0.35 * max(1, len(text.split()))
        end = min(end, length)
        if end <= start:                              # a segment at the very end
            start = max(0.0, end - 0.2)
        out.append((start, end, text))
        previous = end
    out.sort(key=lambda r: r[0])
    return [Segment(round(offset + s, 3), round(offset + e, 3), " " + t,
                    spread(t, offset + s, offset + e)) for s, e, t in out]


__all__ = ["FRAME_S", "SEARCH_S", "SILENCE_RMS", "clock", "seconds", "segments_of", "spoken",
           "spread", "tokens", "wav", "windows"]
