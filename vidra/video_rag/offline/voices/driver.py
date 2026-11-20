"""The voices component: `media.json` + `transcript.raw.json` -> `voices.json`.

The diarizer says who spoke when and keeps no voice. This stage cuts each
speaker's longest turns from the waveform, embeds each with a voice embedder
and averages their unit vectors into one print per speaker. A print belongs to
the video, not to a chunk: a speaker talks across many. Matching prints
across videos is a database read (`match_voices`), never a text search.

Turns under `min_turn_s` are not embedded -- a vector from under ~1.5 s is
mostly noise -- and a speaker with none left keeps its entry with `vector:
None`. How much speech is enough to trust a match is the caller's threshold;
`used_s` says how much each print was made from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from vidra.shared.contracts.documents import (Media, Produced, RawTranscript, Voices,
                                           fingerprint_of, same_video)
from vidra.shared.models.base import VoiceEmbedder, require
from vidra.shared.reporting import logs, progress
from vidra.shared.reporting.errors import Refused
from vidra.shared.storage.files import maybe, read, write

#: Shorter turns are not embedded.
MIN_TURN_S = 1.5
#: The longest turns per speaker, embedded one by one.
MAX_TURNS = 12


def _checked(voice_embedder: Optional[VoiceEmbedder], min_turn_s: float,
             max_turns: int) -> VoiceEmbedder:
    if voice_embedder is None:
        raise Refused("voices needs a voice_embedder; there is no default -- pass "
                      "one, e.g. LocalVoiceEmbedder()")
    require("voice_embedder", voice_embedder)
    if not min_turn_s > 0:
        raise Refused(f"min_turn_s must be positive, not {min_turn_s}")
    if max_turns < 1:
        raise Refused(f"max_turns must be at least 1, not {max_turns}")
    return voice_embedder


def source_hash(raw: RawTranscript, media: Media) -> str:
    """What a print depends on: the turns, and which file they were cut from."""
    return fingerprint_of({"turns": raw.turns, "source": media.source})


def voiceprints(media: Media, raw: RawTranscript,
                voice_embedder: Optional[VoiceEmbedder] = None,
                min_turn_s: float = MIN_TURN_S, max_turns: int = MAX_TURNS,
                previous: Optional[Voices] = None,
                on_progress: Optional[progress.Reporter] = None) -> Voices:
    """One print per diarized speaker, over documents in hand. Reads and writes
    no artifact; decodes the soundtrack of the file `media` names.

    `previous` is an earlier `voices.json`: when it was made from the same
    turns, settings and embedder, it is returned as it is.
    """
    model = _checked(voice_embedder, min_turn_s, max_turns)
    video_id = same_video(media=media, raw_transcript=raw)
    settings = {"min_turn_s": min_turn_s, "max_turns": max_turns}
    digest = source_hash(raw, media)
    if (previous is not None and previous.embedder == model.key
            and previous.source_hash == digest and previous.settings == settings):
        progress.report(on_progress, "voices", 0, 0, len(previous.speakers))
        return previous

    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for turn in raw.turns:
        if turn.get("speaker"):
            by_speaker.setdefault(turn["speaker"], []).append(
                (float(turn["start"]), float(turn["end"])))

    speakers: list[dict[str, Any]] = []
    if by_speaker:
        import numpy as np

        from ..audio.source import load
        track = load(str(media.path), rate=model.rate)
        for n, (speaker, turns) in enumerate(sorted(by_speaker.items())):
            longest = sorted((t for t in turns if t[1] - t[0] >= min_turn_s),
                             key=lambda t: t[0] - t[1])[:max_turns]
            longest.sort()
            vector = None
            if longest:
                clips = [track.samples[int(a * track.rate):int(b * track.rate)]
                         for a, b in longest]
                vectors = np.asarray(model.embed_clips(clips), dtype=np.float64)
                if len(vectors) != len(clips):
                    raise Refused(f"{model!r} returned {len(vectors)} vectors for "
                                  f"{len(clips)} clips")
                vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
                mean = vectors.mean(axis=0)
                vector = [float(v) for v in mean / (np.linalg.norm(mean) or 1.0)]
            speakers.append({
                "speaker": speaker,
                "talk_s": round(sum(b - a for a, b in turns), 3),
                "used_s": round(sum(b - a for a, b in longest), 3),
                "turns": [[round(a, 3), round(b, 3)] for a, b in longest],
                "vector": vector,
            })
            progress.report(on_progress, "voices", len(by_speaker), n + 1, 0, speaker)

    return Voices(video_id=video_id, embedder=model.key, source_hash=digest,
                  settings=settings, speakers=speakers)


def voices(media: str | Path, raw_transcript: str | Path, out: str | Path,
           previous: Optional[str | Path] = None,
           voice_embedder: Optional[VoiceEmbedder] = None,
           min_turn_s: float = MIN_TURN_S, max_turns: int = MAX_TURNS,
           on_progress: Optional[progress.Reporter] = None) -> Produced:
    """One voiceprint per diarized speaker. `voiceprints` plus a read at each
    end. `previous` is usually `out` itself.
    """
    _checked(voice_embedder, min_turn_s, max_turns)
    described = read(media, Media)
    raw = read(raw_transcript, RawTranscript)
    stored = maybe(previous, Voices)
    video_id = same_video(media=described, raw_transcript=raw, previous=stored)

    with logs.timed("voices", video_id) as done:
        document = voiceprints(described, raw, voice_embedder, min_turn_s, max_turns,
                               stored, on_progress)
        reused = document is stored
        where = write(out, document)
        printed = sum(1 for s in document.speakers if s["vector"])
        done(speakers=len(document.speakers), printed=printed, reused=reused,
             embedder=document.embedder)

    skipped = []
    if not raw.turns:
        skipped.append("no speaker turns: run audio with a diarizer")
    return Produced(
        video_id=video_id, component="voices",
        artifacts={"voices": where},
        stats={"speakers": len(document.speakers), "printed": printed,
               "too_short": len(document.speakers) - printed, "reused": reused,
               "embedder": document.embedder,
               "used_s": round(sum(s["used_s"] for s in document.speakers), 1)},
        skipped=skipped,
    )


def load(path: str | Path) -> Voices:
    """Read a `voices.json` back, typed."""
    return read(path, Voices)


__all__ = ["MAX_TURNS", "MIN_TURN_S", "load", "source_hash", "voiceprints", "voices"]
