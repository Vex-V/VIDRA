"""The cut component: a grid + a raw transcript -> `transcript.json`."""

from __future__ import annotations

from pathlib import Path

from vidra.shared.reporting import logs
from vidra.shared.contracts.documents import (Produced, RawTranscript,
                                                  Timeline, Transcript)
from vidra.shared.storage.files import read, write
from vidra.shared.contracts.documents import same_video
from .cutter import stats_for, to_chunks


def apply(timeline: Timeline, raw: RawTranscript) -> Transcript:
    """The grid, onto a transcript already in hand. Reads and writes nothing. The
    video id comes from the transcript.
    """
    chunks = to_chunks(raw, timeline)
    return Transcript(
        video_id=raw.video_id,
        timeline_fingerprint=timeline.fingerprint(),
        model=raw.model,
        chunks=chunks,
        stats=stats_for(raw, chunks),
    )


def cut(timeline: str | Path, raw_transcript: str | Path,
        out: str | Path) -> Produced:
    """Apply the grid at `timeline` to the transcript at `raw_transcript`. `apply`
    plus a read at each end; no model.
    """
    grid = read(timeline, Timeline)
    raw = read(raw_transcript, RawTranscript)
    video_id = same_video(timeline=grid, raw_transcript=raw)

    with logs.timed("cut", video_id) as done:
        transcript = apply(grid, raw)
        stats = transcript.stats
        where = write(out, transcript)
        done(chunks=stats.get("chunks"), words=stats.get("words"))
    return Produced(
        video_id=video_id, component="cut",
        artifacts={"transcript": where},
        stats={**stats, "timeline_fingerprint": transcript.timeline_fingerprint},
    )


def load(path: str | Path) -> Transcript:
    """Read a `transcript.json` back, typed."""
    return read(path, Transcript)
