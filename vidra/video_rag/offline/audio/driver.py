"""The audio component: `media.json` -> `transcript.raw.json`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vidra.shared.reporting import logs
from vidra.shared.contracts.documents import Media, Produced, RawTranscript
from vidra.shared.models.base import Diarizer, Transcriber, require
from vidra.shared.models.roles import resolve
from vidra.shared.storage.files import read, write
#: The reader's `listen` takes the built models and a path; this one a `Media`.
from .reader import listen as _pass
from .source import NoAudio


def listen(media: Media,
           transcriber: Optional[Transcriber] = None,
           diarizer: Optional[Diarizer] = None) -> RawTranscript:
    """Transcribe and diarize the file a `Media` names. Reads and writes no
    artifact.

    `transcriber` is a `Transcriber` (None: `WhisperTranscriber()`), `diarizer`
    a `Diarizer` (None: `PyannoteDiarizer()`; `NoDiarizer()` for none). Both are
    checked before the soundtrack is decoded.
    """
    transcriber = resolve("transcriber", transcriber)
    diarizer = resolve("diarizer", diarizer)
    require("transcriber", transcriber)
    require("diarizer", diarizer)
    if not media.has_audio:
        raise NoAudio(f"{media.video_id} has no audio stream")
    return _pass(media.path, transcriber, diarizer, video_id=media.video_id)


def audio(media: str | Path, out: str | Path,
          transcriber: Optional[Transcriber] = None,
          diarizer: Optional[Diarizer] = None) -> Produced:
    """Transcribe and diarize the whole file; writes no chunk ids. `listen` plus a
    read at each end. The audio decoded is the file at `Media.path`.
    """
    described = read(media, Media)
    with logs.timed("audio", described.video_id) as done:
        raw = listen(described, transcriber, diarizer)
        where = write(out, raw)
        done(segments=raw.stats.get("segments"), words=raw.stats.get("words"),
             speakers=raw.stats.get("speakers"), silent=raw.silent)
    return Produced(
        video_id=described.video_id, component="audio",
        artifacts={"raw_transcript": where},
        stats={**raw.stats, "silent": raw.silent},
        skipped=["transcribe", "diarize"] if raw.silent else [],
    )


def load(path: str | Path) -> RawTranscript:
    """Read a `transcript.raw.json` back, typed."""
    return read(path, RawTranscript)
