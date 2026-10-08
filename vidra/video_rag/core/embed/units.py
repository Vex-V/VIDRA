"""Documents into embeddable units.

One unit is one `(video_id, chunk_id, sampler_id)`. Both modalities produce
them: a description from the picture, a transcript chunk from the soundtrack
with `sampler_id = "transcript"`.

What gets embedded is the summary *and* the structured fields -- every summary
repeats the same setting, the fields do not.

`render` sorts keys at every level. `jsonb` preserves array order but not object
key order, so a document read back from Postgres hands its keys back in a
different order from the file; joining values in iteration order made the same
person into different text, a different hash and a different vector depending
on where it was read from.

Three separators, one per level: `. ` between fields, ` | ` between entities,
`; ` between one entity's attributes. Fields are named, because a bare `", "`
also occurs inside values and made field boundaries invisible.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from vidra.shared.contracts.documents import Descriptions, Transcript
from vidra.shared.contracts.units import Unit, render


def embed_all(units: Sequence[Unit], embedder: Any, batch: int = 64,
              on_batch: Optional[Callable[[int, list[Unit]], None]] = None) -> None:
    """Give every unit its vector, `batch` texts per call. `on_batch(done,
    window)` is called after each call, with how many are done so far."""
    for start in range(0, len(units), batch):
        window = list(units[start:start + batch])
        for unit, vector in zip(window, embedder.embed([u.content for u in window])):
            unit.vector = vector
        if on_batch is not None:
            on_batch(start + len(window), window)


def from_descriptions(document: Descriptions) -> list[Unit]:
    """One unit per (chunk, sampler) that has an answer."""
    units: list[Unit] = []
    for chunk in document.chunks:
        for sampler_id, block in chunk.get("samplers", {}).items():
            structured = block.get("structured") or {}
            content = render(block.get("description", ""), structured)
            if not content:
                continue
            units.append(Unit(document.video_id, chunk["chunk_id"], sampler_id,
                              content, structured,
                              sampler=block.get("sampler")
                              or sampler_id.split(":")[0],
                              question=block.get("question") or sampler_id))
    return units


def from_transcript(document: Transcript) -> list[Unit]:
    """One unit per chunk that has speech, keyed `transcript`.

    Silent chunks are skipped -- they stay in the transcript document because
    `chunk_id` is shared with the video side, but a vector of the empty string
    answers every query equally badly.
    """
    units: list[Unit] = []
    for chunk in document.chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        # `speakers` only, never `turns`: `turns[].text` *is* the transcript,
        # so rendering it appends the whole chunk a second time interleaved
        # with timestamps read as numbers.
        structured = {"speakers": (chunk.get("structured") or {}).get("speakers", [])}
        units.append(Unit(document.video_id, chunk["chunk_id"], "transcript",
                          render(text, structured), structured,
                          sampler="transcript", question="transcript"))
    return units


__all__ = ["embed_all", "Unit", "from_descriptions", "from_transcript", "render"]
