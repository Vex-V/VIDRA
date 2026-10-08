"""Documents into embeddable units.

One unit is one `(video_id, chunk_id, sampler_id)`. Both modalities produce
them: a description from the picture, a transcript chunk from the soundtrack
with `sampler_id = "transcript"`.

What gets embedded is the summary *and* the structured fields, rendered by
`render`: keys sorted at every level, fields named, and three separators, one
per level -- `. ` between fields, ` | ` between entities, `; ` between one
entity's attributes.
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
    """One unit per chunk that has speech, keyed `transcript`. Silent chunks
    are skipped.
    """
    units: list[Unit] = []
    for chunk in document.chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        # `speakers` only, never `turns`: their text is the transcript itself.
        structured = {"speakers": (chunk.get("structured") or {}).get("speakers", [])}
        units.append(Unit(document.video_id, chunk["chunk_id"], "transcript",
                          render(text, structured), structured,
                          sampler="transcript", question="transcript"))
    return units


__all__ = ["embed_all", "Unit", "from_descriptions", "from_transcript", "render"]
