"""`embedded.json` -- what `embed` produced, text and vectors together.

This is the component's output, not a readable copy of something else. While a
`VectorIndex` existed the vectors lived in Qdrant or Postgres and this file
deliberately held none of them; with the index gone it holds both, and that is
what lets a re-run resume without a database being reachable.

Rewritten whole each time, so it cannot hold a stale unit -- which is why
there is no `prune` here. An upserting table keeps whatever it was never told
to remove; a file that is replaced wholesale cannot.

One file per video, naming the embedder that made its vectors. Two embedders
coexist in Postgres, where every row carries the key and every query filters
on it, but not on disk: a file is replaced, not merged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from falconvar.shared.contracts.documents import Embedded
from falconvar.shared.storage.files import read, write as _write
from .units import Unit


def build(video_id: str, units: Sequence[Unit], embedder_key: str = "",
          timeline_fingerprint: str = "") -> Embedded:
    """The document, in chunk order so it reads top to bottom like the video."""
    ordered = sorted(units, key=lambda u: (u.chunk_id, u.sampler_id))
    return Embedded(
        video_id=video_id,
        timeline_fingerprint=timeline_fingerprint,
        embedder=embedder_key,
        units=[{"chunk_id": u.chunk_id,
                "sampler_id": u.sampler_id,
                "text_hash": u.text_hash,
                "characters": len(u.content),
                "content": u.content,
                "sampler": u.sampler, "question": u.question,
                "structured": u.structured,
                # Last, so the readable half of an entry stays readable: a
                # 1536-float list ahead of the text would bury it.
                "vector": list(u.vector) if u.vector else None} for u in ordered],
    )


def write(path: str | Path, video_id: str, units: Sequence[Unit],
          embedder_key: str = "", timeline_fingerprint: str = "") -> str:
    document = build(video_id, units, embedder_key, timeline_fingerprint)
    return _write(path, document)


def load(path: str | Path) -> Embedded:
    return read(path, Embedded)


__all__ = ["build", "load", "write"]
