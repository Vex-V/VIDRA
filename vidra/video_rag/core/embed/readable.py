"""`embedded.json` -- what `embed` produced, text and vectors together.

Rewritten whole each time, under one embedder, so it holds no stale unit.
`embed.load` reads it back.
"""

from __future__ import annotations

from typing import Sequence

from vidra.shared.contracts.documents import Embedded
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
                # The vector last, after the readable fields.
                "vector": list(u.vector) if u.vector else None} for u in ordered],
    )


__all__ = ["build"]
