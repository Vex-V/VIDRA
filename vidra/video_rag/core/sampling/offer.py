"""Offering decimated frames to samplers, chunk by chunk: what both pipelines
do with every frame they decode, and the record a manifest keeps of it.

    offer = Offer(samplers)
    for frame in frames:
        chunk = chunks[chunk_id]
        kept = offer(frame, chunk_id, chunk)      # the samplers that kept it

A sampler is reset whenever the chunk changes, and asked about each frame with
its index inside the chunk. Every keep is recorded under the chunk, in the
manifest's shape.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .reader import Frame
from .samplers.base import Sampler


def empty_chunk(chunk_id: int) -> dict[str, Any]:
    """A manifest chunk no frame has reached yet."""
    return {"chunk_id": chunk_id, "decimated_frames": 0, "samplers": {}}


class Offer:
    """Every decimated frame to every sampler, resetting them at each chunk."""

    def __init__(self, samplers: Sequence[Sampler]) -> None:
        self.samplers = list(samplers)
        #: The chunk being offered, and the next frame's index inside it.
        self.chunk_id: Optional[int] = None
        self.local = 0
        #: Frames offered, and keeps across every sampler.
        self.decimated = 0
        self.sampled = 0

    def __call__(self, frame: Frame, chunk_id: int, chunk: dict[str, Any],
                 **noted: Any) -> list[Sampler]:
        """Offer one frame; record each keep in `chunk`, with `noted` added to
        its record, and return the samplers that kept it."""
        if chunk_id != self.chunk_id:
            self.chunk_id = chunk_id
            self.local = 0
            for sampler in self.samplers:
                sampler.reset(chunk_id)
        self.decimated += 1
        chunk["decimated_frames"] += 1
        kept = []
        for sampler in self.samplers:
            if not sampler.accepts(frame, self.local):
                continue
            record: dict[str, Any] = {"index": frame.index,
                                      "media_ts": round(frame.media_ts, 3),
                                      "chunk_local_index": self.local, **noted}
            # The exact address of the frame.
            if frame.pts is not None:
                record["pts"] = frame.pts
            score = sampler.last_score()
            if score is not None:
                record["score"] = round(score, 4)
            block = chunk["samplers"].setdefault(
                sampler.sampler_id, {"frame_count": 0, "frames": []})
            block["frames"].append(record)
            block["frame_count"] += 1
            kept.append(sampler)
        self.sampled += len(kept)
        self.local += 1
        return kept


__all__ = ["Offer", "empty_chunk"]
