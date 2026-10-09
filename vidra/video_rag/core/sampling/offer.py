"""Offering decimated frames to samplers, chunk by chunk: what both pipelines
do with every frame they decode, and the record a manifest keeps of it.

    offer = Offer(samplers)
    for frame in frames:
        chunk = chunks[chunk_id]
        kept = offer(frame, chunk_id, chunk)      # the samplers that kept it

A sampler is reset whenever the chunk changes, and asked about each frame with
its index inside the chunk. Every keep is recorded under the chunk, in the
manifest's shape, with the sampler's views of it when they are not just the
frame: `{"frame": true}`, or `{"view": n, "label", "meta"}` for the n-th image
made from it. `shown` holds each keeping sampler's views of the last frame.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from .reader import Frame
from .samplers.base import Sampler, View
from vidra.shared.reporting.errors import Refused


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
        #: Frames offered, keeps across every sampler, and images made from kept frames.
        self.decimated = 0
        self.sampled = 0
        self.views = 0
        #: sampler id -> its views of the last frame offered, for each sampler that kept it.
        self.shown: dict[str, list[View]] = {}

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
        self.shown = {}
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
            views = _checked(sampler, sampler.views(frame))
            self.shown[sampler.sampler_id] = views
            if not (len(views) == 1 and views[0].is_frame):
                record["views"] = described(views)
                self.views += sum(1 for v in views if not v.is_frame)
            block = chunk["samplers"].setdefault(
                sampler.sampler_id, {"frame_count": 0, "frames": []})
            block["frames"].append(record)
            block["frame_count"] += 1
            kept.append(sampler)
        self.sampled += len(kept)
        self.local += 1
        return kept


def described(views: Sequence[View]) -> list[dict[str, Any]]:
    """Views as a manifest records them; made images numbered from 1 in order."""
    out: list[dict[str, Any]] = []
    made = 0
    for view in views:
        if view.is_frame:
            out.append({"frame": True})
            continue
        made += 1
        out.append({"view": made, "label": view.label,
                    **({"meta": view.meta} if view.meta else {})})
    return out


def made(views: Sequence[View]) -> list[tuple[int, View]]:
    """`(n, view)` for each image made from the frame, numbered as `described` does."""
    return list(enumerate((v for v in views if not v.is_frame), start=1))


def _checked(sampler: Sampler, views: Any) -> list[View]:
    """A sampler's views, refused unless they are views the store and the
    model can take."""
    who = f"{sampler.sampler_id}.views()"
    if not isinstance(views, list) or not views:
        raise Refused(f"{who} must return a non-empty list of View, "
                      f"not {views!r:.80}")
    for view in views:
        if not isinstance(view, View):
            raise Refused(f"{who} returned a {type(view).__name__}, not a View")
        if view.is_frame:
            continue
        shape = getattr(view.image, "shape", None)
        if shape is None or len(shape) not in (2, 3) or 0 in shape[:2]:
            raise Refused(f"{who}: a view's image must be a non-empty BGR array")
        if not view.label.strip():
            raise Refused(f"{who}: a view needs a label, which the model reads "
                          f"beside it (\"person 1 of 3, enlarged\")")
        try:
            json.dumps(view.meta)
        except (TypeError, ValueError):
            raise Refused(f"{who}: a view's meta must be JSON") from None
    return views


__all__ = ["Offer", "described", "empty_chunk", "made"]
