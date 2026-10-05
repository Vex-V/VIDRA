"""A query, into ranked moments.

Units are fused into one moment per chunk:

    score = 1/(k + best) + 0.5/(k + second),  k = 10

a chunk scores as its best unit plus a discounted second, never a sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

#: The RRF constant for fusing a chunk's units.
MOMENT_K = 10

#: What a second, independent account of the same window is worth.
SECOND_WEIGHT = 0.5


@dataclass
class Moment:
    """One chunk, and the units that spoke for it."""

    #: Which video. Moments are keyed by (video, chunk).
    video_id: str
    #: Which chunk of that video's grid.
    chunk_id: int
    #: Where the chunk starts, in media seconds.
    start_ts: float
    #: Where it ends, in media seconds.
    end_ts: float
    #: A rank fusion, `1/(k+best) + 0.5/(k+second)` at k=10 -- not a similarity.
    #: Read the ranks on each hit.
    score: float
    #: One entry per unit that matched: `sampler_id`, `sampler`, `question`,
    #: `content`, `structured`, `dense_rank` and `text_rank` (None when the lexical
    #: half was silent).
    hits: list[dict[str, Any]] = field(default_factory=list)

    @property
    def samplers(self) -> list[str]:
        return [h["sampler_id"] for h in self.hits]

    def as_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id, "chunk_id": self.chunk_id,
                "start_ts": round(self.start_ts, 3),
                "end_ts": round(self.end_ts, 3),
                "score": round(self.score, 6),
                "samplers": self.samplers,
                # The question behind each pairing.
                "questions": {h["sampler_id"]: h.get("question", "")
                              for h in self.hits},
                "descriptions": {h["sampler_id"]: h["content"] for h in self.hits},
                # The ranks each half gave each unit.
                "ranks": {h["sampler_id"]: {"dense": h.get("dense_rank"),
                                            "text": h.get("text_rank")}
                          for h in self.hits},
                "structured": {h["sampler_id"]: h.get("structured", {})
                               for h in self.hits}}


def to_moments(hits: Sequence[dict[str, Any]],
               video_id: str,
               spans: Any,
               limit: int = 5) -> list[Moment]:
    """Fuse per-unit hits into per-chunk moments, grouped by (video_id, chunk_id).
    `spans` is a per-video map for several videos, a plain list for one; a hit
    carrying its own span uses that.
    """
    # Deterministic order; ties broken by the dense rank.
    ranked = sorted(hits, key=lambda h: (
        -h["score"],
        h.get("dense_rank") if h.get("dense_rank") is not None else 1 << 30,
        h.get("video_id") or "",
        h.get("chunk_id", 0),
        h.get("sampler_id", ""),
    ))
    by_chunk: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = {}
    for rank, hit in enumerate(ranked, start=1):
        key = (hit.get("video_id") or video_id, hit["chunk_id"])
        by_chunk.setdefault(key, []).append((rank, hit))

    def spans_for(vid: str) -> Sequence[tuple[float, float]]:
        if isinstance(spans, dict):
            return spans.get(vid) or []
        return spans or []

    moments: list[Moment] = []
    for (vid, chunk_id), entries in by_chunk.items():
        entries.sort(key=lambda p: p[0])
        best = entries[0][0]
        score = 1.0 / (MOMENT_K + best)
        if len(entries) > 1:
            score += SECOND_WEIGHT / (MOMENT_K + entries[1][0])
        # The backend's own span when it has one, else the grid.
        first = entries[0][1]
        own = spans_for(vid)
        if first.get("start_ts") is not None and first.get("end_ts") is not None:
            start, end = float(first["start_ts"]), float(first["end_ts"])
        else:
            start, end = (own[chunk_id] if chunk_id < len(own) else (0.0, 0.0))
        moments.append(Moment(vid, chunk_id, start, end, score,
                              [hit for _, hit in entries]))
    # Equal scores in a stable order.
    moments.sort(key=lambda m: (-m.score, m.video_id, m.chunk_id))
    return moments[:limit]


__all__ = ["MOMENT_K", "SECOND_WEIGHT", "Moment", "to_moments"]
