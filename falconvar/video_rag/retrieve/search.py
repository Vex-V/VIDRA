"""A query, into ranked moments.

RRF twice: an index fuses vector and text rankings per unit, and this fuses the
units of one chunk into a moment.

    score = 1/(k + best) + 0.5/(k + second),  k = 10

A chunk scores as its best unit plus a discounted second, never a sum. Summing
over every unit a chunk contributed applies RRF to the wrong problem -- it
fuses several rankings of the *same* items, where the term count is constant,
while a chunk contributes one term per sampler that described it. Count then
overwhelms rank.

The bounded form keeps the agreement property on purpose: two accounts at ranks
2 and 3 still beat one at rank 1, which is why the per-sampler split exists.
A third and fourth account add nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

#: Small on purpose. At k=60 the reciprocal barely varies across a short
#: candidate list, which is what let count beat rank.
MOMENT_K = 10

#: What a second, independent account of the same window is worth.
SECOND_WEIGHT = 0.5


@dataclass
class Moment:
    """One chunk, and the units that spoke for it."""

    #: Which video. Part of the key: a chunk id indexes *one* video's grid,
    #: and grouping on the id alone fused chunk 0 of two videos into one
    #: moment with two unrelated accounts.
    video_id: str
    #: Which chunk of that video's grid.
    chunk_id: int
    #: Where the chunk starts, in media seconds.
    start_ts: float
    #: Where it ends, in media seconds.
    end_ts: float
    #: A rank fusion, `1/(k+best) + 0.5/(k+second)` at k=10 -- NOT a
    #: similarity. 0.1326 is the ceiling for a chunk contributing two units
    #: and means "best ranked first, second ranked second". A nonsense query
    #: scores like a real one, because dense always returns nearest
    #: neighbours and there is no relevance floor. Read the ranks, not this.
    score: float
    #: One entry per unit that matched, each a dict carrying `sampler_id`
    #: (the pairing, e.g. `clip:text`), `sampler`, `question`, `content`,
    #: `structured`, `dense_rank` and `text_rank`. `text_rank: None` means
    #: the lexical half was silent on this query.
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
                # Keyed by the pairing, with the two halves beside it, so a
                # caller can group by sampler or by question without parsing
                # an id whose separator is optional.
                "questions": {h["sampler_id"]: h.get("question", "")
                              for h in self.hits},
                "descriptions": {h["sampler_id"]: h["content"] for h in self.hits},
                # The ranks each half gave each unit. Without them a score is
                # uninterpretable: dense always returns nearest neighbours, so
                # a nonsense query scores like a real one -- measured at 0.1146
                # against 0.1294 on this corpus. `text: null` is how a reader
                # sees the lexical half was silent, which is the distinction
                # the number alone cannot carry.
                "ranks": {h["sampler_id"]: {"dense": h.get("dense_rank"),
                                            "text": h.get("text_rank")}
                          for h in self.hits},
                "structured": {h["sampler_id"]: h.get("structured", {})
                               for h in self.hits}}


def to_moments(hits: Sequence[dict[str, Any]],
               video_id: str,
               spans: Any,
               limit: int = 5) -> list[Moment]:
    """Fuse per-unit hits into per-chunk moments.

    **Grouped by `(video_id, chunk_id)`, never by `chunk_id` alone.** Chunk ids
    are indexes into one video's grid, so they are only unique within a video:
    searching two videos at once and grouping on the id would fuse chunk 0 of
    one with chunk 0 of the other into a single "moment" with two unrelated
    accounts of two different clips -- and the agreement bonus would fire on
    them, scoring the collision *above* either real answer. Invisible while the
    scope was one video, which is exactly why it was written that way.

    ``spans`` is a per-video map when the scope is several videos, and a plain
    list when it is one; a hit that carries its own span uses that instead.
    """
    # Deterministic, because RRF ties are common and were being broken by
    # whatever order the backend happened to return.
    #
    # Two ranks fuse to exactly the same score whenever they are symmetric:
    # dense 1 / text 2 and dense 2 / text 1 are both 1/61 + 1/62. Measured on
    # this corpus, that tie alone decided a case the harness scored 1.00
    # against 0.50 -- the two backends broke it opposite ways and neither was
    # more right. Dense rank is the tie-break because it is the half with an
    # opinion on every query; the lexical half is silent on some, so a tie
    # broken by it would be broken by nothing at all.
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
        # The backend's own span when it has one -- the Postgres RPC joins
        # `chunks` and returns it -- falling back to the grid this was handed.
        first = entries[0][1]
        own = spans_for(vid)
        if first.get("start_ts") is not None and first.get("end_ts") is not None:
            start, end = float(first["start_ts"]), float(first["end_ts"])
        else:
            start, end = (own[chunk_id] if chunk_id < len(own) else (0.0, 0.0))
        moments.append(Moment(vid, chunk_id, start, end, score,
                              [hit for _, hit in entries]))
    # Same reason, one level up: equal moment scores are common on a small
    # corpus, and an arbitrary order there is an arbitrary answer.
    moments.sort(key=lambda m: (-m.score, m.video_id, m.chunk_id))
    return moments[:limit]


__all__ = ["MOMENT_K", "SECOND_WEIGHT", "Moment", "to_moments"]
