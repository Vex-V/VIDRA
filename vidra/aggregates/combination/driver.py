"""Several videos' documents -> one record in the same format, to aggregate over.

    week = combine(records=[monday, tuesday], out="data/out/week")

Writes a `Timeline`, `Descriptions`, `Transcript` and `Manifest` under the
usual filenames and returns them as a record, so every aggregator reads it
as it reads a video. Grids are
laid end to end: chunks renumbered after the previous video's, times shifted
onto one clock. `Timeline.params["combined"]` lists each source (its id,
first chunk, chunk count and clock offset); `origin` maps a combined chunk
back. Speakers are prefixed with their video (`monday:SPEAKER_00`). One video
named twice is refused.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from ...shared.reporting import logs
from ...shared.config import paths
from ...shared.contracts.documents import (Descriptions, Manifest, Timeline,
                                           Transcript, fingerprint_of, same_video)
from ...shared.reporting.errors import VidraError
from ...shared.storage import files
#: The documents a combination carries.
from ..core.record import KINDS, Record

#: Manifest stats that are counts, and so add up across videos.
_COUNTS = ("frames_decimated", "frames_sampled", "chunks", "chunks_with_frames",
           "stored_frames", "stored_mb", "elapsed_s")


class CombineError(VidraError, ValueError):
    """Sources that cannot be laid end to end."""


@dataclass
class Part:
    """One video's documents. Only the grid is required."""

    timeline: Timeline
    descriptions: Optional[Descriptions] = None
    transcript: Optional[Transcript] = None
    manifest: Optional[Manifest] = None

    @property
    def video_id(self) -> str:
        return self.timeline.video_id


@dataclass
class Combined:
    """What `merge` returns: the four documents, any of the last three None."""

    timeline: Timeline
    descriptions: Optional[Descriptions] = None
    transcript: Optional[Transcript] = None
    manifest: Optional[Manifest] = None

    def documents(self) -> dict[str, Any]:
        return {kind: getattr(self, kind) for kind in KINDS
                if getattr(self, kind) is not None}


def default_id(video_ids: Sequence[str]) -> str:
    """An id derived from the sources: the same videos in the same order give the
    same id.
    """
    return "combined-" + fingerprint_of({"videos": list(video_ids)})[:8]


def merge(parts: Sequence[Part], video_id: Optional[str] = None) -> Combined:
    """Lay the parts end to end. Reads and writes nothing. `video_id` names the
    combination; by default `default_id` of the sources.
    """
    if not parts:
        raise CombineError("nothing to combine: name at least one video")
    ids = [p.video_id for p in parts]
    twice = sorted({v for v in ids if ids.count(v) > 1})
    if twice:
        raise CombineError(f"{', '.join(twice)} named more than once; every "
                           f"count in the combination would be doubled")
    for part in parts:
        same_video(timeline=part.timeline, descriptions=part.descriptions,
                   transcript=part.transcript, manifest=part.manifest)
    name = paths.check_id(video_id or default_id(ids))

    # 1 · the grid, end to end, with where each source landed.
    spans: list[tuple[float, float]] = []
    sources: list[dict[str, Any]] = []
    first: list[int] = []
    offsets: list[float] = []
    clock = 0.0
    for part in parts:
        grid = part.timeline
        first.append(len(spans))
        offsets.append(clock)
        sources.append({"video_id": part.video_id, "first_chunk": len(spans),
                        "chunks": len(grid), "offset_s": round(clock, 3),
                        "duration_s": round(grid.duration_s, 3),
                        "policy": grid.policy,
                        "timeline_fingerprint": grid.fingerprint()})
        spans += [(start + clock, end + clock) for start, end in grid.spans]
        clock += grid.duration_s
    policies = {p.timeline.policy for p in parts}
    timeline = Timeline(video_id=name, spans=spans,
                        policy=policies.pop() if len(policies) == 1 else "mixed",
                        params={"combined": sources}, derived_from="combined")
    grid_fp = timeline.fingerprint()

    def renumbered(chunks: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
        out = []
        for chunk in chunks:
            moved = copy.deepcopy(chunk)
            moved["chunk_id"] = chunk["chunk_id"] + first[index]
            out.append(moved)
        return out

    def by_source(kind: str) -> dict[str, Any]:
        """Each source's own `model` and `stats`, keyed by video."""
        return {p.video_id: {"model": getattr(getattr(p, kind), "model", None),
                             "stats": getattr(p, kind).stats}
                for p in parts if getattr(p, kind) is not None}

    # 2 · descriptions: renumbered; sampler ids unchanged.
    described = [(i, p) for i, p in enumerate(parts) if p.descriptions is not None]
    descriptions = None
    if described:
        descriptions = Descriptions(
            video_id=name, timeline_fingerprint=grid_fp,
            manifest_fingerprint=fingerprint_of(
                {p.video_id: p.descriptions.manifest_fingerprint for _, p in described}),
            model={"combined": by_source("descriptions")},
            chunks=[c for i, p in described
                    for c in renumbered(p.descriptions.chunks, i)],
            stats={"combined": {v: s["stats"] for v, s in by_source("descriptions").items()},
                   "described": sum(p.descriptions.stats.get("described", 0)
                                    for _, p in described)})

    # 3 · transcript: every chunk of the grid, on the combined clock, speakers
    # prefixed with their video.
    heard = [p for p in parts if p.transcript is not None]
    transcript = None
    if heard:
        chunks: list[dict[str, Any]] = []
        for index, part in enumerate(parts):
            if part.transcript is None:
                chunks += [{"chunk_id": first[index] + c, "text": "", "word_count": 0,
                            "structured": {"speakers": []}, "turns": []}
                           for c in range(len(part.timeline))]
                continue
            for chunk in renumbered(part.transcript.chunks, index):
                for turn in chunk.get("turns") or []:
                    for edge in ("start", "end"):
                        if isinstance(turn.get(edge), (int, float)):
                            turn[edge] = round(turn[edge] + offsets[index], 3)
                    if turn.get("speaker"):
                        turn["speaker"] = f"{part.video_id}:{turn['speaker']}"
                structured = chunk.get("structured") or {}
                if structured.get("speakers"):
                    structured["speakers"] = [f"{part.video_id}:{s}"
                                              for s in structured["speakers"]]
                chunks.append(chunk)
        transcript = Transcript(
            video_id=name, timeline_fingerprint=grid_fp,
            model={"combined": by_source("transcript")}, chunks=chunks,
            stats={"chunks": len(chunks),
                   "chunks_with_speech": sum(1 for c in chunks if c.get("word_count")),
                   "words": sum(c.get("word_count", 0) for c in chunks),
                   "speakers": sum(p.transcript.stats.get("speakers", 0) for p in heard),
                   "combined": {v: s["stats"] for v, s in by_source("transcript").items()}})

    # 4 · manifest: chunks renumbered, counts added; each frame keeps its source.
    ingested = [(i, p) for i, p in enumerate(parts) if p.manifest is not None]
    manifest = None
    if ingested:
        stats: dict[str, Any] = {}
        for key in _COUNTS:
            values = [p.manifest.stats[key] for _, p in ingested if key in p.manifest.stats]
            if values:
                stats[key] = round(sum(values), 3)
        manifest = Manifest(
            video_id=name, timeline_fingerprint=grid_fp,
            source={"combined": {p.video_id: p.manifest.source for _, p in ingested}},
            config={"combined": {p.video_id: p.manifest.config for _, p in ingested}},
            stats=stats,
            chunks=[c for i, p in ingested for c in renumbered(p.manifest.chunks, i)])

    return Combined(timeline, descriptions, transcript, manifest)


def origin(timeline: Timeline, chunk_id: int) -> Optional[dict[str, Any]]:
    """Which video a combined chunk came from, and its own id and clock there.

    None for a grid that is not a combination, or an id outside it.
    """
    for source in (timeline.params or {}).get("combined") or []:
        local = chunk_id - source["first_chunk"]
        if 0 <= local < source["chunks"]:
            start, end = timeline.bounds_of(chunk_id)
            return {"video_id": source["video_id"], "chunk_id": local,
                    "start_ts": round(start - source["offset_s"], 3),
                    "end_ts": round(end - source["offset_s"], 3)}
    return None


def combine(*, records: Sequence["Record"], out: str | Path,
            video_id: Optional[str] = None) -> "Record":
    """Several records laid end to end on one clock: `merge` with a read at each
    end. The combination is written to `out` under the usual filenames and
    returned as a record, which takes the place of any one video's.
    """
    from ..core.record import record

    parts = [Part(**{kind: files.read(where, KINDS[kind])
                     for kind, where in r.paths.items()}) for r in records]
    with logs.timed("combine") as done:
        combined = merge(parts, video_id)
        folder = Path(out)
        written = {kind: files.write(folder / paths.ARTIFACTS[kind], document)
                   for kind, document in combined.documents().items()}
        done(video_id=combined.timeline.video_id, sources=len(parts),
             chunks=len(combined.timeline))
    return record(**{kind: Path(where) for kind, where in written.items()})


__all__ = ["CombineError", "Combined", "KINDS", "Part", "combine", "default_id",
           "merge", "origin"]

