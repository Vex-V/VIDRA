"""Several videos' documents -> one record in the same format, to aggregate over.

    combine(["data/out/monday", "data/out/tuesday"], "data/out/week")
    aggregate("data/out/week/timeline.json", "data/out/week/aggregates",
              descriptions="data/out/week/descriptions.json", ...)

**No aggregator learns what a collection is.** Every one of them reads a
`Context` -- a grid plus the documents cut on it -- and answers about the chunk
ids it finds there. So rather than teaching fourteen aggregators to hold a list
of videos, this builds one more record that looks exactly like a video's: a
`Timeline`, `Descriptions`, `Transcript` and `Manifest`, each the same
dataclass, written under the same filenames. The aggregates then run over it
unchanged -- a summary of the week, entities across both days, stats totalled.

**The grid is laid end to end.** Source `i`'s chunks keep their order and are
renumbered after source `i-1`'s; their times are shifted by the durations
before them. So the combined record has one clock, a chunk id means one thing
in every document, and `span_of` stays arithmetic. The clock is a *virtual*
one: second 312 of the combination is second 12 of the second video, and
`origin` is how a reader turns one into the other.

**Provenance lives in the grid, once.** `Timeline.params["combined"]` lists
each source -- its video id, where its chunks start, how many, where its clock
starts, and the fingerprint of the grid it came from. `params` is part of
`Timeline.fingerprint`, so the provenance is part of the grid's identity: two
combinations of different videos can never share a fingerprint, and every
document cut on the combination records it as usual. Nothing is copied onto
each chunk, because a second copy of "which video" is one that can disagree.

**A speaker is not the same person in two recordings.** Diarization labels are
per recording -- `SPEAKER_00` in one file bears no relation to `SPEAKER_00` in
the next -- so every label is prefixed with its video (`monday:SPEAKER_00`).
Otherwise `speakers` would add two strangers' talk time together and count no
handover between them.

**Entities over a combination link people across the videos**, because the
linker sees every mention at once and the cannot-link rule is per answer, which
a combination keeps. Measured on test + test2 and test + test1 -- videos with
no one in common -- pooling every mention made 2-6 cross-video groups each
(100-311 wrong pairs), and on test2 + test3, the same footage described twice,
it found most of the true cross-video matches with 4-12 wrong pairs. It links
lookalikes across videos as readily as across chunks.

**Refused rather than guessed:** one video named twice (every count would be
doubled), a document that is not its own grid's video, and a combination whose
sources carry none of a document -- `transcript` is simply absent then, as it
is for a silent video.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..shared import logs, paths
from ..shared.contracts.documents import (Descriptions, Manifest, Produced,
                                          Timeline, Transcript, fingerprint_of,
                                          same_video)
from ..shared.errors import FalconvarError
from ..shared.storage import files
#: The documents a combination carries: a record's four, the ones an
#: aggregator reads. `media`, `cuts`, `store` and `embedded` describe a file or
#: feed a search, and a combination is neither.
from .record import KINDS, documents_of

#: Manifest stats that are counts, and so add up across videos.
_COUNTS = ("frames_decimated", "frames_sampled", "chunks", "chunks_with_frames",
           "stored_frames", "stored_mb", "elapsed_s")


class CombineError(FalconvarError, ValueError):
    """Sources that cannot be laid end to end honestly."""


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
    """A name no single video has: stable for the same sources in the same
    order, and inside `paths.check_id`'s alphabet."""
    return "combined-" + fingerprint_of({"videos": list(video_ids)})[:8]


def merge(parts: Sequence[Part], video_id: Optional[str] = None) -> Combined:
    """Lay the parts end to end. Reads and writes nothing.

    `video_id` names the combination; by default one derived from the sources,
    so the same videos in the same order always combine to the same id.
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
        """What each source said about itself, keyed by video: a combination's
        `model` and `stats` are not one model's or one run's."""
        return {p.video_id: {"model": getattr(getattr(p, kind), "model", None),
                             "stats": getattr(p, kind).stats}
                for p in parts if getattr(p, kind) is not None}

    # 2 · descriptions: renumbered, answers untouched. The sampler ids stay as
    #     they were, so an input naming `yolo` reads every video's yolo.
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

    # 3 · transcript: every chunk of the grid, as a transcript always has, so
    #     a video with no soundtrack contributes silent chunks rather than a
    #     hole. Times move onto the combined clock; speakers get their video.
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

    # 4 · manifest: chunks renumbered, counts added. Each frame still names
    #     its own video's store by read index, which is why its source is kept.
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


def combine(sources: Sequence[str | Path | Mapping[str, str | Path]],
            out: str | Path, video_id: Optional[str] = None) -> Produced:
    """`merge` with a read at each end.

    Each source is a video's folder -- the documents found there by their usual
    names -- or `{"timeline": ..., "descriptions": ..., ...}` naming them.
    `out` is the folder the combination is written to, under the same
    filenames, so `aggregate` reads it exactly as it reads a video.
    """
    parts = []
    for source in sources:
        found = documents_of(source)
        if "timeline" not in found:
            raise CombineError(f"{source}: no timeline -- a source needs its grid")
        parts.append(Part(**{kind: files.read(where, KINDS[kind])
                             for kind, where in found.items()}))
    with logs.timed("combine") as done:
        combined = merge(parts, video_id)
        folder = Path(out)
        written = {kind: files.write(folder / paths.ARTIFACTS[kind], document)
                   for kind, document in combined.documents().items()}
        done(video_id=combined.timeline.video_id, sources=len(parts),
             chunks=len(combined.timeline))
    return Produced(
        video_id=combined.timeline.video_id, component="combine",
        artifacts=written,
        stats={"sources": [p.video_id for p in parts],
               "chunks": len(combined.timeline),
               "duration_s": round(combined.timeline.duration_s, 3),
               "documents": sorted(written),
               "out": str(folder)},
        skipped=sorted(set(KINDS) - set(written)))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Lay several videos' documents end to end as one record, "
                    "for the aggregates to run over.")
    ap.add_argument("sources", nargs="+", help="each video's output folder")
    ap.add_argument("--out", required=True, help="the folder to write the combination to")
    ap.add_argument("--video-id", default=None,
                    help="what to call the combination; default derived from "
                         "the sources")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        produced = combine(args.sources, args.out, args.video_id)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1
    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0
    s = produced.stats
    print(f"{produced.video_id}   {len(s['sources'])} videos, {s['chunks']} chunks, "
          f"{s['duration_s']} s")
    print(f"  from        {', '.join(s['sources'])}")
    print(f"  documents   {', '.join(s['documents'])}")
    if produced.skipped:
        print(f"  none of     {', '.join(produced.skipped)}")
    print(f"\ncombined -> {s['out']}")
    return 0


__all__ = ["CombineError", "Combined", "KINDS", "Part", "combine", "default_id",
           "merge", "origin"]


if __name__ == "__main__":
    raise SystemExit(main())
