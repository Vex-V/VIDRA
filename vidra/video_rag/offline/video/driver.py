"""The ingest component: `media.json` + `timeline.json` -> `manifest.json` and
the frame store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from vidra.shared.reporting import logs, progress
from vidra.shared.contracts.documents import (Manifest, Media, Produced,
                                           Timeline)
from vidra.shared.storage.files import read, write
from vidra.shared.contracts.documents import same_video
from ...core.frames import FrameStore
from ...core.sampling.specs import build_samplers, split_specs
#: The pipeline's `ingest` takes built samplers; this module's takes a spec.
from .pipeline import ingest as _pass


def ingest(media: Media, timeline: Timeline,
           sampler: str | Sequence[Any] = "uniform",
           per_second: float = 1.0,
           every_n: Optional[int] = None,
           min_interval_s: float = 0.0,
           max_per_chunk: Optional[int] = None,
           threshold: Optional[float] = None,
           vocabulary: Optional[Sequence[str]] = None,
           confidence: Optional[float] = None,
           languages: Optional[Sequence[str]] = None,
           frames: Optional[FrameStore] = None,
           store_scope: str = "sampled",
           on_progress: Optional[progress.Reporter] = None) -> Manifest:
    """One decode pass over the picture, onto a given grid. Reads the file
    `media.path` names and writes no artifact.

    `frames` is where kept frames go (a `FrameStore`, `MemoryFrames`), or None to
    keep none. Questions are checked against the vocabulary before anything
    decodes.
    """
    from ...core.describe import prompts

    # Built before anything decodes, so a bad argument fails at once.
    built = build_samplers(split_specs(sampler), every_n, min_interval_s,
                           max_per_chunk, threshold, vocabulary, confidence,
                           languages, questions=prompts.questions())
    # Progress is reported per chunk.
    total = len(timeline)
    seen = 0

    def chunk_done(chunk_id: int, chunk: dict) -> None:
        nonlocal seen
        seen += 1
        progress.report(on_progress, "video", total, seen, 0,
                        str(chunk_id), chunk)

    progress.report(on_progress, "video", total, 0)
    return _pass(media, timeline, built, per_second, frames, store_scope,
                 chunk_done if on_progress is not None else None)


def video(media: str | Path, timeline: str | Path, out: str | Path,
          store: Optional[str | Path] = None,
          sampler: str | Sequence[Any] = "uniform",
          per_second: float = 1.0,
          every_n: Optional[int] = None,
          min_interval_s: float = 0.0,
          max_per_chunk: Optional[int] = None,
          threshold: Optional[float] = None,
          vocabulary: Optional[Sequence[str]] = None,
          confidence: Optional[float] = None,
          languages: Optional[Sequence[str]] = None,
          store_scope: str = "sampled",
          prune_store: bool = False,
          on_progress: Optional[progress.Reporter] = None) -> Produced:
    """Sample the picture onto the grid; write the manifest to `out`. `ingest` plus
    a read at each end. `store` is the frame directory; None keeps no frames.
    """
    described = read(media, Media)
    grid = read(timeline, Timeline)
    video_id = same_video(media=described, timeline=grid)

    frames = FrameStore(Path(store)) if store is not None else None
    if frames is None:
        logs.skipped("video", video_id, "store",
                     "no store= was given, so no pixels are kept")

    with logs.timed("video", video_id) as done:
        manifest = ingest(described, grid, sampler, per_second, every_n,
                          min_interval_s, max_per_chunk, threshold, vocabulary,
                          confidence, languages, frames=frames,
                          store_scope=store_scope, on_progress=on_progress)

        pruned: list[int] = []
        if frames is not None and prune_store:
            # After the pass: delete stored frames the new manifest does not name.
            named = {f["index"] for c in manifest.chunks
                     for b in c["samplers"].values() for f in b["frames"]}
            pruned = frames.prune(named)

        where = write(out, manifest)
        done(sampled=manifest.stats.get("frames_sampled"),
             stored=manifest.stats.get("stored_frames"),
             samplers=",".join(manifest.sampler_ids()), pruned=len(pruned))
    artifacts = {"manifest": where}
    if frames is not None and frames.written:
        artifacts["store"] = str(frames.root)

    return Produced(
        video_id=video_id, component="video", artifacts=artifacts,
        stats={**manifest.stats,
               "timeline_fingerprint": manifest.timeline_fingerprint,
               "manifest_fingerprint": manifest.fingerprint(),
               "samplers": manifest.sampler_ids(),
               "pruned_frames": len(pruned)},
        skipped=[] if frames is not None else ["store"],
    )


def load(path: str | Path) -> Manifest:
    """Read a `manifest.json` back, typed."""
    return read(path, Manifest)
