"""The glance component: `manifest.json` + `store/` -> `glances.json`.

Each sampler run's kept frames in a chunk (and the views it made of them) go
to a visual embedder as one group, and its vector is the unit -- no model
answer in between. The document is `embedded.json`'s shape, under the visual
embedder's key plus `:images`, so a database stores and searches it as
moments: rows beside the text units, never ranked with them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from vidra.shared.contracts.documents import (Embedded, Manifest, Produced,
                                           Timeline, fingerprint_of, same_video)
from vidra.shared.contracts.units import IMAGE_QUESTION
from vidra.shared.models.base import image, require, text
from vidra.shared.reporting import logs, progress
from vidra.shared.reporting.errors import Refused
from vidra.shared.storage.files import maybe, read, write
from ...core.describe.frames import FrameSource, store_of
from ...core.frames import FrameStore

if TYPE_CHECKING:
    from vidra.shared.models.base import VisualEmbedder

#: Appended to a visual embedder's key to name the space its units are stored
#: in, so a model that also embeds text never replaces its own text rows.
SUFFIX = ":images"


def space_of(visual_embedder: "VisualEmbedder") -> str:
    """The key a visual embedder's units are stored and searched under."""
    return f"{visual_embedder.key}{SUFFIX}"


def look(manifest: Manifest, timeline: Timeline, frames: FrameStore,
         visual_embedder: Optional["VisualEmbedder"] = None,
         samplers: Optional[Sequence[str]] = None,
         previous: Optional[Embedded] = None,
         batch: int = 16,
         on_progress: Optional[progress.Reporter] = None) -> Embedded:
    """One vector per (chunk, sampler run), over documents and frames in hand.
    Reads and writes no artifact.

    `frames` is the store ingest wrote. `samplers` names the runs to embed
    (None is every one). `previous` is an earlier `glances.json` document:
    units whose images are unchanged under the same model keep their vectors.
    `batch` is units per call to the model.
    """
    if visual_embedder is None:
        raise Refused("glance needs a visual_embedder; there is no default -- pass "
                      "one, e.g. LocalVisualEmbedder()")
    require("visual_embedder", visual_embedder)
    if batch < 1:
        raise Refused(f"batch must be at least 1, not {batch}")
    if manifest.timeline_fingerprint != timeline.fingerprint():
        raise Refused(
            f"{manifest.video_id}: the manifest was built on a different grid "
            f"({manifest.timeline_fingerprint} vs {timeline.fingerprint()}). "
            "Re-run ingest against the current timeline.")

    runs = manifest.sampler_ids()
    if samplers is not None:
        unknown = sorted(set(samplers) - set(runs))
        if unknown:
            raise Refused(f"the manifest has no sampler run {', '.join(unknown)}; "
                          f"it has {', '.join(runs)}")
        runs = [r for r in runs if r in samplers]

    planned = [(chunk["chunk_id"], run_id, block["frames"])
               for chunk in manifest.chunks
               for run_id, block in (chunk.get("samplers") or {}).items()
               if run_id in runs]

    # Refused before any image is read, not cut to fit.
    most = getattr(visual_embedder, "max_images", None)
    if most is not None:
        over = [(c, r, n) for c, r, kept in planned
                if (n := sum(len(f.get("views") or [None]) for f in kept)) > most]
        if over:
            chunk_id, run_id, count = max(over, key=lambda o: o[2])
            raise Refused(
                f"{len(over)} unit(s) hold more images than {visual_embedder!r} takes "
                f"({most}), the most {count} (chunk {chunk_id}, {run_id}); keep fewer "
                f"frames (max_per_chunk) or make fewer views")

    space = space_of(visual_embedder)
    earlier: dict[str, dict[str, Any]] = {}
    if previous is not None and previous.embedder == space:
        earlier = {f"{u['chunk_id']}/{u['sampler_id']}": u
                   for u in previous.units if u.get("vector")}

    units: list[dict[str, Any]] = []
    groups: dict[int, list[dict[str, Any]]] = {}
    with FrameSource(frames, manifest) as source:
        for chunk_id, run_id, kept in planned:
            loaded = source.images_for(chunk_id, run_id)
            parts: list[dict[str, Any]] = []
            for shown in loaded:
                if shown.label:
                    parts.append(text(shown.label))
                parts.append(image(shown.jpeg))
            digest = fingerprint_of({
                "images": [hashlib.sha256(f.jpeg).hexdigest() for f in loaded],
                "labels": [f.label for f in loaded]})
            unit = {"chunk_id": chunk_id, "sampler_id": f"{run_id}:{IMAGE_QUESTION}",
                    "text_hash": digest, "characters": 0, "content": "",
                    "sampler": run_id, "question": IMAGE_QUESTION,
                    "structured": {"frames": sorted({f["index"] for f in kept})},
                    "images": len(loaded), "vector": None}
            stored = earlier.get(f"{chunk_id}/{unit['sampler_id']}")
            if stored is not None and stored.get("text_hash") == digest:
                unit["vector"] = stored["vector"]
            else:
                groups[len(units)] = parts
            units.append(unit)

    pending = list(groups)
    reused = len(units) - len(pending)
    progress.report(on_progress, "glance", len(pending), 0, reused)
    for start in range(0, len(pending), batch):
        window = pending[start:start + batch]
        vectors = visual_embedder.embed_images([groups[i] for i in window])
        if len(vectors) != len(window):
            raise Refused(f"{visual_embedder!r} returned {len(vectors)} vectors for "
                          f"{len(window)} groups")
        for i, vector in zip(window, vectors):
            units[i]["vector"] = [float(v) for v in vector]
        done = start + len(window)
        progress.report(on_progress, "glance", len(pending), done, reused,
                        f"{start}-{done}", [units[i] for i in window])

    units.sort(key=lambda u: (u["chunk_id"], u["sampler_id"]))
    return Embedded(video_id=manifest.video_id,
                    timeline_fingerprint=timeline.fingerprint(),
                    embedder=space, units=units)


def glance(manifest: str | Path, timeline: str | Path, store: str | Path,
           out: str | Path,
           previous: Optional[str | Path] = None,
           visual_embedder: Optional["VisualEmbedder"] = None,
           samplers: Optional[Sequence[str]] = None,
           batch: int = 16,
           on_progress: Optional[progress.Reporter] = None) -> Produced:
    """One vector per (chunk, sampler run), from the frames alone. `look` plus a
    read at each end. `previous` is usually `out` itself.
    """
    if batch < 1:
        raise Refused(f"batch must be at least 1, not {batch}")
    plan = read(manifest, Manifest)
    grid = read(timeline, Timeline)
    stored = maybe(previous, Embedded)
    video_id = same_video(manifest=plan, timeline=grid, previous=stored)

    with logs.timed("glance", video_id) as done:
        document = look(plan, grid, store_of(store), visual_embedder, samplers,
                        stored, batch, on_progress)
        carried = 0 if stored is None or stored.embedder != document.embedder else sum(
            1 for u in document.units
            if stored.stored().get(f"{video_id}:{u['chunk_id']}:{u['sampler_id']}")
            == u["text_hash"])
        where = write(out, document)
        done(units=len(document.units), embedded=len(document.units) - carried,
             unchanged=carried, embedder=document.embedder)

    return Produced(
        video_id=video_id, component="glance",
        artifacts={"glances": where},
        stats={"units": len(document.units),
               "embedded": len(document.units) - carried, "unchanged": carried,
               "images": sum(u["images"] for u in document.units),
               "embedder": document.embedder,
               "samplers": sorted({u["sampler"] for u in document.units})},
    )


def load(path: str | Path) -> Embedded:
    """Read a `glances.json` back, typed: `embedded.json`'s shape."""
    return read(path, Embedded)


__all__ = ["SUFFIX", "glance", "load", "look", "space_of"]
