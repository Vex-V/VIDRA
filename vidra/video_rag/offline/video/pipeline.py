"""One decode pass, feeding every sampler from it.

Per frame: decimate on media time, ask the grid which chunk it is in, reset
the samplers at a boundary, offer the frame to each, release the pixels. The
grid is read, never edited.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

from vidra.shared.contracts.documents import Manifest, Media, Timeline
from ...core.frames import FrameStore
from ...core.sampling.decimate import Decimator
from ...core.sampling.offer import Offer, empty_chunk
from ...core.sampling.reader import read_frames, rotation_of
from ...core.sampling.samplers import Sampler
from vidra.shared.reporting.errors import Refused, UnknownOption


def ingest(media: Media, timeline: Timeline,
           samplers: Sequence[Sampler],
           per_second: float = 1.0,
           store: Optional[FrameStore] = None,
           store_scope: str = "sampled",
           on_chunk: Optional[Callable[[int, dict[str, Any]], None]] = None
           ) -> Manifest:
    """Decode once, offer every decimated frame to every sampler."""
    if not samplers:
        raise Refused("name at least one sampler")
    ids = [s.sampler_id for s in samplers]
    if len(set(ids)) != len(ids):
        # The manifest keys frames by sampler id.
        raise Refused(f"sampler ids must be unique, got {ids}")
    if store_scope not in ("sampled", "decimated"):
        raise UnknownOption("store_scope must be 'sampled' or 'decimated'")

    decimator = Decimator(per_second)
    config = {
        "decimator": decimator.config(),
        "samplers": [s.config() for s in samplers],
        # "Was a store given", not "does it hold anything".
        "frame_store": ({**store.config(), "scope": store_scope}
                        if store is not None else None),
    }

    chunks = [empty_chunk(i) for i in range(len(timeline))]
    offer = Offer(samplers)
    started = time.perf_counter()

    # Read once: the reader applies it, and `recreate` reads it off the manifest.
    rotation = rotation_of(media.path) if media.has_video else 0.0

    for frame in read_frames(media, decimator.accepts, rotation):
        # `nearest`: a frame past the end of the grid belongs to the last chunk.
        chunk_id = timeline.nearest(frame.media_ts)
        if (offer.chunk_id is not None and chunk_id != offer.chunk_id
                and on_chunk is not None):
            on_chunk(offer.chunk_id, chunks[offer.chunk_id])

        # "decimated" keeps every frame the samplers were offered.
        kept = offer(frame, chunk_id, chunks[chunk_id])
        if store is not None and (store_scope == "decimated" or kept):
            store.write(frame.index, frame.image)
        # Pixels are released at once; one frame in memory.
        frame.release()

    if offer.chunk_id is not None and on_chunk is not None:
        on_chunk(offer.chunk_id, chunks[offer.chunk_id])

    elapsed = time.perf_counter() - started
    stats = {
        "frames_decimated": offer.decimated,
        "frames_sampled": offer.sampled,
        "chunks": len(chunks),
        "chunks_with_frames": sum(1 for c in chunks if c["decimated_frames"]),
        "elapsed_s": round(elapsed, 3),
        **({"stored_frames": store.written,
            "stored_mb": round(store.bytes_written / 1024 / 1024, 2)}
           if store is not None else {}),
    }

    return Manifest(
        video_id=media.video_id,
        timeline_fingerprint=timeline.fingerprint(),
        source={"path": media.path, "container": media.container_format,
                "duration_s": media.duration_s,
                **(media.video.as_dict() if media.video else {}),
                # Only when there is one, so an upright video's manifest is unchanged.
                **({"rotation": int(rotation)} if int(rotation) else {})},
        config=config,
        stats=stats,
        chunks=chunks,
    )


__all__ = ["ingest"]
