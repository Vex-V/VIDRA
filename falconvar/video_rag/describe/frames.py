"""Reading pixels back out of wherever ingest kept them.

**Whatever it was handed, and nothing else.** There is no seek-the-video
fallback, and that is deliberate: the store exists so this component has its
frames in hand, and a fallback would quietly do the store's job while leaving
it broken -- silently, since the output is identical and only about 40x
slower. A missing store, or a frame it lacks, raises and names the fix.

**A short frame list is never returned.** A description covering 8 of the 9
frames it claims is indistinguishable from a correct one once written down.

The source is now *given* rather than resolved. It used to build its own path
from a video id, which meant `describe` could only ever read frames the
pipeline's directory layout had put there -- so a caller driving the
components itself had to adopt that layout to use this one. It takes any
`Frames` now, which is how `MemoryFrames` lets a whole run happen with no
filesystem at all. `store_of` is the pipeline's spelling, kept in one place so
`run()` is the only thing that knows where a store lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...shared import paths
from ...shared.contracts.documents import Manifest
from ...shared.errors import Unavailable
from ..frames import Frames, FrameStore


class StoreUnavailable(Unavailable):
    """The frame store is missing, or lacks a frame the manifest names."""


def store_of(video_id: str) -> FrameStore:
    """The store this video's ingest wrote -- the one place a store path is
    derived from a video id.

    Checked here rather than on first read, because "no store at all" and "a
    store missing one frame" are different mistakes with different fixes, and
    the first is worth saying before any describing starts.
    """
    root = paths.artifact(video_id, "store")
    if not root.exists():
        raise StoreUnavailable(
            f"no frame store at {root}. Re-run ingest with a frame store, "
            f"or rebuild it from the manifest and the video.")
    return FrameStore(root)


@dataclass
class LoadedFrame:
    """One frame, as bytes ready to send."""

    index: int
    media_ts: float
    jpeg: bytes

    @property
    def size(self) -> int:
        return len(self.jpeg)


class FrameSource:
    """Frames for one (chunk, sampler), read from whatever holds them.

    Caches by index, because a frame two samplers chose is *described* twice --
    they are different questions -- but only ever read once.
    """

    def __init__(self, frames: Frames, manifest: Manifest) -> None:
        self.frames = frames
        self.manifest = manifest
        self._cache: dict[int, bytes] = {}

    def _read(self, index: int) -> bytes:
        if index in self._cache:
            return self._cache[index]
        try:
            data = self.frames.read(index)
        except KeyError:
            raise StoreUnavailable(
                f"frame {index} is missing, but the manifest names it. "
                f"The frames and the manifest disagree.") from None
        self._cache[index] = data
        return data

    def images_for(self, chunk_id: int, sampler_id: str) -> list[LoadedFrame]:
        records = self.manifest.frames_of(chunk_id, sampler_id)
        if not records:
            return []
        loaded = [LoadedFrame(r["index"], r["media_ts"], self._read(r["index"]))
                  for r in records]
        if len(loaded) != len(records):        # unreachable; the read raises
            raise StoreUnavailable(
                f"chunk {chunk_id}/{sampler_id}: {len(loaded)} of "
                f"{len(records)} frames available")
        return loaded

    def close(self) -> None:
        self._cache.clear()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["FrameSource", "LoadedFrame", "StoreUnavailable", "store_of"]
