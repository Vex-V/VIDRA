"""Reading frames back out of the store ingest wrote.

There is no fallback to seeking the video: a missing store, or a frame it
lacks, raises. A short frame list is never returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vidra.shared.contracts.documents import Manifest
from vidra.shared.reporting.errors import Unavailable
from ..helpers import FrameStore


class StoreUnavailable(Unavailable):
    """The frame store is missing, or lacks a frame the manifest names."""


def store_of(store: str | Path) -> FrameStore:
    """The store at this path, checked before any describing starts."""
    root = Path(store)
    if not root.exists():
        raise StoreUnavailable(
            f"no frame store at {root}. Re-run ingest with a store=, "
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
    """Frames for one (chunk, sampler), read from the store and cached by index."""

    def __init__(self, frames: FrameStore, manifest: Manifest) -> None:
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
