"""Reading frames back out of the store ingest wrote.

There is no fallback to seeking the video: a missing store, or a frame it
lacks, raises. A short frame list is never returned. A frame whose sampler
recorded views is read as those views, in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vidra.shared.contracts.documents import Manifest
from vidra.shared.reporting.errors import Unavailable
from ..frames import FrameStore


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
    #: What a view made from the frame shows ("person 1 of 3, enlarged"); empty
    #: for the frame itself.
    label: str = ""

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

    def _read_view(self, index: int, sampler_id: str, number: int) -> bytes:
        try:
            return self.frames.read_view(index, sampler_id, number)
        except KeyError:
            raise StoreUnavailable(
                f"view {number} of frame {index} ({sampler_id}) is missing, but the "
                f"manifest names it. Re-run ingest: views are made by the sampler "
                f"and recreate does not rebuild them.") from None

    def images_for(self, chunk_id: int, sampler_id: str) -> list[LoadedFrame]:
        """Every image of one sampler run in one chunk, in order: each kept frame,
        or the views its sampler recorded for it."""
        loaded: list[LoadedFrame] = []
        for r in self.manifest.frames_of(chunk_id, sampler_id):
            for view in r.get("views") or [{"frame": True}]:
                if view.get("frame"):
                    loaded.append(LoadedFrame(r["index"], r["media_ts"],
                                              self._read(r["index"])))
                else:
                    loaded.append(LoadedFrame(
                        r["index"], r["media_ts"],
                        self._read_view(r["index"], sampler_id, view["view"]),
                        view["label"]))
        return loaded

    def close(self) -> None:
        self._cache.clear()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["FrameSource", "LoadedFrame", "StoreUnavailable", "store_of"]
