"""The frames themselves -- the one thing components hand each other that is
not a document.

`video` writes pixels and `describe` reads them, and neither may import the
other, so this sits beside the components rather than inside one.

**A frame is addressed by read index**, the reader's count over every frame in
the container. That is the one address that survives the manifest being copied
somewhere else: `pts` is exact *within* a container, but the index is what a
filename can hold and what a dictionary key can be.

Two implementations, and the choice belongs to the caller:

    FrameStore     one JPEG per frame under a directory. What the pipeline
                   uses, what `recovery/` rebuilds, and what survives the
                   process.
    MemoryFrames   the same bytes in a dict. Nothing is written, so a whole
                   run can happen with no filesystem at all -- at the cost of
                   holding every kept frame. Roughly 0.4 MB per frame at 720p,
                   so a minute at 1 fps is tens of megabytes and an hour is
                   hundreds. That is the caller's decision to make, which is
                   why this is a separate class and not a flag.

Both satisfy `Frames`, which is a `Protocol` rather than a base class: the
relationship here is structural, and `FrameStore` predates the protocol by a
long way.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:                            # numpy is ~60 ms to import, and
    import numpy as np                       # `describe` reads bytes, never
                                             # arrays. Annotation only.


@runtime_checkable
class Frames(Protocol):
    """Somewhere kept frames go, and come back from.

    ``written`` and ``bytes_written`` are the accounting the manifest reports,
    and they count *files*, not picks: two samplers routinely keep the same
    frame, and counting it twice overstated one store by 25%.
    """

    #: Frames actually stored during this run.
    written: int
    #: Their total size in bytes, encoded.
    bytes_written: int

    def write(self, index: int, image: Optional["np.ndarray"]) -> Any:
        """Keep one frame. A `None` image is a no-op, not an error."""

    def read(self, index: int) -> bytes:
        """The encoded bytes back. Raises `KeyError` if it was never kept."""

    def __contains__(self, index: int) -> bool: ...

    def config(self) -> dict[str, Any]:
        """What the manifest records about where its frames went."""


def _encode(image: np.ndarray, fmt: str, quality: int) -> bytes:
    import cv2

    params = ([int(cv2.IMWRITE_JPEG_QUALITY), quality]
              if fmt in ("jpg", "jpeg") else [])
    ok, buffer = cv2.imencode(f".{fmt}", image, params)
    if not ok:
        raise OSError(f"could not encode a frame as {fmt}")
    return bytes(buffer)


class FrameStore:
    """Writes one JPEG per kept frame. Nothing else touches this directory.

    The root is an argument rather than derived from a video id, so a caller
    driving the components itself decides where pixels land -- and `describe`
    reads from whatever it is handed rather than resolving the same path a
    second time.
    """

    def __init__(self, root: Path | str, quality: int = 95,
                 fmt: str = "jpg") -> None:
        self.root = Path(root)
        self.quality = quality
        self.format = fmt
        self.written = 0
        self.bytes_written = 0
        #: Read indexes written during this run. Two samplers routinely keep
        #: the same frame -- `uniform:text` at stride 1 offers every decimated
        #: frame, and `clip` picks from that same set -- and the store is
        #: addressed by read index, so the second pick names a file the first
        #: already wrote. Counting it again overstated the store by 25% on a
        #: two-sampler run: 266 frames and 52.75 MB reported against 206 files
        #: and 42.18 MB on disk. Per run rather than per directory, because a
        #: file left by an earlier run should still be rewritten.
        self._indexes: set[int] = set()

    def path_for(self, index: int) -> Path:
        return self.root / f"{index:07d}.{self.format}"

    def write(self, index: int, image: Optional["np.ndarray"]) -> Optional[Path]:
        if image is None:
            return None
        import cv2

        path = self.path_for(index)
        if index in self._indexes:
            # Same index, same pixels, same file. Re-encoding it would cost a
            # JPEG encode to produce the bytes already there.
            return path

        self.root.mkdir(parents=True, exist_ok=True)
        params = ([int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                  if self.format in ("jpg", "jpeg") else [])
        if not cv2.imwrite(str(path), image, params):
            raise OSError(f"could not write {path}")
        self._indexes.add(index)
        self.written += 1
        self.bytes_written += path.stat().st_size
        return path

    def read(self, index: int) -> bytes:
        """The frame's bytes, from disk.

        `KeyError` rather than `FileNotFoundError`, so a caller can answer
        "this source does not hold that frame" the same way whichever
        implementation it was handed. The reader above turns it into a
        `StoreUnavailable` naming the manifest that disagrees.
        """
        path = self.path_for(index)
        if not path.exists():
            raise KeyError(index)
        return path.read_bytes()

    def __contains__(self, index: int) -> bool:
        return self.path_for(index).exists()

    def prune(self, keep: set[int]) -> list[int]:
        """Delete stored frames no longer named, and say which.

        A store accumulates across runs: ingesting with `uniform` and then with
        `clip` leaves the first run's frames behind, because nothing tells the
        directory that a new manifest supersedes the old one. That is disk
        wasted on frames no manifest addresses, and it makes a byte-comparison
        against the store report orphans that are not mismatches.

        Not automatic. Deleting frames is the one irreversible thing this
        component can do, and a caller who is about to describe from an older
        manifest wants them kept -- so it happens when asked.
        """
        if not self.root.exists():
            return []
        removed = []
        for path in sorted(self.root.glob(f"*.{self.format}")):
            try:
                index = int(path.stem)
            except ValueError:
                continue                     # not ours; leave it alone
            if index not in keep:
                path.unlink()
                removed.append(index)
        return removed

    def config(self) -> dict[str, Any]:
        return {"root": str(self.root), "format": self.format,
                "quality": self.quality}


class MemoryFrames:
    """Kept frames in a dict, so a run needs no filesystem.

    Encoded on the way in rather than held as arrays: the encode happens once
    either way, `describe` wants bytes, and a JPEG is roughly a twentieth of
    the raw array. `written` and `bytes_written` mean exactly what they mean on
    `FrameStore`, so the manifest reads the same from both.

    `prune` is present so the two are interchangeable at the one call site
    that prunes, but it is nearly pointless here: nothing accumulates across
    runs, because an instance holds one run's frames and is collected with
    them.
    """

    def __init__(self, quality: int = 95, fmt: str = "jpg") -> None:
        self.quality = quality
        self.format = fmt
        self.written = 0
        self.bytes_written = 0
        self._held: dict[int, bytes] = {}

    def write(self, index: int, image: Optional["np.ndarray"]) -> Optional[int]:
        if image is None:
            return None
        if index in self._held:              # see FrameStore._indexes
            return index
        data = _encode(image, self.format, self.quality)
        self._held[index] = data
        self.written += 1
        self.bytes_written += len(data)
        return index

    def read(self, index: int) -> bytes:
        return self._held[index]

    def __contains__(self, index: int) -> bool:
        return index in self._held

    def __len__(self) -> int:
        return len(self._held)

    def prune(self, keep: set[int]) -> list[int]:
        """Drop what no manifest names. `written` is *not* adjusted, exactly
        as on `FrameStore`: it counts what this run stored, which pruning
        afterwards does not change."""
        dropped = sorted(set(self._held) - keep)
        for index in dropped:
            del self._held[index]
        return dropped

    def config(self) -> dict[str, Any]:
        return {"root": None, "format": self.format, "quality": self.quality,
                "held": "memory"}


__all__ = ["FrameStore", "Frames", "MemoryFrames"]
