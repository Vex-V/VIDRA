"""The frames themselves -- the one thing components hand each other that is
not a document.

`video` writes pixels and `describe` reads them, and neither may import the
other, so this sits beside the components rather than inside one.

**A frame is addressed by read index**, the reader's count over every frame in
the container. That is the one address that survives the manifest being copied
somewhere else: `pts` is exact *within* a container, but the index is what a
filename can hold and what a dictionary key can be.

One implementation: `FrameStore`, one JPEG per frame under a directory. What
`recovery/` rebuilds, and what survives the process.

**There was a second, and a filepath surface is what removed it.**
`MemoryFrames` held the same bytes in a dict so a whole run could happen with
no filesystem at all. Addressed by video id that was a real capability, and
the oracle for it -- the whole chain into an empty data root that had to stay
empty -- was one of the three this factoring rested on.

Addressed by filepath it is not reachable and not coherent. Not reachable
because `video` and `describe` take a directory and build the store from it,
so there is nowhere to hand an object in; not coherent because every document
is written to a path now, so "no filesystem" cannot be true of a run whatever
the pixels do. What is left of it is avoiding JPEG I/O while still writing
eight JSON files, which is a performance choice and not the architectural one
it was.

**And a `Frames` protocol went with it.** It was written when there were two
implementations and a caller chose between them; with one it described
`FrameStore`'s own interface in a second place, and advertised a substitution
the entry points do not offer -- `video(store=)` and `describe(store=)` both
take a directory and build the store themselves, so there is nowhere to hand
an alternative in. A caller whose pixels are somewhere else needs a way past
those two signatures first; the protocol is what that change would restore,
and it is five lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:                            # numpy is ~60 ms to import, and
    import numpy as np                       # `describe` reads bytes, never
                                             # arrays. Annotation only.


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
        #: Frames actually stored during this run, and their encoded size.
        #: The accounting the manifest reports, and it counts *files*, not
        #: picks -- see `_indexes` below.
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


__all__ = ["FrameStore"]
