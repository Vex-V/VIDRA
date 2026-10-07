"""The frame store: one JPEG per kept frame, under a directory.

`video` writes frames and `describe` reads them. A frame is addressed by its
read index, the reader's count over every frame in the container.
`video.recreate` rebuilds a store from a manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:                            # numpy is ~60 ms to import, and
    import numpy as np                       # `describe` reads bytes, never
                                             # arrays. Annotation only.


class FrameStore:
    """Writes one JPEG per kept frame into the directory it is given."""

    def __init__(self, root: Path | str, quality: int = 95,
                 fmt: str = "jpg") -> None:
        self.root = Path(root)
        self.quality = quality
        self.format = fmt
        #: Files written during this run, and their encoded size.
        self.written = 0
        self.bytes_written = 0
        #: Read indexes written during this run: a frame two samplers keep is one file
        #: and is counted once.
        self._indexes: set[int] = set()

    def path_for(self, index: int) -> Path:
        return self.root / f"{index:07d}.{self.format}"

    def write(self, index: int, image: Optional["np.ndarray"]) -> Optional[Path]:
        if image is None:
            return None
        import cv2

        path = self.path_for(index)
        if index in self._indexes:
            # Already written this run.
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

    def write_bytes(self, index: int, data: bytes) -> Path:
        """Store a frame already encoded with `encode`. Counted like `write`."""
        path = self.path_for(index)
        if index in self._indexes:
            return path
        self.root.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._indexes.add(index)
        self.written += 1
        self.bytes_written += len(data)
        return path

    def read(self, index: int) -> bytes:
        """The frame's bytes, from disk. `KeyError` when the store does not hold it."""
        path = self.path_for(index)
        if not path.exists():
            raise KeyError(index)
        return path.read_bytes()

    def __contains__(self, index: int) -> bool:
        return self.path_for(index).exists()

    def prune(self, keep: set[int]) -> list[int]:
        """Delete stored frames `keep` does not name, and return them. Only when asked."""
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


def encode(image: "np.ndarray", quality: int = 95) -> bytes:
    """A frame as the JPEG bytes `FrameStore.write` would put on disk: the same
    encoder and quality, so a frame encoded once can be both stored and sent."""
    import cv2

    ok, data = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise OSError("could not encode a frame as JPEG")
    return data.tobytes()


__all__ = ["FrameStore", "encode"]
