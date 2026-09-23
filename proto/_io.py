"""Reading a document from a path, and writing one to a path.

`shared.storage.files.write` resolves `(video_id, artifact)` to a path and
logs through `PRODUCED_BY`. Nothing here resolves anything: the caller named
the path, so what is left is the atomic write and `from_dict`/`as_dict`.

**The artifact name table stays in the library and is read as data.** A
pipeline composing `folder / ARTIFACTS["raw_transcript"]` gets
`transcript.raw.json`; one spelling the names itself would write
`raw_transcript.json` and nothing would say so. Resolving moved out; naming
did not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, TypeVar

from falconvar.shared import paths
from falconvar.shared.contracts.documents import DOCUMENTS
from falconvar.shared.paths import MissingArtifact
from falconvar.shared.storage import files

T = TypeVar("T")

#: Document class -> (artifact name, filename, the component that writes it).
#: Derived from the library's own tables rather than restated: a second copy
#: of `transcript.raw.json` is a drift waiting to happen.
WRITTEN_BY: dict[type, tuple[str, str, str]] = {
    cls: (name, paths.ARTIFACTS.get(name, ""), paths.PRODUCED_BY.get(name, ""))
    for name, cls in DOCUMENTS.items()
}


def read(path: str | Path, cls: type[T]) -> T:
    """The document at `path`, as `cls`.

    A missing input names the component that makes it, as `paths.require`
    does -- and it can, because the producer is a fact about the *type* the
    caller asked for, not about the path. What it cannot add is
    `Present: media`, which needs a directory to look in; that half is the
    pipeline's now.
    """
    p = Path(path)
    if not p.exists():
        _, filename, producer = WRITTEN_BY.get(cls, ("", "", ""))
        if producer:
            raise MissingArtifact(
                f"no {cls.__name__} at {p} -- `{producer}` writes "
                f"{filename or 'it'}")
        raise MissingArtifact(f"no {cls.__name__} at {p}")
    return cls.from_dict(files.read_json(p))


def maybe(path: Optional[str | Path], cls: type[T]) -> Optional[T]:
    """`read`, or None for an absent argument or a file not written yet."""
    if path is None or not Path(path).exists():
        return None
    return read(path, cls)


def write(path: str | Path, document: Any) -> str:
    """Atomically -- the temp-file-plus-replace every component wrote through.

    Returns the path as a string, which is what goes in the receipt.
    """
    return str(files.write_json(Path(path), document.as_dict()))


def same_video(**documents: Any) -> str:
    """The one id these documents agree on, or a refusal naming the argument.

    Addressed by id this could not happen: `paths.artifact(video_id, name)` is
    a join, so two artifacts read under one id are that video's by
    construction. Addressed by path it can, and a pipeline keeping every file
    in one folder makes it rare rather than impossible -- so the check lives
    here, where a caller wiring components by hand also gets it.
    """
    seen = {name: doc.video_id for name, doc in documents.items()
            if doc is not None}
    ids = set(seen.values())
    if len(ids) > 1:
        detail = ", ".join(f"{name}={vid!r}" for name, vid in sorted(seen.items()))
        raise ValueError(
            f"these documents are not the same video: {detail}. "
            f"Each carries its own id, and a component works on one video.")
    return next(iter(ids)) if ids else ""


__all__ = ["WRITTEN_BY", "maybe", "read", "same_video", "write"]
