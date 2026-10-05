"""A record: the documents an aggregate is taken from, opened as a `Context`.

What `video_rag` wrote for one video, or `combination` for several:

    timeline      required
    descriptions  what the picture was said to hold, per chunk and sampler
    transcript    what was said, per chunk
    manifest      which frames were kept (only `stats` reads it)

Handed over as a folder (the library's filenames) or a mapping naming each
document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Union

from ...shared.config import paths
from ...shared.contracts.documents import (Descriptions, Manifest, Timeline,
                                           Transcript, same_video)
from ...shared.reporting.errors import VidraError
from ...shared.storage import files
from .base import Context

#: The documents a record may hold, and the type each is read as.
KINDS: dict[str, type] = {"timeline": Timeline, "descriptions": Descriptions,
                          "transcript": Transcript, "manifest": Manifest}

#: A record as a caller hands it over.
Source = Union[str, Path, Mapping[str, Union[str, Path]]]


class RecordError(VidraError, ValueError):
    """A record that cannot be opened: no grid, or a name that is no document."""


def documents_of(source: Source) -> dict[str, Path]:
    """A record's document paths: a folder's, by the library's filenames, or a
    mapping's, as named. Only documents that exist are returned."""
    if isinstance(source, Mapping):
        unknown = set(source) - set(KINDS)
        if unknown:
            raise RecordError(f"unknown document(s) {', '.join(sorted(unknown))}; "
                              f"a record names {', '.join(KINDS)}")
        return {kind: Path(where) for kind, where in source.items()
                if where and Path(where).exists()}
    folder = Path(source)
    if not folder.is_dir():
        raise RecordError(f"{folder} is not a folder of documents")
    return {kind: folder / paths.ARTIFACTS[kind] for kind in KINDS
            if (folder / paths.ARTIFACTS[kind]).exists()}


def context(timeline: str | Path,
            descriptions: Optional[str | Path] = None,
            transcript: Optional[str | Path] = None,
            manifest: Optional[str | Path] = None) -> Context:
    """The documents at these paths, joined by `chunk_id`. The grid is required;
    the rest are optional. Documents from two different videos are refused.
    """
    grid = files.read(timeline, Timeline)
    read = {"descriptions": files.maybe(descriptions, Descriptions),
            "transcript": files.maybe(transcript, Transcript),
            "manifest": files.maybe(manifest, Manifest)}
    video_id = same_video(timeline=grid, **read)
    return Context(video_id, grid, read["manifest"], read["descriptions"],
                   read["transcript"])


def open_record(source: Source) -> Context:
    """A folder or a mapping of documents, as one `Context`."""
    found = documents_of(source)
    if "timeline" not in found:
        raise RecordError(f"{source}: no timeline -- a record needs its grid, "
                          f"which `boundaries` writes")
    return context(found["timeline"], found.get("descriptions"),
                   found.get("transcript"), found.get("manifest"))


__all__ = ["KINDS", "RecordError", "Source", "context", "documents_of", "open_record"]
