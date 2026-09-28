"""A record: the documents an aggregate is taken from, opened as a `Context`.

A record is what `video_rag` wrote for one video, or what `combination`
wrote for several -- the same four documents under the same names either way:

    timeline      required: a chunk id means nothing without the grid
    descriptions  what the picture was said to hold, per chunk and sampler
    transcript    what was said, per chunk
    manifest      which frames were kept (only `stats` reads it)

It is handed over as a **folder**, read with the library's own filenames, or as
a **mapping** naming each document outright -- which is how a caller whose
files are not laid out like this project's still uses it. Everything that
reads a record goes through `open_record`, so the two spellings cannot drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Union

from ..shared import paths
from ..shared.contracts.documents import (Descriptions, Manifest, Timeline,
                                          Transcript, same_video)
from ..shared.errors import FalconvarError
from ..shared.storage import files
from .base import Context

#: The documents a record may hold, and the type each is read as.
KINDS: dict[str, type] = {"timeline": Timeline, "descriptions": Descriptions,
                          "transcript": Transcript, "manifest": Manifest}

#: A record as a caller hands it over.
Source = Union[str, Path, Mapping[str, Union[str, Path]]]


class RecordError(FalconvarError, ValueError):
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
    """The documents at these paths, joined by `chunk_id`.

    The grid is required; the rest are optional -- an audio-only video has no
    descriptions, a silent one no transcript -- and an aggregator that needs
    one it was not given says so when asked to run. Documents from two
    different videos are refused by name (`documents.same_video`).
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
