"""The media component: a video file -> a folder holding `media.json`.

The only component that creates a folder: it decides the video id and makes
`<into>/<id>/`. Every later component is handed paths inside it.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

from vidra.shared.reporting import logs
from vidra.shared.config import paths
from vidra.shared.reporting.errors import VidraError, UnknownOption
from vidra.shared.contracts.documents import Media, Produced
from vidra.shared.storage.files import WRITTEN_BY, read, write
from .split import split

#: `media.json`, from the library's filename table.
FILENAME = WRITTEN_BY[Media][1]


class VideoIdTaken(VidraError, FileExistsError):
    """A different video already owns this id, and `on_conflict` said refuse."""


#: What to do when the id is taken by a different file.
ON_CONFLICT = ("new", "replace", "refuse")

#: How far `new` counts before giving up.
MAX_VARIANTS = 999

#: Stands in for a `media.json` that cannot be read: the id counts as taken.
_UNREADABLE = "unreadable"


def media(source: str | Path, into: str | Path,
          video_id: Optional[str] = None,
          on_conflict: str = "new",
          name: Optional[str] = None,
          recorded_at: Optional[str | datetime] = None) -> Produced:
    """Describe the file, make its folder under `into`, write `media.json`.

    The folder's name is the video id: the filename stem by default. `name` and
    `recorded_at` replace the filename and the container's recording time; a run
    that gives neither keeps what an earlier run of the same file was given.

    When a different file already holds the id, `on_conflict` decides:

        new      mint the next free id -- clip-2, clip-3 (default)
        replace  delete everything under the old id and take it
        refuse   raise `VideoIdTaken`

    The same file again (by `Media.source`) always reuses its folder. The receipt
    carries the id used, `requested_id`, and `stats["folder"]`, the folder.
    """
    if on_conflict not in ON_CONFLICT:
        raise UnknownOption(f"unknown on_conflict {on_conflict!r}; "
                            f"known: {', '.join(ON_CONFLICT)}")
    parent = Path(into)

    with logs.timed("media") as done:
        described = _settle(parent, split(source, video_id, name, recorded_at),
                            on_conflict)
        described = _kept(parent, described, name is not None,
                          recorded_at is not None)
        folder = parent / described.video_id
        where = write(folder / FILENAME, described)
        done(video_id=described.video_id, duration_s=described.duration_s,
             has_video=described.has_video, has_audio=described.has_audio)
    return Produced(
        video_id=described.video_id,
        component="media",
        artifacts={"media": where},
        stats={"has_video": described.has_video, "has_audio": described.has_audio,
               "duration_s": described.duration_s,
               "container": described.container_format,
               "source": described.source,
               "name": described.name,
               "recorded_at": described.recorded_at,
               # The folder every later path is built from.
               "folder": str(folder),
               # The id the filename asked for, beside the one used.
               "requested_id": video_id or Path(source).stem},
        # Streams this file does not carry.
        skipped=([] if described.has_video else ["video"])
                + ([] if described.has_audio else ["audio"]),
    )


def _owner(parent: Path, video_id: str) -> Optional[Media]:
    """The `media.json` already in this folder, or None if the id is free."""
    where = parent / paths.check_id(video_id) / FILENAME
    if not where.exists():
        return None
    try:
        return load(where)
    except Exception:                                       # noqa: BLE001
        # An unreadable `media.json` still occupies the id.
        return Media(video_id=video_id, path="", container_format="",
                     duration_s=None, video=None, audio=None,
                     source=_UNREADABLE)


def _is_same(stored: Media, described: Media) -> bool:
    """Whether the id's current owner is this file. A `media.json` with no
    `source` counts as the same file.
    """
    return stored.source is None or stored.source == described.source


def _settle(parent: Path, described: Media, on_conflict: str) -> Media:
    """The id this file should actually be written under."""
    wanted = described.video_id
    stored = _owner(parent, wanted)
    if stored is None or _is_same(stored, described):
        return described

    if on_conflict == "refuse":
        raise VideoIdTaken(
            f"video id {wanted!r} already holds a different file "
            f"({stored.path or 'unreadable media.json'}); pass "
            f"on_conflict='new' to mint another id, 'replace' to delete it, "
            f"or name a video_id yourself")

    if on_conflict == "replace":
        # The whole directory, so nothing from the previous video survives.
        shutil.rmtree(parent / wanted, ignore_errors=True)
        return described

    for n in range(2, MAX_VARIANTS + 1):
        candidate = f"{wanted}-{n}"
        beside = _owner(parent, candidate)
        # Free, or already this file: a re-run lands on the same minted id.
        if beside is None or _is_same(beside, described):
            return replace(described, video_id=candidate)
    raise VideoIdTaken(
        f"{wanted} and {MAX_VARIANTS - 1} variants of it are all taken by "
        f"different files")


def _kept(parent: Path, described: Media, named: bool, timed: bool) -> Media:
    """`described`, with an earlier run's name and time where this one gave none
    (same file, same folder only).
    """
    stored = _owner(parent, described.video_id)
    if stored is None or stored.source != described.source:
        return described
    return replace(
        described,
        name=described.name if named or not stored.name else stored.name,
        recorded_at=(described.recorded_at if timed or not stored.recorded_at
                     else stored.recorded_at))


def load(path: str | Path) -> Media:
    """Read a `media.json` back, typed."""
    return read(path, Media)
