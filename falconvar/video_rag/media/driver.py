"""The media component: a video file -> a folder, holding `media.json`.

**This is the only component that creates anything but a document, and the
asymmetry is the design.** It takes the video and a parent directory, decides
the id, and makes `<into>/<id>/`. Everything after it is handed explicit
filepaths inside that folder -- so a pipeline follows the convention and a
caller wiring one component by hand is not obliged to.

The id staying here is what keeps `on_conflict` working. It guards against two
different files called `clip.mp4` claiming one directory, and only something
that creates directories can guard that: a component handed an output path has
already been told where to write.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Optional

from falconvar.shared import logs, paths
from falconvar.shared.errors import FalconvarError, UnknownOption
from falconvar.shared.contracts.documents import Media, Produced
from falconvar.shared.storage.files import WRITTEN_BY, read, write
from .split import UnusableMedia, split

#: `media.json`, read off the library's table rather than spelled here.
FILENAME = WRITTEN_BY[Media][1]


class VideoIdTaken(FalconvarError, FileExistsError):
    """A different video already owns this id, and `on_conflict` said refuse."""


#: What to do when the id is already taken **by a different file**. The same
#: file arriving again is not a conflict and never reaches this: it reuses its
#: own directory, which is what resume is built on.
ON_CONFLICT = ("new", "replace", "refuse")

#: How far `new` will count before giving up. A bound rather than a `while`:
#: an id that cannot be minted is a caller doing something strange, and
#: spinning is a worse answer than saying so.
MAX_VARIANTS = 999

#: Stands in for a `media.json` that exists but cannot be read. It occupies
#: the id, and it can never compare equal to a real fingerprint, so such a
#: directory is a conflict rather than something to write into.
_UNREADABLE = "unreadable"


def media(source: str | Path, into: str | Path,
          video_id: Optional[str] = None,
          on_conflict: str = "new") -> Produced:
    """Describe the file, make its folder under `into`, write `media.json`.

    `into` is the directory that holds one folder per video. The folder's
    name is the id -- the filename stem by default -- and every later
    component is pointed at files inside it.

    A video id defaults to the filename stem, so two different videos both
    called `clip.mp4` used to claim one folder -- silently. The second
    `media.json` replaced the first while every other artifact stayed,
    leaving a directory whose grid covered 205 s of a 60 s file, both
    documents well-formed. `on_conflict` is what happens instead:

        new      mint the next free id -- clip-2, clip-3 (default)
        replace  delete everything under the old id and take it
        refuse   raise `VideoIdTaken`, so the caller decides

    **Only a *different* file is a conflict.** The same file again reuses its
    own folder whatever this says, because that is the ordinary case -- a
    pipeline begins with it on every run -- and minting there would orphan
    the manifest, the descriptions and the vectors, so a re-run would pay a
    second time for everything already done. `Media.source` is what tells
    the two apart.

    `new` is the default because nothing is lost by it. It does mean the id
    can differ from the one asked for, which is why the receipt carries both
    the id used and `requested_id`, and `stats["home"]` is the folder every
    later path is built from.
    """
    if on_conflict not in ON_CONFLICT:
        raise UnknownOption(f"unknown on_conflict {on_conflict!r}; "
                            f"known: {', '.join(ON_CONFLICT)}")
    parent = Path(into)

    with logs.timed("media") as done:
        described = _settle(parent, split(source, video_id), on_conflict)
        home = parent / described.video_id
        where = write(home / FILENAME, described)
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
               # The folder every later path is built from. A pipeline reads
               # this rather than re-deriving it, because a mint means the id
               # is not the one that was asked for.
               "home": str(home),
               # The id actually used, beside the one the filename asked for,
               # so a mint is visible in the receipt rather than only in a
               # field the caller might not compare.
               "requested_id": video_id or Path(source).stem},
        # What this file does NOT carry. A later component reads this rather
        # than opening the file again to find out.
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
        # An unreadable `media.json` still means the directory is occupied.
        # Treating it as free would write a second video's artifacts in beside
        # whatever is there, which is the failure this whole function exists
        # to stop.
        return Media(video_id=video_id, path="", container_format="",
                     duration_s=None, video=None, audio=None,
                     source=_UNREADABLE)


def _is_same(stored: Media, described: Media) -> bool:
    """Whether the id's current owner is the file we were just handed.

    A `media.json` written before `source` existed carries None, and there is
    nothing to compare -- the file it described may have moved or been
    replaced since. That reads as *the same video*, which is what every such
    directory has been treated as until now, so an existing checkout keeps
    resuming rather than minting a duplicate of everything it has. The
    fingerprint is written on the way past, so it is the last time that id
    cannot answer the question.
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
        # The whole directory, not just `media.json`. Overwriting the one
        # document is exactly the bug: the grid, the frame store and the
        # descriptions of the *previous* video survive it, each still
        # well-formed, and nothing downstream compares them against the media
        # they were derived from.
        shutil.rmtree(parent / wanted, ignore_errors=True)
        return described

    for n in range(2, MAX_VARIANTS + 1):
        candidate = f"{wanted}-{n}"
        beside = _owner(parent, candidate)
        # Free, or already this same file: a re-run of the second `clip.mp4`
        # has to land back on `clip-2` rather than minting `clip-3` every time.
        if beside is None or _is_same(beside, described):
            return replace(described, video_id=candidate)
    raise VideoIdTaken(
        f"{wanted} and {MAX_VARIANTS - 1} variants of it are all taken by "
        f"different files")


def load(path: str | Path) -> Media:
    """Read a `media.json` back. Every later component starts from one."""
    return read(path, Media)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Describe the two streams a media file carries.")
    ap.add_argument("media", type=Path)
    ap.add_argument("into", type=Path,
                    help="the directory that holds one folder per video")
    ap.add_argument("--video-id", default=None, help="defaults to the filename stem")
    ap.add_argument("--on-conflict", default="new", choices=ON_CONFLICT,
                    help="when a DIFFERENT file already holds this id: "
                         "new mints clip-2, replace deletes the old output, "
                         "refuse raises. The same file always reuses its own "
                         "directory")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = media(args.media, args.into, args.video_id,
                         args.on_conflict)
    # `UnusableVideoId` is named rather than caught as the `ValueError` it also
    # is: this is the one driver whose tuple is not already `ValueError`-wide,
    # because the only bad input it had was the file itself. An id can be bad
    # too, and a CLI answering a typo with a traceback is not answering it.
    except (UnusableMedia, VideoIdTaken, paths.UnusableVideoId) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    m = load(produced.artifacts['media'])
    length = f"  {m.duration_s:.3f}s" if m.duration_s else ""
    print(f"{m.path}  [{m.container_format}]{length}")
    if m.video:
        v = m.video
        rate = f"  {v.rate:g} fps" if v.rate else ""
        print(f"  video  {v.codec}  {v.width}x{v.height}{rate}")
        print(f"         time_base={v.time_base}  frames={v.frames}  "
              f"duration={v.duration_s}")
    else:
        print("  video  -- none")
    if m.audio:
        a = m.audio
        print(f"  audio  {a.codec}  {a.rate} Hz  {a.channels} ch")
        print(f"         duration={a.duration_s}")
    else:
        print("  audio  -- none")
    print()
    # Said out loud, because a mint is the one outcome where the id the caller
    # will need afterwards is not the one it asked for.
    if produced.video_id != produced.stats["requested_id"]:
        print(f"  {produced.stats['requested_id']!r} is a different file; "
              f"this one is {produced.video_id!r}")
    print(f"home  -> {produced.stats['home']}")
    print(f"media -> {produced.artifacts['media']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
