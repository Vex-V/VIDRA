"""The media component: a file path -> `media.json`.

The only component that takes a path rather than a video id, because it is the
one that establishes the id. Everything after it is addressed by `video_id`
and a backend, and `shared/paths.py` resolves the rest.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ...shared import logs, paths
from ...shared.errors import FalconvarError, UnknownOption
from ...shared.storage import files
from ...shared.contracts.documents import Media, Produced
from .split import UnusableMedia, split


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


def media(path: str | Path, video_id: Optional[str] = None,
          on_conflict: str = "new") -> Produced:
    """Describe the file, write `media.json`, report what was written.

    A video id defaults to the filename stem, so two different videos both
    called `clip.mp4` used to claim one output directory -- silently. The
    second `media.json` replaced the first while every other artifact stayed,
    leaving a directory whose grid covered 205 s of a 60 s file, both
    documents well-formed. `on_conflict` is what happens instead:

        new      mint the next free id -- clip-2, clip-3 (default)
        replace  delete everything under the old id and take it
        refuse   raise `VideoIdTaken`, so the caller decides

    **Only a *different* file is a conflict.** The same file again reuses its
    own directory whatever this says, because that is the ordinary case --
    `video_rag()` begins with it on every run -- and minting there would
    orphan the manifest, the descriptions and the vectors, so a re-run would
    pay a second time for everything already done. `Media.source` is what
    tells the two apart.

    `new` is the default because nothing is lost by it. It does mean the id
    can differ from the one asked for, which is why `Produced.video_id` is the
    answer and not the argument -- `video_rag()` already reads it back rather
    than trusting what it passed in.
    """
    if on_conflict not in ON_CONFLICT:
        raise UnknownOption(f"unknown on_conflict {on_conflict!r}; "
                            f"known: {', '.join(ON_CONFLICT)}")

    with logs.timed("media") as done:
        described = _settle(split(path, video_id), on_conflict)
        where = files.write(described.video_id, "media",
                              described.as_dict())
        done(video_id=described.video_id, duration_s=described.duration_s,
             has_video=described.has_video, has_audio=described.has_audio)
    return Produced(
        video_id=described.video_id,
        component="media",
        artifacts={"media": where},
        stats={"has_video": described.has_video, "has_audio": described.has_audio,
               "duration_s": described.duration_s,
               "container": described.container_format,
               # The id actually used, beside the one the filename asked for,
               # so a mint is visible in the receipt rather than only in a
               # field the caller might not compare.
               "source": described.source,
               "requested_id": video_id or Path(path).stem},
        # What this file does NOT carry. A later component reads this rather
        # than opening the file again to find out.
        skipped=([] if described.has_video else ["video"])
                + ([] if described.has_audio else ["audio"]),
    )


def _owner(video_id: str) -> Optional[Media]:
    """The `media.json` already under this id, or None if the id is free."""
    if not paths.exists(video_id, "media"):
        return None
    try:
        return load(video_id)
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


def _settle(described: Media, on_conflict: str) -> Media:
    """The id this file should actually be written under."""
    wanted = described.video_id
    stored = _owner(wanted)
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
        shutil.rmtree(paths.home(wanted), ignore_errors=True)
        return described

    for n in range(2, MAX_VARIANTS + 1):
        candidate = f"{wanted}-{n}"
        beside = _owner(candidate)
        # Free, or already this same file: a re-run of the second `clip.mp4`
        # has to land back on `clip-2` rather than minting `clip-3` every time.
        if beside is None or _is_same(beside, described):
            return replace(described, video_id=candidate)
    raise VideoIdTaken(
        f"{wanted} and {MAX_VARIANTS - 1} variants of it are all taken by "
        f"different files")


#: The uniform name every component also answers to: what a dispatch
#: table calls and what a form introspects. The same function object.
#: Named for the component, so a traceback frame says which one failed;
#: eight functions called `run` all read the same in a stack.
run = media


def load(video_id: str) -> Media:
    """Read back what `run` wrote. Every later component starts here."""
    return Media.from_dict(files.read_json(paths.require(video_id, "media")))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Describe the two streams a media file carries.")
    ap.add_argument("media", type=Path)
    ap.add_argument("--video-id", default=None, help="defaults to the filename stem")
    ap.add_argument("--on-conflict", default="new", choices=ON_CONFLICT,
                    help="when a DIFFERENT file already holds this id: "
                         "new mints clip-2, replace deletes the old output, "
                         "refuse raises. The same file always reuses its own "
                         "directory")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = run(args.media, args.video_id, args.on_conflict)
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

    m = load(produced.video_id)
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
    print(f"media -> {produced.artifacts['media']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
