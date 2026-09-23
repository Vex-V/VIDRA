"""The library, one component at a time.

    python example.py [path/to/video.mp4]

Each component of `falconvar.video_rag` gets a section here as it is gone
through: what it takes, what it exposes, and what it leaves behind. Run it and
it does the work and prints all three.

**The module is the unit -- `media.run`, never a bare `media`.** The function
in `media/driver.py` *is* named `media`, so a traceback frame says which
component failed rather than being one of eight frames called `run`, but only
`run` is exported: `load` appears in six of the eight components and `build` in
three, so a bare-name style needs an alias the moment a caller wants a second
thing from one module.

**`run` and `load` are two directions, not two steps.** `run` does the work and
writes; `load` reads the result back, typed. Nothing in a pipeline written by
hand calls `load` to make the pipeline work -- each `run` resolves its own
inputs through the other components' `load`, which is why nothing is passed
between these calls but an id and settings.

**And there is a third name: the verb that does the work on objects.** `run` is
that verb with a read at each end, so a caller holding the documents already
can skip the filesystem entirely -- `media.split` is the one this component
has, and the section below shows the whole chain built that way.

Nothing here needs a checkout. `falconvar.configure(data_root=...)` says where
to write, and without it an installed copy writes under `~/.falconvar`.

So far: 1 · media, what it does when two files want one id, and the same eight
components driven without touching a data root at all.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from falconvar.shared import paths
from falconvar.video_rag import media

SOURCE = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/Chernobyl.mp4")


def component_media(source: Path) -> str:
    """1 · media -- a file path in, `media.json` out.

    The only component that takes a **path** rather than a video id, because it
    is the one that establishes the id. Everything after it is addressed by
    `(video_id, backend)` and `shared/paths.py` resolves the rest -- which is
    what lets one API route run any component, and why `media` is the one
    component that route cannot reach.

        run(path, video_id=None, sink="file", on_conflict="new") -> Produced
        load(video_id)                                           -> Media
        split(path, video_id=None)              -> Media, writing nothing

    That is the whole surface, plus the two errors it raises. `fingerprint`
    and `main` are reached through the modules they live in: a library that
    offers six ways in has to explain all six.
    """
    # `run` describes the file and writes it. `video_id` defaults to the
    # filename stem; it is also a directory name, so it is checked rather than
    # trusted -- see `paths.check_id`.
    produced = media.run(source)
    video_id = produced.video_id

    # `Produced` is a receipt, not the document: what was written, where, and a
    # few headline numbers. `skipped` names the half the file does not carry,
    # so a later component reads that here instead of opening the file again.
    print(f"  run      {video_id}  <- {source}")
    print(f"           stats    {produced.stats}")
    print(f"           wrote    {produced.artifacts['media']}")
    print(f"           skipped  {produced.skipped or '-- carries both'}")

    # `load` is the other direction: the typed, frozen document back.
    described = media.load(video_id)
    print(f"  load     {type(described).__name__}  {described.container_format}"
          f"  {described.duration_s:.3f}s")
    if described.video:
        v = described.video
        print(f"           video    {v.codec} {v.width}x{v.height} {v.rate:g}fps"
              f"  time_base={v.time_base}  frames={v.frames}")
    if described.audio:
        a = described.audio
        print(f"           audio    {a.codec} {a.rate}Hz {a.channels}ch")

    # `duration_s` is the *container's*, and it is the one both halves must
    # agree to use. The streams routinely disagree -- and a grid built from the
    # shorter one leaves the tail of the longer outside every chunk.
    if described.video and described.audio:
        gap = abs((described.video.duration_s or 0) - (described.audio.duration_s or 0))
        print(f"           the two streams differ by {gap:.3f}s; the container "
              f"says {described.duration_s:.3f}s, and that is the one the grid uses")

    # `split` does the same work and writes nothing -- for asking what a file
    # is before committing a directory to it. A second way *in*, which is why
    # it is on the surface where `fingerprint` and `main` are not.
    probed = media.split(source, video_id="not-written")
    print(f"  split    same Media, wrote nothing: "
          f"{paths.exists('not-written', 'media') is False}"
          f"  (has_video={probed.has_video} has_audio={probed.has_audio})")

    # `source` says what file this is, independently of its name or where it
    # sits. It costs one `stat()` on top of a `split` that happened anyway.
    print(f"  source   {described.source}   <- the same bytes under any name")

    return video_id


def collisions(source: Path, scratch: Path) -> None:
    """2 · what happens when two different videos share a filename.

    A video id defaults to the filename stem, so `monday/clip.mp4` and
    `tuesday/clip.mp4` both ask for `clip`. That used to be a silent
    overwrite -- and worse than losing one document, because only `media.json`
    was replaced: the grid, the frame store and the descriptions of the *first*
    video stayed, each still well-formed, leaving a directory whose timeline
    covered 205 s of a 60 s file.

    `on_conflict` decides, and `Media.source` is what makes the decision
    possible: only a **different** file is a conflict.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    same_name = scratch / "clip.mp4"

    # Write this section's output somewhere disposable. `configure` is the
    # library's own knob and it takes precedence over the checkout, so the
    # section below writes two whole videos without touching `data/out/`.
    # Paths are computed per call, so switching it back at the end is enough.
    was = paths.data_root()
    paths.configure(data_root=scratch / "out")

    # The ordinary case, and the reason the fingerprint has to exist: running
    # `media` on a file it has already seen must reuse that directory. Minting
    # here would orphan the manifest, the descriptions and the vectors, so the
    # next run would pay a second time for everything already done.
    shutil.copy(source, same_name)
    first = media.run(same_name)
    again = media.run(same_name)
    print(f"  the same file twice -> {first.video_id!r} then {again.video_id!r}"
          f"   reused: {first.video_id == again.video_id}")

    # A different video arriving under the same filename. `new` is the default
    # because nothing is lost by it.
    other = next(p for p in (Path("samples/test2.mp4"), Path("samples/test1.mp4"))
                 if p.exists() and media.split(p).source != first.stats["source"])
    shutil.copy(other, same_name)
    minted = media.run(same_name)
    print(f"  a different file    -> asked {minted.stats['requested_id']!r}, "
          f"got {minted.video_id!r}   ({other.name})")
    print(f"  the first is intact -> {media.load(first.video_id).duration_s}s "
          f"vs {media.load(minted.video_id).duration_s}s")

    # And it is stable: a re-run of the second file lands back on the id it was
    # given, rather than minting clip-3, clip-4 on every pass.
    print(f"  re-run of that      -> {media.run(same_name).video_id!r}   stable")

    # `refuse` hands the decision back. `replace` deletes the old output
    # *directory*, not just its `media.json` -- overwriting the one document is
    # exactly the corruption above.
    try:
        media.run(same_name, video_id=first.video_id, on_conflict="refuse")
    except media.VideoIdTaken as exc:
        print(f"  on_conflict=refuse  -> VideoIdTaken: {str(exc).split(';')[0]}")

    print(f"  two videos, two ids -> {paths.videos()}")
    paths.configure(data_root=was)


def refusals(source: Path) -> None:
    """What the component refuses, and why each one is a refusal.

    Every deliberate failure is a `FalconvarError` and keeps a builtin base, so
    `except ValueError` already written still fires and an embedding
    application can still catch "anything the library refused" in one clause.
    """
    cases = (
        ("a file that is not there", lambda: media.run("samples/nope.mp4")),
        ("a file that is not media", lambda: media.run("CLAUDE.md")),
        ("a conflict rule nobody has",
         lambda: media.run(source, on_conflict="nope")),
        # An id becomes a directory name. `../../escaped` used to write two
        # levels *above* the data root while `Produced.video_id` read back
        # unchanged, so nothing reported that a run had left its own tree.
        ("an id that climbs out", lambda: media.run(source, video_id="../../escaped")),
        # A leading underscore marks a directory that is not a video. Enforced
        # on read only, it wrote a correct output directory that `videos()` --
        # and so `GET /videos` -- would never list again.
        ("an id that hides itself", lambda: media.run(source, video_id="_hidden")),
    )
    for label, call in cases:
        try:
            call()
            print(f"  {label:<26} -- NOT REFUSED")
        except Exception as exc:                            # noqa: BLE001
            import falconvar
            base = "FalconvarError" if isinstance(exc, falconvar.FalconvarError) else "!"
            print(f"  {label:<26} {type(exc).__name__} ({base})")
            print(f"  {'':<26} {exc}")


def without_a_filesystem(source: Path) -> None:
    """The whole chain on objects, writing nothing anywhere.

    Every component has a verb beside `run` that takes the documents it needs
    and returns the one it makes. Together they are the pipeline with the
    files taken out: no `configure()`, no data root, no output directory --
    which is what lets one component be used without adopting this pipeline's
    layout.

    `MemoryFrames` is what carries the pixels between `video` and `describe`,
    the one thing components hand each other that is not a document. It is a
    separate class rather than a flag because the memory is the caller's to
    accept: this is ~170 MB for a five-minute video at 1 fps.
    """
    from falconvar.video_rag import audio, boundaries, cut, describe, video
    from falconvar.video_rag.frames import MemoryFrames

    described = media.split(source)
    raw = audio.listen(described, transcriber="stub", diarizer="none")
    cuts = boundaries.detect("scene", media=described)
    grid = boundaries.timeline(described, "scene", cuts=cuts)

    held = MemoryFrames()
    manifest = video.ingest(described, grid, "uniform", frames=held)
    transcript = cut.apply(grid, raw)
    answers = describe.answer(manifest, grid, held, describer="stub")

    print(f"  split      {described.video_id}  {described.duration_s:.3f}s")
    print(f"  listen     {raw.stats['segments']} segments, "
          f"{raw.stats['words']} words")
    print(f"  detect     {len(cuts.cuts)} cuts   ({cuts.detector})")
    print(f"  timeline   {len(grid)} chunks   {grid.fingerprint()}")
    print(f"  ingest     {manifest.stats['frames_sampled']} frames, "
          f"{held.bytes_written / 1024 / 1024:.1f} MB in memory")
    print(f"  apply      {transcript.stats['chunks']} chunks, "
          f"{transcript.stats['words']} words placed")
    print(f"  answer     {answers.stats['described']} described")
    print(f"  and on disk, under {paths.out_root()}: nothing")


if __name__ == "__main__":
    if not SOURCE.exists():
        raise SystemExit(f"no such file: {SOURCE}")

    print("\n=== 1 - media ===")
    video_id = component_media(SOURCE)

    # In a temporary directory, because it writes two videos under one
    # filename and the point is what the *output* tree does about it.
    print("\n=== 2 - two videos, one filename ===")
    with tempfile.TemporaryDirectory() as tmp:
        collisions(SOURCE, Path(tmp))

    print("\n=== what it refuses ===")
    refusals(SOURCE)

    # In a temporary root, so "nothing was written" is a claim about a
    # directory nobody else has touched.
    print("\n=== the same components, with no filesystem at all ===")
    with tempfile.TemporaryDirectory() as tmp:
        was = paths.data_root()
        paths.configure(data_root=tmp)
        without_a_filesystem(SOURCE)
        paths.configure(data_root=was)

    print(f"\n=== on disk ===")
    print(f"  {paths.home(video_id)}")
    print(f"  artifacts: {paths.present(video_id)}")
    print(f"  videos():  {paths.videos()}")
