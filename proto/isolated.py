"""The whole chain with the data root poisoned, into a folder of my own.

`paths.artifact`, `home`, `out_root`, `require` and `exists` are replaced with
functions that raise. If anything in `proto` resolves an artifact by id -- for
its own input, for a resume read, or to put a frame store somewhere -- this
stops rather than quietly writing where it would have.

The data root is also pointed at an empty directory, which must still be empty
at the end.

**Both ways in, because the pipeline is a caller like any other.** The
components are driven by hand first, and then `proto.video_rag` is given the
same video and a folder of its own. A driver is exactly the layer that would
be tempted to ask the library where a video lives, so it is the half worth
poisoning the table for.

What is *not* poisoned is `paths.ARTIFACTS` and `paths.DIRECTORIES`, and the
difference is the whole design: those are names read as data, where the five
functions above answer "where does video V keep its things". `layout()` uses
the first and never the second.

    python -m proto.isolated
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="proto-isolated-"))
    forbidden, into = scratch / "data-root", scratch / "mine"
    forbidden.mkdir()
    into.mkdir()

    import falconvar
    falconvar.configure(data_root=str(forbidden))

    from falconvar.shared import paths
    # Read before poisoning: the name table is data a pipeline composes with,
    # which is a different thing from resolving a path for someone.
    names = dict(paths.ARTIFACTS)

    def refuse(*a, **k):
        raise AssertionError(
            f"a filepath component resolved a data root: {a!r} {k!r}")

    for name in ("artifact", "home", "out_root", "require", "exists", "videos"):
        setattr(paths, name, refuse)

    import proto

    first = proto.media.media("samples/test.mp4", into)
    home = Path(first.stats["home"])
    at = {n: home / f for n, f in names.items()}
    store = home / "store"

    steps = [
        ("media", lambda: first),
        ("audio", lambda: proto.audio.audio(at["media"], at["raw_transcript"],
                                            transcriber="stub",
                                            diarizer="none")),
        ("evidence", lambda: proto.boundaries.evidence(
            at["cuts"], "scene", media=at["media"], stride=5)),
        ("boundaries", lambda: proto.boundaries.boundaries(
            at["media"], at["timeline"], policy="scene", cuts=at["cuts"])),
        ("video", lambda: proto.video.video(
            at["media"], at["timeline"], at["manifest"], store=store,
            sampler="uniform", max_per_chunk=2)),
        ("cut", lambda: proto.cut.cut(at["timeline"], at["raw_transcript"],
                                      at["transcript"])),
        ("describe", lambda: proto.describe.describe(
            at["manifest"], at["timeline"], store, at["descriptions"],
            describer="stub")),
        ("embed", lambda: proto.embed.embed(
            at["embedded"], descriptions=at["descriptions"],
            transcript=at["transcript"], timeline=at["timeline"],
            embedder="hash")),
    ]

    for name, step in steps:
        receipt = step()
        where = list(receipt.artifacts.values())
        print(f"  {name:12} {receipt.video_id:6} -> "
              f"{Path(where[0]).name if where else '(nothing)'}")

    # --------------------------------------------- and the same, as a pipeline
    #
    # `transcriber` is not a pipeline argument -- it lives on the component,
    # which is where a caller who wants it goes -- so the entry point is
    # patched to the free backends rather than the driver being asked for
    # them. Everything this is testing happens either side of that call.
    was = proto.audio.audio
    proto.audio.audio = lambda media, out, *a, **k: was(
        media, out, transcriber="stub", diarizer="none")
    try:
        mine = scratch / "mine-pipeline"
        mine.mkdir()
        print("\n  --- the pipeline, same poisoned table ---")
        run = proto.video_rag("samples/test.mp4", mine, policy="vad",
                              describer="stub", embedder="hash",
                              sampler="uniform")
    finally:
        proto.audio.audio = was

    print(f"  {'pipeline':12} {run.video_id:6} -> {run.home}")
    print(f"  {'':12} ran {', '.join(s.component for s in run.steps)}")
    print(f"  {'':12} wrote {', '.join(run.artifacts())}")

    left = sorted(p.name for p in forbidden.iterdir())
    print(f"\n  the data root holds: {left or 'nothing'}")
    if left:
        print("FAIL: something was written to the data root")
        return 1
    print("PASS: no data root")
    return 0


if __name__ == "__main__":
    sys.exit(main())
