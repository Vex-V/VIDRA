"""Both addressings, same settings, same video -- do the documents agree?

`falconvar/video_rag` is untouched, so it is the control. It runs by video id
into a throwaway data root; `proto` runs by filepath, with `media` creating
the folder and every later path composed inside it. Then every document is
compared key by key.

Two levels, because there are two things a caller can use:

    COMPONENTS    eight calls, wired by hand on both sides. The claim is that
                  a filepath component does the same work as an id one.
    THE PIPELINE  `falconvar.video_rag.video_rag` against `proto.video_rag`,
                  which additionally tests that the ordering and the
                  *branching* survived the move -- which policy runs first,
                  which components a run missing a modality skips, and what
                  `previous=` is handed on a second pass.

The pipeline half runs in the two shapes that take opposite branches:
picture-only on a scene grid (no `audio`, no `cut`) and soundtrack-only on a
vad grid (no `video`, no `describe`, no store). Between them every component
runs and every skip is taken.

Nothing here is paid or networked: `stub` transcriber, `none` diarizer,
`stub` describer, `hash` embedder.

    python -m proto.check
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

SOURCE = "samples/test.mp4"

POLICY = "scene"
STRIDE = 5
SAMPLER = "uniform"
MAX_PER_CHUNK = 2

#: Wall-clock, lifted out before comparing. Two runs of the same work cannot
#: agree on these and nothing is wrong when they do not.
TIMINGS = {"elapsed_s", "decode_s", "transcribe_s", "diarize_s", "scan_s",
           "detect_s", "ingest_s", "describe_s", "embed_s", "sample_s",
           "read_s", "encode_s", "ran_at", "seconds", "ms_per_frame"}

#: Differences that are the point rather than a fault. The manifest records
#: where the frames went, and the whole claim here is that the caller decides
#: that -- so the two runs must disagree on it, and agreeing would be the
#: thing worth reporting.
EXPECTED = {".config.frame_store.root"}


def strip(value):
    """The document with every timing key removed, recursively."""
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in TIMINGS}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def differences(a, b, where=""):
    """Every path at which two stripped documents disagree."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append(f"{where}.{key}: only on the right")
            elif key not in b:
                out.append(f"{where}.{key}: only on the left")
            else:
                out += differences(a[key], b[key], f"{where}.{key}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: {len(a)} entries vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += differences(x, y, f"{where}[{i}]")
        return out
    return [] if a == b else [f"{where}: {a!r} vs {b!r}"]


# ----------------------------------------------------------- the two paths

def by_id(root: Path):
    """The control: `falconvar`, unchanged, addressed by video id."""
    import falconvar
    falconvar.configure(data_root=str(root))
    from falconvar.video_rag import (audio, boundaries, cut, describe, embed,
                                     media, video)

    out = []
    first = media.run(SOURCE)
    vid = first.video_id
    out.append(first)
    out.append(audio.run(vid, transcriber="stub", diarizer="none"))
    out.append(boundaries.evidence(vid, POLICY, stride=STRIDE))
    out.append(boundaries.run(vid, policy=POLICY))
    out.append(video.run(vid, sampler=SAMPLER, max_per_chunk=MAX_PER_CHUNK))
    out.append(cut.run(vid))
    out.append(describe.run(vid, describer="stub"))
    out.append(embed.run(vid, embedder="hash"))
    return out, vid


def by_path(into: Path):
    """`proto`: `media` makes the folder, every later path is composed in it.

    This is what the pipeline we have not written yet would do, spelled out.
    The artifact filenames come from the library's own table rather than
    being typed here -- `transcript.raw.json` is not `raw_transcript.json`,
    and a pipeline spelling it itself is one typo from a second convention.
    """
    import proto
    from falconvar.shared.paths import ARTIFACTS

    first = proto.media.media(SOURCE, into)
    home = Path(first.stats["home"])
    at = {name: home / filename for name, filename in ARTIFACTS.items()}
    store = home / "store"

    out = [first]
    out.append(proto.audio.audio(at["media"], at["raw_transcript"],
                                 transcriber="stub", diarizer="none"))
    out.append(proto.boundaries.evidence(at["cuts"], POLICY,
                                         media=at["media"], stride=STRIDE))
    out.append(proto.boundaries.boundaries(at["media"], at["timeline"],
                                           policy=POLICY, cuts=at["cuts"]))
    out.append(proto.video.video(at["media"], at["timeline"], at["manifest"],
                                 store=store, sampler=SAMPLER,
                                 max_per_chunk=MAX_PER_CHUNK))
    out.append(proto.cut.cut(at["timeline"], at["raw_transcript"],
                             at["transcript"]))
    out.append(proto.describe.describe(at["manifest"], at["timeline"], store,
                                       at["descriptions"], describer="stub"))
    out.append(proto.embed.embed(at["embedded"],
                                 descriptions=at["descriptions"],
                                 transcript=at["transcript"],
                                 timeline=at["timeline"], embedder="hash"))
    return out, at, store


# --------------------------------------------------------------- the pipeline

def compare(produced: Path, home: Path) -> int:
    """Every document and every frame in two folders. Returns problems."""
    from proto.driver import layout

    problems = 0
    for artifact, theirs in sorted(layout(home).items()):
        if artifact in ("store", "aggregates"):
            continue
        ours = produced / theirs.name
        if ours.exists() != theirs.exists():
            side = "the id run" if ours.exists() else "the filepath run"
            print(f"  {artifact:15} only {side} wrote it")
            problems += 1
            continue
        if not ours.exists():
            continue
        diff = [d for d in differences(
            strip(json.loads(ours.read_text())),
            strip(json.loads(theirs.read_text())))
            if d.split(":")[0] not in EXPECTED]
        if diff:
            problems += 1
            print(f"  {artifact:15} {len(diff)} difference(s)")
            for line in diff[:6]:
                print(f"       {line}")
        else:
            print(f"  {artifact:15} identical")

    store, control_store = home / "store", produced / "store"
    if store.exists() or control_store.exists():
        ours = sorted(f.name for f in control_store.glob("*.jpg"))
        theirs = sorted(f.name for f in store.glob("*.jpg"))
        same = sum(1 for n in ours if n in theirs
                   and (control_store / n).read_bytes() == (store / n).read_bytes())
        print(f"  {'frames':15} {same}/{len(ours)} byte-identical "
              f"({len(theirs)} on the filepath side)")
        if same != len(ours) or len(ours) != len(theirs):
            problems += 1
    return problems


def free_audio():
    """Both drivers' audio step, on the backends that cost nothing.

    A pipeline deliberately carries no per-stage tuning -- `transcriber` lives
    on the component, which is where a caller who wants it goes. So neither
    driver can be *asked* for a stub, and the two entry points are patched
    instead: identically, on both sides, so the claim under test is untouched
    while this oracle stays free of Whisper and a Hugging Face token.

    Returns the undo.
    """
    import falconvar.video_rag.audio as control
    import proto.audio as ours
    was_control, was_ours = control.run, ours.audio

    control.run = lambda vid, *a, **k: was_control(
        vid, transcriber="stub", diarizer="none")
    ours.audio = lambda media, out, *a, **k: was_ours(
        media, out, transcriber="stub", diarizer="none")

    def undo() -> None:
        control.run, ours.audio = was_control, was_ours
    return undo


def paid(run, component: str) -> int:
    """What a stage actually did, for the two that can resume."""
    key = "described" if component == "describe" else "embedded"
    for receipt in run.steps:
        if receipt.component == component:
            return int(receipt.stats.get(key) or 0)
    return -1                                    # the stage did not run


def pipelines(scratch: Path) -> int:
    """The same extraction through both drivers, in two shapes."""
    import falconvar
    import proto
    from falconvar.video_rag.driver import video_rag as by_id_pipeline

    settings = dict(describer="stub", embedder="hash")
    shapes = [
        # Picture only: evidence decodes and scores, `audio` and `cut` never
        # run, and `embed` is handed descriptions alone.
        ("picture", dict(policy="scene", use_audio=False, sampler=SAMPLER)),
        # Soundtrack only: the grid is derived from the transcript, so `audio`
        # must finish first -- and `video`, `describe` and the store are
        # absent, which `embed` has to tolerate.
        ("soundtrack", dict(policy="vad", use_video=False)),
    ]

    problems = 0
    undo = free_audio()
    try:
        for name, shape in shapes:
            root, into = scratch / f"{name}-id", scratch / f"{name}-path"
            into.mkdir(parents=True)
            spelled = ", ".join(f"{k}={v!r}" for k, v in shape.items())
            print(f"\n=== the pipeline, {name}: {spelled} ===")

            falconvar.configure(data_root=str(root))
            t0 = time.time()
            theirs = by_id_pipeline(SOURCE, **settings, **shape)
            control_s = time.time() - t0

            t0 = time.time()
            ours = proto.video_rag(SOURCE, into, **settings, **shape)
            print(f"  by id {control_s:.1f}s -> {theirs.video_id}   "
                  f"by path {time.time() - t0:.1f}s -> {ours.home.name}")

            ran = [step.component for step in ours.steps]
            print(f"  ran      {', '.join(ran)}")
            print(f"  skipped  {', '.join(ours.skipped) or '-'}")
            if [step.component for step in theirs.steps] != ran:
                print(f"  PROBLEM: the id run ran "
                      f"{[step.component for step in theirs.steps]}")
                problems += 1

            problems += compare(root / "out" / theirs.video_id, ours.home)

            # A second pass over the same folder. `media` fingerprints the
            # same file, so the id is not minted; `previous=` goes to the two
            # components that can resume, which is what the id pipeline did
            # silently by reading their own last output.
            again = proto.video_rag(SOURCE, into, **settings, **shape)
            if again.home != ours.home:
                print(f"  PROBLEM: a re-run minted {again.home.name}")
                problems += 1
            for component in ("describe", "embed"):
                did = paid(again, component)
                if did > 0:
                    print(f"  PROBLEM: a second pass paid for {did} in {component}")
                    problems += 1
            print(f"  re-run   same folder, describe={paid(again, 'describe')} "
                  f"embed={paid(again, 'embed')} (-1 = did not run)")

            # And the same folder told to do it all again, which is the thing
            # addressing by id had no way to ask for.
            fresh = proto.video_rag(SOURCE, into, resume=False, **settings,
                                    **shape)
            for component in ("describe", "embed"):
                did = paid(fresh, component)
                if did == 0:
                    print(f"  PROBLEM: resume=False left {component} doing nothing")
                    problems += 1
            print(f"  no resume describe={paid(fresh, 'describe')} "
                  f"embed={paid(fresh, 'embed')}")
    finally:
        undo()
    return problems


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="proto-"))
    root, into = scratch / "by-id", scratch / "by-path"
    into.mkdir(parents=True)

    print(f"scratch {scratch}\n")

    print("--- by id (falconvar, the control) ---")
    t0 = time.time()
    id_receipts, vid = by_id(root)
    for r in id_receipts:
        print(f"  {r.component:22} {list(r.artifacts)}")
    print(f"  {time.time() - t0:.1f}s, video_id={vid}\n")

    print("--- by filepath (proto) ---")
    t0 = time.time()
    path_receipts, at, store = by_path(into)
    for r in path_receipts:
        print(f"  {r.component:22} {list(r.artifacts)}")
    print(f"  {time.time() - t0:.1f}s, home={path_receipts[0].stats['home']}\n")

    # ------------------------------------------------------- the comparison
    produced = root / "out" / vid
    print("--- documents ---")
    problems = 0
    for artifact, theirs in at.items():
        ours = produced / Path(theirs).name
        if not ours.exists():
            print(f"  {artifact:15} MISSING on the id path: {ours}")
            problems += 1
            continue
        diff = [d for d in differences(
            strip(json.loads(ours.read_text())),
            strip(json.loads(Path(theirs).read_text())))
            if d.split(":")[0] not in EXPECTED]
        if diff:
            problems += 1
            print(f"  {artifact:15} {len(diff)} difference(s)")
            for line in diff[:6]:
                print(f"       {line}")
            if len(diff) > 6:
                print(f"       ... and {len(diff) - 6} more")
        else:
            print(f"  {artifact:15} identical")
    for expected in sorted(EXPECTED):
        print(f"  {'(by design)':15} {expected} differs, as it must")

    # ------------------------------------------------------------ the pixels
    theirs = sorted(p.name for p in store.glob("*.jpg"))
    ours = sorted(p.name for p in (produced / "store").glob("*.jpg"))
    same = sum(1 for n in ours
               if n in theirs
               and (produced / "store" / n).read_bytes() == (store / n).read_bytes())
    print(f"\n--- frames ---\n  {same}/{len(ours)} byte-identical "
          f"({len(theirs)} on the filepath side)")
    if same != len(ours) or len(ours) != len(theirs):
        problems += 1

    # ------------------------------------------------------------- resume
    # `previous=` is an argument now. The diff lives in `encode`, so a caller
    # reaching the verb gets it too -- which is what it could not do before.
    print("\n--- resume, with previous= ---")
    import proto
    again = proto.embed.embed(at["embedded"], descriptions=at["descriptions"],
                              transcript=at["transcript"],
                              previous=at["embedded"],
                              timeline=at["timeline"], embedder="hash")
    print(f"  embed      embedded={again.stats['embedded']} "
          f"unchanged={again.stats['unchanged']}")
    if again.stats["embedded"] or not again.stats["unchanged"]:
        problems += 1
        print("  PROBLEM: a second run re-embedded")

    once_more = proto.describe.describe(at["manifest"], at["timeline"], store,
                                        at["descriptions"],
                                        previous=at["descriptions"],
                                        describer="stub")
    print(f"  describe   described={once_more.stats['described']} "
          f"skipped={once_more.stats.get('skipped')}")
    if once_more.stats["described"]:
        problems += 1
        print("  PROBLEM: a second run re-described")

    after = differences(strip(json.loads(Path(at["embedded"]).read_text())),
                        strip(json.loads((produced / "embedded.json").read_text())))
    print(f"  vectors still identical to the control: {not after}")
    if after:
        problems += 1

    # ------------------------------------------------- a mismatched pairing
    # Addressed by id this could not happen: `paths.artifact` is a join, so
    # two artifacts read under one id are that video's. Addressed by path it
    # can, so the refusal has to be written down -- `_io.same_video`.
    print("\n--- two different videos, one call ---")
    other = proto.media.media("samples/test2.mp4", into)
    elsewhere = Path(other.stats["home"]) / "media.json"
    try:
        proto.boundaries.boundaries(elsewhere, into / "wrong.json",
                                    policy=POLICY, cuts=at["cuts"])
        print("  PROBLEM: accepted a grid built from another video")
        problems += 1
    except ValueError as exc:
        print(f"  refused: {exc}")

    # ------------------------------------------------- a missing input names
    print("\n--- a missing input ---")
    try:
        proto.cut.cut(into / "nope" / "timeline.json", at["raw_transcript"],
                      into / "out.json")
        print("  PROBLEM: read a timeline that is not there")
        problems += 1
    except FileNotFoundError as exc:
        print(f"  {exc}")

    problems += pipelines(scratch)

    print(f"\n{'PASS' if not problems else f'{problems} PROBLEM(S)'}")
    if not problems:
        shutil.rmtree(scratch, ignore_errors=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
