"""What the pipeline does when a run does not need every component.

`check.py` proves the two addressings agree on a run that works. This is the
other half: the branches a driver takes when the request and the file do not
line up, which is most of what `driver.py` is for and none of what a component
can see.

    a stream the file does not carry      skipped, with the reason
    a policy derived from that stream     refused, naming the stream
    a request that contradicts itself     refused before anything runs
    a second file wanting a taken folder  minted, or refused, on request

The fixture is `samples/fixtures/cuts.mp4`, which carries no soundtrack -- the
case a sample with both streams cannot test.

Nothing here is paid or networked: `stub` describer, `hash` embedder, and the
one run that reads a soundtrack never happens.

    python -m proto.branches
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

SILENT = "samples/fixtures/cuts.mp4"       # video only, 60 s
OTHER = "samples/fixtures/slides.mp4"      # a different file, for a collision

FREE = dict(describer="stub", embedder="hash")

_problems: list[str] = []


def ok(claim: str, holds: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if holds else 'FAIL'} {claim}{'   ' + detail if detail else ''}")
    if not holds:
        _problems.append(claim)


def refused(claim: str, call, *expected: str) -> None:
    """The call must raise, and the message must carry every `expected` word."""
    try:
        call()
    except Exception as exc:                              # noqa: BLE001
        message = str(exc)
        missing = [word for word in expected if word not in message]
        ok(claim, not missing, f"-- {message[:110]}")
        if missing:
            _problems.append(f"{claim}: message lacks {missing}")
        return
    ok(claim, False, "-- it was accepted")


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="proto-branches-"))
    into = scratch / "out"
    into.mkdir(parents=True)

    import proto
    from proto.driver import Options, validate

    # ------------------------------------------- a stream the file has not got
    print("\n--- a file with no soundtrack, asked for both ---")
    run = proto.video_rag(SILENT, into, **FREE)
    ran = [step.component for step in run.steps]
    print(f"  ran      {', '.join(ran)}")
    print(f"  skipped  {run.skipped}")
    ok("audio is skipped with the reason",
       run.skipped.get("audio") == "the file carries no audio stream")
    ok("`audio` and `cut` did not run",
       "audio" not in ran and "cut" not in ran)
    ok("the picture still ran end to end",
       {"video", "describe", "embed"} <= set(ran))
    ok("`embed` wrote from descriptions alone",
       (run.home / "embedded.json").exists()
       and not (run.home / "transcript.json").exists())

    # A policy that reads that stream is not a skip -- there is no grid to be
    # had, so it is a refusal, and it names the file rather than the argument.
    refused("a vad grid on a silent file is refused",
            lambda: proto.video_rag(SILENT, into, policy="vad", **FREE),
            "vad", "audio", "none")

    # --------------------------------------------- contradictions, before work
    print("\n--- a request that contradicts itself ---")
    cases = [
        ("a policy from a stream the run is not reading",
         dict(policy="vad", use_audio=False), "soundtrack"),
        ("neither modality", dict(use_video=False, use_audio=False),
         "nothing to do"),
        ("an unknown sampler", dict(sampler="nope"), "unknown sampler"),
        ("an unknown question", dict(sampler="uniform:nope"), "unknown question"),
        ("an unknown policy", dict(policy="nope"), "policy must be one of"),
        ("an unknown conflict rule", dict(on_conflict="nope"), "on_conflict"),
    ]
    for claim, shape, word in cases:
        problems = validate(Options(source=Path(SILENT), into=into, **shape))
        ok(claim, any(word in p for p in problems),
           f"-- {'; '.join(problems)[:90]}")

    missing = validate(Options(source=Path("samples/ghost.mp4"), into=into))
    ok("a source that is not there", any("does not exist" in p for p in missing))

    file_in_the_way = scratch / "a-file"
    file_in_the_way.write_text("not a directory")
    blocked = validate(Options(source=Path(SILENT), into=file_in_the_way))
    ok("`into` pointing at a file",
       any("holds one folder per video" in p for p in blocked))

    # `validate` returns a list and `process` raises it, so a caller gets every
    # problem at once and a run gets none of them.
    refused("`process` raises what `validate` lists",
            lambda: proto.video_rag(SILENT, into, policy="vad",
                                    use_audio=False, **FREE),
            "soundtrack")

    # ------------------------------------- a second file wanting a taken folder
    print("\n--- two different files, one stem ---")
    elsewhere = scratch / "elsewhere"
    elsewhere.mkdir()
    collides = elsewhere / Path(SILENT).name          # same stem, other bytes
    shutil.copyfile(OTHER, collides)

    refused("`refuse` stops before anything else runs",
            lambda: proto.video_rag(collides, into, on_conflict="refuse", **FREE),
            "cuts")
    ok("the refused run wrote nothing new",
       sorted(p.name for p in into.iterdir()) == ["cuts"])

    minted = proto.video_rag(collides, into, **FREE)
    ok("`new` mints a folder beside the first",
       minted.home.name == "cuts-2", f"-- {minted.home.name}")
    ok("the first video's artifacts are untouched",
       (into / "cuts" / "manifest.json").exists()
       and (into / "cuts" / "media.json").exists())
    ok("the receipt is the only place the folder comes from",
       minted.home == into / minted.video_id)

    # The same file again is not a conflict: it is the ordinary second pass,
    # and minting there would orphan everything already paid for.
    same = proto.video_rag(SILENT, into, **FREE)
    ok("the same file resumes rather than minting",
       same.home == run.home, f"-- {same.home.name}")

    print(f"\n{'PASS' if not _problems else f'{len(_problems)} PROBLEM(S)'}")
    if not _problems:
        shutil.rmtree(scratch, ignore_errors=True)
    else:
        print(f"scratch kept at {scratch}")
    return 1 if _problems else 0


if __name__ == "__main__":
    sys.exit(main())
