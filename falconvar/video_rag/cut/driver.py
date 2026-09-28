"""The cut component: a grid + a raw transcript -> `transcript.json`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from falconvar.shared import logs
from falconvar.shared.contracts.documents import (Produced, RawTranscript,
                                                  Timeline, Transcript)
from falconvar.shared.storage.files import read, write
from falconvar.shared.contracts.documents import same_video
from .cutter import stats_for, to_chunks


def apply(timeline: Timeline, raw: RawTranscript) -> Transcript:
    """The grid, onto a transcript already in hand. Reads and writes nothing.

    This is the component's whole work, and `cut` below is this plus a read
    at each end. Nothing here needs a path at all, so a caller with a
    `Timeline` and a `RawTranscript` -- from this pipeline, from an earlier
    run, or built by hand -- can cut one against the other.

    The id comes off the transcript rather than being an argument: `cut`
    re-chunks *that* transcript, and a third opinion about which video it is
    could only ever disagree with the two documents.
    """
    chunks = to_chunks(raw, timeline)
    return Transcript(
        video_id=raw.video_id,
        timeline_fingerprint=timeline.fingerprint(),
        model=raw.model,
        chunks=chunks,
        stats=stats_for(raw, chunks),
    )


def cut(timeline: str | Path, raw_transcript: str | Path,
        out: str | Path) -> Produced:
    """Apply the grid at `timeline` to the transcript at `raw_transcript`.

    Costs no model and can be repeated at will. `apply` plus a read at each
    end, with the paths named rather than derived from a video id.
    """
    grid = read(timeline, Timeline)
    raw = read(raw_transcript, RawTranscript)
    video_id = same_video(timeline=grid, raw_transcript=raw)

    with logs.timed("cut", video_id) as done:
        transcript = apply(grid, raw)
        stats = transcript.stats
        where = write(out, transcript)
        done(chunks=stats.get("chunks"), words=stats.get("words"))
    return Produced(
        video_id=video_id, component="cut",
        artifacts={"transcript": where},
        stats={**stats, "timeline_fingerprint": transcript.timeline_fingerprint},
    )


def load(path: str | Path) -> Transcript:
    """Read a `transcript.json` back, typed."""
    return read(path, Transcript)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Cut a transcript to the grid.")
    ap.add_argument("timeline", help="path to timeline.json")
    ap.add_argument("raw_transcript", help="path to transcript.raw.json")
    ap.add_argument("out", help="where to write transcript.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = cut(args.timeline, args.raw_transcript, args.out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    print(f"{produced.video_id}")
    print(f"  chunks       {s['chunks']}   ({s['chunks_with_speech']} with speech)")
    print(f"  words        {s['words']}/{s['words_in_transcript']} placed")
    if s["words_outside_grid"]:
        print(f"  outside grid {s['words_outside_grid']}")
    print(f"  speakers     {s['speakers']}")
    print(f"\ntranscript -> {produced.artifacts['transcript']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
