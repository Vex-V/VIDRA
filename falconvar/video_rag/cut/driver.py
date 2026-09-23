"""The cut component: `transcript.raw.json` + `timeline.json` -> `transcript.json`."""

from __future__ import annotations

from typing import Optional

from ..boundaries import load as load_timeline
from ..audio import load as load_raw
from ...shared import logs, paths
from ...shared.storage import files
from ...shared.contracts.documents import (Produced, RawTranscript,
                                           Timeline, Transcript)
from .cutter import stats_for, to_chunks


def apply(timeline: Timeline, raw: RawTranscript) -> Transcript:
    """The grid, onto a transcript already in hand. Reads and writes nothing.

    This is the component's whole work, and `run` below is this plus a read
    at each end. Nothing here needs `configure()` or a data root, so a caller
    with a `Timeline` and a `RawTranscript` -- from this pipeline, from an
    earlier run, or built by hand -- can cut one against the other.

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


def cut(video_id: str) -> Produced:
    """Apply the grid. Costs no model and can be repeated at will."""
    with logs.timed("cut", video_id) as done:
        raw = load_raw(video_id)
        timeline = load_timeline(video_id)

        transcript = apply(timeline, raw)
        stats = transcript.stats
        where = files.write(video_id, "transcript", transcript.as_dict())
        done(chunks=stats.get("chunks"), words=stats.get("words"))
    return Produced(
        video_id=video_id, component="cut",        artifacts={"transcript": where},
        stats={**stats, "timeline_fingerprint": transcript.timeline_fingerprint},
    )


#: The uniform name every component also answers to: what a dispatch
#: table calls and what a form introspects. The same function object.
#: See `media/driver.py`.
run = cut


def load(video_id: str) -> Transcript:
    return Transcript.from_dict(
        files.read_json(paths.require(video_id, "transcript")))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Cut a transcript to the grid.")
    ap.add_argument("video_id")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = run(args.video_id)
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
