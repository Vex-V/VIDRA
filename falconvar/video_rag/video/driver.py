"""The ingest component: `media.json` + `timeline.json` -> `manifest.json` and
the frame store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from falconvar.shared.reporting import logs, progress
from falconvar.shared.contracts.documents import (Manifest, Media, Produced,
                                           Timeline)
from falconvar.shared.storage.files import read, write
from falconvar.shared.contracts.documents import same_video
from . import samplers as samplers_mod
#: The pipeline's `ingest` takes built samplers; this module's takes a spec.
from .pipeline import ingest as _pass
from .reader import UnreadableSource
from ..helpers import FrameStore
from falconvar.shared.reporting.errors import Refused, UnknownOption


def split_specs(sampler: str | Sequence[str]) -> list[str]:
    """`"clip:[text,scene],yolo"` -> `["clip:[text,scene]", "yolo"]`, aware of
    brackets.
    """
    if not isinstance(sampler, str):
        return [s.strip() for s in sampler if str(s).strip()]
    out, depth, current = [], 0, []
    for ch in sampler:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(current))
            current = []
            continue
        current.append(ch)
    out.append("".join(current))
    return [s.strip() for s in out if s.strip()]


def parse_spec(spec: str) -> tuple[str, list[str]]:
    """`"clip:[text,scene]"` -> `("clip", ["text", "scene"])`. `clip:a+b` is the
    same as `clip:[a,b]`; a bare name asks the sampler's own question.
    """
    name, sep, rest = spec.partition(":")
    name, rest = name.strip(), rest.strip()
    if not sep or not rest:
        return name, []
    if rest.startswith("[") and rest.endswith("]"):
        rest = rest[1:-1]
    parts = [q.strip() for q in rest.replace("+", ",").split(",")]
    seen, questions = set(), []
    for q in parts:
        if q and q not in seen:          # a repeat would pay for the same call twice
            seen.add(q)
            questions.append(q)
    return name, questions


#: Setting -> the samplers that read it. Settings absent here apply to every
#: sampler.
SAMPLER_SETTINGS: dict[str, tuple[str, ...]] = {
    "every_n": ("uniform",),
    # Samplers that keep a frame when it changed enough.
    "threshold": ("clip", "yolo", "objects", "text"),
    "vocabulary": ("objects",),
    "confidence": ("objects",),
    "languages": ("text",),
}


def build_samplers(specs: Sequence[str], every_n: Optional[int] = None,
                   min_interval_s: float = 0.0,
                   max_per_chunk: Optional[int] = None,
                   threshold: Optional[float] = None,
                   vocabulary: Optional[Sequence[str]] = None,
                   confidence: Optional[float] = None,
                   languages: Optional[Sequence[str]] = None,
                   questions: Optional[Sequence[str]] = None
                   ) -> list[samplers_mod.Sampler]:
    """`["yolo", "clip:[text,scene]"]` -> sampler objects.

    A name may carry one question after a colon or several in a list; unpaired,
    the question is the sampler's own name. Specs naming the same sampler merge
    into one run, so `clip:text,clip:scene` means `clip:[text,scene]`.
    `questions` is the vocabulary to check against; None accepts any name.
    `threshold` is how much a frame must change to be kept; `confidence` is the
    detector's box threshold; `languages` is what the OCR reads.
    """
    given = {"every_n": every_n, "threshold": threshold,
             "vocabulary": vocabulary, "confidence": confidence,
             "languages": languages}

    rate = {"min_interval_s": min_interval_s, "max_per_chunk": max_per_chunk}
    tuned = {} if threshold is None else {"threshold": threshold}
    stride = {} if every_n is None else {"every_n": every_n}
    sure = {} if confidence is None else {"confidence": confidence}
    reads = {} if languages is None else {"languages": list(languages)}

    grouped: dict[str, list[str]] = {}
    for spec in [s.strip() for s in specs if s.strip()]:
        name, asked = parse_spec(spec)
        for question in asked:
            if questions is not None and question not in questions:
                raise UnknownOption(
                    f"{spec!r}: unknown question {question!r}; "
                    f"known: {', '.join(questions)}")
        merged = grouped.setdefault(name, [])
        for question in asked:
            if question not in merged:
                merged.append(question)

    # A setting no chosen sampler reads is refused.
    unreachable = sorted(s for s, value in given.items()
                         if value is not None
                         and not set(SAMPLER_SETTINGS[s]) & set(grouped))
    if unreachable:
        detail = "; ".join(f"{s} is read by "
                           f"{', '.join(SAMPLER_SETTINGS[s])}"
                           for s in unreachable)
        raise Refused(
            f"no chosen sampler reads {', '.join(unreachable)} "
            f"(chosen: {', '.join(sorted(grouped))}) -- {detail}")

    built: list[samplers_mod.Sampler] = []
    for name, asked in grouped.items():
        ask = {"prompts": asked} if asked else {}
        if name == "uniform":
            built.append(samplers_mod.build(name, **ask, **stride, **rate))
        elif name == "objects":
            built.append(samplers_mod.build(name, vocabulary=list(vocabulary)
                                            if vocabulary else None,
                                            **ask, **tuned, **sure, **rate))
        elif name == "text":
            built.append(samplers_mod.build(name, **ask, **tuned, **reads, **rate))
        else:
            # Unset thresholds keep each sampler's own default.
            built.append(samplers_mod.build(name, **ask, **tuned, **rate))
    return built


def ingest(media: Media, timeline: Timeline,
           sampler: str | Sequence[str] = "uniform",
           per_second: float = 1.0,
           every_n: Optional[int] = None,
           min_interval_s: float = 0.0,
           max_per_chunk: Optional[int] = None,
           threshold: Optional[float] = None,
           vocabulary: Optional[Sequence[str]] = None,
           confidence: Optional[float] = None,
           languages: Optional[Sequence[str]] = None,
           frames: Optional[FrameStore] = None,
           store_scope: str = "sampled",
           on_progress: Optional[progress.Reporter] = None) -> Manifest:
    """One decode pass over the picture, onto a given grid. Reads the file
    `media.path` names and writes no artifact.

    `frames` is where kept frames go (a `FrameStore`, `MemoryFrames`), or None to
    keep none. Questions are checked against the vocabulary before anything
    decodes.
    """
    from ..describe import prompts

    # Built before anything decodes, so a bad argument fails at once.
    built = build_samplers(split_specs(sampler), every_n, min_interval_s,
                           max_per_chunk, threshold, vocabulary, confidence,
                           languages, questions=prompts.questions())
    # Progress is reported per chunk.
    total = len(timeline)
    seen = 0

    def chunk_done(chunk_id: int, chunk: dict) -> None:
        nonlocal seen
        seen += 1
        progress.report(on_progress, "video", total, seen, 0,
                        str(chunk_id), chunk)

    progress.report(on_progress, "video", total, 0)
    return _pass(media, timeline, built, per_second, frames, store_scope,
                 chunk_done if on_progress is not None else None)


def video(media: str | Path, timeline: str | Path, out: str | Path,
          store: Optional[str | Path] = None,
          sampler: str | Sequence[str] = "uniform",
          per_second: float = 1.0,
          every_n: Optional[int] = None,
          min_interval_s: float = 0.0,
          max_per_chunk: Optional[int] = None,
          threshold: Optional[float] = None,
          vocabulary: Optional[Sequence[str]] = None,
          confidence: Optional[float] = None,
          languages: Optional[Sequence[str]] = None,
          store_scope: str = "sampled",
          prune_store: bool = False,
          on_progress: Optional[progress.Reporter] = None) -> Produced:
    """Sample the picture onto the grid; write the manifest to `out`. `ingest` plus
    a read at each end. `store` is the frame directory; None keeps no frames.
    """
    described = read(media, Media)
    grid = read(timeline, Timeline)
    video_id = same_video(media=described, timeline=grid)

    frames = FrameStore(Path(store)) if store is not None else None
    if frames is None:
        logs.skipped("video", video_id, "store",
                     "no store= was given, so no pixels are kept")

    with logs.timed("video", video_id) as done:
        manifest = ingest(described, grid, sampler, per_second, every_n,
                          min_interval_s, max_per_chunk, threshold, vocabulary,
                          confidence, languages, frames=frames,
                          store_scope=store_scope, on_progress=on_progress)

        pruned: list[int] = []
        if frames is not None and prune_store:
            # After the pass: delete stored frames the new manifest does not name.
            named = {f["index"] for c in manifest.chunks
                     for b in c["samplers"].values() for f in b["frames"]}
            pruned = frames.prune(named)

        where = write(out, manifest)
        done(sampled=manifest.stats.get("frames_sampled"),
             stored=manifest.stats.get("stored_frames"),
             samplers=",".join(manifest.sampler_ids()), pruned=len(pruned))
    artifacts = {"manifest": where}
    if frames is not None and frames.written:
        artifacts["store"] = str(frames.root)

    return Produced(
        video_id=video_id, component="video", artifacts=artifacts,
        stats={**manifest.stats,
               "timeline_fingerprint": manifest.timeline_fingerprint,
               "manifest_fingerprint": manifest.fingerprint(),
               "samplers": manifest.sampler_ids(),
               "pruned_frames": len(pruned)},
        skipped=[] if frames is not None else ["store"],
    )


run = video


def load(path: str | Path) -> Manifest:
    """Read a `manifest.json` back, typed."""
    return read(path, Manifest)


def main(argv: Optional[list[str]] = None) -> int:
    from falconvar.shared.config import env
    env.load()        # an entry point reads .env; the library never does
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Decide which frames are worth describing.")
    ap.add_argument("media", help="path to media.json")
    ap.add_argument("timeline", help="path to timeline.json")
    ap.add_argument("out", help="where to write manifest.json")
    ap.add_argument("--store", default=None,
                    help="directory for the kept frames. Omit to keep none")
    ap.add_argument("--sampler", default="uniform",
                    help=f"comma-separated; known: "
                         f"{', '.join(samplers_mod.available())}. Any may carry "
                         f"a question after a colon, e.g. `yolo:overview`")
    ap.add_argument("--per-second", type=float, default=1.0,
                    help="frames kept per second of media time (default 1)")
    ap.add_argument("--every-frames", type=int, default=None, dest="every_n",
                    help="uniform: stride over the decimated stream (default 1)")
    ap.add_argument("--min-interval", type=float, default=0.0)
    ap.add_argument("--max-per-chunk", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="change samplers: per-sampler default if unset")
    ap.add_argument("--vocabulary", default=None,
                    help="objects: comma-separated class names")
    ap.add_argument("--confidence", type=float, default=None,
                    help="objects: how sure the detector must be of a box "
                         "(default 0.3). Not --threshold, which is how much "
                         "the frame must have changed")
    ap.add_argument("--languages", default=None,
                    help="text: comma-separated EasyOCR codes (default en)")
    ap.add_argument("--prune-store", action="store_true",
                    help="delete stored frames this manifest does not name. "
                         "A store accumulates across runs; this is the only "
                         "irreversible thing ingest can do, so it is opt-in")
    ap.add_argument("--store-scope", default="sampled",
                    choices=("sampled", "decimated"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    vocab = ([v.strip() for v in args.vocabulary.split(",") if v.strip()]
             if args.vocabulary else None)
    langs = ([l.strip() for l in args.languages.split(",") if l.strip()]
             if args.languages else None)
    try:
        produced = video(args.media, args.timeline, args.out, args.store,
                       args.sampler, args.per_second, args.every_n,
                       args.min_interval, args.max_per_chunk, args.threshold,
                       vocab, args.confidence, langs, args.store_scope,
                       args.prune_store)
    except (KeyError, ValueError, FileNotFoundError, UnreadableSource) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    manifest = load(produced.artifacts['manifest'])
    print(f"{produced.video_id}")
    print(f"  decimated    {s['frames_decimated']}   "
          f"over {s['chunks']} chunks ({s['chunks_with_frames']} with frames)")
    print(f"  sampled      {s['frames_sampled']}")
    print(f"  elapsed      {s['elapsed_s']:.2f}s")
    if "stored_frames" in s:
        print(f"  stored       {s['stored_frames']} frames, {s['stored_mb']:g} MB")
    if s.get("pruned_frames"):
        print(f"  pruned       {s['pruned_frames']} frames no longer named")
    print()
    for sampler_id in s["samplers"]:
        counts = [c["samplers"].get(sampler_id, {}).get("frame_count", 0)
                  for c in manifest.chunks]
        total = sum(counts)
        pct = total / max(s["frames_decimated"], 1) * 100
        print(f"  {sampler_id:<18} {total:>4} frames ({pct:.1f}% of decimated)"
              f"   per chunk {counts[:6]}"
              + (" ..." if len(counts) > 6 else ""))
    print()
    print(f"  timeline fp  {s['timeline_fingerprint']}")
    print(f"  manifest fp  {s['manifest_fingerprint']}")
    print(f"\nmanifest -> {produced.artifacts['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
