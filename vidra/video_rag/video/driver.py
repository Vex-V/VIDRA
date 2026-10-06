"""The ingest component: `media.json` + `timeline.json` -> `manifest.json` and
the frame store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from vidra.shared.reporting import logs, progress
from vidra.shared.contracts.documents import (Manifest, Media, Produced,
                                           Timeline)
from vidra.shared.storage.files import read, write
from vidra.shared.contracts.documents import same_video
from . import samplers as samplers_mod
#: The pipeline's `ingest` takes built samplers; this module's takes a spec.
from .pipeline import ingest as _pass
from ..helpers import FrameStore
from vidra.shared.reporting.errors import Refused, UnknownOption


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
    # Named alone somewhere: its own question is asked even when another spec
    # adds more, so `clip,clip:checkout` is `clip:[clip,checkout]`.
    bare: set[str] = set()
    for spec in [s.strip() for s in specs if s.strip()]:
        name, asked = parse_spec(spec)
        if not asked:
            bare.add(name)
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

    for name, asked in grouped.items():
        if asked and name in bare and name not in asked:
            asked.insert(0, name)

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


def load(path: str | Path) -> Manifest:
    """Read a `manifest.json` back, typed."""
    return read(path, Manifest)
