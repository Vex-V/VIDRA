"""The describe component: `manifest.json` + `store/` -> `descriptions.json`."""

from __future__ import annotations

from typing import Optional, Sequence

from ..boundaries import load as load_timeline
from ..video import load as load_manifest
from ...shared import env, logs, paths, progress
from ...shared.storage import files
from ...shared.contracts.documents import (Descriptions, Manifest,
                                           Produced, Timeline)
from . import base, library, prompts
from .backends import stub  # noqa: F401  -- self-registers
from ..frames import Frames
from .frames import FrameSource, StoreUnavailable, store_of

def answer(manifest: Manifest, timeline: Timeline, frames: Frames,
           describer: Optional[str] = None,
           samplers: Optional[Sequence[str]] = None,
           limit: Optional[int] = None,
           existing: Optional[Descriptions] = None,
           max_output_tokens: Optional[int] = None,
           on_progress: Optional[progress.Reporter] = None) -> Descriptions:
    """One call per (chunk, sampler), over documents and pixels in hand.

    Reads and writes no artifact. `frames` is any `Frames` -- the
    `FrameStore` ingest wrote, or the `MemoryFrames` it filled -- so a whole
    run can happen without a data root.

    `existing` is what `resume` reads from disk on the `run` path: hand it the
    previous `Descriptions` and every pair still current is skipped, exactly
    as it would be. Omit it and everything is described, at cost.

    `describer` is a provider or `provider/model`; None resolves through
    `shared.models.providers` -- FALCONVAR_DESCRIBER, then openai.

    `max_output_tokens` is the ceiling on one answer, and None leaves the
    backend's 2000. It is a real setting rather than a safety margin: the
    `people` schema truncated mid-string at 700 and came back as unparseable
    JSON, and a wider custom shape can do the same at 2000. **It is part of
    the resume key** -- `ModelDescriber.config()` reports it, so changing it
    re-describes everything already stored, at cost. That is correct: a
    truncated answer and a whole one are different answers, and a stored one
    cannot say which it was. A backend that loads no model (`stub`) ignores it.
    """
    env.load()
    # Before a single frame is read: a missing key found by the first call
    # arrives after every frame has been read, which is the failure
    # `workflow.validate` exists to prevent -- and a caller driving the
    # components itself never passes through `validate`. Here rather than in
    # `run` so that both ways in are guarded.
    from ...shared.models import providers
    providers.require("describe", describer)

    # `limit=0` described nothing and reported success. To a caller the word
    # reads as a ceiling, and no ceiling is what `None` means here -- so zero
    # is the one value with two readings, and the run that does nothing looks
    # exactly like the run that had nothing to do.
    if limit is not None and limit < 1:
        raise ValueError(
            f"limit must be at least 1, not {limit}; "
            f"leave it None for no limit")

    if manifest.timeline_fingerprint != timeline.fingerprint():
        raise ValueError(
            f"{manifest.video_id}: the manifest was built on a different grid "
            f"({manifest.timeline_fingerprint} vs {timeline.fingerprint()}). "
            "Re-run ingest against the current timeline.")

    known = prompts.questions()
    unknown = sorted({q for s in manifest.config.get("samplers", [])
                      for q in prompts.questions_of(s, s.get("id", ""))
                      if q not in known})
    if unknown:
        raise ValueError(
            f"manifest names unknown question(s) {', '.join(unknown)}; "
            f"known: {', '.join(prompts.questions())}")

    built = base.build(describer, **({} if max_output_tokens is None
                                     else {"max_output_tokens": max_output_tokens}))
    from .reader import answer as _pass

    with FrameSource(frames, manifest) as source:
        return _pass(manifest, timeline, built, source, samplers, existing,
                     limit, on_progress)


def describe(video_id: str, describer: Optional[str] = None,
             samplers: Optional[Sequence[str]] = None,
             limit: Optional[int] = None,
             resume: bool = True,
             max_output_tokens: Optional[int] = None,
             on_progress: Optional[progress.Reporter] = None) -> Produced:
    """Describe by id, from this video's own frame store. The expensive stage.

    `answer` plus a read at each end; every check lives down there. `resume`
    is the one thing that cannot: it means "read what is already on disk",
    which is what `answer` takes as `existing`.
    """
    with logs.timed("describe", video_id) as done:
        manifest = load_manifest(video_id)
        timeline = load_timeline(video_id)

        existing = None
        if resume and paths.exists(video_id, "descriptions"):
            existing = load(video_id)

        document = answer(manifest, timeline,
                          store_of(video_id),
                          describer, samplers, limit, existing,
                          max_output_tokens, on_progress)

        where = files.write(video_id, "descriptions", document.as_dict())
        done(described=document.stats.get("described"),
             skipped_pairs=document.stats.get("skipped"),
             describer=_named(document.model))
    return Produced(
        video_id=video_id, component="describe",
        artifacts={"descriptions": where},
        # Off the stored model block rather than off a local: `answer` builds
        # the describer now, and the block is what a reader has anyway.
        # `name` is the model backend's key and `describer` the stub's -- the
        # two registries spell it differently, and both are `built.name`.
        stats={**document.stats, "describer": _named(document.model),
               "model": ((document.model.get("params") or {})
                         .get("model", _named(document.model))),
               },
    )


#: The uniform name every component also answers to: what a dispatch
#: table calls and what a form introspects. The same function object.
#: See `media/driver.py`.
run = describe


def _named(model: dict[str, object]) -> str:
    """The describer's own name, whichever key its backend records it under."""
    return str(model.get("name") or model.get("describer") or "")


def prompt_rows(versions: dict[str, str]) -> list[dict[str, object]]:
    """What each question said, at the version a run asked it under.

    `descriptions.model` already records `{question: hash}`, which lets a
    reader *detect* that an answer came from a different prompt version. It
    cannot recover what that version said -- edit an instruction and the old
    text is gone -- so these rows are what make a description's provenance
    readable rather than merely comparable.

    This builds them and writes nothing. It used to write them to Postgres
    itself, from inside the component, which is the layering the sink removal
    undid: `video_rag.driver` hands these to `supabase.write_prompts` when a
    run names a database, and a run that names none never assembles them.
    """
    entries: list[dict[str, object]] = []
    for name, version in sorted(versions.items()):
        entry = library.load()["questions"].get(name) or {}
        shape = library.shape_of(name)
        entries.append({
            "name": name, "version": version,
            "instruction": library.instruction_of(name),
            "shape": shape,
            "summary": shape.get("summary", "standard"),
            "builtin": bool(entry.get("builtin")),
            "about": entry.get("about") or None,
        })
    return entries


def load(video_id: str) -> Descriptions:
    return Descriptions.from_dict(
        files.read_json(paths.require(video_id, "descriptions")))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Describe every (chunk, sampler).")
    ap.add_argument("video_id")
    ap.add_argument("--describer", default=None,
                    help="a provider or provider/model; default "
                         f"FALCONVAR_DESCRIBER, then openai. Known: "
                         f"{', '.join(base.available())}")
    ap.add_argument("--sampler", default=None,
                    help="comma-separated subset to describe")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N calls. Costs money, so this exists")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None, dest="max_output_tokens",
                    help="ceiling on one answer (default 2000). Part of the "
                         "resume key, so changing it re-describes everything")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    samplers = ([s.strip() for s in args.sampler.split(",") if s.strip()]
                if args.sampler else None)
    try:
        produced = run(args.video_id, args.describer, samplers,
                       args.limit, not args.no_resume,
                       args.max_output_tokens)
    except (KeyError, ValueError, FileNotFoundError, StoreUnavailable,
            base.DescriberUnavailable) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    print(f"{produced.video_id}   {s['describer']} ({s['model']})")
    print(f"  described    {s['described']}")
    print(f"  skipped      {s['skipped']}   (already current)")
    print(f"  chunks       {s['chunks']}")
    print(f"  elapsed      {s['elapsed_s']:.2f}s")
    print(f"\ndescriptions -> {produced.artifacts['descriptions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
