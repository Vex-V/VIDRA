"""The describe component: `manifest.json` + `store/` -> `descriptions.json`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from vidra.shared.config import env
from vidra.shared.reporting import logs, progress
from vidra.shared.contracts.documents import (Descriptions, Manifest,
                                           Produced, Timeline)
from vidra.shared.storage.files import maybe, read, write
from vidra.shared.contracts.documents import same_video
from . import base, library, prompts
from .backends import stub  # noqa: F401  -- self-registers
from ..helpers import FrameStore
from .frames import FrameSource, StoreUnavailable, store_of
from vidra.shared.reporting.errors import Refused, UnknownOption

def answer(manifest: Manifest, timeline: Timeline, frames: FrameStore,
           describer: Optional[str] = None,
           samplers: Optional[Sequence[str]] = None,
           limit: Optional[int] = None,
           existing: Optional[Descriptions] = None,
           max_output_tokens: Optional[int] = None,
           on_progress: Optional[progress.Reporter] = None) -> Descriptions:
    """One call per (chunk, sampler, question), over documents and frames in hand.
    Reads and writes no artifact.

    `frames` is the store ingest wrote. `existing` is an earlier `Descriptions`:
    every pair still current is skipped. `describer` is a provider or
    `provider/model`; None resolves to VIDRA_DESCRIBER, then openai.
    `max_output_tokens` caps one answer (None is the backend's 2000) and is part
    of the resume key.
    """
    # Check the key before any frame is read.
    from vidra.shared.models import providers
    providers.require("describe", describer)

    # `limit=0` would describe nothing; refused.
    if limit is not None and limit < 1:
        raise Refused(
            f"limit must be at least 1, not {limit}; "
            f"leave it None for no limit")

    if manifest.timeline_fingerprint != timeline.fingerprint():
        raise Refused(
            f"{manifest.video_id}: the manifest was built on a different grid "
            f"({manifest.timeline_fingerprint} vs {timeline.fingerprint()}). "
            "Re-run ingest against the current timeline.")

    known = prompts.questions()
    unknown = sorted({q for s in manifest.config.get("samplers", [])
                      for q in prompts.questions_of(s, s.get("id", ""))
                      if q not in known})
    if unknown:
        raise UnknownOption(
            f"manifest names unknown question(s) {', '.join(unknown)}; "
            f"known: {', '.join(prompts.questions())}")

    built = base.build(describer, **({} if max_output_tokens is None
                                     else {"max_output_tokens": max_output_tokens}))
    from .reader import answer as _pass

    with FrameSource(frames, manifest) as source:
        return _pass(manifest, timeline, built, source, samplers, existing,
                     limit, on_progress)


def describe(manifest: str | Path, timeline: str | Path,
             store: str | Path, out: str | Path,
             previous: Optional[str | Path] = None,
             describer: Optional[str] = None,
             samplers: Optional[Sequence[str]] = None,
             limit: Optional[int] = None,
             max_output_tokens: Optional[int] = None,
             on_progress: Optional[progress.Reporter] = None) -> Produced:
    """One model call per (chunk, sampler, question). `answer` plus a read at each
    end. `previous` is an earlier `descriptions.json`: every pair still current
    is skipped.
    """
    plan = read(manifest, Manifest)
    grid = read(timeline, Timeline)
    existing = maybe(previous, Descriptions)
    video_id = same_video(manifest=plan, timeline=grid, previous=existing)

    with logs.timed("describe", video_id) as done:
        document = answer(plan, grid, store_of(store), describer, samplers,
                          limit, existing, max_output_tokens, on_progress)

        where = write(out, document)
        done(described=document.stats.get("described"),
             skipped_pairs=document.stats.get("skipped"),
             describer=_named(document.model))
    return Produced(
        video_id=video_id, component="describe",
        artifacts={"descriptions": where},
        # The describer's name from the stored model block.
        stats={**document.stats, "describer": _named(document.model),
               "model": ((document.model.get("params") or {})
                         .get("model", _named(document.model))),
               },
    )


def _named(model: dict[str, object]) -> str:
    """The describer's own name, whichever key its backend records it under."""
    return str(model.get("name") or model.get("describer") or "")


def prompt_rows(versions: dict[str, str]) -> list[dict[str, object]]:
    """What each question said, at the version a run asked it under, as rows for a
    database. Writes nothing.
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


def load(path: str | Path) -> Descriptions:
    """Read a `descriptions.json` back, typed."""
    return read(path, Descriptions)


def main(argv: Optional[list[str]] = None) -> int:
    env.load()        # an entry point reads .env; the library never does
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Describe every (chunk, sampler).")
    ap.add_argument("manifest", help="path to manifest.json")
    ap.add_argument("timeline", help="path to timeline.json")
    ap.add_argument("store", help="the frame store directory")
    ap.add_argument("out", help="where to write descriptions.json")
    ap.add_argument("--previous", default=None,
                    help="an earlier descriptions.json; pairs still "
                         "current are skipped. Omit to describe all")
    ap.add_argument("--describer", default=None,
                    help="a provider or provider/model; default "
                         f"VIDRA_DESCRIBER, then openai. Known: "
                         f"{', '.join(base.available())}")
    ap.add_argument("--sampler", default=None,
                    help="comma-separated subset to describe")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N calls. Costs money, so this exists")
    ap.add_argument("--max-tokens", type=int, default=None, dest="max_output_tokens",
                    help="ceiling on one answer (default 2000). Part of the "
                         "resume key, so changing it re-describes everything")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    samplers = ([s.strip() for s in args.sampler.split(",") if s.strip()]
                if args.sampler else None)
    try:
        produced = describe(args.manifest, args.timeline, args.store,
                            args.out, args.previous, args.describer,
                            samplers, args.limit, args.max_output_tokens)
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
