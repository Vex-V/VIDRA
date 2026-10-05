"""The whole run: extract with `video_rag`, then aggregate what it extracted.

    video_rag    a video in, a searchable index of moments out
    aggregates   higher-level answers over what video_rag extracted

This file calls the two drivers and nothing below them. `Options` is flat:
both tiers' settings in one record.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from . import aggregates
from .shared.config import paths
from .shared.contracts.documents import Produced
from .video_rag import driver as video_rag
from vidra.shared.reporting.errors import Refused
from vidra.shared.models.roles import Models, keys_of, unpack
from vidra.shared.storage.database import Database, as_database

Run = video_rag.Run


@dataclass
class Options:
    """The shape of one run. Tuning lives on the component CLIs."""

    source: Path
    #: The directory that holds one folder per video; None is the data root.
    into: Optional[Path] = None
    video_id: Optional[str] = None
    #: The video's name and recording time; see video_rag's `Options`.
    name: Optional[str] = None
    recorded_at: Optional[str | datetime] = None
    policy: str = "uniform"                  # decides who runs first
    use_video: bool = True
    use_audio: bool = True
    sampler: str = "uniform"                 # what to look at
    describer: Optional[str] = None          # frames -> answers
    embedder: Optional[str] = None           # text -> vectors
    llm: Optional[str] = None                # the `llm` aggregate tier
    tier: str = "free"                       # a cost ceiling
    # Also copy every artifact to this database: a name or a built `Database`.
    database: Optional[str | Database] = None
    #: All three model roles (and keys) as one value; a role set here and as a
    #: field is refused.
    models: Optional[Models] = None


def roles(options: Options) -> dict[str, Optional[str]]:
    """The describer, embedder and llm this run uses, from the fields or `models`."""
    return unpack(options.models, describer=options.describer,
                  embedder=options.embedder, llm=options.llm)


def extraction(options: Options) -> video_rag.Options:
    """The video_rag half of a run, with `into` resolved."""
    chosen = roles(options)
    return video_rag.Options(
        source=options.source,
        into=Path(options.into) if options.into else paths.out_root(),
        video_id=options.video_id, name=options.name,
        recorded_at=options.recorded_at, policy=options.policy,
        use_video=options.use_video, use_audio=options.use_audio,
        sampler=options.sampler, describer=chosen["describer"],
        embedder=chosen["embedder"], database=options.database)


def chosen(options: Options) -> dict[str, bool]:
    """The aggregators a run hands data to: every one up to its tier, on its
    default data.
    """
    if options.tier not in aggregates.TIERS:
        return {}
    return aggregates.up_to(options.tier)


def validate(options: Options) -> list[str]:
    """Everything either tier would refuse, before either runs."""
    if options.tier not in aggregates.TIERS:
        return (video_rag.validate(extraction(options))
                + [f"tier must be one of {', '.join(aggregates.TIERS)}"])
    try:
        used = roles(options)
    except Refused as exc:
        return [str(exc)]
    return (video_rag.validate(extraction(options))
            + aggregates.validate(chosen(options), used["llm"], used["embedder"]))


def process(options: Options,
            on_step: Optional[Callable[[str, Optional[Produced]], None]] = None
            ) -> Run:
    """Extract, then aggregate -- with `options.models`' keys, if it has any."""
    with keys_of(options.models):
        return _process(options, on_step)


def _process(options: Options,
             on_step: Optional[Callable[[str, Optional[Produced]], None]]) -> Run:
    problems = validate(options)
    if problems:
        raise Refused("; ".join(problems))
    say = on_step or (lambda component, produced: None)
    used = roles(options)
    # Built once and shared by both tiers' exports.
    database = as_database(options.database)
    extract = extraction(options)
    extract.database = database

    run = video_rag.process(extract, on_step)

    # The folder extraction settled on is the record the aggregates read; earlier
    # answers there are reused while current.
    say("aggregate", None)
    answers = video_rag.layout(run.home)["aggregates"]
    produced = aggregates.aggregate(
        run.home, answers, previous=answers, llm=used["llm"],
        embedder=used["embedder"], aggregators=chosen(options), database=database)
    run.steps.append(produced)
    run.problems += produced.stats.get("problems") or []
    say(produced.component, produced)
    return run


def main(argv: Optional[list[str]] = None) -> int:
    from vidra.shared.config import env
    env.load()        # an entry point reads .env; the library never does
    import argparse

    ap = argparse.ArgumentParser(
        description="Run the whole pipeline once: extract, then aggregate. "
                    "Per-stage tuning lives on each component's own CLI, e.g. "
                    "`python -m vidra.video_rag.video`.")
    ap.add_argument("source", type=Path)
    ap.add_argument("--into", type=Path, default=None,
                    help="the directory that holds one folder per video; "
                         "default the data root, which VIDRA_DATA and "
                         "vidra.configure() decide")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--name", default=None,
                    help="what to call the video; default the filename")
    ap.add_argument("--recorded-at", default=None,
                    help="when it was recorded, ISO 8601; default the "
                         "container's creation time, if it has one")
    ap.add_argument("--policy", default="uniform")
    ap.add_argument("--sampler", default="uniform",
                    help="comma-separated; any may carry a question after a "
                         "colon, e.g. `yolo:overview`")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--describer", default=None,
                    help="a provider or provider/model, e.g. ollama/gemma3:4b; "
                         "default VIDRA_DESCRIBER, then openai")
    ap.add_argument("--embedder", default=None,
                    help="a provider or provider/model, e.g. local; "
                         "default VIDRA_EMBEDDER, then openai")
    ap.add_argument("--llm", default=None,
                    help="who answers --tier llm: a provider or provider/model; "
                         "default VIDRA_LLM, then openai")
    ap.add_argument("--tier", default="free", choices=aggregates.TIERS)
    ap.add_argument("--database", default=None,
                    help=f"also write a copy to a database; known: "
                         f"{', '.join(video_rag.DATABASES)}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    options = Options(
        source=args.source, into=args.into,
        video_id=args.video_id, name=args.name,
        recorded_at=args.recorded_at, policy=args.policy,
        use_video=not args.no_video, use_audio=not args.no_audio,
        sampler=args.sampler, describer=args.describer,
        embedder=args.embedder, llm=args.llm,
        tier=args.tier, database=args.database)

    problems = validate(options)
    if problems:
        for problem in problems:
            print(f"error: {problem}")
        return 2
    return video_rag.report(options, lambda on_step: process(options, on_step),
                            args.json)


#: The public surface.
__all__ = ["Options", "Run", "extraction", "process", "validate"]


if __name__ == "__main__":
    raise SystemExit(main())
