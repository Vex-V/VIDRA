"""The whole run: extract with `video_rag`, then aggregate what it extracted.

Two tiers, and this file is the only place both are named:

    video_rag    a video in, a searchable index of moments out -- media, audio,
                 boundaries, video, cut, describe, embed, and retrieve over
                 what they built. A complete RAG engine on its own.
    aggregates   higher-level answers over what video_rag extracted -- counts,
                 speakers, summaries, chapters, entities. It never reads the
                 video, only the documents video_rag wrote.

Each tier has one driver, and each driver calls the components in its own
folder: the shape the pipeline always had, one level up. This file calls the
two drivers and nothing below them.

`Options` stays flat, because it is what a request supplies: the API's upload
form and `/capabilities.defaults` read it, and neither should need to know
which tier a field belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import aggregates
from .shared import paths
from .shared.contracts.documents import Produced
from .video_rag import driver as video_rag
from falconvar.shared.errors import Refused

#: Every component a run may invoke, in the order it can run.
COMPONENTS = (*video_rag.COMPONENTS, "aggregate")

Run = video_rag.Run


@dataclass
class Options:
    """The shape of one run. Tuning lives on the component CLIs."""

    source: Path
    #: Where the run writes: the directory that holds one folder per video.
    #: None is the data root, which is what `configure()` and
    #: `FALCONVAR_DATA` decide -- so an application that owns its deployment
    #: says it once, there, and never here. A caller that wants *this* run
    #: somewhere else names it. Both tiers take paths, so the aggregates land
    #: in the same folder as the extraction wherever that is.
    into: Optional[Path] = None
    video_id: Optional[str] = None
    policy: str = "uniform"                  # decides who runs first
    use_video: bool = True
    use_audio: bool = True
    sampler: str = "uniform"                 # what to look at
    describer: Optional[str] = None          # frames -> answers
    embedder: Optional[str] = None           # text -> vectors
    llm: Optional[str] = None                # the `llm` aggregate tier
    tier: str = "free"                       # a cost ceiling
    # Where a *copy* goes. Files are written either way; naming one makes this
    # function read each artifact back and hand it to `storage/supabase.py`.
    database: Optional[str] = None


def extraction(options: Options) -> video_rag.Options:
    """The part of a run video_rag decides.

    `into` is resolved here rather than defaulted in `video_rag.Options`,
    because a component-level default reading the data root would be the
    resolving that left the tier -- and this is a composition root, which is
    the level allowed to know where this deployment keeps its videos.
    """
    return video_rag.Options(
        source=options.source,
        into=Path(options.into) if options.into else paths.out_root(),
        video_id=options.video_id, policy=options.policy,
        use_video=options.use_video, use_audio=options.use_audio,
        sampler=options.sampler, describer=options.describer,
        embedder=options.embedder, database=options.database)


def chosen(options: Options) -> dict[str, bool]:
    """The aggregators a whole run hands data to: every one up to its tier, on
    its own default data. The aggregates pipeline runs only what it is handed,
    and a tier is how this flat, form-shaped `Options` says which."""
    if options.tier not in aggregates.TIERS:
        return {}
    return aggregates.up_to(options.tier)


def validate(options: Options) -> list[str]:
    """Everything either tier would refuse, before either runs."""
    if options.tier not in aggregates.TIERS:
        return (video_rag.validate(extraction(options))
                + [f"tier must be one of {', '.join(aggregates.TIERS)}"])
    return (video_rag.validate(extraction(options))
            + aggregates.validate(chosen(options), options.llm, options.embedder))


def process(options: Options,
            on_step: Optional[Callable[[str, Optional[Produced]], None]] = None
            ) -> Run:
    """Extract, then aggregate."""
    problems = validate(options)
    if problems:
        raise Refused("; ".join(problems))
    say = on_step or (lambda component, produced: None)

    run = video_rag.process(extraction(options), on_step)

    # The folder extraction settled on, off its receipt, is the record the
    # aggregates read. `previous` is its answers folder: a re-run of one video
    # reuses what is still current, as the extraction tier's `resume` does.
    say("aggregate", None)
    answers = video_rag.layout(run.home)["aggregates"]
    produced = aggregates.aggregate(
        run.home, answers, previous=answers, llm=options.llm,
        embedder=options.embedder, aggregators=chosen(options))
    run.steps.append(produced)
    if options.database:
        run.problems += _export_aggregates(produced, options)
    say(produced.component, produced)
    return run


def _export_aggregates(produced: Produced, options: Options) -> list[str]:
    """The aggregates, their definitions, and the video's own vector.

    Here rather than in `aggregates.aggregate` for the reason every other export is
    in `video_rag.driver`: a component produces documents and a pipeline
    decides where copies go. `aggregates` wrote its own Postgres rows and made
    its own whole-video vector, gated on an `index` parameter -- so the tier
    could not be run at all without deciding a destination.

    Each answer is its own file under `aggregates/`, so this cannot go through
    `video_rag.export`: that resolves one path per artifact name.
    """
    from .shared.storage import files, supabase

    problems: list[str] = []
    for answer, where in sorted(produced.artifacts.items()):
        try:
            supabase.write_aggregate(produced.video_id,
                                     files.read_json(Path(where)))
        except Exception as exc:                          # noqa: BLE001
            problems.append(f"{answer} -> {options.database}: {exc}")
    try:
        supabase.write_definitions(
            aggregates.definition_rows(produced.stats.get("definitions") or {}))
    except Exception as exc:                              # noqa: BLE001
        problems.append(f"definitions -> {options.database}: {exc}")

    # The whole video as one vector, from the summary this run has in hand.
    # `embeddings` answers *which twenty seconds* and this answers *which
    # video*, so they never share a ranking -- `/search level=video` is the
    # only thing that reads it.
    if "summary" in (produced.stats.get("aggregates") or []):
        try:
            produced.stats["video_units"] = aggregates.index_summary(
                produced.video_id,
                aggregates.load(produced.artifacts["summary"]).payload,
                options.embedder)
        except Exception as exc:                          # noqa: BLE001
            problems.append(f"video vector -> {options.database}: {exc}")
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Run the whole pipeline once: extract, then aggregate. "
                    "Per-stage tuning lives on each component's own CLI, e.g. "
                    "`python -m falconvar.video_rag.video`.")
    ap.add_argument("source", type=Path)
    ap.add_argument("--into", type=Path, default=None,
                    help="the directory that holds one folder per video; "
                         "default the data root, which FALCONVAR_DATA and "
                         "falconvar.configure() decide")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--policy", default="uniform")
    ap.add_argument("--sampler", default="uniform",
                    help="comma-separated; any may carry a question after a "
                         "colon, e.g. `yolo:overview`")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--describer", default=None,
                    help="a provider or provider/model, e.g. ollama/gemma3:4b; "
                         "default FALCONVAR_DESCRIBER, then openai")
    ap.add_argument("--embedder", default=None,
                    help="a provider or provider/model, e.g. local; "
                         "default FALCONVAR_EMBEDDER, then openai")
    ap.add_argument("--llm", default=None,
                    help="who answers --tier llm: a provider or provider/model; "
                         "default FALCONVAR_LLM, then openai")
    ap.add_argument("--tier", default="free", choices=aggregates.TIERS)
    ap.add_argument("--database", default=None,
                    help=f"also write a copy to a database; known: "
                         f"{', '.join(video_rag.DATABASES)}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    options = Options(
        source=args.source, into=args.into,
        video_id=args.video_id, policy=args.policy,
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


#: Declared, because without it this module published its own imports --
#: `Callable`, `Optional`, `Path`, `dataclass`, `annotations` -- as though they
#: were part of the surface. `extraction` is here because it is how a caller
#: narrows a whole-run `Options` to the extraction half; `main` is not, for the
#: reason no component exports one.
__all__ = ["COMPONENTS", "Options", "Run", "extraction", "process", "validate"]


if __name__ == "__main__":
    raise SystemExit(main())
