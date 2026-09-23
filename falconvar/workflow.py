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
from .shared.contracts.documents import Produced
from .video_rag import driver as video_rag

#: Every component a run may invoke, in the order it can run.
COMPONENTS = (*video_rag.COMPONENTS, "aggregate")

Run = video_rag.Run


@dataclass
class Options:
    """The shape of one run. Tuning lives on the component CLIs."""

    source: Path
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
    """The part of a run video_rag decides."""
    return video_rag.Options(
        source=options.source, video_id=options.video_id, policy=options.policy,
        use_video=options.use_video, use_audio=options.use_audio,
        sampler=options.sampler, describer=options.describer,
        embedder=options.embedder, database=options.database)


def validate(options: Options) -> list[str]:
    """Everything either tier would refuse, before either runs."""
    return (video_rag.validate(extraction(options))
            + aggregates.validate(options.tier, options.llm))


def process(options: Options,
            on_step: Optional[Callable[[str, Optional[Produced]], None]] = None
            ) -> Run:
    """Extract, then aggregate."""
    problems = validate(options)
    if problems:
        raise ValueError("; ".join(problems))
    say = on_step or (lambda component, produced: None)

    run = video_rag.process(extraction(options), on_step)

    say("aggregate", None)
    produced = aggregates.run(run.video_id, options.tier,
                              llm=options.llm, embedder=options.embedder)
    run.steps.append(produced)
    if options.database:
        run.problems += _export_aggregates(produced, options)
    say(produced.component, produced)
    return run


def _export_aggregates(produced: Produced, options: Options) -> list[str]:
    """The aggregates, their definitions, and the video's own vector.

    Here rather than in `aggregates.run` for the reason every other export is
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
                aggregates.load(produced.video_id, "summary").payload,
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
        source=args.source, video_id=args.video_id, policy=args.policy,
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
