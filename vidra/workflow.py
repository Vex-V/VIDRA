"""The whole run: extract with `video_rag`, then aggregate what it extracted.

    video_rag    a video in, a searchable index of moments out
    aggregates   higher-level answers over what video_rag extracted

This file calls the two drivers and nothing below them. `Options` is flat:
both tiers' settings in one record.

The library reads nothing by default; a run is where the choice is made. Every
aggregator up to `tier` runs on everything extraction produced: the counters
on the record, each text aggregator on the transcript and every answer's prose,
each link profile on every answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from . import aggregates
from .shared.config import paths
from .shared.contracts.documents import Produced
from .video_rag.offline import driver as video_rag
from vidra.shared.reporting.errors import Refused
from vidra.shared.models.base import Embedder, LLM, VLM
from vidra.shared.models.roles import Models, unpack
from vidra.shared.storage.database import Database, as_database

Run = video_rag.Run


@dataclass
class Options:
    """The shape of one run. Per-stage tuning lives on each component's function."""

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
    vlm: Optional[VLM] = None                # frames -> answers
    embedder: Optional[Embedder] = None      # text -> vectors
    llm: Optional[LLM] = None                # the `llm` aggregate tier
    tier: str = "free"                       # a cost ceiling
    # Also copy every artifact to this database: a name or a built `Database`.
    database: Optional[str | Database] = None
    #: All three models as one value; a role set here and as a field is refused.
    models: Optional[Models] = None


def roles(options: Options) -> dict[str, Any]:
    """The vlm, embedder and llm this run uses, from the fields or `models`."""
    return unpack(options.models, vlm=options.vlm,
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
        sampler=options.sampler, vlm=chosen["vlm"],
        embedder=chosen["embedder"], database=options.database)


def chosen(options: Options) -> list[str]:
    """The aggregators a run uses: every one whose cost is at most its tier."""
    if options.tier not in aggregates.TIERS:
        return []
    ceiling = aggregates.TIERS.index(options.tier)
    return [name for name in aggregates.available()
            if aggregates.TIERS.index(aggregates.tier_of(name)) <= ceiling]


def inputs_for(names: list[str], folder: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """What each aggregator reads in a run, built from the folder extraction
    wrote, and why any could not run."""
    at = video_rag.layout(folder)
    video = aggregates.record(**{kind: at[kind] for kind in
                                 ("timeline", "transcript", "descriptions", "manifest")
                                 if at[kind].exists()})
    prose = {answer: ["summary"] for answer in video.answer_ids()}
    handed: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for name in names:
        reads = aggregates.driver.wants(name)
        try:
            if reads == "a record":
                handed[name] = video
            elif reads == "sightings":
                handed[name] = video.sightings(profile=name, answers=list(prose))
            else:
                handed[name] = video.excerpt(transcript="transcript" in video.paths,
                                             answers=prose)
        except aggregates.RecordError as why:
            skipped[name] = str(why)
    return handed, skipped


def validate(options: Options) -> list[str]:
    """Everything either tier would refuse, before either runs."""
    if options.tier not in aggregates.TIERS:
        return (video_rag.validate(extraction(options))
                + [f"tier must be one of {', '.join(aggregates.TIERS)}"])
    try:
        used = roles(options)
    except Refused as exc:
        return [str(exc)]
    from .shared.models import base
    from .shared.models.roles import resolve
    names = chosen(options)
    problems = video_rag.validate(extraction(options))
    # The embedder is the extraction's, which `video_rag.validate` checked.
    if any(aggregates.tier_of(n) == "llm" for n in names):
        problems += base.problems("llm", resolve("llm", used["llm"]))
    return problems


def process(options: Options,
            on_step: Optional[Callable[[str, Optional[Produced]], None]] = None
            ) -> Run:
    """Extract, then aggregate."""
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
    answers = video_rag.layout(run.folder)["aggregates"]
    handed, cannot = inputs_for(chosen(options), run.folder)
    produced = aggregates.aggregate(
        out=answers, previous=answers, llm=used["llm"], embedder=used["embedder"],
        database=database, **{name.replace(":", "_"): data
                              for name, data in handed.items()})
    produced.stats["skipped"] = {**cannot, **produced.stats.get("skipped", {})}
    produced.skipped = sorted(produced.stats["skipped"])
    run.steps.append(produced)
    run.problems += produced.stats.get("problems") or []
    say(produced.component, produced)
    return run


#: The public surface.
__all__ = ["Options", "Run", "extraction", "process", "validate"]
