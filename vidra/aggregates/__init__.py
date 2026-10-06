"""aggregates -- higher-level answers over what video_rag extracted.

Reads the documents video_rag wrote, never the video. Name the files once, build
each input from them explicitly, then run one aggregator or several:

    video = aggregates.record(timeline=..., transcript=..., descriptions=...)
    said = video.excerpt(transcript=True)
    seen = video.excerpt(answers={"clip:activity": ["summary", "actors"]})
    people = video.sightings(profile="people", answers=["yolo"])

    aggregates.summary(input=said, out="answers/summary.json", models=models)
    aggregates.aggregate(out="answers", models=models, summary=said, ner=said,
                         entities_people=people, stats=video)

Tiers, cheapest first: `free` (arithmetic: `stats`, `speakers`, `coverage`),
`local` (local models: `ner`, `sentiment`), `llm` (model calls: `summary`,
`chapters`, `events`, custom prompts, and the link profiles
`entities:people` / `objects` / `text`). The model-backed ones are imported
only when used.

`add_prompt` and `add_profile` add definitions of your own (`remove_prompt`,
`remove_profile`, `definition` beside them); they run like the built-ins.
`combine` lays several records end to end as one. `aggregators/` holds the
implementations, `core/` what they share, `database/` the export and `search`.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

from . import definitions
from .definitions import (DefinitionError, ProtectedDefinition, add_profile,
                          add_prompt, definition, remove_profile, remove_prompt)
from .core.base import TIERS, Context, missing
from .core.record import Record, RecordError, record
from .aggregators.coverage import CoverageAggregator
from .aggregators.speakers import SpeakersAggregator
from .aggregators.stats import StatsAggregator
from vidra.shared.reporting.errors import Refused
from vidra.shared.models.base import Embedder, LLM

REGISTRY: dict[str, Any] = {
    cls.name: cls for cls in
    (StatsAggregator, SpeakersAggregator, CoverageAggregator)
}

#: name -> ("module:Class", tier, about), imported on first use.
_LAZY: dict[str, tuple[str, str, str]] = {
    "ner": ("aggregators.ner:NERAggregator", "local",
            "named entities, and which chunks each appears in"),
    "sentiment": ("aggregators.sentiment:SentimentAggregator", "local",
                  "tone per chunk, and where it turns"),
}

#: kind -> the runner every definition of that kind is built with.
RUNNERS: dict[str, str] = {
    "fold": "aggregators.fold:FoldAggregator",
    "spans": "aggregators.spans:SpansAggregator",
    "items": "aggregators.items:ItemsAggregator",
    "link": "aggregators.entities:EntitiesAggregator",
}


def _import(target: str) -> Any:
    module_name, class_name = target.split(":")
    return getattr(importlib.import_module(f".{module_name}", __package__), class_name)


def available() -> list[str]:
    """Every aggregator id: code first, then definitions."""
    return [*REGISTRY, *_LAZY, *definitions.ids()]


def kind_of(name: str) -> Optional[str]:
    """A definition's kind; None for an aggregator written as code."""
    if name in REGISTRY or name in _LAZY:
        return None
    section, definition = definitions.locate(name)
    return "link" if section == "profiles" else definitions.get(section, definition)["kind"]


def tier_of(name: str) -> str:
    if name in REGISTRY:
        return REGISTRY[name].tier
    if name in _LAZY:
        return _LAZY[name][1]
    kind_of(name)                     # raises for an unknown definition
    return "llm"


def about(name: str) -> str:
    """What an aggregator -- or an answer id, `summary~severity` -- is about."""
    from .core.inputs import definition_of
    name = definition_of(name)
    if name in REGISTRY:
        return REGISTRY[name].about
    if name in _LAZY:
        return _LAZY[name][2]
    try:
        return definitions.get(*definitions.locate(name)).get("about", "")
    except definitions.DefinitionError:
        return ""


def takes_inputs(name: str) -> bool:
    return name not in REGISTRY


#: A runner's parameters that name who answers, not settings.
_WHO = ("self", "definition_id", "llm", "embedder")


def _runner_params(name: str) -> list[str]:
    import inspect
    return list(inspect.signature(_import(RUNNERS[kind_of(name)]).__init__).parameters)


def settings_of(name: str) -> list[str]:
    """An aggregator's own settings: its constructor's parameters (`labels` for
    `ner`, `max_spans` / `min_span_s` for a `spans` definition).
    """
    import inspect
    if name in REGISTRY:
        return []
    if name in _LAZY:
        cls = _import(_LAZY[name][0])
        return [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
    return [p for p in _runner_params(name) if p not in _WHO]


def uses_embedder(name: str) -> bool:
    """Whether an aggregator takes an embedder: a link profile or a `spans`
    definition.
    """
    if name in REGISTRY or name in _LAZY:
        return False
    return "embedder" in _runner_params(name)


def build(name: str, llm: Optional[LLM] = None, embedder: Optional[Embedder] = None,
          **settings: Any) -> Any:
    """One aggregator, constructed with its llm, its embedder (if it takes one) and
    its settings.
    """
    if name in REGISTRY:
        if settings:
            raise Refused(f"{name} takes no settings")
        return REGISTRY[name]()
    if name in _LAZY:
        return _import(_LAZY[name][0])(**settings)
    unknown = set(settings) - set(settings_of(name))
    if unknown:
        raise Refused(f"{name} has no setting {', '.join(sorted(unknown))}; "
                      f"it takes {', '.join(settings_of(name)) or 'none'}")
    runner = _import(RUNNERS[kind_of(name)])
    who = {"embedder": embedder} if uses_embedder(name) else {}
    return runner(name, llm, **who, **settings)


from .driver import (AggregateError, Inapplicable, aggregate,  # noqa: E402
                     answer, answers, definition_rows, load, load_all,
                     load_input, validate)
from .components import (chapters, coverage, entities, events, ner,  # noqa: E402
                         prompt, sentiment, speakers, stats, summary)


def __getattr__(name: str) -> Any:
    """PEP 562: `combine`, `merge` and `search` resolve on first use."""
    if name in ("combine", "merge"):
        return getattr(importlib.import_module(".combination", __name__), name)
    if name == "search":
        return importlib.import_module(".database.search", __name__).search
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["REGISTRY", "RUNNERS", "TIERS", "AggregateError", "Context",
           "DefinitionError", "Inapplicable", "ProtectedDefinition", "Record",
           "RecordError", "about", "add_profile", "add_prompt", "aggregate",
           "answer", "answers", "available", "build", "chapters", "combine",
           "coverage", "definition", "definition_rows", "entities", "events",
           "kind_of", "load", "load_all", "load_input", "merge", "missing", "ner", "prompt",
           "record", "remove_profile", "remove_prompt", "search", "sentiment",
           "settings_of", "speakers", "stats", "summary", "takes_inputs",
           "tier_of", "uses_embedder", "validate"]
