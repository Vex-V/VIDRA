"""aggregates -- higher-level answers over what video_rag extracted.

Reads the documents video_rag wrote, never the video. Tiers, cheapest first:
`free` (arithmetic), `local` (local models), `llm` (model calls); the
model-backed ones are imported only when used.

Aggregators written as code: `stats`, `speakers`, `coverage`, `ner`,
`sentiment`. From data: every prompt and link profile in `definitions`
(`summary`, `chapters`, `events`, `entities:people`, and custom ones). Each is
a component with one input and one answer:

    select.select(record, out, "ner", "transcript")    -> an excerpt file
    ner.ner(excerpt, out)                              -> ner.json
    stats.stats(record, out)                           -> stats.json
    prompt.prompt("summary", excerpt, out)             -> summary.json
    entities.entities("people", sightings, out)        -> entities.people.json

    aggregate(record, out, ner="transcript", sentiment=True)   # only those two

`add_prompt` and `add_profile` add definitions of your own (`remove_prompt`,
`remove_profile`, `definition` beside them); they run like the built-ins.

`combine` lays several videos end to end as one record. `core/` holds what
every aggregator shares; `database/` the export and `search`.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

from . import definitions
from .definitions import (DefinitionError, ProtectedDefinition, add_profile,
                          add_prompt, definition, remove_profile, remove_prompt)
from .core.base import TIERS, Context, missing
from .coverage import CoverageAggregator
from .speakers import SpeakersAggregator
from .stats import StatsAggregator
from vidra.shared.reporting.errors import Refused

REGISTRY: dict[str, Any] = {
    cls.name: cls for cls in
    (StatsAggregator, SpeakersAggregator, CoverageAggregator)
}

#: name -> ("module:Class", tier, about), imported on first use.
_LAZY: dict[str, tuple[str, str, str]] = {
    "ner": ("ner:NERAggregator", "local",
            "named entities, and which chunks each appears in"),
    "sentiment": ("sentiment:SentimentAggregator", "local",
                  "tone per chunk, and where it turns"),
}

#: kind -> the runner every definition of that kind is built with.
RUNNERS: dict[str, str] = {
    "fold": "fold:FoldAggregator",
    "spans": "spans:SpansAggregator",
    "items": "items:ItemsAggregator",
    "link": "entities:EntitiesAggregator",
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


def build(name: str, llm: Optional[str] = None, embedder: Optional[str] = None,
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


from .driver import (Inapplicable, aggregate, answer, answers,  # noqa: E402
                     context, definition_rows, load, load_all,
                     load_input, up_to, validate)


def __getattr__(name: str) -> Any:
    """PEP 562: `combine`, `merge` and `search` resolve on first use."""
    if name in ("combine", "merge"):
        return getattr(importlib.import_module(".combination", __name__), name)
    if name == "search":
        return importlib.import_module(".database.search", __name__).search
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["REGISTRY", "RUNNERS", "TIERS", "Context", "DefinitionError", "Inapplicable",
           "ProtectedDefinition", "about", "add_profile", "add_prompt", "aggregate",
           "answer", "answers", "available", "build", "combine", "context",
           "definition", "definition_rows", "kind_of", "load", "load_all",
           "load_input", "merge", "missing", "remove_profile", "remove_prompt",
           "settings_of", "search", "takes_inputs", "tier_of", "up_to",
           "uses_embedder", "validate"]
