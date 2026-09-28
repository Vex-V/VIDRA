"""aggregates -- higher-level answers over what video_rag extracted.

The second tier. Reads the documents video_rag wrote, by the paths it is
handed -- never the video, and never a video_rag component. Answers the
questions embeddings cannot: counts, coverage, who dominated, how much of this
is speech, what the whole video is about, who is who across chunks.

Three tiers, cheapest first. `free` is arithmetic, `local` adds GPU models,
`llm` adds paid calls. Both dear tiers are registered lazily, so importing this
pulls in neither torch nor an API client and needs no key.

Two sources of aggregators. Code: `stats`, `speakers`, `coverage`, `ner`,
`sentiment`. Data: every prompt and link profile in `definitions` -- `summary`,
`chapters`, `events`, `entities:people` and whatever a user has added -- each run
by its kind's runner.

**Every aggregator is a component, and there is a pipeline over them**, the
shape video_rag has. Each takes one input and writes one answer:

    select.select(record, out, "ner", "transcript")    -> an excerpt file
    ner.ner(excerpt, out)                              -> ner.json
    stats.stats(record, out)                           -> stats.json
    prompt.prompt("summary", excerpt, out)             -> summary.json
    entities.entities("people", sightings, out)        -> entities.people.json

    aggregate(record, out, ner="transcript", sentiment=True)   # only those two

**Several videos are one record first.** `combine` lays their documents end to
end in the same formats, and every aggregator then runs over that record
unchanged -- no aggregator knows what a collection is.

**One folder per aggregator, each with a `driver.py`**, as a component of
video_rag has. A tier is a class attribute rather than a directory, because it
says what an aggregator *costs*, not what it is made of -- and grouping by cost
put `linking`, which is 230 lines of its own, three levels from the runner that
is its only caller. What every aggregator shares is `base` (the protocols and
`DefinitionRunner`), `inputs` (what it reads) and `rendering` (how that reads).
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

from . import definitions
from .base import TIERS, Context, missing
from .coverage import CoverageAggregator
from .speakers import SpeakersAggregator
from .stats import StatsAggregator
from falconvar.shared.errors import Refused

REGISTRY: dict[str, Any] = {
    cls.name: cls for cls in
    (StatsAggregator, SpeakersAggregator, CoverageAggregator)
}

#: name -> ("module:Class", tier, about). Resolved on first use, so a `--tier
#: free` run never imports torch.
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
    """Every aggregator id: code first, then definitions. Read now, so a
    definition added through the API is runnable without a restart."""
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
    from .inputs import definition_of
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


def settings_of(name: str) -> list[str]:
    """An aggregator's own settings: its constructor's parameters. `ner` has
    `model`, `labels` and `threshold`; the free ones and the definitions have
    none (a definition's words are its data, and `llm` / `embedder` are who
    answers, not settings of the aggregator). Read off the signature, so a
    setting added to a constructor is reachable without editing a list."""
    import inspect
    if name in _LAZY:
        cls = _import(_LAZY[name][0])
        return [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
    return []


def build(name: str, llm: Optional[str] = None, embedder: Optional[str] = None,
          **settings: Any) -> Any:
    """One aggregator, constructed. Only the llm tier takes a provider, only a
    link profile an embedder, and only a local model settings of its own --
    `labels` for `ner`, a checkpoint for either."""
    if name in REGISTRY:
        if settings:
            raise Refused(f"{name} takes no settings")
        return REGISTRY[name]()
    if name in _LAZY:
        return _import(_LAZY[name][0])(**settings)
    if settings:
        raise Refused(f"{name} is a definition; its words are its settings")
    kind = kind_of(name)
    runner = _import(RUNNERS[kind])
    return runner(name, llm, embedder) if kind == "link" else runner(name, llm)


from .driver import (Inapplicable, aggregate, answer, answers,  # noqa: E402
                     context, definition_rows, index_summary, load, load_all,
                     load_input, up_to, validate)


def __getattr__(name: str) -> Any:
    """PEP 562: `combine` and `merge` resolve on first use.

    Imported eagerly, `python -m falconvar.aggregates.combination` found its
    own module already in `sys.modules` and warned that running it might
    behave unpredictably. And the module is not called `combine`: a package
    attribute and a submodule of one name are one slot, so importing the
    module replaced the function -- `boundaries.grid`'s trap, one tier over.
    """
    if name in ("combine", "merge"):
        return getattr(importlib.import_module(".combination", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["REGISTRY", "RUNNERS", "TIERS", "Context", "Inapplicable", "about",
           "aggregate", "answer", "answers", "available", "build", "combine",
           "context", "definition_rows", "index_summary", "kind_of", "load",
           "load_all", "load_input", "merge", "missing", "settings_of",
           "takes_inputs", "tier_of", "up_to", "validate"]
