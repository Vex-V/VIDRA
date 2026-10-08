"""The sampler spec: `"clip:[text,scene],yolo"`, or sampler objects, -> built
samplers.

Shared by every pipeline that samples, so a spec means the same run whether
the frames come from a file or a stream. A spec is a string, or a list mixing
strings and `Sampler` objects:

    "clip:[text,scene],yolo"                         by name, built-in settings
    [samplers.build("clip", threshold=0.93, prompts=["safety"]), "yolo"]
    [Brightness(step=12, prompts=["overview"])]      a class of your own

A sampler named by a string gets the settings passed beside the spec, each
routed to the samplers whose constructor takes it; an object carries its own.
Questions are checked against a vocabulary the caller passes; nothing here
reads the question library.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from vidra.shared.config.lookup import keyword_parameters
from vidra.shared.reporting.errors import Refused, UnknownOption
from . import samplers as samplers_mod
from .samplers import Sampler

#: One element of a spec: a string naming a sampler, or a sampler object.
Spec = str | Sampler


def split_specs(sampler: Spec | Sequence[Spec]) -> list[Spec]:
    """`"clip:[text,scene],yolo"` -> `["clip:[text,scene]", "yolo"]`, aware of
    brackets. A sampler object, alone or in a list, is kept as it is.
    """
    if isinstance(sampler, Sampler):
        return [sampler]
    if not isinstance(sampler, str):
        return [s if isinstance(s, Sampler) else str(s).strip()
                for s in sampler
                if isinstance(s, Sampler) or str(s).strip()]
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


def spec_text(sampler: Spec | Sequence[Spec]) -> str:
    """A spec as one line, for logs: an object appears as `name(object)`."""
    return ",".join(s if isinstance(s, str) else f"{s.name}(object)"
                    for s in split_specs(sampler))


#: The built-in samplers' settings: setting -> the samplers that read it.
#: Published; a sampler's settings are read off its constructor, and
#: `library_check` holds this table to the built-ins' signatures.
SAMPLER_SETTINGS: dict[str, tuple[str, ...]] = {
    "every_n": ("uniform",),
    # Samplers that keep a frame when it changed enough.
    "threshold": ("clip", "yolo", "objects", "text"),
    "vocabulary": ("objects",),
    "confidence": ("objects",),
    "languages": ("text",),
}

#: Settings every sampler takes: the rate limits the base class enforces.
RATE_SETTINGS = ("min_interval_s", "max_per_chunk")


def settings_of(cls: type) -> set[str]:
    """The settings a sampler class takes: its constructor's keyword parameters,
    less the bookkeeping every sampler has."""
    return {n for n in keyword_parameters(cls) if n not in ("sampler_id", "prompts")}


@dataclass
class _Plan:
    """What a spec asks for, checked: the runs in order, and every problem."""

    #: In spec order: (name, questions) for a string, or the object itself.
    runs: list[Any] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    #: True when every problem is an unknown name (a sampler or a question).
    unknown_only: bool = True

    def add(self, problem: str, unknown: bool = False) -> None:
        self.problems.append(problem)
        self.unknown_only = self.unknown_only and unknown


def _plan(specs: Sequence[Spec], questions: Optional[Sequence[str]]) -> _Plan:
    plan = _Plan()
    known = samplers_mod.available()
    grouped: dict[str, list[str]] = {}
    bare: set[str] = set()
    objects: dict[str, Sampler] = {}

    for spec in specs:
        if isinstance(spec, Sampler):
            name = spec.name
            if name in objects:
                plan.add(f"two {name!r} sampler objects: one run per sampler, so "
                         f"put every question on one (prompts=[...])")
                continue
            objects[name] = spec
            plan.runs.append(spec)
            for question in spec.questions():
                if questions is not None and question not in questions:
                    plan.add(_no_question(name, question, questions,
                                          own=not spec.prompts), unknown=bool(spec.prompts))
            try:
                json.dumps(spec.config())
            except (TypeError, ValueError) as exc:
                plan.add(f"{name}.config() must be plain JSON, since the manifest "
                         f"records it: {exc}")
            continue
        if not isinstance(spec, str):
            plan.add(f"a sampler spec is a string or a Sampler, not "
                     f"{type(spec).__name__}")
            continue
        name, asked = parse_spec(spec)
        if name not in known:
            plan.add(f"unknown sampler {name!r} in {spec!r}; known: "
                     f"{', '.join(known)}", unknown=True)
            continue
        for question in asked:
            if questions is not None and question not in questions:
                plan.add(f"unknown question {question!r} in {spec!r}; "
                         f"known: {', '.join(questions)}", unknown=True)
        if name not in grouped:
            grouped[name] = []
            plan.runs.append((name, grouped[name]))
        if not asked:
            bare.add(name)
        for question in asked:
            if question not in grouped[name]:
                grouped[name].append(question)

    for name in sorted(set(grouped) & set(objects)):
        plan.add(f"{name!r} is given twice, by name and as an object: one run "
                 f"per sampler, so put its questions on the object (prompts=[...])")

    # Named alone somewhere: its own question is asked even when another spec
    # adds more, so `clip,clip:checkout` is `clip:[clip,checkout]`.
    for name, asked in grouped.items():
        if name in bare and name not in asked:
            if asked:
                asked.insert(0, name)
            if questions is not None and name not in questions:
                plan.add(_no_question(name, name, questions, own=True))
    return plan


def _no_question(name: str, question: str, questions: Sequence[str],
                 own: bool) -> str:
    if own:
        return (f"sampler {name!r} has no question of its own: pair it with one "
                f"({name}:overview, or prompts=[...] on an object), or add one "
                f"named {name!r} with describe.add_question")
    return f"unknown question {question!r} for sampler {name!r}; known: {', '.join(questions)}"


def problems(sampler: Spec | Sequence[Spec],
             questions: Optional[Sequence[str]] = None) -> list[str]:
    """What is wrong with a spec, as messages; empty is valid. Checks names,
    questions and duplicates, not settings: what `validate` asks."""
    return _plan(split_specs(sampler), questions).problems


def build_samplers(specs: Sequence[Spec], every_n: Optional[int] = None,
                   min_interval_s: float = 0.0,
                   max_per_chunk: Optional[int] = None,
                   threshold: Optional[float] = None,
                   vocabulary: Optional[Sequence[str]] = None,
                   confidence: Optional[float] = None,
                   languages: Optional[Sequence[str]] = None,
                   questions: Optional[Sequence[str]] = None
                   ) -> list[Sampler]:
    """`["yolo", "clip:[text,scene]", Brightness(step=12)]` -> sampler objects,
    one per run, in spec order.

    A name may carry one question after a colon or several in a list; unpaired,
    the question is the sampler's own name. Specs naming the same sampler merge
    into one run, so `clip:text,clip:scene` means `clip:[text,scene]`. A sampler
    object is copied, so every run has its own state; it carries its own
    settings and questions. `questions` is the vocabulary to check against;
    None accepts any name.

    The settings go to the samplers named by string whose constructor takes
    them: `threshold` is how much a frame must change to be kept, `confidence`
    the detector's box threshold, `languages` what the OCR reads. A setting no
    named sampler takes is refused.
    """
    plan = _plan(list(specs), questions)
    if plan.problems:
        error = UnknownOption if plan.unknown_only else Refused
        raise error("; ".join(plan.problems))

    given: dict[str, Any] = {k: v for k, v in {
        "every_n": every_n, "threshold": threshold, "vocabulary": vocabulary,
        "confidence": confidence, "languages": languages}.items() if v is not None}
    for listed in ("vocabulary", "languages"):
        if listed in given:
            given[listed] = list(given[listed])
    rate = {"min_interval_s": min_interval_s, "max_per_chunk": max_per_chunk}

    named = [run for run in plan.runs if not isinstance(run, Sampler)]
    takes = {name: settings_of(samplers_mod.class_of(name)) for name, _ in named}
    unreachable = sorted(s for s in given
                         if not any(s in takes[name] for name, _ in named))
    if unreachable:
        readers = {s: sorted(n for n in samplers_mod.available()
                             if s in settings_of(samplers_mod.class_of(n)))
                   for s in unreachable}
        detail = "; ".join(f"{s} is read by {', '.join(readers[s]) or 'no sampler'}"
                           for s in unreachable)
        objects = [run.name for run in plan.runs if isinstance(run, Sampler)]
        hint = (" (a sampler object carries its own settings: set them on it)"
                if objects else "")
        raise Refused(
            f"no sampler named in the spec reads {', '.join(unreachable)} "
            f"(named: {', '.join(n for n, _ in named) or 'none'}) -- {detail}{hint}")

    built: list[Sampler] = []
    for run in plan.runs:
        if isinstance(run, Sampler):
            # Its own copy: per-chunk state is not shared between runs, while
            # anything it loaded (a model) is.
            built.append(copy.copy(run))
            continue
        name, asked = run
        settings = {s: v for s, v in given.items() if s in takes[name]}
        ask = {"prompts": asked} if asked else {}
        built.append(samplers_mod.build(name, **ask, **rate, **settings))
    return built


__all__ = ["RATE_SETTINGS", "SAMPLER_SETTINGS", "Spec", "build_samplers",
           "parse_spec", "problems", "settings_of", "spec_text", "split_specs"]
