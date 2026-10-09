"""The sampler spec: `"clip:[text,scene],yolo"`, or sampler objects, -> built
samplers.

Shared by every pipeline that samples. A spec is a string, or a list mixing
strings and `Sampler` objects:

    "clip:[text,scene],yolo"                         by name, default settings
    [samplers.build("clip", threshold=0.93, prompts=["safety"]), "yolo"]
    [Brightness(step=12, prompts=["overview"])]      a class of your own

A sampler named by a string is built with its defaults; settings are given
only on an object, so each lands on the sampler it was written on.
Questions are checked against a vocabulary the caller passes; nothing here
reads the question library.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

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
    parts = [q.strip() for q in rest.split(",")]
    seen, questions = set(), []
    for q in parts:
        if q and q not in seen:          # each question once
            seen.add(q)
            questions.append(q)
    return name, questions


def spec_text(sampler: Spec | Sequence[Spec]) -> str:
    """A spec as one line, for logs: an object appears as `name(object)`."""
    return ",".join(s if isinstance(s, str) else f"{s.name}(object)"
                    for s in split_specs(sampler))


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
            if "+" in question:
                plan.add(f"{spec!r}: questions are listed in brackets, "
                         f"{name}:[{question.replace('+', ',')}]")
            elif questions is not None and question not in questions:
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


def build_samplers(specs: Sequence[Spec],
                   questions: Optional[Sequence[str]] = None) -> list[Sampler]:
    """`["yolo", "clip:[text,scene]", Brightness(step=12)]` -> sampler objects,
    one per run, in spec order.

    A name may carry one question after a colon or several in a list; unpaired,
    the question is the sampler's own name. Specs naming the same sampler merge
    into one run, so `clip:text,clip:scene` means `clip:[text,scene]`. A sampler
    object is copied, so every run has its own state; it carries its own
    settings and questions, and a name is built with its defaults.
    `questions` is the vocabulary to check against; None accepts any name.
    """
    plan = _plan(list(specs), questions)
    if plan.problems:
        error = UnknownOption if plan.unknown_only else Refused
        raise error("; ".join(plan.problems))

    built: list[Sampler] = []
    for run in plan.runs:
        if isinstance(run, Sampler):
            # Its own copy: per-chunk state is not shared between runs, while
            # anything it loaded (a model) is.
            built.append(copy.copy(run))
            continue
        name, asked = run
        built.append(samplers_mod.build(name, **({"prompts": asked} if asked else {})))
    return built


__all__ = ["Spec", "build_samplers", "parse_spec", "problems", "spec_text",
           "split_specs"]
