"""The sampler spec: sampler objects -> the runs a pipeline makes.

Shared by every pipeline that samples. A spec is one `Sampler` or a list of
them, and None is `Uniform()`:

    [Clip(prompts=["text", "scene"]), Yolo()]
    Clip(threshold=0.93)
    [Brightness(step=12, prompts=["overview"])]      a class of your own

Each object is one run: one pass over the frames, answering every question on
it. Questions are checked against a vocabulary the caller passes; nothing here
reads the question library.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from vidra.shared.reporting.errors import Refused, UnknownOption
from . import samplers as samplers_mod
from .samplers import Sampler

#: What a pipeline's `sampler` takes.
Spec = Sampler | Sequence[Sampler] | None

#: A sampler name: it is written into answer ids (`name:question`) and frame
#: file names, so it is kept to what both can carry.
NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


def as_list(sampler: Spec) -> list[Any]:
    """The spec as a list: None is `[Uniform()]`, one object a list of one."""
    if sampler is None:
        return [samplers_mod.Uniform()]
    if isinstance(sampler, (Sampler, str)):
        return [sampler]
    return list(sampler)


def spec_text(sampler: Spec) -> str:
    """A spec as one line, for logs: `clip[text,scene],yolo`."""
    return ",".join(f"{s.name}[{','.join(s.prompts)}]" if s.prompts else s.name
                    for s in as_list(sampler) if isinstance(s, Sampler))


@dataclass
class _Plan:
    """What a spec asks for, checked: every problem, and whether each is an
    unknown question (and so an `UnknownOption`)."""

    problems: list[str] = field(default_factory=list)
    unknown_only: bool = True

    def add(self, problem: str, unknown: bool = False) -> None:
        self.problems.append(problem)
        self.unknown_only = self.unknown_only and unknown


def _plan(specs: list[Any], questions: Optional[Sequence[str]],
          configs: bool = False) -> _Plan:
    plan = _Plan()
    if not specs:
        plan.add("no sampler: pass at least one, or None for Uniform()")
    names: set[str] = set()
    for spec in specs:
        if isinstance(spec, str):
            plan.add(f"a sampler is an object, not the string {spec!r}: "
                     f"samplers.Clip(), samplers.Yolo(prompts=['overview']), ... "
                     f"from vidra.video_rag import samplers")
            continue
        if not isinstance(spec, Sampler):
            plan.add(f"a sampler is a Sampler object, not {type(spec).__name__}")
            continue
        name = spec.name
        if not isinstance(name, str) or not NAME.match(name):
            plan.add(f"{type(spec).__name__}.name must be lowercase letters, "
                     f"digits, '_' or '-', starting with a letter, not {name!r}: "
                     f"it is written into answer ids as name:question")
            continue
        if name in samplers_mod.BUILTIN and not isinstance(
                spec, samplers_mod.builtin(name)):
            plan.add(f"{name!r} is a built-in sampler's name; give "
                     f"{type(spec).__name__} a name of its own")
            continue
        if name in names:
            plan.add(f"two {name!r} samplers: one run per sampler, so put every "
                     f"question on one (prompts=[...])")
            continue
        names.add(name)
        for question in spec.questions():
            if questions is not None and question not in questions:
                plan.add(_no_question(name, question, questions,
                                      own=not spec.prompts),
                         unknown=bool(spec.prompts))
        if not configs:
            continue
        try:
            # Reading the config builds the sampler's models (they are deferred),
            # so this runs only when a run is about to use them.
            json.dumps(spec.config())
        except (TypeError, ValueError) as exc:
            plan.add(f"{name}.config() must be plain JSON, since the manifest "
                     f"records it: {exc}")
    return plan


def _no_question(name: str, question: str, questions: Sequence[str],
                 own: bool) -> str:
    if own:
        return (f"sampler {name!r} has no question of its own: give it one "
                f"(prompts=['overview']), or add one named {name!r} with "
                f"describe.add_question")
    return f"unknown question {question!r} for sampler {name!r}; known: {', '.join(questions)}"


def problems(sampler: Spec, questions: Optional[Sequence[str]] = None) -> list[str]:
    """What is wrong with a spec, as messages; empty is valid. Checks kinds,
    names, questions and duplicates: what `validate` asks. Loads no model."""
    return _plan(as_list(sampler), questions).problems


def build_samplers(sampler: Spec,
                   questions: Optional[Sequence[str]] = None) -> list[Sampler]:
    """The samplers a run uses, one per object, in order, each a copy so that
    per-chunk state is never shared between runs while anything an object
    loaded (a model) is. Each sampler's models are loaded here, on the object
    passed in, so a later run with it loads nothing. `questions` is the
    vocabulary to check against; None accepts any name.
    """
    specs = as_list(sampler)
    plan = _plan(specs, questions, configs=True)
    if plan.problems:
        error = UnknownOption if plan.unknown_only else Refused
        raise error("; ".join(plan.problems))
    return [copy.copy(s) for s in specs]


__all__ = ["NAME", "Spec", "as_list", "build_samplers", "problems", "spec_text"]
