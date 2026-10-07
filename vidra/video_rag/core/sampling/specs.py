"""The sampler spec: `"clip:[text,scene],yolo"` -> built samplers.

Shared by every pipeline that samples, so a spec means the same run whether
the frames come from a file or a stream. Questions are checked against a
vocabulary the caller passes; nothing here reads the question library.
"""

from __future__ import annotations

from typing import Optional, Sequence

from . import samplers as samplers_mod
from vidra.shared.reporting.errors import Refused, UnknownOption


def split_specs(sampler: str | Sequence[str]) -> list[str]:
    """`"clip:[text,scene],yolo"` -> `["clip:[text,scene]", "yolo"]`, aware of
    brackets.
    """
    if not isinstance(sampler, str):
        return [s.strip() for s in sampler if str(s).strip()]
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


#: Setting -> the samplers that read it. Settings absent here apply to every
#: sampler.
SAMPLER_SETTINGS: dict[str, tuple[str, ...]] = {
    "every_n": ("uniform",),
    # Samplers that keep a frame when it changed enough.
    "threshold": ("clip", "yolo", "objects", "text"),
    "vocabulary": ("objects",),
    "confidence": ("objects",),
    "languages": ("text",),
}


def build_samplers(specs: Sequence[str], every_n: Optional[int] = None,
                   min_interval_s: float = 0.0,
                   max_per_chunk: Optional[int] = None,
                   threshold: Optional[float] = None,
                   vocabulary: Optional[Sequence[str]] = None,
                   confidence: Optional[float] = None,
                   languages: Optional[Sequence[str]] = None,
                   questions: Optional[Sequence[str]] = None
                   ) -> list[samplers_mod.Sampler]:
    """`["yolo", "clip:[text,scene]"]` -> sampler objects.

    A name may carry one question after a colon or several in a list; unpaired,
    the question is the sampler's own name. Specs naming the same sampler merge
    into one run, so `clip:text,clip:scene` means `clip:[text,scene]`.
    `questions` is the vocabulary to check against; None accepts any name.
    `threshold` is how much a frame must change to be kept; `confidence` is the
    detector's box threshold; `languages` is what the OCR reads.
    """
    given = {"every_n": every_n, "threshold": threshold,
             "vocabulary": vocabulary, "confidence": confidence,
             "languages": languages}

    rate = {"min_interval_s": min_interval_s, "max_per_chunk": max_per_chunk}
    tuned = {} if threshold is None else {"threshold": threshold}
    stride = {} if every_n is None else {"every_n": every_n}
    sure = {} if confidence is None else {"confidence": confidence}
    reads = {} if languages is None else {"languages": list(languages)}

    grouped: dict[str, list[str]] = {}
    # Named alone somewhere: its own question is asked even when another spec
    # adds more, so `clip,clip:checkout` is `clip:[clip,checkout]`.
    bare: set[str] = set()
    for spec in [s.strip() for s in specs if s.strip()]:
        name, asked = parse_spec(spec)
        if not asked:
            bare.add(name)
        for question in asked:
            if questions is not None and question not in questions:
                raise UnknownOption(
                    f"{spec!r}: unknown question {question!r}; "
                    f"known: {', '.join(questions)}")
        merged = grouped.setdefault(name, [])
        for question in asked:
            if question not in merged:
                merged.append(question)

    # A setting no chosen sampler reads is refused.
    unreachable = sorted(s for s, value in given.items()
                         if value is not None
                         and not set(SAMPLER_SETTINGS[s]) & set(grouped))
    if unreachable:
        detail = "; ".join(f"{s} is read by "
                           f"{', '.join(SAMPLER_SETTINGS[s])}"
                           for s in unreachable)
        raise Refused(
            f"no chosen sampler reads {', '.join(unreachable)} "
            f"(chosen: {', '.join(sorted(grouped))}) -- {detail}")

    for name, asked in grouped.items():
        if asked and name in bare and name not in asked:
            asked.insert(0, name)

    built: list[samplers_mod.Sampler] = []
    for name, asked in grouped.items():
        ask = {"prompts": asked} if asked else {}
        if name == "uniform":
            built.append(samplers_mod.build(name, **ask, **stride, **rate))
        elif name == "objects":
            built.append(samplers_mod.build(name, vocabulary=list(vocabulary)
                                            if vocabulary else None,
                                            **ask, **tuned, **sure, **rate))
        elif name == "text":
            built.append(samplers_mod.build(name, **ask, **tuned, **reads, **rate))
        else:
            # Unset thresholds keep each sampler's own default.
            built.append(samplers_mod.build(name, **ask, **tuned, **rate))
    return built


__all__ = ["SAMPLER_SETTINGS", "build_samplers", "parse_spec", "split_specs"]
