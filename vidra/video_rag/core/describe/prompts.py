"""What to ask about a chunk, and what shape to expect back.

Logic only: the instructions and shapes are data in `prompts.json` (and
`data/prompts.json`), merged by `library.py`. A question resolves to a shape,
and the shape decides the response schema.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from . import library


def __getattr__(name: str) -> Any:
    """`SYSTEM` reads through to the data."""
    if name == "SYSTEM":
        return library.load()["system"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def questions() -> list[str]:
    """The valid question names."""
    return library.questions()


def schema_for(question: str) -> dict[str, Any]:
    """The strict response schema for one question: every property required."""
    shape = library.shape_of(question)
    fields = dict(shape.get("fields") or {})
    summaries = library.load()["summaries"]
    summary = summaries.get(shape.get("summary", "standard")) or summaries["standard"]
    properties = {"summary": summary, **fields}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def version_of(question: str) -> str:
    """A hash of one question: its instruction, its shape and the system preamble."""
    payload = json.dumps({
        "system": library.load()["system"],
        "instruction": library.instruction_of(question),
        "shape": library.shape_of(question),
        "summaries": library.load()["summaries"],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def versions(names: Sequence[str]) -> dict[str, str]:
    """`{question: hash}` for the questions a run actually asks."""
    return {name: version_of(name) for name in sorted(set(names))}


def question_for(context: dict[str, Any]) -> str:
    """Which question these frames are for: the one the caller names, else the
    sampler config's `prompt`, else the sampler's name. Unknown questions fall
    back to the general one.
    """
    if context.get("question"):
        return str(context["question"])
    config = context.get("sampler_config") or {}
    asked = config.get("prompts") or []
    if len(asked) == 1:
        return asked[0]
    return config.get("prompt") or context.get("sampler") or ""


def questions_of(sampler_config: dict[str, Any], run_id: str = "") -> list[str]:
    """The questions asked about one sampler run's frames. Empty `prompts` means
    one question named after the sampler.
    """
    asked = list(sampler_config.get("prompts") or [])
    if asked:
        return asked
    single = sampler_config.get("prompt")          # documents written before lists
    if single:
        return [single]
    return [sampler_config.get("name") or run_id]


def answer_id(sampler_name: str, question: str) -> str:
    """The key one answer is stored under: `name:question`, or the bare name when
    the question is the sampler's own.
    """
    return sampler_name if question == sampler_name else f"{sampler_name}:{question}"


def questions_in(manifest) -> list[str]:
    """Every question this manifest asks, from its sampler config."""
    return sorted({q for s in manifest.config.get("samplers", [])
                   for q in questions_of(s, s.get("id", ""))})


def span_of(context: dict[str, Any]) -> str:
    return f"{context['start_ts']:.1f}s to {context['end_ts']:.1f}s"


def vocabulary_of(context: dict[str, Any]) -> str:
    """The open-vocabulary list the detector was given, as prose."""
    detector = (context.get("sampler_config") or {}).get("detector") or {}
    words = detector.get("vocabulary") or []
    return ", ".join(words) if words else "no vocabulary recorded"


def for_sampler(question: str, context: dict[str, Any], frame_count: int) -> str:
    """The instruction for one (chunk, sampler) call, with its placeholders filled."""
    return library.instruction_of(question).format(
        n=frame_count,
        span=span_of(context),
        vocabulary=vocabulary_of(context),
    )


def frame_label(index: int, media_ts: float, position: int, total: int) -> str:
    """What precedes each image: its position, timestamp and frame index."""
    return f"Frame {position} of {total} -- t={media_ts:.2f}s (index {index}):"


def view_label(label: str, media_ts: float, position: int) -> str:
    """What precedes an image made from a frame: which frame, and what it shows."""
    return f"Frame {position}, {label} -- t={media_ts:.2f}s:"


#: Added to the instruction when any image is a view made from a frame.
VIEWS = ("Some images are views made from a frame -- a crop, an enlargement or a "
         "marked copy -- labelled with what they show. They are the same moment as "
         "their frame, not further frames: use them for detail.")
