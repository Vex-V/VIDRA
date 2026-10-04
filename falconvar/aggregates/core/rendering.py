"""How what `inputs` read reaches an aggregator: times, batches and pieces.

Times come from the grid, never from a model. `batched` divides rows for the
llm kinds; `pieces` cuts text for the local models.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .base import Context


def batched(items: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def resolve_span(context: Context, chunk_ids: Sequence[int]
                 ) -> tuple[float, float]:
    """The span covering these chunks, from the grid."""
    valid = [c for c in chunk_ids if 0 <= c < len(context.timeline)]
    if not valid:
        return 0.0, 0.0
    return (min(context.span_of(c)[0] for c in valid),
            max(context.span_of(c)[1] for c in valid))


def plain(row: Any) -> str:
    """A row's text without labels or times: what a model should read."""
    return " ".join(said for _, said in row.parts)


def pieces(text: str, limit: int) -> list[str]:
    """Text cut at sentence ends, then at spaces, into pieces of at most `limit`
    characters. Nothing is dropped.
    """
    out: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                out.append(current)
                current = ""
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current = f"{current} {sentence}"
        else:
            out.append(current)
            current = sentence
    if current:
        out.append(current)
    return [p for p in out if p]


__all__ = ["batched", "pieces", "plain", "resolve_span"]
