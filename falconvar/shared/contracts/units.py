"""One embeddable unit, and how a document becomes its text.

Both tiers make units: `embed` one per (chunk, sampler), `aggregates` one per
video from its summary. So this is a contract, not a video_rag feature -- it
sits beside `documents` for the reason `documents` sits here at all.

`render` sorts keys at every level. `jsonb` preserves array order but not
object key order, so a document read back from Postgres hands its keys back in
a different order from the file; joining values in iteration order made the
same person into different text, a different hash and a different vector
depending on where it was read from.

Three separators, one per level: `. ` between fields, ` | ` between entities,
`; ` between one entity's attributes. Fields are named, because a bare `", "`
also occurs inside values and made field boundaries invisible.

**An aggregate renders a field through this too.** A field read by a summary
and the same field read by a search must be the same string, or the text a
summary is built from is not the text that was indexed -- and nothing would
report the difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .documents import fingerprint_of


def render(summary: str, structured: dict[str, Any]) -> str:
    """One string per unit: the summary, then every structured field, named."""
    parts: list[str] = []
    if summary and summary.strip():
        parts.append(summary.strip())
    for key in sorted(structured):
        value = structured[key]
        rendered = _render_value(value)
        if rendered:
            parts.append(f"{key}: {rendered}")
    return ". ".join(parts)


def _render_value(value: Any) -> str:
    if value is None or value == "" or value == []:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        # Sorted here too. Determinism had been handled one level deep and not
        # two, which is exactly how the Postgres/file divergence survived.
        return "; ".join(f"{k} {_render_value(value[k])}" for k in sorted(value)
                         if _render_value(value[k]))
    if isinstance(value, list):
        return " | ".join(r for r in (_render_value(v) for v in value) if r)
    return str(value)


@dataclass
class Unit:
    """One embeddable thing, keyed by a hash of its own text."""

    #: Which video this text came from.
    video_id: str
    #: Which chunk of its grid. With `video_id` this is the key a moment is
    #: grouped on -- a chunk id indexes one video's grid, and grouping on it
    #: alone fused two videos' chunk 0 into one moment.
    chunk_id: int
    #: The pairing that produced it, e.g. `clip:text`, or the bare sampler name
    #: when the question *is* the strategy's own.
    sampler_id: str
    #: The text that gets embedded: the summary and the structured fields
    #: together. Measured, both beats either -- dense MRR 0.705 against 0.528
    #: for the summary alone and 0.636 for the fields alone.
    content: str
    #: The answer's fields, kept as payload so a search can filter on them.
    #: Only useful as a filter where the values are a vocabulary: `role` is
    #: free text, so one video produced `cashier`, `customer` and `child
    #: customer`.
    structured: dict[str, Any] = field(default_factory=dict)
    #: The embedding, or None when this unit was not re-embedded because its
    #: text is unchanged. A backend handed a None writes nothing, which is why
    #: the vectors must exist before any upsert.
    vector: Optional[list[float]] = None
    #: The two halves of `sampler_id`, carried rather than parsed. Filtering by
    #: question is the query a person actually makes -- "the text on screen",
    #: not "what the CLIP sampler said" -- and it is not expressible as a
    #: suffix match, because a bare id like `clip` means question == strategy.
    sampler: str = ""
    #: The question half of `sampler_id`, as its own field so the-text-on-
    #: screen-wherever-asked is one equality -- which is the query a person
    #: actually makes.
    question: str = ""

    @property
    def text_hash(self) -> str:
        """Identity is the text. Re-embedding is then "what changed", not
        "what is here", which is what makes a re-run cost nothing."""
        return fingerprint_of({"content": self.content})

    @property
    def key(self) -> str:
        return f"{self.video_id}:{self.chunk_id}:{self.sampler_id}"

    def as_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id, "chunk_id": self.chunk_id,
                "sampler_id": self.sampler_id, "content": self.content,
                "structured": self.structured, "text_hash": self.text_hash,
                "sampler": self.sampler, "question": self.question,
                "vector": self.vector}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Unit":
        return cls(d["video_id"], d["chunk_id"], d["sampler_id"], d["content"],
                   d.get("structured", {}), d.get("vector"),
                   d.get("sampler", ""), d.get("question", ""))


__all__ = ["Unit", "render"]
