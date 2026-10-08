"""One embeddable unit, and how a document becomes its text.

`render` joins a summary and its structured fields, keys sorted at every
level. Three separators: `. ` between fields, ` | ` between entities, `; `
between an entity's attributes. Aggregates render fields through it too.
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
        # Sorted at every level.
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
    #: Which chunk of its grid.
    chunk_id: int
    #: The pairing that produced it, e.g. `clip:text`, or the bare sampler name.
    sampler_id: str
    #: The text that gets embedded: the summary and the structured fields.
    content: str
    #: The answer's fields, kept as payload for filtering.
    structured: dict[str, Any] = field(default_factory=dict)
    #: The embedding, or None when the unit was not re-embedded.
    vector: Optional[list[float]] = None
    #: The sampler half of `sampler_id`.
    sampler: str = ""
    #: The question half of `sampler_id`.
    question: str = ""

    @property
    def text_hash(self) -> str:
        """A hash of the content."""
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
