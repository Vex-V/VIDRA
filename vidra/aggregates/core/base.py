"""The aggregator protocols, `Context`, and `DefinitionRunner`.

`Context` joins every document by `chunk_id`. A tier is a cost ceiling:
`free` is arithmetic, `local` adds local models, `llm` adds model calls; they
run cheapest first. `stats`, `speakers` and `coverage` take no input; every
other aggregator reads an input (see `inputs`) and answers once per input.
`depends_on` names a source the video must have, else it is skipped with the
reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

from ...shared.contracts.documents import (Descriptions, Manifest, Timeline,
                                           Transcript, fingerprint_of)
#: Re-exported, the one class shared with audio.
from ...shared.reporting.errors import ModelUnavailable
from ...shared.models.base import LLM

#: Cheapest first.
TIERS = ("free", "local", "llm")


@dataclass
class Context:
    """Every finished document, joined by `chunk_id`."""

    video_id: str
    timeline: Timeline
    manifest: Optional[Manifest] = None
    descriptions: Optional[Descriptions] = None
    transcript: Optional[Transcript] = None

    @property
    def sources(self) -> set[str]:
        """What this video actually has -- answer ids, plus `transcript`."""
        found: set[str] = set()
        if self.descriptions is not None:
            for chunk in self.descriptions.chunks:
                found |= set(chunk.get("samplers", {}))
        if self.transcript is not None and any(
                c.get("word_count") for c in self.transcript.chunks):
            found.add("transcript")
        return found

    def chunk_ids(self) -> list[int]:
        return list(range(len(self.timeline)))

    def span_of(self, chunk_id: int) -> tuple[float, float]:
        return self.timeline.bounds_of(chunk_id)

    def text_of(self, chunk_id: int) -> dict[str, str]:
        """Everything said about one chunk, by answer id."""
        out: dict[str, str] = {}
        if self.descriptions is not None:
            for chunk in self.descriptions.chunks:
                if chunk["chunk_id"] == chunk_id:
                    for sid, block in chunk.get("samplers", {}).items():
                        out[sid] = block.get("description", "")
        if self.transcript is not None:
            text = self.transcript.text_of(chunk_id)
            if text:
                out["transcript"] = text
        return out

    def inputs_fingerprint(self) -> str:
        """A hash of every chunk's text, for aggregators that take no input."""
        return fingerprint_of({
            "timeline": self.timeline.fingerprint(),
            "text": {str(i): self.text_of(i) for i in self.chunk_ids()},
        })


class Aggregator(Protocol):
    """Counts what extraction produced. Takes no input."""

    name: str
    tier: str
    about: str
    depends_on: Sequence[str]

    def run(self, context: Context) -> dict[str, Any]: ...


class Reader(Protocol):
    """Answers once per input. `read` is separate from `run` so a stored answer can
    be reused before anything is paid for.
    """

    name: str
    tier: str
    about: str
    depends_on: Sequence[str]
    takes_inputs: bool
    #: What the answer depends on besides the text read.
    version: str

    def read(self, context: Context, one: Any) -> Any: ...

    def run(self, context: Context, read: Any) -> dict[str, Any]: ...


def missing(aggregator: Any, context: Context) -> Optional[str]:
    """Why this aggregator cannot run here, or None."""
    for need in getattr(aggregator, "depends_on", ()):
        if need not in context.sources:
            return f"needs {need}, which this video has no output for"
    return None


#: How many lines go into one fold.
BATCH = 25

#: Lines one `spans` naming call or one `items` call reads.
WINDOW = 100


def schema(name: str, properties: dict[str, Any]) -> dict[str, Any]:
    """`{name, schema}` for a strict structured call: every key required."""
    return {"name": name.replace(":", "_").replace("~", "_"),
            "schema": {"type": "object", "additionalProperties": False,
                       "required": list(properties), "properties": properties}}


def listing(key: str, item: dict[str, Any]) -> dict[str, Any]:
    """A property holding a list of objects, every key required."""
    return {key: {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": list(item), "properties": item}}}


class DefinitionRunner:
    """An aggregator built from a definition (`summary`, `chapters`, `events`, a
    link profile). A subclass is a kind (`fold`, `spans`, `items`, `link`);
    `_run` is what a kind implements.
    """

    tier = "llm"
    depends_on: tuple[str, ...] = ()
    takes_inputs = True

    def __init__(self, definition_id: str, llm: Optional[LLM] = None) -> None:
        from ...shared.models.base import calls
        from ...shared.models.roles import resolve
        from .. import definitions
        self.name = definition_id
        self.section, self.definition = definitions.locate(definition_id)
        self.entry = definitions.get(self.section, self.definition)
        self.about = self.entry.get("about", "")
        self.version = definitions.version_of(self.section, self.definition)
        self.llm = calls("llm", resolve("llm", llm))

    @property
    def model_key(self) -> str:
        return self.llm.key

    def properties(self) -> dict[str, Any]:
        """The definition's own fields, compiled by the shared field builder."""
        from ...shared.contracts.fields import compile_fields
        return compile_fields(self.entry["fields"])

    def read(self, context: Any, one: Any) -> Any:
        from .inputs import read as read_input
        return read_input(context, one)

    def run(self, context: Any, read: Any) -> dict[str, Any]:
        import asyncio
        return asyncio.run(self._run(context, read))

    async def _run(self, context: Any, read: Any) -> dict[str, Any]:
        raise NotImplementedError


__all__ = ["BATCH", "TIERS", "WINDOW", "Aggregator", "Context", "DefinitionRunner",
           "ModelUnavailable", "Reader", "listing", "missing", "schema"]
