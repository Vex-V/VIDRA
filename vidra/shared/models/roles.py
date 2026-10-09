"""Which model fills each role, chosen once and handed to every call.

    models = Models(vlm=OpenAI("gpt-5.4-mini"),
                    llm=Anthropic("claude-haiku-4-5"),
                    embedder=LocalEmbedder())

    video_rag("x.mp4", "data/out", models=models)
    aggregates.aggregate(out=out, models=models, summary=video.excerpt(transcript=True))
    search("the reactor", "x", models=models)

Pipelines and searches take `models=` and nothing else; a component takes
the model it calls by its role's name (`describe(vlm=...)`).

Four roles: the vlm (frames -> answers), the llm (text -> text), the
embedder (text -> vectors, the same wherever an index is built and read) and
the visual_embedder (frames -> vectors, beside or instead of the vlm). Each is
a model object -- a ready-made one or a subclass of `VLM`, `LLM`, `Embedder`
or `VisualEmbedder`. A role left None is the default when the call is made,
except the visual_embedder, which has none: None leaves frames unembedded. A
role that is given is checked at construction (the right kind, a key, an API
key).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..reporting.errors import Refused
from .base import ROLES, Embedder, LLM, VLM, VisualEmbedder, require


def default(role: str) -> Any:
    """The model a role uses when none is given: OpenAI's. The visual_embedder
    has none."""
    if role == "visual_embedder":
        return None
    if role == "embedder":
        from .embedders.remote import OpenAIEmbedder
        return OpenAIEmbedder()
    from .llm import OpenAI
    return OpenAI()


def resolve(role: str, model: Any) -> Any:
    """`model`, or the role's default when it is None."""
    return default(role) if model is None else model


@dataclass(frozen=True)
class Models:
    """A model per role; `None` is the default."""

    #: Frames -> answers: a vision-language model.
    vlm: Optional[VLM] = None
    #: Text -> text: the llm aggregates and the entity accounts.
    llm: Optional[LLM] = None
    #: Text -> vectors. Must be the same wherever an index is built and read.
    embedder: Optional[Embedder] = None
    #: Frames -> vectors, and a query into their space. None embeds no frames.
    visual_embedder: Optional[VisualEmbedder] = None

    def __post_init__(self) -> None:
        for role in ROLES:
            model = getattr(self, role)
            if model is not None:
                require(role, model)

    def resolved(self) -> dict[str, Any]:
        """The model each role uses right now, defaults filled in."""
        return {role: resolve(role, getattr(self, role)) for role in ROLES}


def given(models: Optional[Models]) -> Models:
    """`models` as a pipeline uses it: None is every role at its default."""
    if models is None:
        return Models()
    if not isinstance(models, Models):
        raise Refused(f"models must be a Models, not {type(models).__name__} -- "
                      f"e.g. Models(vlm=OpenAI())")
    return models


__all__ = ["Models", "default", "given", "resolve"]
