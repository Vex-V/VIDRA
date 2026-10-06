"""Which model fills each role, chosen once and handed to every call.

    models = Models(vlm=OpenAI("gpt-5.4-mini"),
                    llm=Anthropic("claude-haiku-4-5"),
                    embedder=LocalEmbedder())

    video_rag("x.mp4", "data/out", models=models)
    aggregates.aggregate(out=out, models=models, summary=video.excerpt(transcript=True))
    search("the reactor", "x", models=models)

Three roles: the vlm (frames -> answers), the llm (text -> text) and the
embedder (text -> vectors, the same wherever an index is built and read). Each
is a model object -- a ready-made one or a subclass of `VLM`, `LLM` or
`Embedder`. A role left None is the default when the call is made; a role
that is given is checked at construction (the right kind, a key, an API key).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..reporting.errors import Refused
from .base import ROLES, Embedder, LLM, VLM, require


def default(role: str) -> Any:
    """The model a role uses when none is given: OpenAI's."""
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

    def __post_init__(self) -> None:
        for role in ROLES:
            model = getattr(self, role)
            if model is not None:
                require(role, model)

    def resolved(self) -> dict[str, Any]:
        """The model each role uses right now, defaults filled in."""
        return {role: resolve(role, getattr(self, role)) for role in ROLES}


def unpack(models: Optional[Models], **given: Any) -> dict[str, Any]:
    """The models a call should use, from `models` and its own keywords. A role
    set in both places is refused; one set in neither stays None.
    """
    if models is None:
        return dict(given)
    if not isinstance(models, Models):
        raise Refused(f"models must be a Models, not {type(models).__name__}")
    out: dict[str, Any] = {}
    for name, value in given.items():
        held = getattr(models, name)
        if value is not None and held is not None:
            raise Refused(f"{name} is set twice, as {value!r} and on models as "
                          f"{held!r}; pass it one way")
        out[name] = value if value is not None else held
    return out


__all__ = ["Models", "default", "resolve", "unpack"]
