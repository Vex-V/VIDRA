"""Which model fills each role, chosen once and handed to every call.

    models = Models(vlm=OpenAI("gpt-6-luna"),
                    llm=Anthropic("claude-haiku-4-5"),
                    embedder=LocalEmbedder())

    video_rag("x.mp4", "data/out", models=models)
    aggregates.aggregate(out=out, models=models, summary=video.excerpt(transcript=True))
    search("the reactor", "x", models=models)

Pipelines and searches take `models=` and nothing else; a component takes
the model it calls by its role's name (`describe(vlm=...)`).

Seven roles: the vlm (frames -> answers), the llm (text -> text), the
embedder (text -> vectors, the same wherever an index is built and read), the
visual_embedder (frames -> vectors, beside or instead of the vlm), the
voice_embedder (each diarized speaker -> a voiceprint), the transcriber (the
soundtrack -> words with times) and the diarizer (the soundtrack -> who spoke
when). Each is a model object -- a ready-made one or a subclass of `VLM`,
`LLM`, `Embedder`, `VisualEmbedder`, `VoiceEmbedder`, `Transcriber` or
`Diarizer`. A role left None is the default when the call is made -- OpenAI's
for the first three, Whisper and pyannote in this process for the last two --
except the visual_embedder and the voice_embedder, which have none: None
leaves frames, or voices, unembedded. A role that is given is checked at
construction (the right kind, a key, an API key).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..reporting.errors import Refused
from .base import (ROLES, Diarizer, Embedder, LLM, Transcriber, VLM, VisualEmbedder,
                   VoiceEmbedder, require)


def default(role: str) -> Any:
    """The model a role uses when none is given: OpenAI's, or Whisper and
    pyannote for the soundtrack. The visual_embedder and the voice_embedder
    have none. Building one loads nothing."""
    if role in ("visual_embedder", "voice_embedder"):
        return None
    if role == "transcriber":
        from .speech.whisper import WhisperTranscriber
        return WhisperTranscriber()
    if role == "diarizer":
        from .speech.pyannote import PyannoteDiarizer
        return PyannoteDiarizer()
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
    #: Each diarized speaker -> a voiceprint (`voices`). None makes no prints.
    voice_embedder: Optional[VoiceEmbedder] = None
    #: The soundtrack -> words with times (`audio`). None is WhisperTranscriber().
    transcriber: Optional[Transcriber] = None
    #: The soundtrack -> who spoke when (`audio`). None is PyannoteDiarizer();
    #: NoDiarizer() for none.
    diarizer: Optional[Diarizer] = None

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
