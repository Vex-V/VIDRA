"""The Embedder protocol, and resolving a name to one.

One vector space per embedder: its key is `provider:model:dims`, and vectors
from different keys are never compared. A query and a document are embedded
with the prefixes the model was trained with (e5, nomic, bge, mxbai); a
prefix set by hand changes the key.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Optional, Protocol, Sequence
from ...reporting.errors import Unavailable


class EmbedderUnavailable(Unavailable):
    """No client, no key, or a model this account cannot reach."""


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    @property
    def key(self) -> str: ...
    def config(self) -> dict[str, Any]: ...


def query_vector(embedder: Any, text: str) -> list[float]:
    """The vector for a search, through the query side where there is one."""
    side = getattr(embedder, "embed_query", None)
    return side(text) if side else embedder.embed([text])[0]


_BGE = "Represent this sentence for searching relevant passages: "

#: Model id fragment -> (query prefix, document prefix). First match wins.
PREFIXES: tuple[tuple[str, str, str], ...] = (
    ("multilingual-e5", "query: ", "passage: "),
    ("e5-", "query: ", "passage: "),
    ("nomic-embed-text", "search_query: ", "search_document: "),
    ("bge-small-en", _BGE, ""),
    ("bge-base-en", _BGE, ""),
    ("bge-large-en", _BGE, ""),
    ("mxbai-embed-large", _BGE, ""),
)


def prefixes_for(model: str) -> tuple[str, str]:
    lowered = model.lower()
    for fragment, query, document in PREFIXES:
        if fragment in lowered:
            return query, document
    return "", ""


def key_for(name: str, model: str, dims: int,
            query_prefix: str, document_prefix: str) -> str:
    """`name:model:dims`, plus a suffix when the prefixes are not the model's own."""
    key = f"{name}:{model}:{dims}"
    if (query_prefix, document_prefix) != prefixes_for(model):
        digest = hashlib.sha1(f"{query_prefix}\0{document_prefix}".encode()).hexdigest()
        key += f":p{digest[:6]}"
    return key


class HashEmbedder:
    """Deterministic vectors from a hash. No model, no network: for exercising the
    pipeline without a key. Its key says `hash`.
    """

    name = "hash"

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vector = [0.0] * self.dims
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dims
                vector[index] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            out.append([v / norm for v in vector])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def key(self) -> str:
        return f"{self.name}:none:{self.dims}"

    def config(self) -> dict[str, Any]:
        return {"embedder": self.name, "dims": self.dims}


def build(name: Optional[str] = None, **kwargs) -> Embedder:
    """An embedder from a provider name, `provider/model`, `hash`, or None for the
    default.
    """
    from .. import providers

    chosen, model = providers.choose("embed", name)
    if chosen == providers.OFFLINE["embed"]:
        return HashEmbedder(**({"dims": kwargs["dims"]} if kwargs.get("dims") else {}))
    provider = providers.get(chosen)
    if provider.protocol == "local":
        from .local import LocalEmbedder
        return LocalEmbedder(model, **kwargs)
    from .remote import RemoteEmbedder
    return RemoteEmbedder(chosen, model, **kwargs)


def available() -> list[str]:
    from .. import providers
    return providers.names("embed")


__all__ = ["Embedder", "EmbedderUnavailable", "HashEmbedder", "PREFIXES",
           "available", "build", "key_for", "prefixes_for", "query_vector"]
