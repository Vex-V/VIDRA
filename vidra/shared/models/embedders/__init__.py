"""The ready-made embedders, and what every embedder shares.

    OpenAIEmbedder("text-embedding-3-small")     OpenAI, or any server's /embeddings
    LocalEmbedder("BAAI/bge-small-en-v1.5")      a Hugging Face model in this process
    HashEmbedder()                               no model: for exercising a pipeline

One vector space per embedder: its key is `name:model:dims`, and vectors from
different keys are never compared. A query and a document are embedded with
the prefixes the model was trained with (e5, nomic, bge, mxbai); a prefix set
by hand changes the key.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

from ...reporting.errors import Unavailable
from ..base import Embedder


class EmbedderUnavailable(Unavailable):
    """No client, no key, or a model this account cannot reach."""


def query_vector(embedder: Any, text: str) -> list[float]:
    """The vector for a search, through the query side."""
    return embedder.embed_query(text)


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


class HashEmbedder(Embedder):
    """Deterministic vectors from a hash of each word. No model, no network: for
    exercising the pipeline without a key. Its key says `hash`.
    """

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims
        self.key = f"hash:none:{dims}"

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


def __getattr__(name: str) -> Any:
    """PEP 562: the two that import an SDK resolve on first use."""
    if name == "OpenAIEmbedder":
        from .remote import OpenAIEmbedder
        return OpenAIEmbedder
    if name == "LocalEmbedder":
        from .local import LocalEmbedder
        return LocalEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Embedder", "EmbedderUnavailable", "HashEmbedder", "LocalEmbedder",
           "OpenAIEmbedder", "PREFIXES", "key_for", "prefixes_for", "query_vector"]
