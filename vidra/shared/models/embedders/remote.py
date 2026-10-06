"""Vectors from an API: OpenAI, or any server serving OpenAI's /embeddings.

    OpenAIEmbedder("text-embedding-3-small")
    OpenAIEmbedder("nomic-embed-text", base_url="http://localhost:11434/v1", name="ollama")
    OpenAIEmbedder("voyage-3.5", base_url="https://api.voyageai.com/v1", name="voyage",
                   api_key=..., input_type=True)

A model's width is known (OpenAI's) or probed once per process with a short
string, and every batch is checked against it.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from ..base import Embedder
from . import EmbedderUnavailable, key_for, prefixes_for

#: Widths that need no probe.
KNOWN_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072,
              "text-embedding-ada-002": 1536}

#: Where OpenAI's own key is read from, first set wins.
KEY_VARS = ("OPENAI_API_KEY", "OPENAI_API")

_PROBED: dict[tuple[str, str, str, int], int] = {}


class OpenAIEmbedder(Embedder):
    """`name` is the first part of the key (`openai:text-embedding-3-small:1536`).
    With no `base_url` this is OpenAI and needs a key (`api_key=`, else
    `OPENAI_API_KEY`); with one, the key is optional. `dims` asks the server for
    that width (sent as `dimensions`). `input_type=True` sends `query` or
    `document` with each call, as Voyage wants.
    """

    def __init__(self, model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 name: str = "openai", dims: Optional[int] = None,
                 input_type: bool = False,
                 query_prefix: Optional[str] = None,
                 document_prefix: Optional[str] = None,
                 client: Any = None) -> None:
        if not model:
            raise EmbedderUnavailable("OpenAIEmbedder needs a model name")
        self.label = name
        self.model_id = model
        self.base_url = base_url
        self.input_type = input_type
        self._api_key = api_key
        auto_query, auto_document = prefixes_for(model)
        self.query_prefix = auto_query if query_prefix is None else query_prefix
        self.document_prefix = auto_document if document_prefix is None else document_prefix
        #: The width asked for; otherwise known or probed.
        self.requested = int(dims) if dims else None
        self._dims = self.requested or (KNOWN_DIMS.get(model) if base_url is None
                                        else None)
        self._client = client

    def __repr__(self) -> str:                    # never the API key
        where = f", base_url={self.base_url!r}" if self.base_url else ""
        return f"OpenAIEmbedder({self.model_id!r}{where})"

    def _key(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        return next((os.environ[v] for v in KEY_VARS if os.environ.get(v)), None)

    def problems(self) -> list[str]:
        if self.base_url is None and not self._key():
            return [f"no key for openai:{self.model_id}: pass api_key= or set "
                    f"{' or '.join(KEY_VARS)} in the environment"]
        return []

    def _connect(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            found = self._key()
            if found is None and self.base_url is None:
                raise EmbedderUnavailable(self.problems()[0])
            options: dict[str, Any] = {"api_key": found or "not-needed"}
            if self.base_url:
                options["base_url"] = self.base_url
            self._client = OpenAI(**options)
        return self._client

    def _call(self, texts: Sequence[str], side: str) -> list[list[float]]:
        extra: dict[str, Any] = {}
        if self.requested:
            extra["dimensions"] = self.requested
        if self.input_type:
            extra["extra_body"] = {"input_type": side}
        try:
            response = self._connect().embeddings.create(model=self.model_id,
                                                         input=list(texts), **extra)
        except EmbedderUnavailable:
            raise
        except Exception as exc:                            # noqa: BLE001
            where = self.base_url or self.label
            raise EmbedderUnavailable(
                f"{self.label} could not embed with {self.model_id!r} ({where}): "
                f"{exc}") from None
        data = sorted(response.data, key=lambda d: getattr(d, "index", 0))
        vectors = [list(d.embedding) for d in data]
        if len(vectors) != len(texts):
            raise EmbedderUnavailable(
                f"{self.label} returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors

    def _checked(self, vectors: list[list[float]]) -> list[list[float]]:
        for vector in vectors:
            if self._dims is None:
                self._dims = len(vector)
            elif len(vector) != self._dims:
                raise EmbedderUnavailable(
                    f"{self.label}:{self.model_id} returned {len(vector)} dimensions "
                    f"where {self._dims} were expected -- a different model behind "
                    "the same name writes into a space it does not belong to")
        return vectors

    @property
    def dims(self) -> int:
        if self._dims is None:
            probe = (self.label, self.base_url or "", self.model_id, self.requested or 0)
            if probe not in _PROBED:
                _PROBED[probe] = len(self._call(["width probe"], "document")[0])
            self._dims = _PROBED[probe]
        return self._dims

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._checked(self._call([self.document_prefix + t for t in texts],
                                        "document"))

    def embed_query(self, text: str) -> list[float]:
        return self._checked(self._call([self.query_prefix + text], "query"))[0]

    @property
    def key(self) -> str:
        return key_for(self.label, self.model_id, self.dims,
                       self.query_prefix, self.document_prefix)


__all__ = ["KNOWN_DIMS", "OpenAIEmbedder"]
