"""Vectors from a Hugging Face model loaded into this process, with
`sentence-transformers`.

    LocalEmbedder()                              BAAI/bge-small-en-v1.5
    LocalEmbedder("intfloat/multilingual-e5-small", device="cpu")

No key, no server. Weights land in `weights/embedders/`. A model is loaded on
first use, once per (model, device) per process.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional, Sequence

from ...config import paths, settings
from ..base import Embedder
from ..devices import default_device
from . import EmbedderUnavailable, key_for, prefixes_for

#: The model when none is named.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def cache_dir() -> Path:
    """Where sentence-transformers caches a checkpoint."""
    return paths.weights_root() / "embedders"

_MODELS: dict[tuple[str, str], Any] = {}
_LOCK = threading.Lock()


def _load(model: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    with _LOCK:
        if (model, device) not in _MODELS:
            try:
                # The configured Hugging Face token, if any (only a gated model's first download
                # needs one).
                _MODELS[(model, device)] = SentenceTransformer(
                    model, device=device, cache_folder=str(cache_dir()),
                    token=settings.hf_token())
            except Exception as exc:                     # noqa: BLE001
                raise EmbedderUnavailable(f"could not load {model!r}: {exc}") from None
        return _MODELS[(model, device)]


class LocalEmbedder(Embedder):
    """`device` defaults to cuda, then mps, then the CPU. `dims`, if given, is
    checked against what the model makes."""

    concurrency = 1

    def __init__(self, model: str = DEFAULT_MODEL, dims: Optional[int] = None,
                 device: Optional[str] = None, batch: int = 32,
                 query_prefix: Optional[str] = None,
                 document_prefix: Optional[str] = None) -> None:
        self.model_id = model or DEFAULT_MODEL
        self.device = device or default_device()
        self.batch = batch
        auto_query, auto_document = prefixes_for(self.model_id)
        self.query_prefix = auto_query if query_prefix is None else query_prefix
        self.document_prefix = auto_document if document_prefix is None else document_prefix
        self._wanted_dims = int(dims) if dims else None
        self._loaded: Any = None

    def __repr__(self) -> str:
        return f"LocalEmbedder({self.model_id!r})"

    def _model(self) -> Any:
        if self._loaded is None:
            loaded = _load(self.model_id, self.device)
            made = int(loaded.get_embedding_dimension())
            if self._wanted_dims and self._wanted_dims != made:
                raise EmbedderUnavailable(
                    f"{self.model_id} makes {made}-dimension vectors, not "
                    f"{self._wanted_dims}")
            self._loaded = loaded
        return self._loaded

    @property
    def dims(self) -> int:
        return int(self._model().get_embedding_dimension())

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self._model().encode(list(texts), batch_size=self.batch,
                                    normalize_embeddings=True,
                                    convert_to_numpy=True).tolist()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]

    @property
    def key(self) -> str:
        return key_for("local", self.model_id, self.dims,
                       self.query_prefix, self.document_prefix)


__all__ = ["DEFAULT_MODEL", "LocalEmbedder", "cache_dir"]
