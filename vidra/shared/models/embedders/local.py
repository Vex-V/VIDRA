"""Vectors from a Hugging Face model loaded into this process, with
`sentence-transformers`.

No key, no server. Weights land in `weights/embedders/`. A model is loaded
once per (model, device) per process.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional, Sequence

from ...config import paths, settings
from .. import providers
from ..devices import default_device
from . import EmbedderUnavailable, key_for, prefixes_for

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


class LocalEmbedder:
    name = "local"

    def __init__(self, model: Optional[str] = None, dims: Optional[int] = None,
                 device: Optional[str] = None, batch: int = 32,
                 query_prefix: Optional[str] = None,
                 document_prefix: Optional[str] = None) -> None:
        self.model = model or providers.get("local").embed_model or ""
        self.device = device or default_device()
        self.batch = batch
        auto_query, auto_document = prefixes_for(self.model)
        self.query_prefix = auto_query if query_prefix is None else query_prefix
        self.document_prefix = auto_document if document_prefix is None else document_prefix
        self._model = _load(self.model, self.device)
        self.dims = int(self._model.get_embedding_dimension())
        if dims and int(dims) != self.dims:
            raise EmbedderUnavailable(
                f"{self.model} makes {self.dims}-dimension vectors, not {dims}")

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self._model.encode(list(texts), batch_size=self.batch,
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
        return key_for(self.name, self.model, self.dims,
                       self.query_prefix, self.document_prefix)

    def config(self) -> dict[str, Any]:
        return {"embedder": self.name, "model": self.model, "dims": self.dims,
                "device": self.device, "query_prefix": self.query_prefix,
                "document_prefix": self.document_prefix}


__all__ = ["LocalEmbedder", "cache_dir"]
