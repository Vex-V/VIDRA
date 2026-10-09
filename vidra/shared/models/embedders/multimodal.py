"""Text and images in one space, from one model loaded into this process with
`sentence-transformers`: EmbeddingGemma 2.

    model = LocalMultimodalEmbedder()               google/embeddinggemma-2
    Models(embedder=model, visual_embedder=model)   both roles, one space

A group of images (a chunk's frames and their views) is one input -- each
view's label before its image -- and one vector, made by the model rather than
averaged. Needs `sentence-transformers>=6.1` and `transformers>=5.18`, which
older versions refuse to load. Weights land in `weights/embedders/`.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Sequence

from ...config import settings
from ..base import Embedder, VisualEmbedder
from ..devices import default_device
from . import EmbedderUnavailable

#: The model when none is named.
DEFAULT_MODEL = "google/embeddinggemma-2"

#: Where an image goes in an input's text.
IMAGE_TOKEN = "<|image|>"

_MODELS: dict[tuple[str, str], Any] = {}
_LOCK = threading.Lock()


def _load(model: str, device: str) -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    from .local import cache_dir

    with _LOCK:
        if (model, device) not in _MODELS:
            half = device.startswith("cuda") and torch.cuda.is_bf16_supported()
            try:
                _MODELS[(model, device)] = SentenceTransformer(
                    model, device=device, cache_folder=str(cache_dir()),
                    token=settings.hf_token(),
                    model_kwargs={"torch_dtype": torch.bfloat16} if half else {})
            except Exception as exc:                     # noqa: BLE001
                raise EmbedderUnavailable(
                    f"could not load {model!r}: {exc} -- it needs "
                    "sentence-transformers>=6.1 and transformers>=5.18") from None
        return _MODELS[(model, device)]


class LocalMultimodalEmbedder(Embedder, VisualEmbedder):
    """Fills the embedder role, the visual_embedder role, or both. `dims`
    shortens the vectors (768, 512, 256 or 128 for EmbeddingGemma 2); a
    search must use the same. `query_prompt` and `document_prompt` are the
    model's own prompt names."""

    concurrency = 1
    #: At the model's default image budget, 280 tokens each in an 8K context.
    max_images = 29

    def __init__(self, model: str = DEFAULT_MODEL, dims: Optional[int] = None,
                 device: Optional[str] = None, batch: int = 8,
                 query_prompt: str = "SearchQuery",
                 document_prompt: str = "Document") -> None:
        self.model_id = model or DEFAULT_MODEL
        self.device = device or default_device()
        self.batch = batch
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self._dims = int(dims) if dims else None

    def __repr__(self) -> str:
        return f"LocalMultimodalEmbedder({self.model_id!r})"

    def _model(self) -> Any:
        return _load(self.model_id, self.device)

    @property
    def dims(self) -> int:
        return self._dims or int(self._model().get_embedding_dimension())

    @property
    def key(self) -> str:
        return f"local:{self.model_id}:{self.dims}"

    def _encode(self, inputs: list[Any], prompt: Optional[str] = None) -> list[list[float]]:
        return self._model().encode(inputs, batch_size=self.batch, prompt_name=prompt,
                                    truncate_dim=self._dims, normalize_embeddings=True,
                                    convert_to_numpy=True).tolist()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(list(texts), self.document_prompt) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self.query_prompt)[0]

    def embed_images(self, groups: Sequence[Sequence[dict[str, Any]]]) -> list[list[float]]:
        import io

        from PIL import Image

        inputs = []
        for group in groups:
            words, pictures = [], []
            for part in group:
                if part.get("type") == "image":
                    words.append(IMAGE_TOKEN)
                    pictures.append(Image.open(io.BytesIO(part["data"])).convert("RGB"))
                elif part.get("type") == "text":
                    words.append(part["text"])
            if not pictures:
                raise EmbedderUnavailable("a group holds no image part")
            inputs.append({"text": " ".join(words), "image": pictures})
        return self._encode(inputs) if inputs else []


__all__ = ["DEFAULT_MODEL", "LocalMultimodalEmbedder"]
