"""Image vectors from a CLIP-style model loaded into this process, with
`transformers`: an image tower for frames and a text tower for queries, in
one space.

    LocalVisualEmbedder()                                  openai/clip-vit-base-patch32
    LocalVisualEmbedder("google/siglip2-base-patch16-224")

The model sees one image at a time, so a group of images (a chunk's frames and
their views) is the mean of their unit vectors, normalised; the key says so.
A query is cut at the text tower's length (77 tokens for CLIP). Weights land
in `weights/embedders/`; a model is loaded on first use, once per (model,
device) per process.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Sequence

from ...config import settings
from ..base import VisualEmbedder
from ..devices import default_device
from . import EmbedderUnavailable

#: The model when none is named.
DEFAULT_MODEL = "openai/clip-vit-base-patch32"

_MODELS: dict[tuple[str, str], tuple[Any, Any]] = {}
_LOCK = threading.Lock()


def _load(model: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoProcessor

    from .local import cache_dir

    with _LOCK:
        if (model, device) not in _MODELS:
            try:
                where = {"cache_dir": str(cache_dir()), "token": settings.hf_token()}
                loaded = AutoModel.from_pretrained(model, **where).to(device).eval()
                _MODELS[(model, device)] = (loaded, AutoProcessor.from_pretrained(model, **where))
            except Exception as exc:                     # noqa: BLE001
                raise EmbedderUnavailable(f"could not load {model!r}: {exc}") from None
        return _MODELS[(model, device)]


def _features(output: Any) -> Any:
    """The projected embedding, whether a model returns it bare or as
    `pooler_output`."""
    return output if hasattr(output, "shape") else output.pooler_output


def _unit(matrix: Any) -> Any:
    import numpy as np
    return matrix / np.clip(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12, None)


class LocalVisualEmbedder(VisualEmbedder):
    """`device` defaults to cuda, then mps, then the CPU. `batch` is images per
    forward pass."""

    concurrency = 1

    def __init__(self, model: str = DEFAULT_MODEL, device: Optional[str] = None,
                 batch: int = 32) -> None:
        self.model_id = model or DEFAULT_MODEL
        self.device = device or default_device()
        self.batch = batch

    def __repr__(self) -> str:
        return f"LocalVisualEmbedder({self.model_id!r})"

    def _model(self) -> tuple[Any, Any]:
        return _load(self.model_id, self.device)

    @property
    def dims(self) -> int:
        config = self._model()[0].config
        return int(getattr(config, "projection_dim", None)
                   or config.vision_config.hidden_size)

    @property
    def key(self) -> str:
        return f"local:{self.model_id}:{self.dims}:mean"

    def _images(self, jpegs: Sequence[bytes]) -> Any:
        import cv2
        import numpy as np
        import torch

        model, processor = self._model()
        out = []
        for start in range(0, len(jpegs), self.batch):
            pictures = []
            for data in jpegs[start:start + self.batch]:
                decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    raise EmbedderUnavailable("an image part is not an image OpenCV reads")
                pictures.append(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                pixels = processor(images=pictures, return_tensors="pt")["pixel_values"]
                features = _features(model.get_image_features(
                    pixel_values=pixels.to(self.device)))
            out.append(features.float().cpu().numpy())
        return _unit(np.concatenate(out, axis=0))

    def embed_images(self, groups: Sequence[Sequence[dict[str, Any]]]) -> list[list[float]]:
        jpegs = [[p["data"] for p in group if p.get("type") == "image"] for group in groups]
        if not all(jpegs):
            raise EmbedderUnavailable("a group holds no image part")
        if not jpegs:
            return []
        flat = self._images([j for group in jpegs for j in group])
        vectors, at = [], 0
        for group in jpegs:
            # Text parts are labels; this model reads only the images.
            vectors.append(_unit(flat[at:at + len(group)].mean(axis=0)).tolist())
            at += len(group)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        import torch

        model, processor = self._model()
        siglip = "siglip" in str(model.config.model_type)
        # SigLIP was trained on text padded to its full length, unmasked.
        length = int(getattr(model.config.text_config, "max_position_embeddings", 64))
        with torch.no_grad():
            tokens = processor(text=[text], return_tensors="pt", truncation=True,
                               max_length=length,
                               padding="max_length" if siglip else True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()
                      if k in ("input_ids", "attention_mask")}
            if siglip:
                tokens.pop("attention_mask", None)
            features = _features(model.get_text_features(**tokens))
        return _unit(features.float().cpu().numpy())[0].tolist()


__all__ = ["DEFAULT_MODEL", "LocalVisualEmbedder"]
