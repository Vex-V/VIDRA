"""Whole-frame appearance change, via CLIP cosine similarity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from ..reader import Frame
from .base import Sampler
from falconvar.shared.reporting.errors import Refused, UnknownOption

if TYPE_CHECKING:
    from .perception.embedders import FrameEmbedder


class ClipChangeSampler(Sampler):
    """Samples once the scene has changed enough to be worth describing.

    Each decimated frame is embedded with CLIP and kept when its cosine
    similarity falls below `threshold`. Two modes:

      `reference`   -- compare against the last kept frame (slow change
                       accumulates)
      `consecutive` -- compare against the previous frame evaluated (only sudden
                       change)

    The useful threshold depends on the footage; 0.96 is the default.
    """

    name = "clip"

    def __init__(
        self,
        embedder: Optional["FrameEmbedder"] = None,
        threshold: float = 0.96,
        mode: str = "reference",
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(min_interval_s, max_per_chunk, sampler_id, prompts)
        if mode not in ("reference", "consecutive"):
            raise UnknownOption("mode must be 'reference' or 'consecutive'")
        if not 0.0 <= threshold <= 1.0:
            raise Refused("threshold must be a cosine similarity in [0, 1]")
        if embedder is None:
            from .perception.embedders import CLIPEmbedder

            embedder = CLIPEmbedder()
        self.embedder = embedder
        self.threshold = threshold
        self.mode = mode
        self._reference: Optional[np.ndarray] = None
        self._last_score: Optional[float] = None

    def on_reset(self, chunk_id: int) -> None:
        # The reference resets at every chunk.
        self._reference = None
        self._last_score = None

    def last_score(self) -> Optional[float]:
        return self._last_score

    def describe(self, frame: Frame) -> np.ndarray:
        if frame.image is None:
            raise Refused("ClipChangeSampler needs pixels; frame.image is None")
        return self.embedder.embed_one(frame.image)

    def compare(self, current: np.ndarray, reference: np.ndarray) -> float:
        # Embeddings are L2-normalised, so the dot product is the cosine.
        return float(np.dot(current, reference))

    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        embedding = self.describe(frame)

        if self._reference is None:
            self._last_score = None
            self._reference = embedding
            return True

        similarity = self.compare(embedding, self._reference)
        self._last_score = similarity

        if self.mode == "consecutive":
            # The reference is "the previous frame", so it moves every time
            # regardless of what is decided below.
            self._reference = embedding

        keep = similarity < self.threshold
        if keep and self.mode == "reference":
            self._reference = embedding
        return keep

    def config(self) -> dict:
        return {
            **self._base_config(),
            "threshold": self.threshold,
            "mode": self.mode,
            "embedder": self.embedder.config(),
        }
