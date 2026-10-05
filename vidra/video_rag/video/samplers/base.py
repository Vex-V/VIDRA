"""The sampler contract.

A sampler sees the decimated stream one frame at a time and answers yes or
no; it cannot look ahead or revisit a frame. The base class enforces:

    min_interval_s  smallest gap between two kept frames
    max_per_chunk   ceiling on frames kept from one chunk
    prompts         which questions describe should ask about these frames

Rate limits apply before `propose` runs, and every chunk keeps at least one
frame. Prompts are opaque strings here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from ..reader import Frame
from vidra.shared.reporting.errors import Refused


class Sampler(ABC):
    """Base class for frame-selection strategies."""

    name: str = "base"

    def __init__(self, min_interval_s: float = 0.0,
                 max_per_chunk: Optional[int] = None,
                 sampler_id: Optional[str] = None,
                 prompts: Optional[Sequence[str]] = None) -> None:
        if min_interval_s < 0:
            raise Refused("min_interval_s must be >= 0")
        if max_per_chunk is not None and max_per_chunk < 1:
            raise Refused("max_per_chunk must be >= 1; every chunk keeps a frame")
        self.min_interval_s = min_interval_s
        self.max_per_chunk = max_per_chunk
        self._sampler_id = sampler_id
        self.prompts: list[str] = list(prompts or [])
        self._kept_in_chunk = 0
        self._last_kept_ts: Optional[float] = None

    @property
    def sampler_id(self) -> str:
        """The manifest key for this run of this sampler: the strategy's name."""
        return self._sampler_id or self.name

    def reset(self, chunk_id: int) -> None:
        """Called once when a chunk opens, before any frame is offered: samplers
        forget everything at a boundary.
        """
        self._kept_in_chunk = 0
        self._last_kept_ts = None
        self.on_reset(chunk_id)

    def on_reset(self, chunk_id: int) -> None:
        """Subclass hook for clearing strategy state at a boundary."""

    def accepts(self, frame: Frame, chunk_local_index: int) -> bool:
        """Final decision for one frame. Do not override -- implement propose."""
        if self.max_per_chunk is not None and self._kept_in_chunk >= self.max_per_chunk:
            # Chunk is full.
            return False

        first_of_chunk = self._last_kept_ts is None
        if (not first_of_chunk
                and frame.media_ts - self._last_kept_ts < self.min_interval_s):
            return False

        # The strategy sees every allowed frame; the first frame of a chunk is kept
        # regardless.
        keep = self.propose(frame, chunk_local_index) or first_of_chunk
        if keep:
            self._kept_in_chunk += 1
            self._last_kept_ts = frame.media_ts
        return keep

    @abstractmethod
    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        """The strategy's opinion, before rate limits and the chunk guarantee.
        `chunk_local_index` counts decimated frames in the current chunk from 0.
        """

    # The two halves of a change sampler: describe a frame, compare two.

    def describe(self, frame: Frame) -> Any:
        """The model's output for this frame. None for positional samplers."""
        return None

    def compare(self, current: Any, reference: Any) -> Optional[float]:
        """Similarity in [0, 1]. Lower means more changed."""
        return None

    def last_score(self) -> Optional[float]:
        """What the last decision was based on, recorded in the manifest."""
        return None

    def _base_config(self) -> dict[str, Any]:
        # `prompts` is omitted when empty.
        config: dict[str, Any] = {
            "id": self.sampler_id,
            "name": self.name,
            "min_interval_s": self.min_interval_s,
            "max_per_chunk": self.max_per_chunk,
        }
        if self.prompts:
            config["prompts"] = list(self.prompts)
        return config

    def questions(self) -> list[str]:
        """The questions to ask about this run's frames; empty means one named after
        the strategy.
        """
        return list(self.prompts) or [self.name]

    def config(self) -> dict[str, Any]:
        """Serialised into the manifest so a run can be reproduced."""
        return self._base_config()


__all__ = ["Sampler"]
