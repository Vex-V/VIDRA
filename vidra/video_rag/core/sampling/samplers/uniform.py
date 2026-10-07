"""Positional sampling: every Nth decimated frame.

The stride counts `chunk_local_index`, the frame's position in its chunk's
decimated stream, so the cadence in seconds follows from `per_second`
(`every_n=3` is one frame every 3 s at `per_second=1`). `min_interval_s` sets
a cadence in seconds instead. Default 1: every decimated frame.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..reader import Frame
from .base import Sampler
from vidra.shared.reporting.errors import Refused


class UniformSampler(Sampler):
    """Every ``every_n``th decimated frame."""

    name = "uniform"

    def __init__(self, every_n: int = 1, min_interval_s: float = 0.0,
                 max_per_chunk: Optional[int] = None,
                 sampler_id: Optional[str] = None,
                 prompts: Optional[Sequence[str]] = None) -> None:
        every_n = int(every_n)
        if every_n < 1:
            raise Refused("every_n must be >= 1; it is a frame stride")
        # The stride is applied in `propose`.
        super().__init__(min_interval_s, max_per_chunk, sampler_id, prompts)
        self.every_n = every_n

    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        # Index 0 is always kept.
        return chunk_local_index % self.every_n == 0

    def config(self) -> dict[str, Any]:
        return {**self._base_config(), "every_n": self.every_n}


__all__ = ["UniformSampler"]
