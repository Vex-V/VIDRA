"""The Describer protocol, and building one from a VLM.

A describer takes frames and a context and returns a summary plus the
structured fields its question's shape holds. `ModelDescriber` is the one
there is: it asks whichever `VLM` it is handed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .frames import LoadedFrame
from vidra.shared.reporting.errors import Unavailable


class DescriberUnavailable(Unavailable):
    """No client, no key, or a model this account cannot reach."""


@dataclass
class Description:
    """One answer about one (chunk, sampler)."""

    summary: str
    fields: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class Describer(Protocol):
    #: How many (chunk, sampler) runs are described at once.
    concurrency: int

    async def describe(self, images: Sequence[LoadedFrame],
                       context: dict[str, Any]) -> Description: ...
    def config(self) -> dict[str, Any]: ...


def build(vlm: Any = None, **kwargs) -> Describer:
    """A describer asking `vlm` (None is the default VLM). Checked here, before
    any frame is read: the right kind of model, a key, an API key."""
    from vidra.shared.models.base import calls
    from vidra.shared.models.roles import resolve
    from .backends.model import ModelDescriber
    return ModelDescriber(calls("vlm", resolve("vlm", vlm)), **kwargs)


__all__ = ["Describer", "DescriberUnavailable", "Description", "build"]
