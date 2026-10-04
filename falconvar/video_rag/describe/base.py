"""The Describer protocol, and a registry.

A describer takes frames and a context and returns a summary plus the
structured fields its question's shape holds: the `stub`, which loads
nothing, or `ModelDescriber` for any provider. Resolved lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

from .frames import LoadedFrame
from falconvar.shared.reporting.errors import Unavailable


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


#: Describers that are not a model provider: the stub.
_REGISTRY: dict[str, Any] = {}


def register(cls) -> Any:
    _REGISTRY[cls.name] = cls
    return cls


def build(name: Optional[str] = None, **kwargs) -> Describer:
    """A provider, `provider/model`, `stub`, or None for the default."""
    from falconvar.shared.models import providers

    chosen, _ = providers.choose("describe", name)
    if chosen == providers.OFFLINE["describe"]:
        from .backends import stub  # noqa: F401  -- self-registers
    if chosen in _REGISTRY:
        return _REGISTRY[chosen]()
    from .backends.model import ModelDescriber
    return ModelDescriber(name, **kwargs)


def available() -> list[str]:
    from falconvar.shared.models import providers
    return sorted(set(_REGISTRY) | set(providers.names("describe")))


__all__ = ["Describer", "DescriberUnavailable", "Description", "available",
           "build", "register"]
