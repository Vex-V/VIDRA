"""Who answers each kind of model call, chosen once and handed to every call.

    models = Models(describer="openai", embedder="local", llm="anthropic",
                    keys={"openai": vault.get("oa")})

    video_rag("x.mp4", "data/out", models=models)
    aggregates.aggregate(out=out, models=models, summary=video.excerpt(transcript=True))
    search("the reactor", "x", models=models)

Three roles: the describer (frames -> answers), the embedder (text -> vectors,
the same wherever an index is built and read) and the llm (text -> text). A
role left None resolves when the call is made; a named role is checked at
construction. `keys` is `{provider: key}`, outranks the environment and is
never printed or written. The pipelines hand the keys down for the length of
their call; `with models:` does the same around component calls.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional

from ..reporting.errors import Refused

#: Field -> the role `providers` knows it by.
ROLES = {"describer": "describe", "embedder": "embed", "llm": "llm"}


@dataclass(frozen=True)
class Models:
    """A provider or `provider/model` per role; `None` is the default."""

    #: Frames -> answers. A vision model.
    describer: Optional[str] = None
    #: Text -> vectors. Must be the same wherever an index is built and read.
    embedder: Optional[str] = None
    #: Text -> text: the llm aggregates and the entity accounts.
    llm: Optional[str] = None
    #: `{provider: key}`, outranking the environment. Never printed or compared.
    keys: Optional[Mapping[str, str]] = field(default=None, repr=False,
                                              compare=False, hash=False)

    def __post_init__(self) -> None:
        from . import providers
        if self.keys is not None:
            if not isinstance(self.keys, Mapping):
                raise Refused("keys must be a {provider: key} mapping")
            known = providers.providers()
            for name, key in self.keys.items():
                if name not in known:
                    # An unknown provider name is refused.
                    raise Refused(f"keys names an unknown provider {name!r}; "
                                  f"known: {', '.join(sorted(known))}")
                if not isinstance(key, str) or not key.strip():
                    raise Refused(f"the key for {name} must be a non-empty string")
            # A copy, so a caller's later changes do not apply.
            object.__setattr__(self, "keys", dict(self.keys))
        with providers.keys(self.keys):
            for name, role in ROLES.items():
                spec = getattr(self, name)
                if spec is None:
                    continue
                if not isinstance(spec, str) or not spec.strip():
                    raise Refused(f"{name} must be a provider or provider/model, "
                                  f"not {spec!r}")
                providers.choose(role, spec)      # unknown, or cannot serve it
                providers.require(role, spec)     # no key, here or in the environment

    def __enter__(self) -> "Models":
        """Hand this value's keys to every model call inside the block."""
        from . import providers
        self.__dict__.setdefault("_entered", []).append(providers.use_keys(self.keys))
        return self

    def __exit__(self, *exc: Any) -> None:
        from . import providers
        providers.release(self.__dict__["_entered"].pop())

    def resolved(self) -> dict[str, str]:
        """What each role resolves to right now, as `provider/model`."""
        from . import providers
        out = {}
        for name, role in ROLES.items():
            chosen, model = providers.choose(role, getattr(self, name))
            out[name] = f"{chosen}/{model}" if model else chosen
        return out


def unpack(models: Optional[Models], **given: Optional[str]) -> dict[str, Optional[str]]:
    """The role strings a call should use, from `models` and its own keywords. A
    role set in both places is refused.
    """
    if models is None:
        return dict(given)
    if not isinstance(models, Models):
        raise Refused(f"models must be a Models, not {type(models).__name__}")
    out: dict[str, Any] = {}
    for name, value in given.items():
        held = getattr(models, name)
        if value is not None and held is not None:
            raise Refused(f"{name} is set twice, as {value!r} and on models as "
                          f"{held!r}; pass it one way")
        out[name] = value if value is not None else held
    return out


@contextmanager
def keys_of(models: Optional[Models]) -> Iterator[None]:
    """A pipeline's whole call, with `models`' keys in reach. No-op for None."""
    from . import providers
    with providers.keys(models.keys if isinstance(models, Models) else None):
        yield


__all__ = ["Models", "ROLES", "keys_of", "unpack"]
