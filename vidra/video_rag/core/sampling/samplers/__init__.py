"""Sampling strategies, by name, and how to add one of your own.

A name is a strategy: which frames to keep. What is asked about them is a
question, paired as `name:question`. Model-backed samplers are imported only
when built.

    samplers.build("clip", threshold=0.93, prompts=["safety"])   # a configured object
    samplers.register(Brightness)          # a class of your own, usable by name
    samplers.available()                   # every name a spec may use

A sampler of your own subclasses `Sampler` and fills in `propose`; pass an
object of it in a spec (`sampler=[Brightness(step=12, prompts=["overview"])]`),
or register the class to name it in a spec string.
"""

from __future__ import annotations

import inspect
import re
from typing import Type

from .base import Sampler
from .uniform import UniformSampler
from vidra.shared.config.lookup import imported
from vidra.shared.reporting.errors import Refused, UnknownOption

#: name -> "module:ClassName", imported on first use.
_LAZY: dict[str, str] = {
    "clip": "scene:ClipChangeSampler",
    "yolo": "people:PersonChangeSampler",
    "objects": "objects:ObjectChangeSampler",
    "text": "ocr:TextChangeSampler",
}
_REGISTRY: dict[str, Type[Sampler]] = {"uniform": UniformSampler}

#: The shipped samplers. Their names cannot be taken by a registered class.
BUILTIN = frozenset({"uniform", *_LAZY})

#: A sampler name: what a spec string can carry (`name:question` splits on the
#: colon, `,` `+` `[` `]` are the spec's own syntax).
NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


def register(cls: Type[Sampler]) -> Type[Sampler]:
    """Make a sampler class usable by name in a spec string. Usable as a class
    decorator. Registering a name again replaces the class (a notebook cell run
    twice); a built-in name is refused."""
    if not (inspect.isclass(cls) and issubclass(cls, Sampler)):
        raise Refused(f"register takes a Sampler subclass, not {cls!r}")
    if inspect.isabstract(cls):
        raise Refused(f"{cls.__name__} does not implement propose(frame, "
                      f"chunk_local_index)")
    name = getattr(cls, "name", "")
    if not isinstance(name, str) or not NAME.match(name):
        raise Refused(f"{cls.__name__}.name must be lowercase letters, digits, "
                      f"'_' or '-', starting with a letter, not {name!r}: it is "
                      f"written in spec strings as name:question")
    if name in BUILTIN:
        raise Refused(f"{name!r} is a built-in sampler and cannot be replaced; "
                      f"give {cls.__name__} a name of its own")
    _REGISTRY[name] = cls
    return cls


def class_of(name: str) -> Type[Sampler]:
    """The class a name builds, imported if it is a built-in not yet loaded."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name not in _LAZY:
        raise UnknownOption(f"unknown sampler {name!r}; known: {', '.join(available())}")
    _REGISTRY[name] = imported(_LAZY[name], __package__)
    return _REGISTRY[name]


def build(name: str, **settings) -> Sampler:
    """A sampler object, by name, with its settings: what a spec string builds,
    configured. Pass the object in a spec to use these settings in a run."""
    return class_of(name)(**settings)


def available() -> list[str]:
    """Every name a spec may use: the built-ins and every registered class."""
    return sorted(set(_REGISTRY) | set(_LAZY))


__all__ = ["BUILTIN", "Sampler", "available", "build", "class_of", "register"]
