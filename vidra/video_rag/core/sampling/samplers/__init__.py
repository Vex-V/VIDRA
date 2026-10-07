"""Sampling strategies, by name.

A name is a strategy: which frames to keep. What is asked about them is a
question, paired as `name:question`. Model-backed samplers are imported only
when built.
"""

from __future__ import annotations

import importlib
from typing import Type

from .base import Sampler
from .uniform import UniformSampler
from vidra.shared.reporting.errors import UnknownOption

#: name -> "module:ClassName", imported on first use.
_LAZY: dict[str, str] = {
    "clip": "scene:ClipChangeSampler",
    "yolo": "people:PersonChangeSampler",
    "objects": "objects:ObjectChangeSampler",
    "text": "ocr:TextChangeSampler",
}
_REGISTRY: dict[str, Type[Sampler]] = {}


def register(cls: Type[Sampler]) -> Type[Sampler]:
    _REGISTRY[cls.name] = cls
    return cls


register(UniformSampler)


def _resolve(name: str) -> Type[Sampler]:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name not in _LAZY:
        raise UnknownOption(f"unknown sampler {name!r}; known: {', '.join(available())}")
    module_name, class_name = _LAZY[name].split(":")
    module = importlib.import_module(f".{module_name}", __package__)
    return register(getattr(module, class_name))


def build(name: str, **kwargs) -> Sampler:
    """Instantiate a registered sampler by name."""
    return _resolve(name)(**kwargs)


def available() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))


__all__ = ["Sampler", "UniformSampler", "available", "build", "register"]
