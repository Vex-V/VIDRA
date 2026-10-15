"""Sampling strategies: which frames to keep, as objects you build and pass in.

    from vidra.video_rag.samplers import Clip, Yolo
    video_rag("x.mp4", "data/out", sampler=[Clip(threshold=0.93), Yolo()])

Each class is one strategy. Its settings are its constructor's arguments, plus
the base's `min_interval_s`, `max_per_chunk` and `prompts` (the questions asked
about the frames it keeps; by default the one named after the strategy).
Model-backed samplers are imported on first use, so `import vidra` stays light.

A sampler of your own subclasses `Sampler`, gives itself a `name` and fills in
`propose`; pass an object of it like any other.
"""

from __future__ import annotations

import importlib
from typing import Type

from .base import Sampler, View
from .uniform import Uniform

#: The shipped samplers: name -> (module, class), imported on first use.
_BUILTIN: dict[str, tuple[str, str]] = {
    "uniform": ("uniform", "Uniform"),
    "clip": ("scene", "Clip"),
    "yolo": ("people", "Yolo"),
    "objects": ("objects", "Objects"),
    "text": ("ocr", "Text"),
}
_CLASSES = {cls: module for module, cls in _BUILTIN.values()}

#: The shipped sampler names. A class of your own may not take one.
BUILTIN = frozenset(_BUILTIN)


def builtin(name: str) -> Type[Sampler]:
    """The shipped class behind a name, imported if it is not loaded yet."""
    module, cls = _BUILTIN[name]
    return getattr(importlib.import_module(f".{module}", __name__), cls)


def __getattr__(name: str):
    if name in _CLASSES:
        return getattr(importlib.import_module(f".{_CLASSES[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Clip", "Objects", "Sampler", "Text", "Uniform", "View", "Yolo"]
