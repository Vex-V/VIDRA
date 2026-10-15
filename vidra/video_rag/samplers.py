"""The samplers: which frames to keep, as objects.

    from vidra.video_rag.samplers import Clip, Yolo
    video_rag("x.mp4", "data/out", sampler=[Clip(threshold=0.93), Yolo()])

The classes live in `core.sampling.samplers`; this is where a caller imports
them. Model-backed ones are imported on first use.
"""

from __future__ import annotations

from typing import Any

from .core.sampling import samplers as _samplers
from .core.sampling.samplers import Sampler, Uniform, View


def __getattr__(name: str) -> Any:
    if name in _samplers.__all__:
        return getattr(_samplers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_samplers.__all__)
