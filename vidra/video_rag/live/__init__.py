"""The live pipeline: a stream in, an answer per kept frame as soon as it is
sampled, and `offline`'s documents when the stream ends.

    driver     video_rag_live() · validate (the same arguments) · Observation
    source     a stream opened with PyAV, on a clock starting at its first frame;
               `frames`: that stream's frames, for code of your own
    observe    the sampling thread: decimate, chunk, samplers, context frames
    sender     `send`: push a file to a port at real-time pace, for testing

Chunking is uniform only: a chunk is `int(media_ts // chunk_s)`, an id and a
point where the samplers reset, never a batch. There is no audio.
"""

from __future__ import annotations

from typing import Any

_LAZY = {name: ("driver", name) for name in (
    "LiveRun", "Observation", "StopStream", "validate", "video_rag_live")}
_LAZY.update({"send": ("sender", "send"), "StreamUnavailable": ("source", "StreamUnavailable"),
              "frames": ("source", "frames"), "Arrival": ("source", "Arrival")})


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib
        module, attribute = _LAZY[name]
        return getattr(importlib.import_module(f".{module}", __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = sorted(_LAZY)
