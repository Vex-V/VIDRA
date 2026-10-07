"""The live pipeline: a stream in, an answer per kept frame as soon as it is
sampled, and `offline`'s documents when the stream ends.

    driver     Options · validate · process · video_rag_live() · Observation
    source     a stream opened with PyAV, on a clock starting at its first frame
    observe    the sampling thread: decimate, chunk, samplers, context frames
    send       push a file to a port at real-time pace, for testing

Chunking is uniform only: a chunk is `int(media_ts // chunk_s)`, an id and a
point where the samplers reset, never a batch. There is no audio.
"""

from __future__ import annotations

from typing import Any

_LAZY = {name: ("driver", name) for name in (
    "LiveRun", "Observation", "Options", "StopStream", "process", "validate",
    "video_rag_live")}
_LAZY.update({"send": ("send", "send"), "StreamUnavailable": ("source", "StreamUnavailable")})


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib
        module, attribute = _LAZY[name]
        return getattr(importlib.import_module(f".{module}", __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = sorted(_LAZY)
