"""video_rag -- a video in, a searchable index of moments out, and the search.

    media        1  what streams the file carries
    audio        2  the soundtrack, scanned whole
    boundaries 3+4  the grid, from the picture or the soundtrack
    video        5  which frames each sampler keeps
    cut          6  the transcript, onto the grid
    describe     7  one model answer per (chunk, sampler:question)
    embed        8  both modalities to vectors
    retrieve        a query to ranked moments

Every component takes the paths it reads and the path it writes:

    media.media(source, into, video_id=None, on_conflict="new")
    audio.audio(media, out, transcriber=..., ...)
    boundaries.evidence(out, policy, media=, raw_transcript=, ...)
    boundaries.boundaries(media, out, policy=, cuts=, ...)
    boundaries.retune(cuts, out, threshold)
    video.video(media, timeline, out, store=, sampler=, ...)
    cut.cut(timeline, raw_transcript, out)
    describe.describe(manifest, timeline, store, out, previous=, ...)
    embed.embed(out, descriptions=, transcript=, previous=, timeline=, ...)
    retrieve.search(query, video_id=, grids=, ...)

Each also has a verb over documents in hand (`split`, `listen`, `detect`,
`timeline`, `ingest`, `apply`, `answer`, `encode`). `video_rag(source, into,
...)` runs them all: `media` makes `<into>/<id>/` and `driver.layout()` names
every path inside it. `helpers/` holds the frame store both `video` and
`describe` use.
"""

from __future__ import annotations

from typing import Any

#: Resolved on first use, so importing the package does not import every
#: component.
_LAZY = {"video_rag": ("driver", "video_rag"),
         "process": ("driver", "process"),
         "Options": ("driver", "Options"),
         "Run": ("driver", "Run"),
         "validate": ("driver", "validate"),
         "layout": ("driver", "layout"),
         "search": ("retrieve", "search")}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib
        module, attribute = _LAZY[name]
        return getattr(importlib.import_module(f".{module}", __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = ["Options", "Run", "layout", "process", "search", "validate",
           "video_rag"]
