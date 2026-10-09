"""video_rag -- a video in, a searchable index of moments out, and the search.

    core/       what every pipeline uses: samplers, the question vocabulary,
                the frame store, search
    offline/    the batch pipeline: a whole file, every stage in order
    live/       the streaming pipeline: an answer per kept frame as it arrives

The batch stages, in order:

    media        1  what streams the file carries
    audio        2  the soundtrack, scanned whole
    boundaries 3+4  the grid, from the picture or the soundtrack
    video        5  which frames each sampler keeps
    cut          6  the transcript, onto the grid
    describe     7  one model answer per (chunk, sampler:question)
    glance      7b  kept frames to vectors, with no answer (beside or instead)
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
    glance.glance(manifest, timeline, store, out, previous=, visual_embedder=)
    embed.embed(out, descriptions=, transcript=, previous=, timeline=, ...)
    retrieve.search(query, video_id=, grids=, ...)

Each also has a verb over documents in hand (`split`, `listen`, `detect`,
`timeline`, `ingest`, `apply`, `answer`, `look`, `encode`). `video_rag(source, into,
...)` runs them all: `media` makes `<into>/<id>/` and `layout()` names every
path inside it.

Every name here is reached as `from vidra.video_rag import ...`, whichever
folder it lives in; a component is the module itself.
"""

from __future__ import annotations

from typing import Any, Optional

#: Resolved on first use. An attribute of None is the module itself.
_LAZY: dict[str, tuple[str, Optional[str]]] = {
    "video_rag": ("offline.driver", "video_rag"),
    "Run": ("offline.driver", "Run"),
    "validate": ("offline.driver", "validate"),
    "layout": ("offline.driver", "layout"),
    "search": ("core.retrieve", "search"),
    "search_observations": ("core.retrieve", "search_observations"),
    "video_rag_live": ("live.driver", "video_rag_live"),
    "live": ("live", None),
    **{name: (f"offline.{name}", None) for name in (
        "media", "audio", "boundaries", "video", "cut", "describe", "glance",
        "embed")},
    "retrieve": ("core.retrieve", None),
    # Build a configured sampler, register one of your own, list them.
    "samplers": ("core.sampling.samplers", None),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib
        module, attribute = _LAZY[name]
        found = importlib.import_module(f".{module}", __name__)
        return found if attribute is None else getattr(found, attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = ["Run", "layout", "search", "search_observations", "validate", "video_rag",
           "video_rag_live"]
