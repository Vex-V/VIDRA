"""video_rag -- a video in, a searchable index of moments out, and the search.

The extraction half of FalCONvar, and a complete RAG engine on its own:

    media        1  what streams the file carries
    audio        2  the soundtrack, scanned whole
    boundaries 3+4  THE GRID, from the picture or the soundtrack
    video        5  which frames each sampler keeps
    cut          6  the transcript, onto the grid
    describe     7  one model answer per (chunk, sampler:question)
    embed        8  both modalities to vectors, keyed by a hash of the text
    retrieve        a query to ranked moments over what embed built

**Every component is addressed by filepath.** It takes the paths it reads and
the path it writes, and resolves nothing. Everything below a driver --
`split`, `scenes`, `speech`, `grid`, `pipeline`, `cutter`, `reader`, `units`,
the samplers, the backends -- never knew about a video id in the first place;
the drivers were the only place that did.

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

and one pipeline over all of them, which is the only thing here that decides
a path:

    video_rag(source, into, policy=, sampler=, use_audio=, ...)

Each component takes the paths it reads, the path it writes, and the same
tuning arguments as before. Each returns the same `Produced`, and `artifacts` still
carries where the output went.

**`media` is the only one that creates a folder, and that asymmetry is the
design.** It takes the video and a parent directory, decides the id, and makes
`<into>/<id>/`. That is what keeps `on_conflict` working: guarding against two
different `clip.mp4` files claiming one directory needs something that creates
directories, and a component handed an output path has already been told where
to write. `Produced.stats["home"]` is the folder, and `driver.layout()` composes every
later path inside it.

**Naming stays in the library; resolving does not.** Every component reads
and writes through `shared.storage.files`, whose `WRITTEN_BY` is derived from
`shared.paths.ARTIFACTS` and `PRODUCED_BY` rather than restated -- so a
pipeline composing `home / ARTIFACTS["raw_transcript"]` gets
`transcript.raw.json`, and a missing input still names the component that
writes it, because the producer is a fact about the type that was asked for.

`helpers/` is what neither component may own and no tier may keep: pixels are
not a document, so `FrameStore` sits there rather than inside `video`, which
writes them, or `describe`, which reads them.

**`run` is gone.** Each module's entry point is named for its component, which
is what the function was always called -- `run` was an alias beside it. The
object verbs stay: `split`, `listen`, `detect`, `timeline`, `ingest`, `apply`,
`answer`, `encode` take documents and return documents, and the filepath
function is that verb with a read at each end. It has to be, because the
filepath layer is built *on* the verb rather than instead of it.

Two things that were hidden are now arguments, and both read better for it:
`describe(previous=)` replaces `resume=True` silently reading its own last
output, and `embed`'s resume diff moved down into `encode(previous=)` so both
ways in get it.

`retrieve` has no filepath input at all -- it takes a query and reads a
database. Its one layout dependency is gone: `spans_of` took the grid from a
`timeline.json` it resolved under a data root, and now takes one the caller
names through `grids=`, or the database.

**The pipeline is a caller like any other.** `driver.py` runs the eight
components in dependency order, and everything it does a user could do by
hand -- it composes paths from `layout()`, decides which components this
file and this policy need, and hands `previous=` to the two that can resume.
That is the point of the shape: the pipeline is the convenient way, not the
only way, and the components do not know it exists.

Nothing here touches `shared.paths` to resolve anything -- `ARTIFACTS` and
`PRODUCED_BY` are read as *data*, which is a different thing from asking
where a video lives.

**What the move from `proto/` was verified against.** Every document, byte
for byte, against the id-addressed tier that this replaced -- component by
component and through the pipeline in two shapes, 299/299 frames identical,
with `paths.artifact`, `home`, `out_root`, `require`, `exists` and `videos`
replaced by functions that raise and an empty data root that stayed empty.
Those three oracles were `check.py`, `isolated.py` and `branches.py`, and
they are in the commit before the move rather than in the package.
"""

from __future__ import annotations

from typing import Any

#: Lazily, so `import falconvar.video_rag` stays two modules and 1.3 ms.
#: Binding `process` here eagerly would import every component -- 53 modules,
#: 352 ms and `av` -- for anyone who only wanted one of them. PEP 562 makes
#: `from falconvar.video_rag import video_rag` resolve through this instead.
#:
#: The components are not in here and do not need to be: `from
#: falconvar.video_rag import media` imports the submodule itself, which is
#: the spelling every caller uses anyway.
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
