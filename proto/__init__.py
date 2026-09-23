"""The components, addressed by filepath instead of by video id.

A copy of `falconvar/video_rag/`'s eight component packages with their drivers
rewritten. Everything below a driver -- `split`, `scenes`, `speech`, `grid`,
`pipeline`, `cutter`, `reader`, `units`, the samplers, the backends -- is
untouched, because none of it ever knew about a video id. The drivers were
the only place that did.

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

**Naming stays in the library; resolving does not.** `_io.WRITTEN_BY` is
derived from `shared.paths.ARTIFACTS` and `PRODUCED_BY` rather than restated,
so a pipeline composing `home / ARTIFACTS["raw_transcript"]` gets
`transcript.raw.json` -- and a missing input still names the component that
writes it, because the producer is a fact about the type that was asked for.

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

**Three oracles, all of them runnable:**

    python -m proto.check      every document byte-compared against
                               `falconvar`, which is untouched and so is the
                               control -- component by component, and then the
                               whole pipeline in two shapes
    python -m proto.isolated   the same chain with `paths.artifact`, `home`,
                               `out_root`, `require`, `exists` and `videos`
                               replaced by functions that raise, into an empty
                               data root that must stay empty
    python -m proto.branches   what the pipeline does when a run does not need
                               every component: a missing stream, a policy
                               derived from it, a contradictory request, and a
                               second file wanting a taken folder
"""

from __future__ import annotations

from . import audio, boundaries, cut, describe, embed, media, retrieve, video
from .driver import Options, Run, layout, process, video_rag

__all__ = ["Options", "Run", "audio", "boundaries", "cut", "describe", "embed",
           "layout", "media", "process", "retrieve", "video", "video_rag"]
