"""The retrieve component: a query -> ranked moments. Writes nothing.

The query is embedded with the embedder that built the index; every row
carries its embedder key and only rows in that space are searched. Ranking is
the database's (`Database.search`; for Supabase the `vr_search` RPC). With a
visual embedder too, the frames' space is searched apart and the two rankings
are fused by rank, never by similarity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from vidra.shared.config.paths import MissingArtifact
from vidra.shared.contracts.documents import Timeline
from vidra.shared.models import embedders as embedders_mod
from vidra.shared.models.base import require
from vidra.shared.models.roles import Models, given, resolve
from vidra.shared.reporting.errors import Refused
from vidra.shared.storage.database import Database, as_database
from vidra.shared.storage.files import read
from .moments import Moment, to_moments


def scope_of(video_id: Optional[str | Sequence[str]] = None,
             video_ids: Optional[Sequence[str]] = None) -> Optional[list[str]]:
    """The set of videos to search, from a string or a sequence. None means every
    one.
    """
    if video_ids is not None:
        chosen = [v for v in video_ids if v]
        return chosen or None
    if isinstance(video_id, str) and video_id:
        return [video_id]
    if isinstance(video_id, (list, tuple, set)):
        chosen = [v for v in video_id if v]
        return chosen or None
    return None


def spans_of(video_id: str,
             timeline: Optional[str | Path] = None,
             database: Optional[str | Database] = None) -> list[tuple[float, float]]:
    """The grid: from a `timeline.json` if one is named, else from the database."""
    if timeline is not None and Path(timeline).exists():
        return read(timeline, Timeline).spans
    return as_database(database or "supabase").spans(video_id)


def chunks_in(spans: Sequence[tuple[float, float]],
              after: Optional[float], before: Optional[float]) -> list[int]:
    """Which chunks overlap a time window."""
    lo = float("-inf") if after is None else after
    hi = float("inf") if before is None else before
    return [i for i, (start, end) in enumerate(spans) if end > lo and start < hi]


def search(query: str, video_id: Optional[str | Sequence[str]] = None,
           moments: int = 5,
           sampler: Optional[str] = None,
           candidates: int = 20,
           question: Optional[str] = None,
           strategy: Optional[str] = None,
           chunk_ids: Optional[Sequence[int]] = None,
           window: int = 0,
           after: Optional[float] = None,
           before: Optional[float] = None,
           structured: Optional[dict[str, Any]] = None,
           video_ids: Optional[Sequence[str]] = None,
           grids: Optional[Mapping[str, str | Path]] = None,
           database: Optional[str | Database] = None,
           models: Optional[Models] = None
           ) -> tuple[list[Moment], list[str]]:
    """Ranked moments, and notes about the ranking.

        # every video in the index
        moments, notes = retrieve.search("the audience laughed")

        # one video
        moments, notes = retrieve.search("the audience laughed", video_id="talk")

        # a window inside one, in media seconds
        moments, notes = retrieve.search("the key insight", video_id="talk",
                                         after=120.0, before=300.0)

        # by question, wherever it was asked
        moments, notes = retrieve.search("the diagram", video_id="talk",
                                         question="text")

        # several named videos
        moments, notes = retrieve.search("the budget",
                                         video_ids=["q1", "q2"])

    Read `notes` when `moments` is empty: they say why. A `Moment.score` is a rank
    fusion, not a similarity. `database` is a built `Database` or a name
    (`supabase` by default). `models.embedder` must be the one the index was
    built with; None is OpenAI's default.

    `models.visual_embedder` also searches the frames `glance` embedded, in
    their own ranking; a chunk both rankings find gets the second-account
    bonus. Given alone, only the frames are searched -- the default text
    embedder is used only when no visual embedder is named.
    """
    if not (query or "").strip():
        raise Refused("a search needs a query; this one is empty")
    for name, value, least in (("moments", moments, 1), ("candidates", candidates, 1),
                               ("window", window, 0)):
        if value < least:
            raise Refused(f"{name} must be at least {least}, not {value}")
    if after is not None and before is not None and before <= after:
        raise Refused(f"before ({before}) must be later than after ({after})")

    scope = scope_of(video_id, video_ids)
    chosen = given(models)
    visual = chosen.visual_embedder
    built = (resolve("embedder", chosen.embedder)
             if chosen.embedder is not None or visual is None else None)
    spaces: list[tuple[str, Any, Any]] = []
    if built is not None:
        require("embedder", built)
        spaces.append((built.key, built, embedders_mod.query_vector))
    if visual is not None:
        require("visual_embedder", visual)
        from ...offline.glance.driver import space_of
        spaces.append((space_of(visual), visual, lambda model, q: model.embed_query(q)))
    target = as_database(database or "supabase")
    grid = _Grids(target, grids or {})
    notes: list[str] = []

    narrowed = _chunks_wanted(scope, chunk_ids, window, after, before, grid, target)
    if narrowed is not None and not narrowed:
        return [], ["no chunk matches that window"]

    # Some models embed a query differently from a passage.
    ranked = []
    for key, model, embed_query in spaces:
        ranked.append((key, target.search(embed_query(model, query), query, key,
                                          candidates, scope, sampler, question,
                                          strategy, narrowed, structured)))
    keys = " or ".join(key for key, _ in ranked)
    if not any(found for _, found in ranked):
        if any(f is not None for f in (sampler, question, strategy,
                                       narrowed, structured, scope)):
            # Nothing matched the filters, as opposed to nothing indexed in this space.
            return [], [f"nothing matched those filters in {keys} -- "
                        "if that embedder never indexed these videos, "
                        "embed them with it first"]
        raise MissingArtifact(
            f"any video: nothing indexed for {keys}. Run embed (or glance) with "
            f"this embedder, and a pipeline naming a database, first -- a "
            f"different embedder writes different rows.")
    if len(ranked) == 1:
        hits = ranked[0][1]
    else:
        # Each space's own ranking, re-scored by position: scores from two
        # spaces are not on one scale.
        hits = [{**hit, "score": 1.0 / (60 + rank), "space": key}
                for key, found in ranked for rank, hit in enumerate(found, start=1)]
        for key, found in ranked:
            if not found:
                notes.append(f"nothing indexed for {key} in this scope; only "
                             f"the other space ranked")
    if any(f is not None for f in (sampler, question, strategy, structured)):
        notes.append("filtering gives up the agreement signal: a chunk "
                     "contributes fewer terms, so scores fall -- to a single "
                     "1/(k+rank) when only one unit per chunk survives")
    if scope is None or len(scope) > 1:
        # Moments from several videos share one ranking.
        notes.append("scope is several videos: moments are keyed by "
                     "(video_id, chunk_id), since a chunk id only means "
                     "something inside one grid")
    return to_moments(hits, grid, moments, scope[0] if scope and len(scope) == 1 else ""), notes


class _Grids:
    """Each video's grid, read once and only when something needs it: a time
    window, or a hit that does not carry its own span."""

    def __init__(self, database: Database, files: Mapping[str, str | Path]) -> None:
        self.database = database
        self.files = files
        self.read: dict[str, list[tuple[float, float]]] = {}

    def __call__(self, video_id: str) -> list[tuple[float, float]]:
        if video_id not in self.read:
            self.read[video_id] = spans_of(video_id, self.files.get(video_id),
                                           self.database)
        return self.read[video_id]


def _chunks_wanted(scope: Optional[Sequence[str]], chunk_ids: Optional[Sequence[int]],
                   window: int, after: Optional[float], before: Optional[float],
                   grid: _Grids, database: Database) -> Optional[list[int]]:
    """The chunk ids a search is narrowed to, or None for no narrowing. A time
    window and a chunk set are the same filter; over several videos a window is
    the union of each one's chunks."""
    timed = after is not None or before is not None
    if not chunk_ids and not timed:
        return None
    # Every grid in scope; with no scope, every video the database lists.
    known = list(scope) if scope else database.video_ids()
    # An empty list is no constraint, as with `video_ids`.
    wanted = {int(c) for c in chunk_ids} if chunk_ids else None
    if timed:
        in_window = {c for vid in known for c in chunks_in(grid(vid), after, before)}
        wanted = in_window if wanted is None else wanted & in_window
    if window:
        # The neighbours of each matched chunk, `window` either side.
        longest = max((len(grid(vid)) for vid in known), default=0)
        widened = {c + step for c in wanted for step in range(-window, window + 1)}
        wanted = {c for c in widened if 0 <= c < longest} or wanted
    return sorted(wanted)


__all__ = ["chunks_in", "scope_of", "search", "spans_of"]
