"""The retrieve component: a query -> ranked moments. Writes nothing.

The query is embedded with the embedder that built the index; every row
carries its embedder key and only rows in that space are searched. Ranking is
the database's (`Database.search`; for Supabase the `vr_search` RPC).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from vidra.shared.contracts.documents import Timeline
from vidra.shared.storage.files import read
from vidra.shared.models import embedders as embedders_mod
from .search import Moment, to_moments
from vidra.shared.reporting.errors import Refused
from vidra.shared.models.base import Embedder, require
from vidra.shared.models.roles import Models, resolve, unpack
from vidra.shared.storage.database import Database, as_database
from vidra.shared.config.paths import MissingArtifact


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
           embedder: Optional[Embedder] = None,
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
    (`supabase` by default). `embedder` (or `models.embedder`) must be the one
    the index was built with; None is OpenAI's default.
    """
    if not (query or "").strip():
        # An empty query is refused here rather than at the provider.
        raise Refused("a search needs a query; this one is empty")

    scope = scope_of(video_id, video_ids)
    built = resolve("embedder", unpack(models, embedder=embedder)["embedder"])
    require("embedder", built)
    return _search(built, query, scope, moments, sampler, question,
                   candidates, strategy, chunk_ids, window, after, before,
                   structured, grids, as_database(database or "supabase"))


def _search(built, query: str,
            scope: Optional[Sequence[str]],
            moments: int, sampler: Optional[str], question: Optional[str],
            candidates: int, strategy: Optional[str],
            chunk_ids: Optional[Sequence[int]], window: int,
            after: Optional[float], before: Optional[float],
            structured: Optional[dict[str, Any]],
            grids: Optional[Mapping[str, str | Path]],
            database: Database
            ) -> tuple[list[Moment], list[str]]:
    notes: list[str] = []
    # A grid per video in scope: a chunk id means nothing without its video.
    known = list(scope) if scope else _all_videos(database)
    spans = {vid: spans_of(vid, (grids or {}).get(vid), database)
             for vid in known}
    one = spans.get(known[0], []) if len(known) == 1 else spans

    # A time window and a chunk set are the same filter.
    wanted: Optional[set[int]] = None
    if chunk_ids:
        # An empty list is no constraint, as with `video_ids`.
        wanted = set(int(c) for c in chunk_ids)
    if after is not None or before is not None:
        # A window over several videos is the union of each one's chunks.
        in_window: set[int] = set()
        for vid in known:
            in_window |= set(chunks_in(spans.get(vid, []), after, before))
        wanted = in_window if wanted is None else (wanted & in_window)
    if wanted is not None and window:
        # The neighbours of each matched chunk, `window` either side.
        longest = max((len(v) for v in spans.values()), default=0)
        widened = {c + step for c in wanted
                   for step in range(-window, window + 1)}
        wanted = {c for c in widened if 0 <= c < longest} or wanted
    narrowed = sorted(wanted) if wanted is not None else None
    if narrowed is not None and not narrowed:
        return [], notes + ["no chunk matches that window"]

    # Some models embed a query differently from a passage.
    vector = embedders_mod.query_vector(built, query)
    hits = database.search(vector, query, built.key, candidates, scope,
                           sampler, question, strategy, narrowed, structured)
    if not hits:
        if any(f is not None for f in (sampler, question, strategy,
                                       narrowed, structured, scope)):
            # Nothing matched the filters, as opposed to nothing indexed in this space.
            return [], notes + [f"nothing matched those filters in {built.key} -- "
                                "if that embedder never indexed these videos, "
                                "embed them with it first"]
        where = ", ".join(scope) if scope else "any video"
        raise MissingArtifact(
            f"{where}: nothing indexed for {built.key}. Run embed with this "
            f"embedder, and a pipeline naming a database, first -- a "
            f"different embedder writes different rows.")
    if any(f is not None for f in (sampler, question, strategy, structured)):
        notes.append("filtering gives up the agreement signal: a chunk "
                     "contributes fewer terms, so scores fall -- to a single "
                     "1/(k+rank) when only one unit per chunk survives")
    if scope is None or len(known) > 1:
        # Moments from several videos share one ranking.
        notes.append(f"scope is {len(known)} videos: moments are keyed by "
                     "(video_id, chunk_id), since a chunk id only means "
                     "something inside one grid")
    return to_moments(hits, known[0] if len(known) == 1 else "",
                      one, moments), notes


def _all_videos(database: Database) -> list[str]:
    """Every video the database has a grid for."""
    try:
        return database.video_ids()
    except Exception:                                    # noqa: BLE001
        return []
