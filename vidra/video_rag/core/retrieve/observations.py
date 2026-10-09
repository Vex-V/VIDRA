"""A query, into ranked frames: search over a live run's answers.

`search` ranks chunks; this ranks the individual answers a live run wrote, one
per kept frame and question, and can do it while the run is still going. A time
filter is either media seconds (`after`, `before`) or when the frames arrived
(`since`, `until`) -- "anything like a spill in the last ten minutes" is the
second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from vidra.shared.models import embedders as embedders_mod
from vidra.shared.models.base import require
from vidra.shared.models.roles import Models, given, resolve
from vidra.shared.reporting.errors import Refused
from vidra.shared.storage.database import Database, as_database
from .driver import scope_of


@dataclass
class FrameHit:
    """One live answer that matched: a kept frame, and what was said about it."""

    video_id: str
    chunk_id: int
    #: `name:question`, or the bare sampler name for its own question.
    sampler_id: str
    sampler: str
    question: str
    #: The kept frame, its time on the media clock, and every frame shown.
    frame_index: int
    media_ts: float
    frames: list[int]
    description: str
    structured: dict[str, Any]
    #: When the kept frame arrived (ISO 8601 UTC).
    seen_at: Optional[str]
    #: A rank fusion for ordering, not a similarity.
    score: float
    dense_rank: Optional[int] = None
    text_rank: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "extra"}


def _stamp(value: Optional[str | datetime], name: str) -> Optional[str]:
    """A time as ISO 8601 UTC; a naive datetime is taken as UTC."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            raise Refused(f"{name} must be a datetime or ISO 8601, not {value!r}") from None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def search_observations(query: str,
                        video_id: Optional[str | Sequence[str]] = None,
                        limit: int = 10,
                        sampler: Optional[str] = None,
                        question: Optional[str] = None,
                        strategy: Optional[str] = None,
                        after: Optional[float] = None,
                        before: Optional[float] = None,
                        since: Optional[str | datetime] = None,
                        until: Optional[str | datetime] = None,
                        structured: Optional[dict[str, Any]] = None,
                        video_ids: Optional[Sequence[str]] = None,
                        database: Optional[str | Database] = None,
                        models: Optional[Models] = None) -> list[FrameHit]:
    """Ranked live answers, best first.

        # anything like a spill on dock-3 in the last ten minutes
        search_observations("liquid spilled on the floor", "dock-3",
                            since=datetime.now(timezone.utc) - timedelta(minutes=10),
                            models=models, database=db)

    `models.embedder` must be the one the live run used.
    `database` is a built `Database` or a name (`supabase` by default); a
    `Folder` reads each run's `observations.jsonl` in place.
    """
    if not (query or "").strip():
        raise Refused("a search needs a query; this one is empty")
    if limit < 1:
        raise Refused(f"limit must be at least 1, not {limit}")
    if after is not None and before is not None and before <= after:
        raise Refused(f"before ({before}) must be later than after ({after})")
    lo, hi = _stamp(since, "since"), _stamp(until, "until")
    if lo is not None and hi is not None and hi <= lo:
        raise Refused(f"until ({hi}) must be later than since ({lo})")

    built = resolve("embedder", given(models).embedder)
    require("embedder", built)
    target = as_database(database or "supabase")
    vector = embedders_mod.query_vector(built, query)
    rows = target.search_observations(
        vector, query, built.key, limit, scope_of(video_id, video_ids), sampler,
        question, strategy, after, before, lo, hi, structured)
    known = set(FrameHit.__dataclass_fields__) - {"extra"}
    return [FrameHit(**{k: r.get(k) for k in known if k in r},
                     extra={k: v for k, v in r.items() if k not in known})
            for r in rows]


__all__ = ["FrameHit", "search_observations"]
