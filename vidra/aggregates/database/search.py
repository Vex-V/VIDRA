"""Search what the aggregates embedded: which video, which part, who.

    aggregates.search("the reactor explosion", level="source")   # which video
    aggregates.search("the reactor explosion", level="span")     # which part
    aggregates.search("woman with long pink hair", level="entity")  # who

Each level is searched alone. Every result names its source and the videos it
covers.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ...shared.reporting.errors import Refused
from ...shared.models.base import require
from ...shared.models.roles import Models, resolve, unpack
from .export import LEVELS


def search(query: str, level: str = "source", embedder: Optional[Any] = None,
           limit: int = 5, source_ids: Optional[Sequence[str]] = None,
           database: Optional[Any] = None,
           models: Optional[Models] = None) -> list[dict[str, Any]]:
    """Ranked summaries (`source`), chapters (`span`) or entities (`entity`).

    `embedder` must be the one the aggregates were exported with (`models`
    carries it). `source_ids=None` is every source. Each result has `source_id`,
    `video_ids`, `aggregate_id`, `item_id`, `content`, `start_ts`/`end_ts` where
    it has a span, and its ranks.
    """
    from ...shared.models import embedders
    from ...shared.storage.database import as_database

    if level not in LEVELS:
        raise Refused(f"unknown level {level!r}; known: {', '.join(LEVELS)}")
    if not str(query or "").strip():
        raise Refused("a search needs a query; this one is empty")
    if limit < 1:
        raise Refused("limit must be 1 or more")
    built = resolve("embedder", unpack(models, embedder=embedder)["embedder"])
    require("embedder", built)
    # Some models embed a query differently from a passage.
    vector = embedders.query_vector(built, query)
    return as_database(database or "supabase").search_aggregates(
        vector, query, built.key, level, limit, source_ids)


__all__ = ["LEVELS", "search"]
