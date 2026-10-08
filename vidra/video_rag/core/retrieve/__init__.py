"""retrieve -- a query, into ranked moments; or, over a live run's answers,
into ranked frames."""

from __future__ import annotations

#: The public surface: entry points, errors and return types.
from .driver import search
from .observations import FrameHit, search_observations
from .search import Moment

#: Searching summaries, chapters and entities is `vidra.aggregates.search`.
__all__ = ["FrameHit", "Moment", "search", "search_observations"]
