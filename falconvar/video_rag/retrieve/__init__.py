"""retrieve -- a query, into ranked moments."""

from __future__ import annotations

#: The public surface: entry points, errors and return types.
from .driver import search
from .search import Moment

#: Searching summaries, chapters and entities is `falconvar.aggregates.search`.
__all__ = ["Moment", "search"]
