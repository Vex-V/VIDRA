"""`stats` -- counts and coverage: chunks, samplers, words, frames.

    aggregates.stats(record=video, out=...)      a record in, one answer out
"""

from .driver import StatsAggregator

__all__ = ["StatsAggregator"]
