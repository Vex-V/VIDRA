"""`stats` -- counts and coverage: chunks, samplers, words, frames.

    stats.stats(record, out)      a record in, one answer out
"""

from .driver import StatsAggregator, stats

__all__ = ["StatsAggregator", "stats"]
