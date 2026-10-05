"""`coverage` -- which chunks have an account, and from which modality.

    aggregates.coverage(record=video, out=...)      a record in, one answer out
"""

from .driver import CoverageAggregator

__all__ = ["CoverageAggregator"]
