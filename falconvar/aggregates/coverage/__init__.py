"""`coverage` -- which chunks have an account, and from which modality.

    coverage.coverage(record, out)      a record in, one answer out
"""

from .driver import CoverageAggregator, coverage

__all__ = ["CoverageAggregator", "coverage"]
