"""`sentiment` -- tone per chunk, and where it turns.

    aggregates.sentiment(input=excerpt, out=...)      an excerpt in, one answer out

Imported only when asked for by name; see `ner`.
"""

from .driver import SentimentAggregator

__all__ = ["SentimentAggregator"]
