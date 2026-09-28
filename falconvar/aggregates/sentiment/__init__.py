"""`sentiment` -- tone per chunk, and where it turns.

    sentiment.sentiment(excerpt, out)      an excerpt in, one answer out

Imported only when asked for by name; see `ner`.
"""

from .driver import SentimentAggregator, sentiment

__all__ = ["SentimentAggregator", "sentiment"]
