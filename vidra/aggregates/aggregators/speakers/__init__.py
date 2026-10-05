"""`speakers` -- who spoke, for how long, and how often the voice changed.

    aggregates.speakers(record=video, out=...)      a record in, one answer out
"""

from .driver import SpeakersAggregator

__all__ = ["SpeakersAggregator"]
