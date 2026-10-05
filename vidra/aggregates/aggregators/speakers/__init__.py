"""`speakers` -- who spoke, for how long, and how often the voice changed.

    speakers.speakers(record, out)      a record in, one answer out
"""

from .driver import SpeakersAggregator, speakers

__all__ = ["SpeakersAggregator", "speakers"]
