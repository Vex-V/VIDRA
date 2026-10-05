"""`select` -- what one aggregate reads, taken out of a record as a file.

An `Excerpt` for a text aggregator, `Sightings` for a link profile. See
`driver` for why choosing what to read is a component of its own.
"""

from .driver import load, pick, select

__all__ = ["load", "pick", "select"]
