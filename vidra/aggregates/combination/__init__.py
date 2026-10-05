"""`combination` -- several videos laid end to end as one record.

    combination.combine([folder, ...], out)      the folders in, one record out

A component like every other aggregator folder, with its own CLI:
`python -m vidra.aggregates.combination a b --out ab`.
"""

from .driver import (CombineError, Combined, KINDS, Part, combine, default_id,
                     merge, origin)

__all__ = ["CombineError", "Combined", "KINDS", "Part", "combine", "default_id",
           "merge", "origin"]
