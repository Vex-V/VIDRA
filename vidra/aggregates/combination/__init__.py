"""`combination` -- several videos laid end to end as one record.

    aggregates.combine(records=[monday, tuesday], out=...)   records in, one record out
"""

from .driver import (CombineError, Combined, KINDS, Part, combine, default_id,
                     merge, origin)

__all__ = ["CombineError", "Combined", "KINDS", "Part", "combine", "default_id",
           "merge", "origin"]
