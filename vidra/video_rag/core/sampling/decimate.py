"""Rate reduction at the head of the pipeline: `per_second` frames per second of
media time.

Buckets on media time, never every Nth frame, so a lossy file stays aligned.
Decides from a timestamp alone, before any pixels are converted.
"""

from __future__ import annotations

from typing import Any, Optional
from vidra.shared.reporting.errors import Refused


class Decimator:
    """Keeps the first frame seen in each slice of media time.

        1    -> one frame per second      (buckets at 0.0, 1.0, 2.0 ...)
        4    -> four per second           (0.0, 0.25, 0.5, 0.75 ...)
        0.5  -> one every two seconds
    """

    def __init__(self, per_second: float = 1.0) -> None:
        if per_second <= 0:
            raise Refused("per_second must be positive")
        self.per_second = per_second
        self._last_bucket: Optional[int] = None

    def bucket_of(self, media_ts: float) -> int:
        return int(media_ts * self.per_second)

    def accepts(self, media_ts: float) -> bool:
        """True if this timestamp opens a new slice of media time."""
        bucket = self.bucket_of(media_ts)
        if self._last_bucket is None or bucket > self._last_bucket:
            self._last_bucket = bucket
            return True
        return False

    def config(self) -> dict[str, Any]:
        return {"per_second": self.per_second}


__all__ = ["Decimator"]
