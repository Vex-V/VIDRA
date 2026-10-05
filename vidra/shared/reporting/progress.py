"""What a component reports while it is still running.

    def tick(p: Progress):
        print(f"{p.component} {p.completed}/{p.total}  {p.current}")

    describe.describe(..., on_progress=tick)

Events can arrive out of order (concurrent calls finish in any order);
`completed` and `total` are correct. The callback runs on the event loop, so
it should be quick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

#: What every `on_progress` parameter takes.
Reporter = Callable[["Progress"], None]


@dataclass(frozen=True)
class Progress:
    """One unit of a component's work, finished."""

    #: Which component is reporting (as `Produced.component`).
    component: str
    #: How many units this run will attempt.
    total: int
    #: How many have finished, counting up.
    completed: int
    #: How many were skipped because they were already done.
    skipped: int = 0
    #: What just finished, e.g. `3:clip:text`, a chunk id, or a batch range.
    current: Optional[str] = None
    #: That unit's result, when there is one.
    result: Any = None

    @property
    def fraction(self) -> float:
        """Finished over attempted, in [0, 1]. Zero total reads as done."""
        return 1.0 if not self.total else self.completed / self.total


def report(on_progress: Optional[Reporter], component: str, total: int,
           completed: int, skipped: int = 0, current: Optional[str] = None,
           result: Any = None) -> None:
    """Call back, if a callback was given."""
    if on_progress is None:
        return
    on_progress(Progress(component=component, total=total, completed=completed,
                         skipped=skipped, current=current, result=result))


__all__ = ["Progress", "Reporter", "report"]
