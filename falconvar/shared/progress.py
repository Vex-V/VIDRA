"""What a component reports while it is still running.

`Produced` says what a component did once it is finished, which is the right
answer for a step that takes 30 ms and no answer at all for one that takes
four minutes of paid calls. A poller reading `stage` learned only which
component was running; inside it there was nothing.

    def tick(p: Progress):
        print(f"{p.component} {p.completed}/{p.total}  {p.current}")

    describe.run(video_id, on_progress=tick)

**A callback, not a generator.** `describe` gathers its calls under the
provider's concurrency -- 5.9x on the measured case -- so a synchronous
generator would have to serialise that or buffer through a queue, and the
buffering loses the one thing a generator was wanted for, which is stopping
part way. A callback that raises stops the run at the next await, and `limit=`
already expresses "only this many, and resume later" without stopping anything
mid-flight.

**Events can arrive out of order**, for the same reason: completion order is
not manifest order. `completed` and `total` are monotonic and correct;
`current` is whichever call just landed.

**The callback runs on the event loop.** A slow one serialises the gather it
is reporting on, so it is for reporting and not for work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

#: What every `on_progress` parameter takes.
Reporter = Callable[["Progress"], None]


@dataclass(frozen=True)
class Progress:
    """One unit of a component's work, finished."""

    #: Which component is reporting, in the same vocabulary as
    #: `Produced.component`.
    component: str
    #: How many units this run will attempt. Known before the first one
    #: starts, because every component plans its work up front.
    total: int
    #: How many have finished, counting up.
    completed: int
    #: How many were not attempted because they were already done -- resumed
    #: descriptions, unchanged embeddings. Counted separately from `completed`
    #: because "nothing to do" and "everything done" look identical otherwise,
    #: which is the failure `limit=0` had.
    skipped: int = 0
    #: What just finished, in whatever words the component addresses its work:
    #: `3:clip:text` for a description, a chunk id for an ingest, a batch
    #: range for an embed. None before anything has.
    current: Optional[str] = None
    #: That unit's result, when there is one worth handing over. Deliberately
    #: untyped: a description, a manifest chunk and a batch of units are
    #: different things, and a common shape would be a lie.
    result: Any = None

    @property
    def fraction(self) -> float:
        """Finished over attempted, in [0, 1]. Zero total reads as done."""
        return 1.0 if not self.total else self.completed / self.total


def report(on_progress: Optional[Reporter], component: str, total: int,
           completed: int, skipped: int = 0, current: Optional[str] = None,
           result: Any = None) -> None:
    """Call back, if anyone asked. The no-callback case costs one `is None`."""
    if on_progress is None:
        return
    on_progress(Progress(component=component, total=total, completed=completed,
                         skipped=skipped, current=current, result=result))


__all__ = ["Progress", "Reporter", "report"]
