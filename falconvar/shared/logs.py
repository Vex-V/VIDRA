"""What a run says about itself, for whoever is listening.

One logger per component, `falconvar.<component>`, and **no handler anywhere**
-- a library that configures logging configures it for the application that
imported it. `falconvar/__init__.py` attaches a `NullHandler` and nothing
else, so the default is silence and a caller opts in:

    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("falconvar").setLevel(logging.DEBUG)

Every record carries structured fields on `extra`, so a handler that wants
JSON has them without parsing the message:

    video_id     which video
    component    the same vocabulary as `Produced.component`
    event        read | write | skip | call | done
    duration_ms  for a timed event
    artifact     for a read or a write
    reason       for a skip

**Levels, and one departure from the obvious.** DEBUG is what was read, what
was skipped and why. INFO is a component starting and finishing, with its
headline numbers. ERROR is a failure.

WARNING is *not* used for a setting resolved by its default -- `providers`
choosing openai is the documented answer, published by `/capabilities`, not a
fallback, and warning on a working default teaches people to filter warnings
out. It is reserved for what this tree already reports-and-continues: the
best-effort Supabase write, and a `prompts_error`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

ROOT = "falconvar"

#: The fields every record may carry. Named here so a formatter can ask for
#: them without a record that lacks one raising inside logging itself.
FIELDS = ("video_id", "component", "event", "duration_ms", "artifact", "reason")


def logger(component: str) -> logging.Logger:
    """`falconvar.describe`, and so on. One per component, by its own name."""
    return logging.getLogger(f"{ROOT}.{component}")


#: Attribute names a `LogRecord` already owns. `extra` holding one of these
#: does not shadow it -- `logging` raises `KeyError: Attempt to overwrite
#: 'name' in LogRecord` and takes the run down with it. A component's stats
#: are its own business and may legitimately be called `module` or `name`, so
#: a collision is renamed rather than refused: a logging call must never be
#: the thing that fails a stage that has already done its work.
RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)))


def _extra(component: str, event: str, video_id: Optional[str],
           rest: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """`rest` is a dict and not `**kwargs` on purpose: a component's stats are
    its own vocabulary, and `done(video_id=...)` is a perfectly reasonable
    thing to write -- as `**kwargs` it collided with this function's own
    parameter and took the run down from inside a logging call."""
    fields = {"component": component, "event": event, "video_id": video_id}
    for key, value in (rest or {}).items():
        if value is None:
            continue
        # A stat may not overwrite the three structural fields either: a
        # record claiming a component it did not come from is worse than one
        # with an awkwardly named number on it.
        taken = key in RESERVED or key in ("component", "event", "video_id")
        fields[f"stat_{key}" if taken else key] = value
    return fields


def read(component: str, video_id: Optional[str], artifact: str) -> None:
    logger(component).debug("read %s", artifact,
                            extra=_extra(component, "read", video_id,
                                         {"artifact": artifact}))


def wrote(component: str, video_id: Optional[str], artifact: str,
          where: str) -> None:
    logger(component).debug("wrote %s -> %s", artifact, where,
                            extra=_extra(component, "write", video_id,
                                         {"artifact": artifact}))


def skipped(component: str, video_id: Optional[str], what: str,
            reason: str) -> None:
    logger(component).debug("skipped %s: %s", what, reason,
                            extra=_extra(component, "skip", video_id,
                                         {"artifact": what, "reason": reason}))


class timed:
    """A component's whole run, as one INFO in and one INFO out.

        with logs.timed("describe", video_id) as done:
            ...
            done(described=12, skipped=3)

    The closing call takes the headline numbers, because they do not exist
    until the work does -- and a finish line with no numbers is the thing a
    poller could already infer from the next start line.

    A failure logs ERROR with the elapsed time and re-raises. Nothing is
    swallowed: this reports, it does not handle.
    """

    __slots__ = ("component", "video_id", "started", "_stats")

    def __init__(self, component: str, video_id: Optional[str] = None) -> None:
        self.component = component
        self.video_id = video_id
        self.started = 0.0
        self._stats: dict[str, Any] = {}

    def __call__(self, video_id: Optional[str] = None, **stats: Any) -> None:
        """The headline numbers, and -- for `media` -- the id itself.

        `media` is the one component that does not know its video id until it
        has run: a different file wanting a taken id is given a new one. So
        the id is settable here rather than only at construction, and it lands
        in the structural `video_id` field where a handler looks for it.
        """
        if video_id is not None:
            self.video_id = video_id
        self._stats.update(stats)

    def __enter__(self) -> "timed":
        self.started = time.perf_counter()
        logger(self.component).info(
            "started", extra=_extra(self.component, "start", self.video_id))
        return self

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> None:
        elapsed = round((time.perf_counter() - self.started) * 1000, 1)
        log = logger(self.component)
        if kind is not None:
            log.error("failed after %s ms: %s", elapsed, value,
                      extra=_extra(self.component, "fail", self.video_id,
                                   {"duration_ms": elapsed,
                                    "reason": f"{kind.__name__}: {value}"[:300]}))
            return
        detail = " ".join(f"{k}={v}" for k, v in self._stats.items())
        log.info("done in %s ms  %s", elapsed, detail,
                 extra=_extra(self.component, "done", self.video_id,
                              {"duration_ms": elapsed, **self._stats}))


__all__ = ["FIELDS", "ROOT", "logger", "read", "skipped", "timed", "wrote"]
