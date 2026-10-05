"""Logging: one logger per component, `vidra.<component>`, and no handler.

`vidra/__init__.py` attaches a `NullHandler`, so the default is silence:

    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("vidra").setLevel(logging.DEBUG)

Every record carries structured fields on `extra`:

    video_id     which video
    component    the same vocabulary as `Produced.component`
    event        read | write | skip | call | done
    duration_ms  for a timed event
    artifact     for a read or a write
    reason       for a skip

DEBUG is what was read and skipped; INFO is a component starting and
finishing; WARNING is a failed database write that was continued past; ERROR
is a failure.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

ROOT = "vidra"


def logger(component: str) -> logging.Logger:
    """`vidra.describe`, and so on. One per component, by its own name."""
    return logging.getLogger(f"{ROOT}.{component}")


#: Attribute names a `LogRecord` already owns; an `extra` key with one of these
#: names is renamed.
RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)))


def _extra(component: str, event: str, video_id: Optional[str],
           rest: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The `extra` dict for one record; `rest` holds a component's own stats."""
    fields = {"component": component, "event": event, "video_id": video_id}
    for key, value in (rest or {}).items():
        if value is None:
            continue
        # A stat may not overwrite the structural fields.
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

    A failure logs ERROR with the elapsed time and re-raises.
    """

    __slots__ = ("component", "video_id", "started", "_stats")

    def __init__(self, component: str, video_id: Optional[str] = None) -> None:
        self.component = component
        self.video_id = video_id
        self.started = 0.0
        self._stats: dict[str, Any] = {}

    def __call__(self, video_id: Optional[str] = None, **stats: Any) -> None:
        """Record the headline numbers, and optionally the video id (which `media`
        only knows once it has run).
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


__all__ = ["ROOT", "logger", "read", "skipped", "timed", "wrote"]
