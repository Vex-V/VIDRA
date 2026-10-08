"""The sampling half of a live run, on its own thread: decode, decimate, find
the chunk, offer the frame to every sampler, and hand each kept frame on as
soon as its context is complete.

A chunk is `int(media_ts // chunk_s)` -- known the moment a frame arrives, so
nothing waits for a chunk to close. Samplers reset when the chunk changes,
exactly as in `offline`, so the same frames are kept from the same file.

Context: `before` decimated frames are kept in a ring and sent ahead of the
kept frame; `after` makes the kept frame wait for that many more decimated
frames, which delays its answer by `after / per_second` seconds. Every frame a
model is shown is written to the store, so an answer names frames that exist.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from ..core.describe import prompts
from ..core.describe.frames import LoadedFrame
from ..core.frames import FrameStore, encode
from ..core.sampling.decimate import Decimator
from ..core.sampling.offer import Offer, empty_chunk
from ..core.sampling.samplers import Sampler
from .source import Arrival, Source


@dataclass
class Ask:
    """One kept frame, with its context, waiting for its questions to be answered."""

    chunk_id: int
    #: The sampler run that kept it, its name and its configuration.
    run_id: str
    name: str
    config: dict[str, Any]
    #: `(answer_id, question)` for every question this run asks.
    asked: list[tuple[str, str]]
    frame: LoadedFrame
    arrival: Arrival
    before: list[LoadedFrame] = field(default_factory=list)
    after: list[LoadedFrame] = field(default_factory=list)
    need_after: int = 0

    def images(self) -> list[LoadedFrame]:
        """Every frame the model is shown, in time order."""
        return [*self.before, self.frame, *self.after]


@dataclass
class Tally:
    """What the sampling thread saw, read by the driver when the run ends."""

    #: [{chunk_id, decimated_frames, samplers: {id: {frame_count, frames[]}}}],
    #: by chunk id; a chunk a gap skipped is absent.
    chunks: dict[int, dict[str, Any]] = field(default_factory=dict)
    decimated: int = 0
    sampled: int = 0
    #: Why reading stopped: `ended`, `stopped` or `stop_after_s`.
    ended: str = "ended"
    error: Optional[BaseException] = None


def observe(source: Source, samplers: Sequence[Sampler], decimator: Decimator,
            chunk_s: float, context: tuple[int, int], store: FrameStore,
            post: Callable[[Ask], None], tally: Tally,
            stop: Any, stop_after_s: Optional[float] = None) -> None:
    """Read `source` until it ends or `stop` is set, posting one `Ask` per frame
    a sampler keeps. Runs on the sampling thread; `post` must be thread-safe.
    """
    before_n, after_n = context
    ring: deque[LoadedFrame] = deque(maxlen=before_n or None)
    waiting: list[Ask] = []
    offer = Offer(samplers)
    asked_by = {s.sampler_id: [(prompts.answer_id(s.name, q), q)
                               for q in prompts.questions_of(s.config(), s.sampler_id)]
                for s in samplers}

    def kept(frame: LoadedFrame) -> LoadedFrame:
        store.write_bytes(frame.index, frame.jpeg)
        return frame

    for frame, arrival in source.frames(decimator.accepts):
        if stop.is_set():
            tally.ended = "stopped"
            break
        if stop_after_s is not None and frame.media_ts >= stop_after_s:
            tally.ended = "stop_after_s"
            break
        chunk_id = int(frame.media_ts // chunk_s)
        chunk = tally.chunks.setdefault(chunk_id, empty_chunk(chunk_id))

        jpeg: Optional[bytes] = None

        def loaded() -> LoadedFrame:
            nonlocal jpeg
            if jpeg is None:
                jpeg = encode(frame.image, store.quality)
            return LoadedFrame(frame.index, frame.media_ts, jpeg)

        # This frame completes the context of frames kept before it.
        if waiting:
            this = kept(loaded())
            still: list[Ask] = []
            for ask in waiting:
                ask.after.append(this)
                if len(ask.after) >= ask.need_after:
                    post(ask)
                else:
                    still.append(ask)
            waiting = still

        for sampler in offer(frame, chunk_id, chunk, seen_at=arrival.seen_at):
            ask = Ask(chunk_id=chunk_id, run_id=sampler.sampler_id,
                      name=sampler.name, config=sampler.config(),
                      asked=asked_by[sampler.sampler_id], frame=kept(loaded()),
                      arrival=arrival, before=[kept(f) for f in ring],
                      need_after=after_n)
            if after_n:
                waiting.append(ask)
            else:
                post(ask)

        if before_n:
            ring.append(loaded())
        tally.decimated, tally.sampled = offer.decimated, offer.sampled
        frame.release()

    # The stream ended before their context did: sent with what there is.
    for ask in waiting:
        post(ask)


__all__ = ["Ask", "Tally", "observe"]
