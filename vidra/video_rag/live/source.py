"""A live source: decoded frames, on a clock that starts at the first frame.

Anything PyAV can open is a source -- `tcp://0.0.0.0:9000?listen=1` (wait for
a sender), `udp://...`, `rtmp://...`, `rtsp://camera/...`, or a file, which is
then read as fast as it decodes.

A stream's timestamps start wherever the sender's clock was, and jump when it
reconnects or wraps. `media_ts` is counted from the first frame, a jump forward
larger than `gap_s` is recorded as a gap, and a jump backward continues the
clock where it was rather than going back in time. Wall time is recorded beside
each frame as a label (`seen_at`); no decision is made on it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from vidra.shared.contracts.documents import VideoStream
from vidra.shared.reporting.errors import Unavailable
from ..core.sampling.reader import Frame


class StreamUnavailable(Unavailable):
    """The source could not be opened, carries no picture, or stopped answering."""


@dataclass
class Arrival:
    """When a frame reached us, beside where it sits on the media clock."""

    #: `time.perf_counter()` at decode; lag is measured against it.
    seen: float
    #: The same moment as wall time, ISO 8601 UTC.
    seen_at: str
    #: True when the clock jumped just before this frame (a gap or a reconnect).
    gap_before: bool = False


class Source:
    """One opened stream. `frames()` yields until the sender stops or `stop()`."""

    def __init__(self, url: str, open_timeout_s: Optional[float] = 60.0,
                 read_timeout_s: Optional[float] = 10.0, gap_s: float = 2.0,
                 record: Optional[Path] = None) -> None:
        self.url = url
        self.open_timeout_s = open_timeout_s
        self.read_timeout_s = read_timeout_s
        self.gap_s = gap_s
        self.record = record
        self.container: Any = None
        self.stream: Any = None
        self._recording: Any = None
        self._recorded: Any = None
        self._stopping = False
        #: `(last_ts, next_ts)` for every jump of more than `gap_s`.
        self.gaps: list[tuple[float, float]] = []
        #: Jumps backward, continued rather than followed.
        self.rewinds = 0
        #: Packets the decoder could not read, and recording writes that failed:
        #: skipped and counted, since a damaged packet is routine on a network.
        self.corrupt = 0
        self.unrecorded = 0
        #: When the first frame arrived, ISO 8601 UTC.
        self.started_at: Optional[str] = None
        #: The media clock at the newest decoded frame, and one frame's length.
        self.last_ts: Optional[float] = None
        self.step = 1.0 / 30.0

    # ------------------------------------------------------------- opening
    def open(self) -> "Source":
        import av

        try:
            self.container = av.open(
                self.url, timeout=(self.open_timeout_s, self.read_timeout_s))
        except Exception as exc:                             # noqa: BLE001
            raise StreamUnavailable(
                f"cannot open {self.url} ({type(exc).__name__}: {exc})") from None
        if not self.container.streams.video:
            self.close()
            raise StreamUnavailable(f"{self.url} carries no video stream")
        self.stream = self.container.streams.video[0]
        # Required for decode speed.
        self.stream.thread_type = "AUTO"
        if self.record is not None:
            self.record.parent.mkdir(parents=True, exist_ok=True)
            self._recording = av.open(str(self.record), "w")
            self._recorded = self._recording.add_stream_from_template(self.stream)
        return self

    def describe(self, duration_s: Optional[float] = None) -> VideoStream:
        """The picture stream as `media.json` records it."""
        codec = self.stream.codec_context
        rate = self.stream.average_rate or self.stream.guessed_rate
        return VideoStream(
            index=self.stream.index, codec=codec.name,
            rate=float(rate) if rate else None,
            time_base=(str(self.stream.time_base)
                       if self.stream.time_base is not None else None),
            width=codec.width, height=codec.height, frames=None,
            duration_s=duration_s)

    @property
    def container_format(self) -> str:
        return self.container.format.name if self.container is not None else ""

    # ------------------------------------------------------------- reading
    def frames(self, keep: Callable[[float], bool]) -> Iterator[tuple[Frame, Arrival]]:
        """Yield the frames `keep` accepts, pixels attached, with when each arrived.
        `keep` is handed a `media_ts`; a declined frame is never converted.
        """
        time_base = self.stream.time_base
        rate = self.stream.average_rate or self.stream.guessed_rate
        step = self.step = 1.0 / float(rate) if rate else 1.0 / 30.0
        origin: Optional[int] = None
        offset = 0.0
        last: Optional[float] = None
        index = 0
        import av
        try:
            for packet in self.container.demux(self.stream):
                if self._stopping:
                    return
                try:
                    decoded = packet.decode()
                except av.error.FFmpegError:
                    self.corrupt += 1
                    decoded = []
                # Decoded before muxing: muxing re-points the packet at the output.
                if self._recording is not None and packet.dts is not None:
                    packet.stream = self._recorded
                    try:
                        self._recording.mux(packet)
                    except av.error.FFmpegError:
                        self.unrecorded += 1
                for av_frame in decoded:
                    seen = time.perf_counter()
                    if self.started_at is None:
                        self.started_at = _now()
                    if av_frame.pts is not None and time_base is not None:
                        if origin is None:
                            origin = av_frame.pts
                        raw = float(Fraction(av_frame.pts - origin) * time_base)
                        pts = av_frame.pts
                    else:
                        raw = index * step
                        pts = None
                    media_ts = raw + offset
                    gap = False
                    if last is not None:
                        if media_ts < last:
                            # Backward: a reconnect or a wrap. Continue the clock.
                            offset += last + step - media_ts
                            media_ts = last + step
                            self.rewinds += 1
                            gap = True
                        elif media_ts - last > self.gap_s:
                            self.gaps.append((round(last, 3), round(media_ts, 3)))
                            gap = True
                    last = self.last_ts = media_ts
                    if keep(media_ts):
                        image = av_frame.to_ndarray(format="bgr24")
                        yield (Frame(index=index, media_ts=media_ts, pts=pts,
                                     image=image),
                               Arrival(seen=seen, seen_at=_now(), gap_before=gap))
                    index += 1
        except Exception as exc:                             # noqa: BLE001
            if self._stopping:
                return
            if isinstance(exc, (av.error.ExitError, av.error.EOFError)):
                # The sender went away, or stopped answering for read_timeout_s.
                return
            raise StreamUnavailable(
                f"{self.url} failed while reading ({type(exc).__name__}: {exc})"
            ) from None

    @property
    def duration_s(self) -> float:
        """How much media has been read: the newest frame's time plus its length."""
        return 0.0 if self.last_ts is None else self.last_ts + self.step

    def stop(self) -> None:
        """Stop after the frame being read; safe from any thread."""
        self._stopping = True

    def close(self) -> None:
        if self._recording is not None:
            try:
                self._recording.close()
            finally:
                self._recording = None
        if self.container is not None:
            try:
                self.container.close()
            finally:
                self.container = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


__all__ = ["Arrival", "Source", "StreamUnavailable"]
