"""Push a file to a live source at the pace it was recorded: the test sender.

    send("samples/test.mp4", "tcp://127.0.0.1:9000")        # real time
    send("samples/test.mp4", "tcp://127.0.0.1:9000", speed=4.0)

The receiver listens first (`video_rag_live("tcp://0.0.0.0:9000?listen=1",
...)`); the file is remuxed, not re-encoded, into MPEG-TS, which survives being
cut into network packets. Only the picture is sent: live reads no audio.
"""

from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from vidra.shared.reporting.errors import Refused
from .source import StreamUnavailable


def send(path: str | Path, url: str, speed: float = 1.0,
         connect_timeout_s: float = 30.0, format: str = "mpegts") -> dict[str, Any]:
    """Send `path`'s picture to `url`, paced at `speed` times real time. Retries
    the connection for `connect_timeout_s`, since a receiver may start late.
    Returns what was sent.
    """
    import av

    if speed <= 0:
        raise Refused(f"speed must be positive, not {speed}")
    source = av.open(str(path))
    if not source.streams.video:
        source.close()
        raise Refused(f"{path} carries no video stream")
    stream = source.streams.video[0]

    deadline = time.monotonic() + connect_timeout_s
    while True:
        try:
            out = av.open(url, "w", format=format)
            break
        except Exception as exc:                             # noqa: BLE001
            if time.monotonic() > deadline:
                source.close()
                raise StreamUnavailable(
                    f"nothing is listening at {url} ({type(exc).__name__}: {exc})"
                ) from None
            time.sleep(0.25)

    sent = 0
    first = None
    closed = False
    started = time.perf_counter()
    try:
        target = out.add_stream_from_template(stream)
        for packet in source.demux(stream):
            if packet.dts is None:
                continue
            # Paced on decode order, which only moves forward.
            ts = packet.dts
            if first is None:
                first = ts
            due = float(Fraction(ts - first) * stream.time_base) / speed
            wait = due - (time.perf_counter() - started)
            if wait > 0:
                time.sleep(wait)
            packet.stream = target
            try:
                out.mux(packet)
            except (av.error.FFmpegError, OSError):
                # The receiver closed the connection: its run is over, so is ours.
                closed = True
                break
            sent += 1
    finally:
        try:
            out.close()
        except (av.error.FFmpegError, OSError):
            closed = True
        source.close()
    return {"packets": sent, "elapsed_s": round(time.perf_counter() - started, 3),
            "url": url, "speed": speed, "closed_by_receiver": closed}


__all__ = ["send"]
