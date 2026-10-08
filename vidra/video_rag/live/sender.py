"""Push a file to a live source at the pace it was recorded: the test sender.

    send("samples/test.mp4", "tcp://127.0.0.1:9000")        # real time
    send("samples/test.mp4", "tcp://127.0.0.1:9000", speed=4.0)

The receiver listens first (`video_rag_live("tcp://0.0.0.0:9000?listen=1",
...)`); the file is remuxed, not re-encoded: into MPEG-TS for the codecs it
carries (H.264, HEVC, MPEG-2/4), into Matroska for the rest (VP9, AV1, ...),
both of which survive being cut into network packets. Only the picture is
sent: live reads no audio.
"""

from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from vidra.shared.reporting.errors import Refused
from .source import StreamUnavailable

#: Video codecs MPEG-TS carries; anything else goes in Matroska.
MPEGTS_CODECS = {"h264", "hevc", "mpeg1video", "mpeg2video", "mpeg4"}


def send(path: str | Path, url: str, speed: float = 1.0,
         connect_timeout_s: float = 120.0, format: Optional[str] = None
         ) -> dict[str, Any]:
    """Send `path`'s picture to `url`, paced at `speed` times real time. Retries
    the connection for `connect_timeout_s`. `format` is the container on the
    wire; None picks MPEG-TS or Matroska by the codec. Returns what was sent.

    Run it in its own process, as a camera would be.
    """
    import av

    if speed <= 0:
        raise Refused(f"speed must be positive, not {speed}")
    with av.open(str(path)) as probe:
        if not probe.streams.video:
            raise Refused(f"{path} carries no video stream")
        codec = probe.streams.video[0].codec_context.name
    try:
        av.Codec(codec, "w")
    except Exception:                                        # noqa: BLE001
        # Copying a stream needs its codec writable in this build.
        raise Refused(f"{path} is {codec}, which this PyAV cannot write, so it "
                      f"cannot be sent without re-encoding; re-encode it to "
                      f"H.264 first") from None
    if format is None:
        format = "mpegts" if codec in MPEGTS_CODECS else "matroska"

    # Retried from the first packet until one is written: PyAV connects on
    # the first write, not on open.
    deadline = time.monotonic() + connect_timeout_s
    while True:
        source = av.open(str(path))
        stream = source.streams.video[0]
        out = av.open(url, "w", format=format)
        target = out.add_stream_from_template(stream)
        packets = (p for p in source.demux(stream) if p.dts is not None)
        first_packet = next(packets, None)
        if first_packet is None:
            out.close()
            source.close()
            raise Refused(f"{path} has no packets to send")
        first = first_packet.dts
        first_packet.stream = target
        try:
            out.mux(first_packet)
            break
        except (av.error.FFmpegError, OSError) as exc:
            for closing in (out, source):
                try:
                    closing.close()
                except (av.error.FFmpegError, OSError):
                    pass
            if time.monotonic() > deadline:
                raise StreamUnavailable(
                    f"nothing is listening at {url} ({type(exc).__name__}: {exc})"
                ) from None
            time.sleep(0.25)

    sent = 1
    closed = False
    started = time.perf_counter()
    try:
        for packet in packets:
            # Paced on decode order, which only moves forward.
            due = float(Fraction(packet.dts - first) * stream.time_base) / speed
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
            "url": url, "speed": speed, "format": format,
            "closed_by_receiver": closed}

__all__ = ["send"]
