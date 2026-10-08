"""Sequential decode, converting only the frames something asked for.

The decimator decides from `media_ts` alone, before a frame is converted to
an array, so declined frames cost only the decode. A generator: one frame in
flight. `Frame.image` is released when no longer needed; copy anything kept.
Rotation is read with OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import av
import numpy as np

from vidra.shared.contracts.documents import Media
from vidra.shared.reporting.errors import VidraError

#: Container rotation is applied by the reader (PyAV does not), read with
#: OpenCV.
_ROTATIONS = {90: 0, 180: 1, 270: 2}       # cv2.ROTATE_* resolved lazily


class UnreadableSource(VidraError, RuntimeError):
    """The file cannot be opened, or carries no video stream."""


@dataclass
class Frame:
    """One frame the pipeline is going to look at.

    `media_ts` is the position on the media clock. `pts` is the same position in
    the container's timebase and addresses the frame exactly. `index` counts
    every frame read, kept or not; it names the stored file. `image` is BGR and
    borrowed: copy anything that outlives the loop.
    """

    index: int
    media_ts: float
    pts: Optional[int] = None
    image: Optional[np.ndarray] = field(default=None, repr=False)

    def release(self) -> None:
        """Drop the pixels."""
        self.image = None


def rotation_of(path: str) -> float:
    """Container rotation in degrees, via OpenCV. 0.0 when unknown."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return 0.0
        return float(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0)
    finally:
        cap.release()


def rotator(degrees: float) -> Optional[Callable[[Any], Any]]:
    """What turns a decoded image upright for a container rotation of `degrees`,
    or None when it is already upright (or the angle is not a quarter turn)."""
    turn = _ROTATIONS.get(int(degrees or 0))
    if turn is None:
        return None
    import cv2
    code = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE)[turn]
    return lambda image: cv2.rotate(image, code)


def read_frames(media: Media,
                keep: Callable[[float], bool],
                rotation: Optional[float] = None) -> Iterator[Frame]:
    """Yield only the frames `keep` accepts, with pixels attached. `keep` is handed
    a `media_ts`; a declined frame is never converted.
    """
    if not media.has_video:
        raise UnreadableSource(f"{media.path} has no video stream")

    try:
        container = av.open(media.path)
    except Exception as exc:                             # noqa: BLE001
        raise UnreadableSource(
            f"cannot open {media.path} ({type(exc).__name__})") from None

    rotate = rotator(rotation if rotation is not None else rotation_of(media.path))

    index = 0
    try:
        stream = container.streams.video[0]
        # Required for decode speed.
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        fps = media.video.rate or 30.0

        for av_frame in container.decode(video=0):
            # Exact rational arithmetic; the float is for comparison.
            if av_frame.pts is not None and time_base is not None:
                pts = av_frame.pts
                media_ts = float(pts * time_base)
            else:
                pts = None
                media_ts = index / fps

            if keep(media_ts):
                image = av_frame.to_ndarray(format="bgr24")   # the 6.2 ms
                if rotate is not None:
                    image = rotate(image)
                yield Frame(index=index, media_ts=media_ts, pts=pts, image=image)
            index += 1
    finally:
        container.close()


__all__ = ["Frame", "UnreadableSource", "read_frames", "rotation_of", "rotator"]
