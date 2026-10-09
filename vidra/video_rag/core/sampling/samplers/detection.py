"""Detect, describe, compare: the shape every detection sampler shares. The
descriptor (see `descriptors`) decides what is compared.

`send` says what the model is shown for a kept frame: the frame (`frame`),
the frame and then its `crops` largest detections (`both`), or the
detections alone (`crops`). Each is cut out with `crop_margin` around it and
enlarged so its shorter side reaches `crop_size` (at most `MAX_SCALE` times).
A kept frame with nothing detected is shown as itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from ..reader import Frame
from .base import Sampler, View
from vidra.shared.reporting.errors import Refused

if TYPE_CHECKING:
    from .perception.descriptors import RegionDescriptor
    from .perception.detectors import Detection, ObjectDetector

#: The most a crop is enlarged.
MAX_SCALE = 4.0
#: What a kept frame can be shown as.
SEND = ("frame", "both", "crops")


class DetectionChangeSampler(Sampler):
    """Samples when detected regions stop looking like the last kept frame.

    The score is the weakest best match, one direction only: for each region
    visible now, its best match in the reference frame; the worst of those is the
    score. Regions leaving do not trigger a sample; regions arriving or changing
    do. Region count is never a trigger.
    """

    name = "detection"

    def __init__(
        self,
        detector: Optional["ObjectDetector"] = None,
        descriptor: Optional["RegionDescriptor"] = None,
        threshold: float = 0.83,
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
        send: str = "frame",
        crops: int = 3,
        crop_size: Optional[int] = None,
        crop_margin: float = 0.1,
    ) -> None:
        super().__init__(min_interval_s, max_per_chunk, sampler_id, prompts)
        if not 0.0 <= threshold <= 1.0:
            raise Refused("threshold must be a similarity in [0, 1]")
        if detector is None or descriptor is None:
            raise Refused("detector and descriptor are required")
        if send not in SEND:
            from vidra.shared.reporting.errors import UnknownOption
            raise UnknownOption(f"send must be one of {', '.join(SEND)}, not {send!r}")
        if send == "frame" and (crops != 3 or crop_size is not None or crop_margin != 0.1):
            raise Refused("crops, crop_size and crop_margin shape crops, and "
                          "send='frame' makes none; set send='both' or 'crops'")
        if crops < 1:
            raise Refused(f"crops must be at least 1, not {crops}")
        if crop_size is not None and crop_size < 1:
            raise Refused(f"crop_size is pixels, at least 1, not {crop_size}")
        if crop_margin < 0:
            raise Refused(f"crop_margin must be >= 0, not {crop_margin}")
        self.detector = detector
        self.descriptor = descriptor
        self.threshold = threshold
        self.send = send
        self.crops = crops
        self.crop_size = crop_size
        self.crop_margin = crop_margin
        self._reference: Optional[object] = None
        self._last_score: Optional[float] = None
        #: The last frame detected on, and what was found: a kept frame's crops
        #: come from the detection that kept it.
        self._detected: tuple[Optional[int], list["Detection"]] = (None, [])

    def on_reset(self, chunk_id: int) -> None:
        self._reference = None
        self._last_score = None

    def last_score(self) -> Optional[float]:
        return self._last_score

    def describe(self, frame: Frame):
        if frame.image is None:
            raise Refused(f"{type(self).__name__} needs pixels; frame.image is None")
        detections = self.detector.detect(frame.image)
        self._detected = (frame.index, detections)
        return self.descriptor.describe(frame.image, detections)

    def views(self, frame: Frame) -> list[View]:
        """As `send` says: the frame, then or instead up to `crops` of its largest
        detections, left to right. The frame alone when nothing was detected."""
        if self.send == "frame":
            return [View.FRAME]
        index, found = self._detected
        if index != frame.index:
            found = self.detector.detect(frame.image)
        chosen = sorted(sorted(found, key=lambda d: -d.area)[:self.crops],
                        key=lambda d: (d.x1, d.y1))
        views = [View.FRAME] if self.send == "both" else []
        for number, detection in enumerate(chosen, start=1):
            cut = detection.crop(frame.image, self.crop_margin)
            if cut is None:
                continue
            image, scale = _enlarged(cut, self.crop_size)
            label = (f"{detection.label} {number} of {len(chosen)} from the left"
                     + (f", enlarged {scale:.1f}x" if scale > 1 else ""))
            views.append(View(image, label, {
                "box": [round(detection.x1), round(detection.y1),
                        round(detection.x2), round(detection.y2)],
                "detected": detection.label,
                "confidence": round(float(detection.confidence), 3),
                "scale": round(scale, 3)}))
        return views or [View.FRAME]

    def compare(self, current, reference) -> Optional[float]:
        """The weakest best match, one-directional."""
        if self.descriptor.count(current) == 0:
            return 1.0                      # nothing in shot; reference untouched
        if self.descriptor.count(reference) == 0:
            return 0.0                      # something appeared where there was nothing
        similarity = self.descriptor.similarity(current, reference)
        return float(similarity.max(axis=1).min())

    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        current = self.describe(frame)
        count = self.descriptor.count(current)

        if self._reference is None:
            self._last_score = None
            self._reference = current
            return True

        if count == 0:
            # Nothing in shot: the reference is kept.
            self._last_score = 1.0
            return False

        if self.descriptor.count(self._reference) == 0:
            # Something has appeared where there was nothing.
            self._last_score = 0.0
            self._reference = current
            return True

        similarity = self.descriptor.similarity(current, self._reference)
        score = float(similarity.max(axis=1).min())
        self._last_score = score

        keep = score < self.threshold
        if keep:
            # Only a kept frame moves the reference, so change accumulates.
            self._reference = current
        return keep

    def config(self) -> dict:
        return {
            **self._base_config(),
            "threshold": self.threshold,
            "detector": self.detector.config(),
            "descriptor": self.descriptor.config(),
            # Only when crops are made.
            **({"send": self.send,
                "crops": {"count": self.crops, "size": self.crop_size,
                          "margin": self.crop_margin}} if self.send != "frame" else {}),
        }


def _enlarged(image: Any, size: Optional[int]) -> tuple[Any, float]:
    """`image` scaled up so its shorter side reaches `size`, at most `MAX_SCALE`
    times, and the scale used; never scaled down."""
    import numpy as np

    short = min(image.shape[:2])
    scale = min(MAX_SCALE, size / short) if size else 1.0
    if scale <= 1.0:
        return np.ascontiguousarray(image), 1.0
    import cv2
    return cv2.resize(image, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC), scale


