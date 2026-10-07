"""Detect, describe, compare: the shape every detection sampler shares. The
descriptor (see `descriptors`) decides what is compared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from ..reader import Frame
from .base import Sampler
from vidra.shared.reporting.errors import Refused

if TYPE_CHECKING:
    from .perception.descriptors import RegionDescriptor
    from .perception.detectors import ObjectDetector


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
    ) -> None:
        super().__init__(min_interval_s, max_per_chunk, sampler_id, prompts)
        if not 0.0 <= threshold <= 1.0:
            raise Refused("threshold must be a similarity in [0, 1]")
        if detector is None or descriptor is None:
            raise Refused("detector and descriptor are required")
        self.detector = detector
        self.descriptor = descriptor
        self.threshold = threshold
        self._reference: Optional[object] = None
        self._last_score: Optional[float] = None

    def on_reset(self, chunk_id: int) -> None:
        self._reference = None
        self._last_score = None

    def last_score(self) -> Optional[float]:
        return self._last_score

    def describe(self, frame: Frame):
        if frame.image is None:
            raise Refused(f"{type(self).__name__} needs pixels; frame.image is None")
        detections = self.detector.detect(frame.image)
        return self.descriptor.describe(frame.image, detections)

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
        }


