"""Sampling on which of a named vocabulary of objects is in shot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from .detection import DetectionChangeSampler

if TYPE_CHECKING:
    from .perception.detectors import ObjectDetector


class ObjectChangeSampler(DetectionChangeSampler):
    """Objects, compared by presence and position; no embedder.

    Detection uses an open vocabulary (YOLO-World) set by `vocabulary`. Matching
    is class-aware, so a cart never counts as a bag. `send` can show the model
    the largest detections cut out, beside the frame or instead of it.
    """

    name = "objects"

    def __init__(
        self,
        detector: Optional["ObjectDetector"] = None,
        vocabulary: Optional[Sequence[str]] = None,
        threshold: float = 0.30,
        class_aware: bool = True,
        confidence: float = 0.30,
        metric: str = "proximity",
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
        send: str = "frame",
        crops: int = 3,
        crop_size: Optional[int] = None,
        crop_margin: float = 0.1,
    ) -> None:
        from .perception.descriptors import BoxGeometryDescriptor

        if detector is None:
            from .perception.detectors import OpenVocabDetector

            detector = OpenVocabDetector(vocabulary=vocabulary, confidence=confidence)
        super().__init__(
            detector=detector,
            descriptor=BoxGeometryDescriptor(class_aware=class_aware, metric=metric),
            threshold=threshold,
            min_interval_s=min_interval_s,
            max_per_chunk=max_per_chunk,
            sampler_id=sampler_id,
            prompts=prompts,
            send=send,
            crops=crops,
            crop_size=crop_size,
            crop_margin=crop_margin,
        )


