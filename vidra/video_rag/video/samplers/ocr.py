"""Sampling on the writing in shot changing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from .detection import DetectionChangeSampler

if TYPE_CHECKING:
    from .perception.detectors import ObjectDetector


class TextChangeSampler(DetectionChangeSampler):
    """Text, compared by where it is and what it looks like.

    EasyOCR finds the regions; nothing is read. `TextLayoutDescriptor` masks the
    frame to every text region and compares the result as a whole. Meant for
    screens and slides.
    """

    name = "text"

    def __init__(
        self,
        detector: Optional["ObjectDetector"] = None,
        threshold: float = 0.92,
        grid: int = 128,
        languages: Sequence[str] = ("en",),
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
    ) -> None:
        from .perception.descriptors import TextLayoutDescriptor

        if detector is None:
            from .perception.detectors import TextRegionDetector

            detector = TextRegionDetector(languages=languages)
        super().__init__(
            detector=detector,
            descriptor=TextLayoutDescriptor(grid=grid),
            threshold=threshold,
            min_interval_s=min_interval_s,
            max_per_chunk=max_per_chunk,
            sampler_id=sampler_id,
            prompts=prompts,
        )
