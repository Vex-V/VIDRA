"""Sampling on who is in shot: how many, where, and how that changes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from .detection import DetectionChangeSampler

if TYPE_CHECKING:
    from .perception.detectors import ObjectDetector
    from .perception.embedders import FrameEmbedder


class PersonChangeSampler(DetectionChangeSampler):
    """People, compared by appearance.

    YOLO locates each person, CLIP embeds each person crop (upsampled to 224),
    and the crop embeddings are compared; no identity tracking. The score is the
    minimum over people, so crowded frames sample more: `min_interval_s` limits
    that. Default threshold 0.83. `crop_pad` pads the crops CLIP compares;
    `send`, `crops`, `crop_size` and `crop_margin` shape the crops the model is
    shown.
    """

    name = "yolo"

    def __init__(
        self,
        detector: Optional["ObjectDetector"] = None,
        embedder: Optional["FrameEmbedder"] = None,
        threshold: float = 0.83,
        crop_pad: float = 0.08,
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
        send: str = "frame",
        crops: int = 3,
        crop_size: Optional[int] = None,
        crop_margin: float = 0.1,
    ) -> None:
        from .perception.descriptors import CropEmbeddingDescriptor

        if detector is None:
            from .perception.detectors import YoloPersonDetector

            detector = YoloPersonDetector()
        super().__init__(
            detector=detector,
            descriptor=CropEmbeddingDescriptor(embedder=embedder, crop_pad=crop_pad),
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


