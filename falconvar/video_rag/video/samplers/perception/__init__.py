"""Perception: detectors and descriptors that turn pixels into something
comparable.

A sampler decides what to keep; these find regions and describe them.
Nothing here imports a sampler. Every model weight (YOLO, CLIP, EasyOCR) is
loaded here, lazily.
"""

from .descriptors import (BoxGeometryDescriptor, CropEmbeddingDescriptor,
                          RegionDescriptor, TextLayoutDescriptor)
from .detectors import (Detection, ObjectDetector, OpenVocabDetector,
                        TextRegionDetector, YoloPersonDetector)
from .embedders import CLIPEmbedder, FrameEmbedder

__all__ = [
    "BoxGeometryDescriptor",
    "CLIPEmbedder",
    "CropEmbeddingDescriptor",
    "Detection",
    "FrameEmbedder",
    "ObjectDetector",
    "OpenVocabDetector",
    "RegionDescriptor",
    "TextLayoutDescriptor",
    "TextRegionDetector",
    "YoloPersonDetector",
]
