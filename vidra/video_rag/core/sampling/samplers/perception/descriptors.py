"""How a change sampler compares two frames' detections.

    people   a CLIP embedding of each crop (appearance)
    objects  box geometry (presence and position)
    text     the frame masked to where text was found, compared as a whole

Each descriptor returns an opaque per-frame description and a similarity
matrix between two of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import cv2
import numpy as np

from .detectors import Detection
from vidra.shared.reporting.errors import UnknownOption


class RegionDescriptor(ABC):
    """Describes detected regions and compares two frames' worth of them."""

    name: str = "base"

    @abstractmethod
    def describe(self, image: np.ndarray, detections: Sequence[Detection]) -> Any:
        """Turn detections into whatever this descriptor compares."""

    @abstractmethod
    def count(self, described: Any) -> int: ...

    @abstractmethod
    def similarity(self, current: Any, reference: Any) -> np.ndarray:
        """(n_current, n_reference) similarities in [0, 1]."""

    def config(self) -> dict:
        return {"name": self.name}


class CropEmbeddingDescriptor(RegionDescriptor):
    """Appearance, via a CLIP embedding of each crop (upsampled to 224)."""

    name = "crop_embedding"

    def __init__(self, embedder=None, crop_pad: float = 0.08) -> None:
        if embedder is None:
            from .embedders import CLIPEmbedder

            embedder = CLIPEmbedder()
        self.embedder = embedder
        self.crop_pad = crop_pad

    def describe(self, image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        crops = [c for d in detections if (c := d.crop(image, self.crop_pad)) is not None]
        if not crops:
            return np.zeros((0, 1), dtype=np.float32)
        return self.embedder.embed(crops)

    def count(self, described: np.ndarray) -> int:
        return len(described)

    def similarity(self, current: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return current @ reference.T

    def config(self) -> dict:
        return {
            "name": self.name,
            "crop_pad": self.crop_pad,
            "embedder": self.embedder.config(),
        }


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (n, 4) box arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _proximity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Graded box similarity: centre distance in box-sized units, times size
    agreement. Identical boxes score 1.0, a box displaced by its own diagonal 0.5.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    centre_a = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], axis=1)
    centre_b = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
    distance = np.linalg.norm(centre_a[:, None, :] - centre_b[None, :, :], axis=2)
    diag_a = np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])
    diag_b = np.hypot(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
    scale = np.maximum(0.5 * (diag_a[:, None] + diag_b[None, :]), 1e-9)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    size_agreement = np.minimum(area_a[:, None], area_b[None, :]) / np.maximum(
        np.maximum(area_a[:, None], area_b[None, :]), 1e-9
    )
    return (1.0 / (1.0 + distance / scale)) * size_agreement


class BoxGeometryDescriptor(RegionDescriptor):
    """Presence and position only; no pixels compared. Optionally two regions must
    share a class to match.
    """

    name = "box_geometry"

    def __init__(self, class_aware: bool = True, metric: str = "proximity") -> None:
        if metric not in ("proximity", "iou"):
            raise UnknownOption("metric must be 'proximity' or 'iou'")
        self.class_aware = class_aware
        self.metric = metric

    def describe(self, image: np.ndarray, detections: Sequence[Detection]) -> dict:
        if not detections:
            return {"boxes": np.zeros((0, 4)), "labels": []}
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float64)
        return {"boxes": boxes, "labels": [d.label for d in detections]}

    def count(self, described: dict) -> int:
        return len(described["boxes"])

    def similarity(self, current: dict, reference: dict) -> np.ndarray:
        fn = _proximity_matrix if self.metric == "proximity" else _iou_matrix
        sim = fn(current["boxes"], reference["boxes"])
        if self.class_aware and sim.size:
            same = np.array(
                [[c == r for r in reference["labels"]] for c in current["labels"]]
            )
            sim = sim * same
        return sim

    def config(self) -> dict:
        return {"name": self.name, "class_aware": self.class_aware, "metric": self.metric}


class TextLayoutDescriptor(RegionDescriptor):
    """One descriptor for the whole frame's text: the frame masked to every text
    region.
    """

    name = "text_layout"

    def __init__(self, grid: int = 128, dilate: int = 2) -> None:
        self.grid = grid
        self.dilate = dilate

    def describe(self, image: np.ndarray, detections: Sequence[Detection]) -> dict:
        if not detections:
            return {"vector": np.zeros((0, self.grid * self.grid), dtype=np.float32)}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        h, w = mask.shape
        for d in detections:
            pad_x = d.width * 0.02 * self.dilate
            pad_y = d.height * 0.02 * self.dilate
            x1 = int(max(0, d.x1 - pad_x))
            y1 = int(max(0, d.y1 - pad_y))
            x2 = int(min(w, d.x2 + pad_x))
            y2 = int(min(h, d.y2 + pad_y))
            mask[y1:y2, x1:x2] = 1
        masked = (gray * mask).astype(np.float32)
        small = cv2.resize(masked, (self.grid, self.grid), interpolation=cv2.INTER_AREA)
        vector = small.reshape(-1)
        vector -= vector.mean()
        norm = np.linalg.norm(vector)
        vector = vector / norm if norm > 1e-6 else np.zeros_like(vector)
        return {"vector": vector[None, :]}

    def count(self, described: dict) -> int:
        return len(described["vector"])

    def similarity(self, current: dict, reference: dict) -> np.ndarray:
        if current["vector"].size == 0 or reference["vector"].size == 0:
            return np.zeros((len(current["vector"]), len(reference["vector"])))
        return np.clip(current["vector"] @ reference["vector"].T, 0.0, 1.0)

    def config(self) -> dict:
        return {"name": self.name, "grid": self.grid, "dilate": self.dilate}


