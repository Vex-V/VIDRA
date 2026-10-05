"""Object detectors: where the people, objects or text are in a frame. The
samplers decide what to do about it.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from vidra.shared.config import paths
from vidra.shared.models.devices import default_device

# Weights live under the weights folder rather than wherever a command ran.
# `weights/clip/ViT-B-32.pt` is the CLIP text encoder YOLO-World embeds its
# vocabulary with (separate from the HuggingFace CLIP the scene sampler uses).



def weight_path(name: str) -> str:
    """The local copy of a weight file if there is one, else the bare name (which
    ultralytics downloads).
    """
    local = paths.weights_root() / name
    return str(local) if local.exists() else name


@dataclass(frozen=True)
class Detection:
    """One detected object in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = "person"

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def crop(self, image: np.ndarray, pad: float = 0.08) -> Optional[np.ndarray]:
        """Cut this detection out of the frame, padded by `pad` for context."""
        h, w = image.shape[:2]
        px, py = self.width * pad, self.height * pad
        x1 = int(max(0, round(self.x1 - px)))
        y1 = int(max(0, round(self.y1 - py)))
        x2 = int(min(w, round(self.x2 + px)))
        y2 = int(min(h, round(self.y2 + py)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return image[y1:y2, x1:x2]


class ObjectDetector(ABC):
    """Finds regions of interest in a frame."""

    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]: ...

    def config(self) -> dict:
        return {"name": self.name}


class YoloPersonDetector(ObjectDetector):
    """Ultralytics YOLO restricted to the person class. `min_height` drops people
    too small to crop usefully.
    """

    name = "yolo"

    def __init__(
        self,
        model: str = "yolo11n.pt",
        confidence: float = 0.35,
        device: Optional[str] = None,
        min_height: int = 64,
        imgsz: int = 640,
    ) -> None:
        os.environ.setdefault("YOLO_VERBOSE", "False")
        from ultralytics import YOLO

        self.model_name = model
        self.confidence = confidence
        self.min_height = min_height
        self.imgsz = imgsz
        self.device = device or default_device()
        self._model = YOLO(weight_path(model))
        self._model.to(self.device)

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self._model.predict(
            image,
            classes=[0],  # COCO class 0 is person
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        out = []
        for (x1, y1, x2, y2), c in zip(xyxy, conf):
            det = Detection(float(x1), float(y1), float(x2), float(y2), float(c))
            if det.height >= self.min_height:
                out.append(det)
        return out

    def config(self) -> dict:
        return {
            "name": self.name,
            "model": self.model_name,
            "confidence": self.confidence,
            "min_height": self.min_height,
            "imgsz": self.imgsz,
            "device": self.device,
        }


class OpenVocabDetector(ObjectDetector):
    """YOLO-World: classes given as text rather than fixed at training time.
    Confidence runs lower than a closed-vocabulary detector (0.15-0.40 is normal).
    """

    name = "openvocab"

    DEFAULT_VOCAB = (
        "shopping bag",
        "plastic bag",
        "shopping cart",
        "shopping basket",
        "cardboard box",
        "bottle",
        "grocery item",
        "cash register",
    )

    def __init__(
        self,
        vocabulary: Optional[Sequence[str]] = None,
        model: str = "yolov8s-worldv2.pt",
        confidence: float = 0.20,
        device: Optional[str] = None,
        min_height: int = 0,
        imgsz: int = 640,
    ) -> None:
        os.environ.setdefault("YOLO_VERBOSE", "False")
        from ultralytics import YOLOWorld

        self.vocabulary = list(vocabulary or self.DEFAULT_VOCAB)
        self.model_name = model
        self.confidence = confidence
        self.min_height = min_height
        self.imgsz = imgsz
        self.device = device or default_device()
        self._model = YOLOWorld(weight_path(model))
        self._model.set_classes(self.vocabulary)
        self._model.to(self.device)

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self._model.predict(
            image,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        out = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            label = self.vocabulary[k] if k < len(self.vocabulary) else str(k)
            det = Detection(float(x1), float(y1), float(x2), float(y2), float(c), label)
            if det.height >= self.min_height:
                out.append(det)
        return out

    def config(self) -> dict:
        return {
            "name": self.name,
            "model": self.model_name,
            "vocabulary": self.vocabulary,
            "confidence": self.confidence,
            "min_height": self.min_height,
            "device": self.device,
        }


class TextRegionDetector(ObjectDetector):
    """EasyOCR's detector only: where text is, not what it says. Every region is
    weighted equally.
    """

    name = "text"

    def __init__(
        self,
        languages: Sequence[str] = ("en",),
        gpu: Optional[bool] = None,
        min_height: int = 0,
        low_text: float = 0.4,
        # Above EasyOCR's 0.7 default, so graphics are not taken for text.
        text_threshold: float = 0.85,
        link_threshold: float = 0.4,
        canvas_size: int = 1280,
        mag_ratio: float = 1.0,
    ) -> None:
        import easyocr

        self.languages = list(languages)
        # EasyOCR picks cuda, then mps, itself when told there is a GPU.
        self.gpu = default_device() != "cpu" if gpu is None else gpu
        self.min_height = min_height
        self.low_text = low_text
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)

    def detect(self, image: np.ndarray) -> list[Detection]:
        horizontal, free = self._reader.detect(
            image,
            low_text=self.low_text,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            canvas_size=self.canvas_size,
            mag_ratio=self.mag_ratio,
        )
        out: list[Detection] = []

        def keep(det: Detection) -> None:
            if det.height >= self.min_height and det.area > 0:
                out.append(det)

        # detect() returns a list-per-image; unwrap it.
        for box in (horizontal[0] if horizontal else []):
            x_min, x_max, y_min, y_max = box
            keep(Detection(float(x_min), float(y_min), float(x_max), float(y_max), 1.0, "text"))
        for quad in (free[0] if free else []):
            pts = np.array(quad, dtype=np.float64)
            keep(Detection(
                float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()), 1.0, "text",
            ))
        return out

    def config(self) -> dict:
        return {
            "name": self.name,
            "languages": self.languages,
            "gpu": self.gpu,
            "min_height": self.min_height,
            "low_text": self.low_text,
            "text_threshold": self.text_threshold,
        }


