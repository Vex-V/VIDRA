"""Object detectors: where the people, objects or text are in a frame. The
samplers decide what to do about it.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from vidra.shared.config import paths
from vidra.shared.models.devices import default_device
from vidra.shared.reporting.errors import ModelUnavailable, Refused

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
    """Where text is, not what it says: PP-OCRv3's DB detector on OpenCV's dnn
    module (`cv2.dnn.TextDetectionModel_DB`), on the CPU. The frame is scaled
    so its longer side is `size`, the shorter rounded to a multiple of 32.
    `binary_threshold` decides which pixels are text, `polygon_threshold`
    which regions are kept, `unclip_ratio` how far a region is grown. Each
    region is returned as its bounding box, every one weighted equally. The
    model is OpenCV's model zoo English detector (2.4 MB), downloaded to the
    weights folder on first use; it found Greek as well.
    """

    #: Where the default model is fetched from on first use.
    URL = ("https://huggingface.co/opencv/text_detection_ppocr/resolve/main/"
           "text_detection_en_ppocrv3_2023may.onnx")
    MODEL = "ocr/text_detection_en_ppocrv3_2023may.onnx"

    name = "text"

    def __init__(
        self,
        model: str = MODEL,
        size: int = 1280,
        binary_threshold: float = 0.3,
        polygon_threshold: float = 0.5,
        unclip_ratio: float = 2.0,
        max_candidates: int = 200,
        min_height: int = 0,
    ) -> None:
        import threading

        import cv2

        if size < 32 or size % 32:
            raise Refused(f"size must be a multiple of 32, not {size}")
        for label, value in (("binary_threshold", binary_threshold),
                             ("polygon_threshold", polygon_threshold)):
            if not 0.0 < value < 1.0:
                raise Refused(f"{label} must be in (0, 1), not {value}")
        self.model_name = model
        self.size = size
        self.binary_threshold = binary_threshold
        self.polygon_threshold = polygon_threshold
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates
        self.min_height = min_height
        path = _model_file(model, self.URL if model == self.MODEL else None)
        try:
            net = cv2.dnn.readNetFromONNX(str(path), cv2.dnn.ENGINE_NEW)
        except cv2.error as exc:
            raise ModelUnavailable(f"OpenCV could not load {path.name}: {exc}") from None
        self._model = cv2.dnn.TextDetectionModel_DB(net)
        self._model.setBinaryThreshold(binary_threshold)
        self._model.setPolygonThreshold(polygon_threshold)
        self._model.setUnclipRatio(unclip_ratio)
        self._model.setMaxCandidates(max_candidates)
        # A copied sampler shares this model, and a network keeps its input
        # between calls.
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        scale = self.size / max(height, width)
        shape = (max(32, round(width * scale / 32) * 32),
                 max(32, round(height * scale / 32) * 32))
        with self._lock:
            self._model.setInputParams(1.0 / 255.0, shape,
                                       (122.67891434, 116.66876762, 104.00698793))
            quads, scores = self._model.detect(image)
        out = []
        for quad, score in zip(quads, scores):
            pts = np.asarray(quad, dtype=np.float64)
            det = Detection(float(max(0.0, pts[:, 0].min())), float(max(0.0, pts[:, 1].min())),
                            float(min(width, pts[:, 0].max())), float(min(height, pts[:, 1].max())),
                            float(score), "text")
            if det.height >= self.min_height and det.area > 0:
                out.append(det)
        return out

    def config(self) -> dict:
        return {
            "name": self.name,
            "model": self.model_name,
            "size": self.size,
            "binary_threshold": self.binary_threshold,
            "polygon_threshold": self.polygon_threshold,
            "unclip_ratio": self.unclip_ratio,
            "max_candidates": self.max_candidates,
            "min_height": self.min_height,
        }


def _model_file(name: str, url: Optional[str]) -> Path:
    """`name` as given when absolute, else under the weights folder; downloaded
    from `url` when missing, through a partial file so an interrupted download
    is never loaded as a model."""
    path = Path(name)
    if not path.is_absolute():
        path = paths.weights_root() / path
    if not path.exists() and url is not None:
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        try:
            urllib.request.urlretrieve(url, partial)
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise ModelUnavailable(f"could not download {url}: {exc}") from None
        partial.replace(path)
    if not path.exists():
        raise ModelUnavailable(f"no model file at {path}: give the path to a DB "
                               "text detector exported to ONNX")
    return path


