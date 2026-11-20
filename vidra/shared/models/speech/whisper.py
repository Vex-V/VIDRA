"""faster-whisper over a whole track, in this process.

    WhisperTranscriber("small")              cuda when CTranslate2 sees one, else the CPU
    WhisperTranscriber("medium", language="en", device="cpu")

Making one loads nothing: the model is loaded on the first `transcribe`, once
per (model, device, compute type) per process, so one object -- or several
alike -- serves every video. Requires `vidra[audio]`.
"""

from __future__ import annotations

import threading
from importlib.util import find_spec
from typing import Any, Optional

from ...reporting.errors import ModelUnavailable, Refused
from ..base import Transcriber
from . import Segment, Transcript, Word, cuda

#: The default model.
DEFAULT_MODEL = "small"

#: Compute type by device.
DEFAULT_COMPUTE = {"cuda": "float16", "cpu": "int8"}

_MODELS: dict[tuple[str, str, str], Any] = {}
_LOCK = threading.Lock()


def _device() -> str:
    """cuda when CTranslate2 sees a GPU, else the CPU. Asked of CTranslate2
    itself, not torch, which faster-whisper does not need; never mps, which
    CTranslate2 does not have."""
    cuda.enable()
    import ctranslate2
    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


class WhisperTranscriber(Transcriber):
    """Whole track in, segments with word timestamps out. `vad_filter` drops
    long silences before decoding: on the reference video it is the difference
    between 10 segments and 0."""

    concurrency = 1

    def __init__(self, model: str = DEFAULT_MODEL,
                 device: Optional[str] = None,
                 compute_type: Optional[str] = None,
                 language: Optional[str] = None,
                 vad_filter: bool = True,
                 model_obj: Any = None) -> None:
        if device == "mps":
            raise Refused("Whisper (CTranslate2) runs on cuda or cpu, not mps; leave "
                          "device unset and it uses the CPU on a Mac")
        self.model_name = model or DEFAULT_MODEL
        self.language = language
        self.vad_filter = vad_filter
        given = "given" if model_obj is not None else None
        self._device = device or given
        self._compute_type = compute_type or given
        self._model = model_obj

    @property
    def key(self) -> str:
        return f"whisper:{self.model_name}"

    def __repr__(self) -> str:
        return f"WhisperTranscriber({self.model_name!r})"

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = _device()
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type or DEFAULT_COMPUTE.get(self.device, "int8")

    def problems(self) -> list[str]:
        if self._model is None and find_spec("faster_whisper") is None:
            return ['Whisper needs faster-whisper: pip install "vidra[audio]"']
        return []

    def _loaded(self) -> Any:
        if self._model is not None:
            return self._model
        device, compute = self.device, self.compute_type
        with _LOCK:
            if (self.model_name, device, compute) not in _MODELS:
                # Seeing a GPU says nothing about CTranslate2's CUDA libraries,
                # and without them it fails at the first encode, after the load.
                lost = cuda.missing() if device == "cuda" else []
                if lost:
                    raise ModelUnavailable(
                        f"Whisper on cuda needs CUDA 12's cuBLAS and cuDNN 9, and "
                        f"{', '.join(lost)} cannot be loaded: pip install "
                        f"nvidia-cublas-cu12 \"nvidia-cudnn-cu12>=9,<10\" (the [audio] "
                        f"extra does), or pass device='cpu'")
                from faster_whisper import WhisperModel
                try:
                    _MODELS[(self.model_name, device, compute)] = WhisperModel(
                        self.model_name, device=device, compute_type=compute)
                except Exception as exc:                 # noqa: BLE001
                    raise ModelUnavailable(
                        f"could not load Whisper {self.model_name!r} on {device} "
                        f"({compute}): {exc}") from None
            return _MODELS[(self.model_name, device, compute)]

    def transcribe(self, samples: Any) -> Transcript:
        segments, info = self._loaded().transcribe(
            samples, language=self.language,
            word_timestamps=True, vad_filter=self.vad_filter)
        out: list[Segment] = []
        for seg in segments:                  # a generator: the work is here
            out.append(Segment(
                start=float(seg.start), end=float(seg.end), text=seg.text,
                words=[Word(float(w.start), float(w.end), w.word)
                       for w in (seg.words or [])]))
        return Transcript(segments=out, language=info.language,
                          language_probability=float(info.language_probability))

    def config(self) -> dict[str, Any]:
        return {"transcriber": "whisper", "model": self.model_name,
                "device": self.device, "compute_type": self.compute_type,
                "language": self.language, "vad_filter": self.vad_filter}


__all__ = ["DEFAULT_MODEL", "WhisperTranscriber"]
