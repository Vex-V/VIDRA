"""faster-whisper over a whole track. Imported only when asked for by name."""

from __future__ import annotations

from typing import Any, Optional

from . import cuda
from ..models import ModelUnavailable, Segment, Transcript, Word
from ..source import Track

#: The default model.
DEFAULT_MODEL = "small"

#: Compute type by device.
DEFAULT_COMPUTE = {"cuda": "float16", "cpu": "int8"}


class WhisperTranscriber:
    """A `Transcriber`. Whole track in, segments with word timestamps out."""

    name = "whisper"

    def __init__(self, model: str = DEFAULT_MODEL,
                 device: Optional[str] = None,
                 compute_type: Optional[str] = None,
                 language: Optional[str] = None,
                 vad_filter: bool = True,
                 model_obj: Any = None) -> None:
        self.model_name = model
        self.language = language
        self.vad_filter = vad_filter

        if model_obj is not None:
            self.device = self.compute_type = "given"
            self._model = model_obj
            return

        # Preload the CUDA libraries before CTranslate2 asks for them by name (Windows).
        cuda.enable()
        import torch
        from faster_whisper import WhisperModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or DEFAULT_COMPUTE.get(self.device, "int8")
        try:
            self._model = WhisperModel(model, device=self.device,
                                       compute_type=self.compute_type)
        except Exception as exc:                         # noqa: BLE001
            raise ModelUnavailable(
                f"could not load Whisper {model!r} on {self.device} "
                f"({self.compute_type}): {exc}") from None

    def transcribe(self, track: Track) -> Transcript:
        segments, info = self._model.transcribe(
            track.samples, language=self.language,
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
        return {"transcriber": self.name, "model": self.model_name,
                "device": self.device, "compute_type": self.compute_type,
                "language": self.language, "vad_filter": self.vad_filter}


__all__ = ["DEFAULT_MODEL", "WhisperTranscriber"]
