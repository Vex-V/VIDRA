"""Voiceprints from a pyannote speaker-embedding model loaded into this process.

    LocalVoiceEmbedder()            pyannote/wespeaker-voxceleb-resnet34-LM, 256-d

The default is the model inside pyannote's own diarization pipeline, so a
machine that has diarized has it in the Hugging Face cache already; it is not
gated. A clip is embedded whole. Weights stay in Hugging Face's cache
(`HF_HOME`), as the diarizer's do. A model is loaded on first use, once per
(model, device) per process.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Sequence

from ...config import settings
from ..base import VoiceEmbedder
from ..devices import default_device
from . import EmbedderUnavailable

#: The model when none is named.
DEFAULT_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

#: Widths known without loading the model, so the key (and a resume check)
#: costs no load. Any other model is loaded to read its width.
KNOWN_DIMS = {DEFAULT_MODEL: 256}

_MODELS: dict[tuple[str, str], Any] = {}
_LOCK = threading.Lock()


def _load(model: str, device: str) -> Any:
    with _LOCK:
        if (model, device) not in _MODELS:
            try:
                import torch
                from pyannote.audio import Inference, Model

                loaded = Model.from_pretrained(model, token=settings.hf_token())
                if loaded is None:                       # pyannote's refused download
                    raise RuntimeError("the download was refused")
                _MODELS[(model, device)] = (
                    loaded, Inference(loaded, window="whole", device=torch.device(device)))
            except ModuleNotFoundError as exc:
                raise EmbedderUnavailable(
                    f"{exc}: voiceprints need pyannote.audio (pip install \"vidra[audio]\")") from None
            except Exception as exc:                     # noqa: BLE001
                raise EmbedderUnavailable(f"could not load {model!r}: {exc}") from None
        return _MODELS[(model, device)]


class LocalVoiceEmbedder(VoiceEmbedder):
    """`device` defaults to cuda, then mps, then the CPU."""

    concurrency = 1
    rate = 16000

    def __init__(self, model: str = DEFAULT_MODEL, device: Optional[str] = None) -> None:
        self.model_id = model or DEFAULT_MODEL
        self.device = device or default_device()

    def __repr__(self) -> str:
        return f"LocalVoiceEmbedder({self.model_id!r})"

    @property
    def dims(self) -> int:
        if self.model_id in KNOWN_DIMS:
            return KNOWN_DIMS[self.model_id]
        return int(_load(self.model_id, self.device)[0].dimension)

    @property
    def key(self) -> str:
        return f"pyannote:{self.model_id}:{self.dims}"

    def embed_clips(self, clips: Sequence[Any]) -> list[list[float]]:
        import numpy as np
        import torch

        _, inference = _load(self.model_id, self.device)
        out = []
        for clip in clips:
            samples = np.ascontiguousarray(clip, dtype=np.float32)
            vector = inference({"waveform": torch.from_numpy(samples).unsqueeze(0),
                                "sample_rate": self.rate})
            out.append([float(v) for v in np.asarray(vector).ravel()])
        return out


__all__ = ["DEFAULT_MODEL", "KNOWN_DIMS", "LocalVoiceEmbedder"]
