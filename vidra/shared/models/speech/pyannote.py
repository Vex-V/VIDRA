"""Speaker turns over a whole track, via pyannote (4.x), in this process.

    PyannoteDiarizer()                        pyannote/speaker-diarization-3.1

Uses `.exclusive_speaker_diarization`, with overlapping speech resolved so
each word has one speaker, unless `exclusive=False`. Making one loads nothing:
the pipeline is loaded on the first `diarize`, once per (model, device) per
process. Requires `vidra[audio]`.
"""

from __future__ import annotations

import os
import threading
from importlib.util import find_spec
from typing import Any, Optional

from ...reporting.errors import ModelUnavailable
from ..base import Diarizer
from ..devices import default_device
from . import Diarization, Turn

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")

_PIPELINES: dict[tuple[str, str], Any] = {}
_LOCK = threading.Lock()


def _token(explicit: Optional[str] = None) -> Optional[str]:
    """The token to download with: given here, then
    `vidra.configure(hf_token=...)`, then the variables, else None (Hugging
    Face's own lookup).
    """
    from vidra.shared.config import settings
    return (explicit or settings.hf_token()
            or next((os.environ[v] for v in TOKEN_VARS if os.environ.get(v)), None))


class PyannoteDiarizer(Diarizer):
    """Whole track in, speaker turns out. `device` defaults to cuda, then mps,
    then the CPU. No token is needed to load a model already in the Hugging
    Face cache."""

    concurrency = 1

    def __init__(self, model: str = DEFAULT_MODEL,
                 device: Optional[str] = None,
                 token: Optional[str] = None,
                 exclusive: bool = True,
                 pipeline: Any = None) -> None:
        self.model_name = model or DEFAULT_MODEL
        self.exclusive = exclusive
        self._token = token
        self._pipeline = pipeline
        self._device = "given" if pipeline is not None and device is None else device

    @property
    def key(self) -> str:
        return f"pyannote:{self.model_name}"

    def __repr__(self) -> str:                    # never the token
        return f"PyannoteDiarizer({self.model_name!r})"

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = default_device()
        return self._device

    def problems(self) -> list[str]:
        if self._pipeline is None and find_spec("pyannote") is None:
            return ['pyannote needs pyannote.audio: pip install "vidra[audio]"']
        return []

    def _loaded(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        device = self.device
        with _LOCK:
            if (self.model_name, device) not in _PIPELINES:
                import torch
                from pyannote.audio import Pipeline

                key = _token(self._token)
                try:
                    pipeline = Pipeline.from_pretrained(self.model_name, token=key)
                    if pipeline is None:
                        # pyannote returns None for a refused download.
                        raise RuntimeError("the download was refused")
                    pipeline.to(torch.device(device))
                except Exception as exc:                 # noqa: BLE001
                    from huggingface_hub import get_token
                    how = ("" if key or get_token() else
                           " No Hugging Face token was found: pass "
                           "vidra.configure(hf_token=...), set HF_TOKEN, or run "
                           "`hf auth login`.")
                    raise ModelUnavailable(
                        f"could not load {self.model_name!r}: {exc}.{how} It is "
                        f"gated: the first download needs a token from an account "
                        f"that accepted its terms on huggingface.co. After that it "
                        f"loads from the cache without one.") from None
                _PIPELINES[(self.model_name, device)] = pipeline
            return _PIPELINES[(self.model_name, device)]

    def diarize(self, samples: Any) -> Diarization:
        import numpy as np
        import torch

        waveform = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
        output = self._loaded()({"waveform": waveform.unsqueeze(0),
                                 "sample_rate": self.rate})
        annotation = (output.exclusive_speaker_diarization if self.exclusive
                      else output.speaker_diarization)
        turns = [Turn(str(label), float(seg.start), float(seg.end))
                 for seg, _, label in annotation.itertracks(yield_label=True)]
        turns.sort(key=lambda t: t.start)
        return Diarization(turns)

    def config(self) -> dict[str, Any]:
        return {"diarizer": "pyannote", "diarizer_model": self.model_name,
                "device": self.device, "exclusive": self.exclusive}


__all__ = ["DEFAULT_MODEL", "PyannoteDiarizer"]
