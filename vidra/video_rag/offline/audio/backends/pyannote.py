"""Speaker turns over a whole track, via pyannote (4.x).

Uses `.exclusive_speaker_diarization`, with overlapping speech resolved so
each word has one speaker, unless `exclusive=False`.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..models import Diarization, ModelUnavailable, Turn
from ....shared.models.devices import default_device
from ..source import Track

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _token(explicit: Optional[str] = None) -> Optional[str]:
    """The token to download with: given here, then
    `vidra.configure(hf_token=...)`, then the variables, else None (Hugging
    Face's own lookup).
    """
    from vidra.shared.config import settings
    return (explicit or settings.hf_token()
            or next((os.environ[v] for v in TOKEN_VARS if os.environ.get(v)), None))


class PyannoteDiarizer:
    """A `Diarizer`. Whole track in, speaker turns out."""

    name = "pyannote"

    def __init__(self, model: str = DEFAULT_MODEL,
                 device: Optional[str] = None,
                 token: Optional[str] = None,
                 exclusive: bool = True,
                 pipeline: Any = None) -> None:
        self.model_name = model
        self.exclusive = exclusive

        if pipeline is not None:
            self.device, self._pipeline = "given", pipeline
            return
        import torch
        from pyannote.audio import Pipeline

        # No token is required to load a model already in the Hugging Face cache.
        key = _token(token)
        self.device = device or default_device()
        try:
            self._pipeline = Pipeline.from_pretrained(model, token=key)
            if self._pipeline is None:
                # pyannote returns None for a refused download.
                raise RuntimeError("the download was refused")
            self._pipeline.to(torch.device(self.device))
        except Exception as exc:                         # noqa: BLE001
            from huggingface_hub import get_token
            how = ("" if key or get_token() else
                   " No Hugging Face token was found: pass "
                   "vidra.configure(hf_token=...), set HF_TOKEN, or run "
                   "`hf auth login`.")
            raise ModelUnavailable(
                f"could not load {model!r}: {exc}.{how} It is gated: the first "
                f"download needs a token from an account that accepted its "
                f"terms on huggingface.co. After that it loads from the cache "
                f"without one.") from None

    def diarize(self, track: Track) -> Diarization:
        # A silent track has no speakers.
        if track.silent:
            return Diarization([])

        import torch

        output = self._pipeline({
            "waveform": torch.from_numpy(track.samples).unsqueeze(0),
            "sample_rate": track.rate,
        })
        annotation = (output.exclusive_speaker_diarization if self.exclusive
                      else output.speaker_diarization)
        turns = [Turn(str(label), float(seg.start), float(seg.end))
                 for seg, _, label in annotation.itertracks(yield_label=True)]
        turns.sort(key=lambda t: t.start)
        return Diarization(turns)

    def config(self) -> dict[str, Any]:
        return {"diarizer": self.name, "diarizer_model": self.model_name,
                "device": self.device, "exclusive": self.exclusive}


__all__ = ["DEFAULT_MODEL", "PyannoteDiarizer"]
