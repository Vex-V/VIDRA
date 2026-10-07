"""The waveform, decoded once, whole, with PyAV: 16 kHz mono float32, what both
Whisper and pyannote want.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import av
import numpy as np
from vidra.shared.reporting.errors import VidraError

#: The sample rate both models expect.
SAMPLE_RATE = 16000

#: Below this RMS a track is treated as silent.
SILENCE_RMS = 1e-3


class NoAudio(VidraError):
    """The file carries no audio stream at all."""


@dataclass
class Track:
    """One decoded waveform, with its RMS and peak."""

    samples: np.ndarray                      # float32, mono, SAMPLE_RATE
    rate: int
    rms: float
    peak: float

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.rate

    @property
    def silent(self) -> bool:
        return self.rms < SILENCE_RMS

    def as_dict(self) -> dict[str, Any]:
        return {"rate": self.rate, "duration_s": round(self.duration_s, 3),
                "samples": int(len(self.samples)), "rms": round(self.rms, 6),
                "peak": round(self.peak, 6), "silent": self.silent}


def load(path: str, rate: int = SAMPLE_RATE) -> Track:
    """Decode the whole audio stream to mono float32 at ``rate``."""
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise NoAudio(f"{path} has no audio stream")
        # Required for decode speed.
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        blocks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                blocks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):            # flush
            blocks.append(out.to_ndarray().reshape(-1))

    samples = (np.concatenate(blocks) if blocks
               else np.zeros(0, dtype=np.float32)).astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return Track(samples=samples, rate=rate, rms=rms, peak=peak)


__all__ = ["SAMPLE_RATE", "SILENCE_RMS", "NoAudio", "Track", "load"]
