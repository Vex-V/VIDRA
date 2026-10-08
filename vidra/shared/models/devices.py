"""Which device an in-process torch model runs on when the caller names none.

Every torch-backed model -- CLIP, both YOLO detectors, the local embedder,
pyannote -- asks here. Whisper does not: CTranslate2 runs on cuda or the CPU
and has no `mps`. torch is imported inside the call.
"""

from __future__ import annotations


def default_device() -> str:
    """`cuda`, then `mps` (Apple silicon), then `cpu`: the first torch can use."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


__all__ = ["default_device"]
