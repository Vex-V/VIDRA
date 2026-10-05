"""3+4 · boundaries -- where the chunk boundaries fall, from either modality.

    scenes.py   the picture: frame-to-frame difference, thresholded into cuts
    speech.py   the soundtrack: silences or speaker changes in a transcript
    grid.py     cuts + a duration -> spans, with the guards applied
"""

from __future__ import annotations

#: The public surface: entry points, errors and return types.
from .driver import (EVIDENCE_SETTINGS, boundaries, detect, evidence, load,
                     retune, timeline)
from .grid import POLICIES

__all__ = ["EVIDENCE_SETTINGS", "POLICIES", "boundaries", "detect",
           "evidence", "load", "retune", "timeline"]
