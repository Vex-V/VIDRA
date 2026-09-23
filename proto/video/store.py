"""Where this component's frames go.

`FrameStore` moved up to `video_rag/frames.py` when `describe` stopped
resolving the store path for itself: `video` writes pixels and `describe`
reads them, and a module either of them owned would be an edge between two
components that exchange everything else as files.

Kept as a name because it is where this package has always looked.
"""

from __future__ import annotations

from ..frames import FrameStore

__all__ = ["FrameStore"]
