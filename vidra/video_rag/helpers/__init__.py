"""What more than one video_rag component uses that is not a document.

    frames    `FrameStore`, one JPEG per kept frame under a directory
"""

from __future__ import annotations

from .frames import FrameStore

__all__ = ["FrameStore"]
