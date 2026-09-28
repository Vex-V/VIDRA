"""What more than one component needs and neither of them may own.

A component hands the next one a *document*, and `shared/contracts` is where
those shapes live. This is the other kind: something two components share
that is not a document, and so has nowhere in `shared` to be either --
`aggregates` never touches a frame.

    frames    `FrameStore`, one JPEG per kept frame under a directory

`video` writes pixels and `describe` reads them. Either owning the class
would be an edge between two components that exchange everything else as
files -- the same argument that keeps a document's dataclass out of the
component that produces it.

It is also why this is a folder rather than one module: the next thing that
is shared, not a document, and not `aggregates`' business lands here beside
it rather than inside whichever component happened to need it first.
"""

from __future__ import annotations

from .frames import FrameStore

__all__ = ["FrameStore"]
