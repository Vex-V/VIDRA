"""What every registry here does: import a class named as text when it is
first asked for, and read what its constructor accepts.

    imported("folder:Folder", __package__)    # `.folder`, relative to the caller
    imported("vidra.shared.storage.folder:Folder")
    keyword_parameters(WhisperTranscriber)    # ["model", "device", ...]

A registry names a class as `module:Class` so the heavy module behind it (a
model, a client library) is imported only by a run that uses it.
"""

from __future__ import annotations

from typing import Any, Optional


def imported(target: str, package: Optional[str] = None) -> Any:
    """`module:attribute`, imported now. With `package`, `module` is relative to
    it, as `.module` would be in that package."""
    import importlib

    module, _, attribute = target.partition(":")
    found = importlib.import_module(f".{module}" if package else module, package)
    return getattr(found, attribute)


def keyword_parameters(cls: type) -> list[str]:
    """What `cls(...)` accepts by name, in order: its constructor's parameters
    less `self`, `*args` and `**kwargs`. A class with no constructor of its own
    accepts nothing."""
    import inspect

    if cls.__init__ is object.__init__:
        return []
    named = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return [p.name for p in inspect.signature(cls.__init__).parameters.values()
            if p.name != "self" and p.kind in named]


__all__ = ["imported", "keyword_parameters"]
