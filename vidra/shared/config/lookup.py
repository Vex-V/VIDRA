"""What every registry here does: import a class named as text when it is
first asked for, and read what its constructor accepts.

    imported("folder:Folder", __package__)    # `.folder`, relative to the caller
    imported("vidra.shared.storage.folder:Folder")
    keyword_parameters(WhisperTranscriber)    # ["model", "device", ...]

And what a `validate` beside a function shares with it: the same arguments.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence


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


def bound(function: Callable[..., Any], args: Sequence[Any], kwargs: dict[str, Any],
          drop: Sequence[str] = ()) -> dict[str, Any]:
    """The arguments a call to `function` would have, defaults filled in, less
    `drop`. A call `function` would not accept raises its `TypeError`."""
    import inspect

    arguments = inspect.signature(function).bind(*args, **kwargs)
    arguments.apply_defaults()
    return {k: v for k, v in arguments.arguments.items() if k not in drop}


def signature_without(function: Callable[..., Any], drop: Sequence[str],
                      returns: str) -> Any:
    """`function`'s signature less `drop`, returning `returns`: what a
    `validate` taking the same arguments shows to `help` and the docs check."""
    import inspect

    signature = inspect.signature(function)
    return signature.replace(
        parameters=[p for p in signature.parameters.values() if p.name not in drop],
        return_annotation=returns)


__all__ = ["bound", "imported", "keyword_parameters", "signature_without"]
