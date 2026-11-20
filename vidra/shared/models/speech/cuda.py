"""Making the CUDA libraries findable on Windows.

CTranslate2 (under faster-whisper) loads `cublas64_12.dll` by name at its
first encode, and Windows does not search `site-packages/nvidia/*/bin`.
Loading each library by absolute path first lets that request resolve. Call
`enable()` before constructing any CUDA-backed audio model.
"""

from __future__ import annotations

import ctypes
import site
import sys
from pathlib import Path
from typing import Optional

#: Loaded in dependency order.
NEEDED = ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll")

_done: Optional[list[str]] = None


def enable() -> list[str]:
    """Preload the vendored CUDA libraries. Idempotent; a no-op off Windows."""
    global _done
    if _done is not None:
        return _done
    if sys.platform != "win32":
        _done = []
        return _done

    # Wherever Python imports `nvidia` from (a namespace package), then the
    # environment's site-packages.
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    roots = [Path(p) for p in (spec.submodule_search_locations or [])] if spec else []
    roots += [Path(p) / "nvidia" for p in site.getsitepackages()]
    roots += [Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"]
    loaded: list[str] = []
    for name in NEEDED:
        for root in roots:
            if not root.is_dir():
                continue
            for dll in sorted(root.glob(f"*/bin/{name}")):
                try:
                    ctypes.WinDLL(str(dll))
                except OSError:
                    continue          # a missing optional dep is not fatal here
                loaded.append(name)
                break
            if name in loaded:
                break
    _done = loaded
    return loaded


def missing() -> list[str]:
    """The libraries CTranslate2 will ask for by name that cannot be loaded,
    after `enable()`. Empty off Windows, and when a system CUDA 12 serves them.
    """
    if sys.platform != "win32":
        return []
    enable()
    lost = []
    for name in NEEDED:
        try:
            ctypes.WinDLL(name)
        except OSError:
            lost.append(name)
    return lost
