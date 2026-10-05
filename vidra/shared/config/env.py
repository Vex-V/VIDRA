"""Reading a `.env` into the environment: by an entry point, or when asked.

The library never calls this by itself. Every `python -m vidra...` CLI
calls `load()` first; in code, use `vidra.configure(env_file=...)` or
`load_dotenv()`. With no path, `load()` reads the checkout's `.env`, or the
working directory's when installed. A variable already set is never
overridden. Keys only: the data root is not read from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import paths

_loaded = False


def env_file() -> Optional[Path]:
    """The `.env` a CLI would read, or None if there is none."""
    candidate = (paths.CHECKOUT / ".env" if paths.CHECKOUT is not None
                 else Path.cwd() / ".env")
    return candidate if candidate.exists() else None


def load(path: Optional[Path | str] = None, force: bool = False) -> bool:
    """Read `.env` into the environment. True if anything was read.

    `path` names a file explicitly; without it, see `env_file`.
    """
    global _loaded
    if _loaded and not force and path is None:
        return True

    found = Path(path).expanduser() if path is not None else env_file()
    if found is None or not found.exists():
        _loaded = True
        return False

    from dotenv import load_dotenv
    load_dotenv(found, override=False)
    _loaded = True
    return True


__all__ = ["env_file", "load"]
