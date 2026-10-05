"""Writing a document to disk, and reading one back.

Every component writes its documents here. Writes are atomic (temp file plus
`os.replace`). `read` and `write` take a path and a document type: `from_dict`
on the way in, `as_dict` on the way out, with a log record each. A missing
input names the component that writes it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, TypeVar

from ..config import paths
from ..contracts.documents import DOCUMENTS
from ..reporting.errors import VidraError
from ..config.paths import MissingArtifact

T = TypeVar("T")


class WrongDocument(VidraError, ValueError):
    """A file that is not the document asked for: a folder where a file
    belongs, or an excerpt handed where a timeline was wanted."""

#: Document class -> (artifact name, filename, the component that writes it).
WRITTEN_BY: dict[type, tuple[str, str, str]] = {
    cls: (name, paths.ARTIFACTS.get(name, ""), paths.PRODUCED_BY.get(name, ""))
    for name, cls in DOCUMENTS.items()
}


def write_json(path: Path, document: dict[str, Any]) -> Path:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read(path: str | Path, cls: type[T]) -> T:
    """The document at `path`, as `cls`. A missing file names the component that
    writes it.
    """
    p = Path(path)
    name, filename, producer = WRITTEN_BY.get(cls, ("", "", ""))
    if not p.exists():
        if producer:
            raise MissingArtifact(
                f"no {cls.__name__} at {p} -- `{producer}` writes "
                f"{filename or 'it'}")
        raise MissingArtifact(f"no {cls.__name__} at {p}")
    if p.is_dir():
        raise WrongDocument(f"{p} is a folder, not a {cls.__name__} file"
                            + (f" -- `{producer}` writes one" if producer else ""))
    raw = read_json(p)
    # A document of the wrong kind is refused by its `document` tag.
    kind = raw.get("document") if isinstance(raw, dict) else None
    if kind and name and kind != name:
        raise WrongDocument(f"{p} holds `{kind}`, not `{name}`"
                            + (f" -- `{producer}` writes that" if producer else ""))
    document = cls.from_dict(raw)
    if producer:
        from ..reporting import logs
        logs.read(producer, getattr(document, "video_id", None), name)
    return document


def maybe(path: Optional[str | Path], cls: type[T]) -> Optional[T]:
    """`read`, or None for no path or a file not written yet."""
    if path is None or not Path(path).exists():
        return None
    return read(path, cls)


def write(path: str | Path, document: Any) -> str:
    """Write one document to `path`. Returns the path as a string."""
    name, _, producer = WRITTEN_BY.get(type(document), ("", "", ""))
    where = str(write_json(Path(path), document.as_dict()))
    if producer:
        from ..reporting import logs
        logs.wrote(producer, getattr(document, "video_id", None), name, where)
    return where


__all__ = ["WRITTEN_BY", "WrongDocument", "maybe", "read", "read_json", "write",
           "write_json"]
