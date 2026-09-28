"""Writing a document to disk, and reading one back.

Every component writes here and nowhere else. A component's job is to produce
a document; deciding that a copy should also land in Postgres is a question
about a *deployment*, and the pipeline owns it -- see `supabase.py`.

That was not always true. `sinks.write` used to fan one document out to a file
and a database, so every component took a `sink`, published it through
`/capabilities`, and had to know which artifacts had a row mapping at all.
It generated the same class of bug repeatedly: `embed` accepted a `sink` it
never read and a generated form offered it as a working control; `support()`
had to answer `any` rather than `all` because `video` writes a manifest *and*
a directory of JPEGs; and the best-effort-versus-raise question had a
different answer in the docstring than in the code. One concern threaded
through fourteen files.

The file write is atomic -- temp file plus `os.replace` -- because components
hand off through files, so a partially written document would be the next
component's input rather than a corrupt log line.

**`read` and `write` take a path and a type.** They used to take a video id
and an artifact name and resolve one -- which is the resolving that left the
components, so what is left here is the typed layer over the two json
functions: `from_dict` on the way in, `as_dict` on the way out, and the two
log records at the two choke points.

They are in `shared` rather than in a tier because both tiers read documents.
`aggregates` reads its inputs through here too, since it moved to paths; it
writes each answer with `write_json`, because an answer is one of many files
in a folder rather than an artifact with a name of its own.

**A missing input still names the component that makes it.** `paths.require`
could say it because it had the artifact name; this can because the producer
is a fact about the *type* the caller asked for, and `WRITTEN_BY` derives
that from the library's own tables. What it cannot add is `Present: media`,
which needs a directory to look in -- that half is a pipeline's now.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, TypeVar

from .. import paths
from ..contracts.documents import DOCUMENTS
from ..errors import FalconvarError
from ..paths import MissingArtifact

T = TypeVar("T")


class WrongDocument(FalconvarError, ValueError):
    """A file that is not the document asked for: a folder where a file
    belongs, or an excerpt handed where a timeline was wanted."""

#: Document class -> (artifact name, filename, the component that writes it).
#: Derived from the library's own tables rather than restated: a second copy
#: of `transcript.raw.json` is a drift waiting to happen.
WRITTEN_BY: dict[type, tuple[str, str, str]] = {
    cls: (name, paths.ARTIFACTS.get(name, ""), paths.PRODUCED_BY.get(name, ""))
    for name, cls in DOCUMENTS.items()
}


def write_json(path: Path, document: dict[str, Any]) -> Path:
    """Atomically. A reader never sees a torn document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read(path: str | Path, cls: type[T]) -> T:
    """The document at `path`, as `cls`.

    The one line every typed document read passes through, which is what
    makes it worth a log record and worth a better error than the
    interpreter's `[Errno 2] No such file or directory`.
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
    # Every document names its own kind. Read as another, one failed with the
    # interpreter's `KeyError: 'chunks'` from deep inside `from_dict` -- an
    # excerpt handed where a timeline belongs. A document with no tag
    # predates it and is read as before.
    kind = raw.get("document") if isinstance(raw, dict) else None
    if kind and name and kind != name:
        raise WrongDocument(f"{p} holds `{kind}`, not `{name}`"
                            + (f" -- `{producer}` writes that" if producer else ""))
    document = cls.from_dict(raw)
    if producer:
        from .. import logs
        logs.read(producer, getattr(document, "video_id", None), name)
    return document


def maybe(path: Optional[str | Path], cls: type[T]) -> Optional[T]:
    """`read`, or None for an argument nobody passed or a file not written yet.

    Both absences mean the same thing to every caller that takes one -- a
    `previous=` to resume from, cuts a policy may not need -- so they are one
    answer rather than a branch at each call site.
    """
    if path is None or not Path(path).exists():
        return None
    return read(path, cls)


def write(path: str | Path, document: Any) -> str:
    """One document, to the path the caller named. Returns it as a string.

    The one line every document write passes through, which is what makes it
    worth a log record: the document's type names the component, so a reader
    learns who wrote what without the caller having to say.
    """
    name, _, producer = WRITTEN_BY.get(type(document), ("", "", ""))
    where = str(write_json(Path(path), document.as_dict()))
    if producer:
        from .. import logs
        logs.wrote(producer, getattr(document, "video_id", None), name, where)
    return where


__all__ = ["WRITTEN_BY", "WrongDocument", "maybe", "read", "read_json", "write",
           "write_json"]
