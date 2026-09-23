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
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .. import paths


def write_json(path: Path, document: dict[str, Any]) -> Path:
    """Atomically. A reader never sees a torn document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(video_id: str, artifact: str, document: dict[str, Any]) -> str:
    """One document, to the path this artifact name resolves to.

    The one line every document write passes through, which is what makes it
    worth a log record: `PRODUCED_BY` names the component, so a reader learns
    who wrote what without the caller having to say.
    """
    from .. import logs
    component = paths.PRODUCED_BY.get(artifact, "shared")
    where = str(write_json(paths.artifact(video_id, artifact), document))
    logs.wrote(component, video_id, artifact, where)
    return where


__all__ = ["read_json", "write", "write_json"]
