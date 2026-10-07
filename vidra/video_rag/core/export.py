"""Copying a run's documents to a database, whichever pipeline wrote them.

A pipeline writes files; when a run names a `database`, each document is read
back and handed to that database's `write_<artifact>`. A failed write is
reported, not raised: the files are the run's result, the copy is a copy.
"""

from __future__ import annotations

from pathlib import Path

from vidra.shared.contracts.documents import Produced
from vidra.shared.reporting import logs
from vidra.shared.storage.database import Database, as_database
from .describe import library


def prompt_rows(versions: dict[str, str]) -> list[dict[str, object]]:
    """What each question said, at the version a run asked it under, as rows for a
    database. Writes nothing.
    """
    entries: list[dict[str, object]] = []
    for name, version in sorted(versions.items()):
        entry = library.load()["questions"].get(name) or {}
        shape = library.shape_of(name)
        entries.append({
            "name": name, "version": version,
            "instruction": library.instruction_of(name),
            "shape": shape,
            "summary": shape.get("summary", "standard"),
            "builtin": bool(entry.get("builtin")),
            "about": entry.get("about") or None,
        })
    return entries


def export(produced: Produced, database: str | Database) -> list[str]:
    """Write a component's artifacts to a database. Returns what failed.

    Reads each artifact file back from the receipt and hands it to the
    database's `write_<artifact>`. Directories are skipped, and so is an
    artifact whose hook the database left alone; a failure is reported rather
    than raised.
    """
    target = as_database(database)
    from vidra.shared.storage import files

    problems: list[str] = []
    for artifact, where in sorted(produced.artifacts.items()):
        if not where or Path(where).is_dir():
            continue
        hook = f"write_{artifact}"
        if not hasattr(Database, hook):
            # A new artifact with no hook is a bug here, not the caller's.
            raise RuntimeError(f"{produced.component} produced {artifact!r} and "
                               f"Database has no {hook}; add one")
        if not target.implements(hook):
            continue
        try:
            getattr(target, hook)(produced.video_id, files.read_json(Path(where)))
        except Exception as exc:                          # noqa: BLE001
            message = f"{artifact} -> {target.name}: {exc}"
            problems.append(message)
            logs.logger(produced.component).warning(
                "%s", message,
                extra={"component": produced.component, "event": "export",
                       "video_id": produced.video_id, "artifact": artifact,
                       "reason": str(exc)[:300]})

    # The prompt text behind each question hash this run used.
    if produced.component == "describe" and target.implements("write_prompts"):
        try:
            document = files.read_json(Path(produced.artifacts["descriptions"]))
            target.write_prompts(
                prompt_rows((document.get("model") or {}).get("prompts") or {}))
        except Exception as exc:                          # noqa: BLE001
            problems.append(f"prompts -> {target.name}: {exc}")
    return problems


__all__ = ["export", "prompt_rows"]
