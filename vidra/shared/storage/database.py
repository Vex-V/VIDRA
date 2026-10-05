"""The database base class, and the names backends are built by.

    db = Supabase()                                  # SUPABASE_* from the environment
    db = Supabase(url="https://x.supabase.co", key="...")

    video_rag("x.mp4", "data/out", database=db)      # copies go here
    search("the reactor", "x", database=db)          # and are read from here

Each backend is its own module (`supabase.py`, `folder.py`) implementing
these methods; `DATABASES` maps a name to it. A pipeline writes to a
database; components never do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from ..reporting.errors import UnknownOption


class Database(ABC):
    """Where a run's copies go, and what a search reads. Every method raises on
    failure; the pipeline decides to report and continue.
    """

    #: The name a run reports problems under, and the string that builds one.
    name: str = ""

    # --------------------------------------------------------------- writing

    @abstractmethod
    def write(self, video_id: str, artifact: str, document: dict[str, Any]) -> bool:
        """One artifact's document. False when it has no row representation."""

    @abstractmethod
    def write_prompts(self, rows: list[dict[str, Any]]) -> int:
        """The questions a describe run asked, at the version it asked them."""

    @abstractmethod
    def write_definitions(self, rows: list[dict[str, Any]]) -> int:
        """The aggregate definitions a run used, at the version it used them."""

    @abstractmethod
    def write_source(self, source: dict[str, Any]) -> None:
        """What a set of aggregate answers is about: `source_id`, `video_ids`,
        `members` (each video's chunk and time offsets), duration, chunks."""

    @abstractmethod
    def write_answer(self, source_id: str, answer: dict[str, Any],
                     items: list[dict[str, Any]], mentions: list[dict[str, Any]]) -> None:
        """One aggregate answer, the items it places in time and, for a linked entity,
        its sightings. Rows built by `aggregates.database.export`.
        """

    @abstractmethod
    def write_aggregate_units(self, source_id: str, units: list[dict[str, Any]],
                              embedder_key: str) -> int:
        """Embedded summaries, chapters and entities, each with its `level`."""

    # --------------------------------------------------------------- reading

    @abstractmethod
    def search(self, vector: Sequence[float], query: str, embedder_key: str,
               limit: int = 20, video_ids: Optional[Sequence[str]] = None,
               sampler: Optional[str] = None, question: Optional[str] = None,
               strategy: Optional[str] = None,
               chunk_ids: Optional[Sequence[int]] = None,
               structured: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Ranked units for one query; `video_ids=None` is every video."""

    @abstractmethod
    def search_aggregates(self, vector: Sequence[float], query: str, embedder_key: str,
                          level: str, limit: int = 5,
                          source_ids: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
        """Summaries (`source`), chapters (`span`) or entities (`entity`),
        ranked within that one level."""

    @abstractmethod
    def spans(self, video_id: str) -> list[tuple[float, float]]:
        """One video's grid, as the database holds it."""

    @abstractmethod
    def video_ids(self) -> list[str]:
        """Every video the database holds."""

    # ------------------------------------------------------------- lifetime

    def close(self) -> None:
        """Let go of any connection. Safe to call twice."""

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


#: Name -> "module:Class" for its backend.
DATABASES: dict[str, str] = {
    "folder": "vidra.shared.storage.folder:Folder",
    "supabase": "vidra.shared.storage.supabase:Supabase",
}


def backend(name: str) -> type[Database]:
    """The class a name builds."""
    if name not in DATABASES:
        raise UnknownOption(f"unknown database {name!r}; pass a Database, or one "
                            f"of: {', '.join(DATABASES)}")
    import importlib
    module, _, attribute = DATABASES[name].partition(":")
    return getattr(importlib.import_module(module), attribute)


def as_database(value: "Optional[str | Database]") -> Optional[Database]:
    """A `Database` from one already built, a name (built with its defaults from
    the environment), or None.
    """
    if value is None or isinstance(value, Database):
        return value
    if not isinstance(value, str):
        raise UnknownOption(f"a database is a Database or a name, not "
                            f"{type(value).__name__}")
    return backend(value)()


__all__ = ["DATABASES", "Database", "as_database", "backend"]
