"""The database base class, and the names backends are built by.

    db = Supabase()                                  # SUPABASE_* from the environment
    db = Supabase(url="https://x.supabase.co", key="...")

    video_rag("x.mp4", "data/out", database=db)      # copies go here
    search("the reactor", "x", database=db)          # and are read from here

A database is a set of hooks, one per thing a run saves and one per thing a
search reads. Subclass `Database` and fill in only the ones you want:

    class VectorsOnly(Database):
        name = "vectors"
        def write_embedded(self, video_id, document):
            for unit in document["units"]:
                index.upsert(f"{video_id}/{unit['chunk_id']}/{unit['sampler_id']}",
                             unit["vector"], unit)

    video_rag("x.mp4", "data/out", database=VectorsOnly())

A write hook left alone does nothing, so a run simply does not save that
artifact. A read hook left alone raises `Unsupported`: a search answered with
nothing would look like a search that found nothing.

`Supabase` and `Folder` are backends built this way; `DATABASES` maps a name
to each. A pipeline writes to a database; components never do.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..reporting.errors import Unavailable, UnknownOption


class Unsupported(Unavailable, NotImplementedError):
    """This database does not implement that read."""


class Database:
    """Where a run's copies go, and what a search reads.

    The write hooks are called by the pipelines as each stage finishes, with
    the stage's document as a dict -- exactly what its JSON file holds. Every
    hook raises on failure; the pipeline reports the failure and continues.
    """

    #: The name a run reports problems under, and the string that builds one.
    name: str = ""

    # ------------------------------------------------------- video_rag writes
    #
    # One per artifact, named `write_<artifact>`, so the pipeline calls each by
    # the artifact's own name. `document` is the artifact's JSON as a dict.

    def write_media(self, video_id: str, document: dict[str, Any]) -> None:
        """`media.json`: the file, its container and its streams."""

    def write_raw_transcript(self, video_id: str, document: dict[str, Any]) -> None:
        """`transcript.raw.json`: words, segments and speaker turns, uncut."""

    def write_cuts(self, video_id: str, document: dict[str, Any]) -> None:
        """`cuts.json`: the boundary evidence (scene scores, speech cuts)."""

    def write_timeline(self, video_id: str, document: dict[str, Any]) -> None:
        """`timeline.json`: the chunk grid, `chunks[]` with `start_ts`/`end_ts`."""

    def write_manifest(self, video_id: str, document: dict[str, Any]) -> None:
        """`manifest.json`: per chunk, the frames each sampler kept."""

    def write_transcript(self, video_id: str, document: dict[str, Any]) -> None:
        """`transcript.json`: the transcript cut onto the grid, per chunk."""

    def write_descriptions(self, video_id: str, document: dict[str, Any]) -> None:
        """`descriptions.json`: per chunk and sampler, the model's answer."""

    def write_embedded(self, video_id: str, document: dict[str, Any]) -> None:
        """`embedded.json`: `embedder` (the space's key) and `units[]`, each
        with `chunk_id`, `sampler_id`, `sampler`, `question`, `content`,
        `structured`, `text_hash` and `vector`."""

    def write_prompts(self, rows: list[dict[str, Any]]) -> None:
        """The questions a describe run asked, at the version it asked them."""

    def write_observations(self, video_id: str, rows: list[dict[str, Any]]) -> None:
        """A live run's answers, as they arrive: one row per (kept frame,
        question), each with `chunk_id`, `sampler_id`, `sampler`, `question`,
        `frame_index`, `media_ts`, `frames`, `description`, `structured`,
        `content`, `text_hash`, `seen_at`, `answered_at`, `lag_s`,
        `gap_before`, `embedder` and `vector`. Called many times per run, with
        whatever arrived since the last call."""

    # ------------------------------------------------------ aggregates writes
    #
    # Rows built by `aggregates.database.export`.

    def write_source(self, source: dict[str, Any]) -> None:
        """What a set of aggregate answers is about: `source_id`, `video_ids`,
        `members` (each video's chunk and time offsets), duration, chunks."""

    def write_answer(self, source_id: str, answer: dict[str, Any],
                     items: list[dict[str, Any]], mentions: list[dict[str, Any]]) -> None:
        """One aggregate answer, the items it places in time and, for a linked
        entity, its sightings."""

    def write_aggregate_units(self, source_id: str, units: list[dict[str, Any]],
                              embedder_key: str) -> None:
        """Embedded summaries, chapters and entities, each with its `level` and
        `vector`. Embedding them costs a call, so it happens only when this
        hook is implemented."""

    def write_definitions(self, rows: list[dict[str, Any]]) -> None:
        """The aggregate definitions a run used, at the version it used them."""

    # --------------------------------------------------------------- reading
    #
    # Needed only to search through this database.

    def search(self, vector: Sequence[float], query: str, embedder_key: str,
               limit: int = 20, video_ids: Optional[Sequence[str]] = None,
               sampler: Optional[str] = None, question: Optional[str] = None,
               strategy: Optional[str] = None,
               chunk_ids: Optional[Sequence[int]] = None,
               structured: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Ranked units for one query; `video_ids=None` is every video."""
        raise self._unsupported("search")

    def search_aggregates(self, vector: Sequence[float], query: str, embedder_key: str,
                          level: str, limit: int = 5,
                          source_ids: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
        """Summaries (`source`), chapters (`span`) or entities (`entity`),
        ranked within that one level."""
        raise self._unsupported("search_aggregates")

    def search_observations(self, vector: Sequence[float], query: str,
                            embedder_key: str, limit: int = 20,
                            video_ids: Optional[Sequence[str]] = None,
                            sampler: Optional[str] = None,
                            question: Optional[str] = None,
                            strategy: Optional[str] = None,
                            after: Optional[float] = None,
                            before: Optional[float] = None,
                            since: Optional[str] = None,
                            until: Optional[str] = None,
                            structured: Optional[dict[str, Any]] = None
                            ) -> list[dict[str, Any]]:
        """Ranked live answers, one per kept frame and question. `after` and
        `before` are media seconds; `since` and `until` are ISO 8601 times the
        frames arrived."""
        raise self._unsupported("search_observations")

    def spans(self, video_id: str) -> list[tuple[float, float]]:
        """One video's grid, as the database holds it."""
        raise self._unsupported("spans")

    def video_ids(self) -> list[str]:
        """Every video the database holds."""
        raise self._unsupported("video_ids")

    # ------------------------------------------------------------- lifetime

    def close(self) -> None:
        """Let go of any connection. Safe to call twice."""

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------------- helpers

    def implements(self, hook: str) -> bool:
        """Whether this database fills in `hook`, rather than inheriting the
        default. The pipelines skip the work behind a hook nobody implemented."""
        if not hasattr(Database, hook):
            raise UnknownOption(f"a Database has no hook {hook!r}")
        return getattr(type(self), hook) is not getattr(Database, hook)

    def _unsupported(self, hook: str) -> Unsupported:
        return Unsupported(f"{type(self).__name__} does not implement {hook}(); "
                           f"subclass it and define {hook} to read through it")


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


__all__ = ["DATABASES", "Database", "Unsupported", "as_database", "backend"]
