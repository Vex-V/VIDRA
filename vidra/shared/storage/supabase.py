"""The Supabase backend: its row writers, its readers, and the `Supabase` class.

    db = Supabase()                              # SUPABASE_* from the environment
    video_rag("x.mp4", "data/out", database=db)  # a copy of every artifact
    search("the reactor", "x", database=db)      # read back from here

video_rag's tables are `vr_` (`db/supabase/video_rag.sql`): `write` turns each
artifact into rows, and `write_prompts` records each question's wording. The
aggregates' are `ag_` (`db/supabase/aggregates.sql`): `write_source`,
`write_answer`, `write_aggregate_units` and `write_definitions`. Searches go
through `search_embeddings` and `search_aggregate_units`. The only module that
knows a table name; `db.py` is the client and the batched upsert.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from . import db
from .database import Database

#: The moment index: one row per (video, chunk, sampler, embedder).
EMBEDDINGS = "vr_embeddings"
#: The aggregate index: summaries, chapters and entities, by level.
AGGREGATE_EMBEDDINGS = "ag_embeddings"


def write_media(video_id: str, document: dict[str, Any],
                api: Any = None) -> None:
    api = api or db.client()
    db.upsert("vr_videos", [{
        "video_id": video_id,
        "path": document["path"],
        "name": document.get("name"),
        "recorded_at": document.get("recorded_at"),
        "container": document["container_format"],
        "duration_s": document.get("duration_s"),
        "has_video": document.get("video") is not None,
        "has_audio": document.get("audio") is not None,
        "video_stream": document.get("video"),
        "audio_stream": document.get("audio"),
    }], api)


def write_timeline(video_id: str, document: dict[str, Any],
                   api: Any = None) -> None:
    api = api or db.client()
    # A grid needs a `vr_videos` row, so a combination's is refused.
    db.upsert("vr_timelines", [{
        "video_id": video_id,
        "policy": document["policy"],
        "derived_from": document["derived_from"],
        "params": document.get("params", {}),
        "fingerprint": document["fingerprint"],
        "duration_s": document["duration_s"],
        "chunk_count": document["chunk_count"],
    }], api)
    chunks = document.get("chunks", [])
    db.upsert("vr_chunks", [{
        "video_id": video_id, "chunk_id": c["chunk_id"],
        "start_ts": c["start_ts"], "end_ts": c["end_ts"],
    } for c in chunks], api)
    # After the upserts: delete chunks past the end of a grid that shrank.
    db.delete_stale_chunks("vr_chunks", video_id, len(chunks), api)


def write_raw_transcript(video_id: str, document: dict[str, Any],
                         api: Any = None) -> None:
    api = api or db.client()
    db.upsert("vr_transcripts", [{
        "video_id": video_id,
        "model": document.get("model", {}),
        "track": document.get("track", {}),
        "stats": document.get("stats", {}),
        "segments": document.get("segments", []),
        "words": document.get("words", []),
        "turns": document.get("turns", []),
    }], api)


def write_transcript(video_id: str, document: dict[str, Any],
                     api: Any = None) -> None:
    api = api or db.client()
    # The header row exists from `raw_transcript`; add the grid and the chunks.
    db.upsert("vr_transcripts", [{
        "video_id": video_id,
        "timeline_fingerprint": document.get("timeline_fingerprint"),
        "model": document.get("model", {}),
        "stats": document.get("stats", {}),
    }], api)
    chunks = document.get("chunks", [])
    db.upsert("vr_transcript_chunks", [{
        "video_id": video_id, "chunk_id": c["chunk_id"],
        "text": c.get("text", ""), "word_count": c.get("word_count", 0),
        "structured": c.get("structured", {}), "turns": c.get("turns", []),
    } for c in chunks], api)


def write_manifest(video_id: str, document: dict[str, Any],
                   api: Any = None) -> None:
    api = api or db.client()
    """The frames first, the row that claims them second.

    There is no transaction across two REST writes, so an ordering is the only
    guard there is. Observed: the `manifests` row landed and `chunk_samplers`
    was refused, leaving a manifest that claimed a run with no sampled frames --
    a partial state that reads exactly like a valid one, since a video whose
    samplers kept nothing is a thing that can happen.

    Written the other way round, the same failure leaves rows nobody points at
    and no manifest claiming them, so `paths`/`db.fetch_manifest` report the
    truth: this video has not been ingested here yet.
    """
    # One row per (chunk, sampler run); `questions` lists what was asked of it.
    by_id = {s["id"]: s for s in document.get("config", {}).get("samplers", [])}
    rows = []
    for chunk in document.get("chunks", []):
        for run_id, block in chunk.get("samplers", {}).items():
            config = by_id.get(run_id, {})
            name = config.get("name") or run_id.split(":")[0]
            asked = (list(config.get("prompts") or [])
                     or ([config["prompt"]] if config.get("prompt") else [name]))
            rows.append({
                "video_id": video_id, "chunk_id": chunk["chunk_id"],
                "sampler_id": run_id,
                "questions": asked,
                "frame_count": block.get("frame_count", 0),
                "frames": block.get("frames", []),
            })
    db.upsert("vr_chunk_samplers", rows, api)

    db.upsert("vr_manifests", [{
        "video_id": video_id,
        "timeline_fingerprint": document["timeline_fingerprint"],
        "manifest_fingerprint": document["manifest_fingerprint"],
        "source": document.get("source", {}),
        "config": document.get("config", {}),
        "stats": document.get("stats", {}),
    }], api)


def write_descriptions(video_id: str, document: dict[str, Any],
                       api: Any = None) -> None:
    api = api or db.client()
    rows = []
    for chunk in document.get("chunks", []):
        for sampler_id, block in chunk.get("samplers", {}).items():
            rows.append({
                "video_id": video_id, "chunk_id": chunk["chunk_id"],
                "sampler_id": sampler_id,
                "question": block.get("question", sampler_id),
                "frame_indexes": block.get("frame_indexes", []),
                "frame_count": block.get("frame_count", 0),
                "description": block.get("description"),
                "structured": block.get("structured", {}),
                "model": document.get("model", {}),
                "elapsed_s": block.get("elapsed_s"),
                "timeline_fingerprint": document.get("timeline_fingerprint"),
                "manifest_fingerprint": document.get("manifest_fingerprint"),
            })
    db.upsert("vr_descriptions", rows, api)
    # No stale-row delete: descriptions are not removed by a re-ingest.


# ------------------------------------------------------------- aggregates
#
# Rows built by `aggregates.database.export`; these only store them.

def write_source(source: dict[str, Any], api: Any = None) -> None:
    """What a set of answers is about: one video, or several end to end."""
    db.upsert("ag_sources", [source], api or db.client())


def write_answer(source_id: str, answer: dict[str, Any],
                 items: list[dict[str, Any]], mentions: list[dict[str, Any]],
                 api: Any = None) -> None:
    """One answer, the items it places in time, and an entity's sightings.

    Upserted, then rows the answer no longer holds are deleted. A removed item
    takes its mentions with it.
    """
    api = api or db.client()
    key = {"source_id": source_id, "aggregate_id": answer["aggregate_id"]}
    db.upsert("ag_answers", [{**answer, "source_id": source_id}], api)
    db.upsert("ag_items", [{**key, **item} for item in items], api)
    db.delete_except("ag_items", key, "item_id",
                     [item["item_id"] for item in items], api)
    db.upsert("ag_mentions", [{**key, **m} for m in mentions], api)
    db.delete_except("ag_mentions", key, "mention_key",
                     [m["mention_key"] for m in mentions], api)


def write_aggregate_units(source_id: str, units: list[dict[str, Any]],
                          embedder_key: str, api: Any = None) -> int:
    """The embedded summary, chapters and entities, in one embedder's space.
    Rows an answer no longer holds in this space are removed after the upsert.
    """
    api = api or db.client()
    rows = [{"source_id": source_id, "aggregate_id": u["aggregate_id"],
             "item_id": u["item_id"], "level": u["level"], "embedder": embedder_key,
             "text_hash": u["text_hash"], "content": u["content"],
             "start_ts": u.get("start_ts"), "end_ts": u.get("end_ts"),
             "embedding": list(u["vector"])} for u in units if u.get("vector")]
    written = db.upsert_vectors(AGGREGATE_EMBEDDINGS, rows, api)
    for answer_id in sorted({r["aggregate_id"] for r in rows}):
        db.delete_except(AGGREGATE_EMBEDDINGS,
                         {"source_id": source_id, "aggregate_id": answer_id,
                          "embedder": embedder_key}, "item_id",
                         [r["item_id"] for r in rows if r["aggregate_id"] == answer_id], api)
    return written


def write_definitions(entries: list[dict[str, Any]], api: Any = None) -> int:
    """Record the aggregate definitions a run used, keyed (name, version).
    Append-only.
    """
    if not entries:
        return 0
    return db.upsert("ag_definitions", entries, api)


def write_prompts(entries: list[dict[str, Any]], api: Any = None) -> int:
    """Record the prompt versions a run used, keyed (name, version). Append-only."""
    if not entries:
        return 0
    return db.upsert("vr_prompts", entries, api)


def write_embeddings(video_id: str, document: dict[str, Any],
                     api: Any = None) -> int:
    """`embedded.json` into the moment index, vectors and all, under the
    document's embedder. Rows the document no longer holds are deleted after
    the upsert.
    """
    api = api or db.client()
    embedder = document.get("embedder", "")
    units = document.get("units", [])
    rows = [{
        "video_id": video_id, "chunk_id": u["chunk_id"],
        "sampler_id": u["sampler_id"], "embedder": embedder,
        # Both halves of the sampler id, so each filter is an equality.
        "sampler": u.get("sampler", ""), "question": u.get("question", ""),
        "text_hash": u.get("text_hash", ""), "content": u.get("content", ""),
        "structured": u.get("structured", {}),
        "embedding": list(u.get("vector") or []),
    } for u in units if u.get("vector")]
    written = db.upsert_vectors(EMBEDDINGS, rows, api)

    live = {(u["chunk_id"], u["sampler_id"]) for u in units}
    stored = (api.table(EMBEDDINGS).select("chunk_id,sampler_id")
              .eq("video_id", video_id).eq("embedder", embedder)
              .execute().data or [])
    for row in stored:
        if (row["chunk_id"], row["sampler_id"]) not in live:
            (api.table(EMBEDDINGS).delete()
                .eq("video_id", video_id).eq("chunk_id", row["chunk_id"])
                .eq("sampler_id", row["sampler_id"])
                .eq("embedder", embedder).execute())
    return written


#: Artifact name -> the function that writes its rows. An artifact absent here
#: has no rows.
WRITERS: dict[str, Callable[..., Any]] = {
    "media": write_media,
    "timeline": write_timeline,
    "raw_transcript": write_raw_transcript,
    "transcript": write_transcript,
    "manifest": write_manifest,
    "descriptions": write_descriptions,
    "embedded": write_embeddings,
}


def writer_for(artifact: str) -> Optional[Callable[..., Any]]:
    return WRITERS.get(artifact)


def write(video_id: str, artifact: str, document: dict[str, Any],
          api: Any = None) -> bool:
    """One document, by artifact name. False when it has no row mapping."""
    handler = WRITERS.get(artifact)
    if handler is None:
        return False
    handler(video_id, document, api)
    return True


# ------------------------------------------------------------------ reading

def search_embeddings(vector: Sequence[float], query: str, embedder_key: str,
                      limit: int = 20,
                      video_ids: Optional[Sequence[str]] = None,
                      sampler: Optional[str] = None,
                      question: Optional[str] = None,
                      strategy: Optional[str] = None,
                      chunk_ids: Optional[Sequence[int]] = None,
                      structured: Optional[dict[str, Any]] = None,
                      api: Any = None) -> list[dict[str, Any]]:
    """Ranked units for one query, from the `vr_search` RPC (vector and text
    rankings fused by RRF). `video_ids=None` searches every video.
    """
    api = api or db.client(write=False)
    try:
        response = api.rpc("vr_search", {
            "p_query_vector": list(vector),
            "p_query_text": query,
            "p_video_ids": list(video_ids) if video_ids else None,
            "p_embedder": embedder_key,
            "p_sampler": sampler,
            "p_question": question,
            "p_strategy": strategy,
            "p_chunk_ids": list(chunk_ids) if chunk_ids else None,
            "p_structured": structured or None,
            "p_limit": limit,
        }).execute()
    except Exception as exc:                             # noqa: BLE001
        raise db.DatabaseUnavailable(
            f"vr_search failed ({exc}). The ranking lives in that "
            "RPC; if it is missing, run db/supabase/video_rag.sql") from None

    return [{
        "video_id": r.get("video_id", ""),
        "chunk_id": r["chunk_id"], "sampler_id": r["sampler_id"],
        "sampler": r.get("sampler", ""), "question": r.get("question", ""),
        "content": r.get("content", ""),
        "structured": r.get("structured", {}),
        "score": float(r.get("score", 0.0)),
        "dense_rank": r.get("vector_rank"),
        "text_rank": r.get("text_rank"),
        # Times from the grid, joined by the RPC.
        "start_ts": r.get("start_ts"), "end_ts": r.get("end_ts"),
    } for r in (response.data or [])]


def search_aggregate_units(vector: Sequence[float], query: str, embedder_key: str,
                           level: str, limit: int = 5,
                           source_ids: Optional[Sequence[str]] = None,
                           api: Any = None) -> list[dict[str, Any]]:
    """Ranked summaries, chapters or entities, one level at a time: the `ag_search`
    RPC.
    """
    api = api or db.client(write=False)
    try:
        response = api.rpc("ag_search", {
            "p_embedder": embedder_key, "p_level": level,
            "p_query_vector": list(vector), "p_query_text": query,
            "p_source_ids": list(source_ids) if source_ids else None,
            "p_limit": limit,
        }).execute()
    except Exception as exc:                             # noqa: BLE001
        raise db.DatabaseUnavailable(
            f"ag_search failed ({exc}); if it is missing, run "
            "db/supabase/aggregates.sql") from None
    return [{"source_id": r["source_id"], "aggregate_id": r["aggregate_id"],
             "item_id": r["item_id"], "level": r["level"],
             "content": r.get("content", ""), "label": r.get("label"),
             "video_ids": r.get("video_ids") or [],
             "start_ts": r.get("start_ts"), "end_ts": r.get("end_ts"),
             "score": float(r.get("score", 0.0)),
             "dense_rank": r.get("vector_rank"), "text_rank": r.get("text_rank")}
            for r in (response.data or [])]


def read_spans(video_id: str, api: Any = None) -> list[tuple[float, float]]:
    """One video's grid, from `vr_chunks`."""
    rows = ((api or db.client(write=False)).table("vr_chunks")
            .select("chunk_id,start_ts,end_ts")
            .eq("video_id", video_id).order("chunk_id").execute().data or [])
    return [(float(r["start_ts"]), float(r["end_ts"])) for r in rows]


def read_video_ids(api: Any = None) -> list[str]:
    """Every video the database holds a grid for."""
    rows = ((api or db.client(write=False)).table("vr_timelines")
            .select("video_id").execute().data or [])
    return [r["video_id"] for r in rows]



# ------------------------------------------------------------ the backend

class Supabase(Database):
    """Postgres through Supabase's REST API.

    `url` and `key` default to `SUPABASE_URL` and the secret key; `read_key`,
    used by searches, to the publishable key. Nothing connects until the first
    call; building one only checks that a URL and a key exist.
    """

    name = "supabase"

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None,
                 read_key: Optional[str] = None) -> None:
        self.url = url or db._first(db.URL_VARS)
        self._key = key
        self._read_key = read_key
        if not self.url:
            raise db.DatabaseUnavailable(
                "no Supabase URL: pass url= or set SUPABASE_URL in .env")
        if not (key or read_key or db._first(db.SECRET_VARS)
                or db._first(db.PUBLISHABLE_VARS)):
            raise db.DatabaseUnavailable(
                "no Supabase key: pass key= or set one of "
                + ", ".join(db.SECRET_VARS + db.PUBLISHABLE_VARS) + " in .env")
        self._writer: Any = None
        self._reader: Any = None

    def __repr__(self) -> str:                 # never a key
        return f"Supabase(url={self.url!r})"

    def writer(self) -> Any:
        """The client writes go through, built once and kept."""
        if self._writer is None:
            self._writer = db.client(self.url, self._key, write=True)
        return self._writer

    def reader(self) -> Any:
        """The client searches go through: the read-only key."""
        if self._reader is None:
            self._reader = db.client(self.url, self._read_key, write=False)
        return self._reader

    def write(self, video_id: str, artifact: str, document: dict[str, Any]) -> bool:
        if writer_for(artifact) is None:
            return False
        return write(video_id, artifact, document, self.writer())

    def write_prompts(self, rows: list[dict[str, Any]]) -> int:
        return write_prompts(rows, self.writer())

    def write_definitions(self, rows: list[dict[str, Any]]) -> int:
        return write_definitions(rows, self.writer())

    def write_source(self, source: dict[str, Any]) -> None:
        write_source(source, self.writer())

    def write_answer(self, source_id: str, answer: dict[str, Any],
                     items: list[dict[str, Any]], mentions: list[dict[str, Any]]) -> None:
        write_answer(source_id, answer, items, mentions, self.writer())

    def write_aggregate_units(self, source_id: str, units: list[dict[str, Any]],
                              embedder_key: str) -> int:
        return write_aggregate_units(source_id, units, embedder_key, self.writer())

    def search(self, vector: Sequence[float], query: str, embedder_key: str,
               limit: int = 20, video_ids: Optional[Sequence[str]] = None,
               sampler: Optional[str] = None, question: Optional[str] = None,
               strategy: Optional[str] = None,
               chunk_ids: Optional[Sequence[int]] = None,
               structured: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        return search_embeddings(vector, query, embedder_key, limit,
                                          video_ids, sampler, question, strategy,
                                          chunk_ids, structured, self.reader())

    def search_aggregates(self, vector: Sequence[float], query: str, embedder_key: str,
                          level: str, limit: int = 5,
                          source_ids: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
        return search_aggregate_units(vector, query, embedder_key, level, limit,
                                      source_ids, self.reader())

    def spans(self, video_id: str) -> list[tuple[float, float]]:
        return read_spans(video_id, self.reader())

    def video_ids(self) -> list[str]:
        return read_video_ids(self.reader())

    def close(self) -> None:
        # Dropping the clients is the whole of closing; they are rebuilt on next use.
        self._writer = self._reader = None


__all__ = ["AGGREGATE_EMBEDDINGS", "EMBEDDINGS", "Supabase", "WRITERS", "read_spans",
           "read_video_ids", "search_aggregate_units", "search_embeddings", "write",
           "write_aggregate_units", "write_answer", "write_definitions",
           "write_descriptions", "write_embeddings", "write_manifest", "write_media",
           "write_prompts", "write_raw_transcript", "write_source", "write_timeline",
           "write_transcript", "writer_for"]
