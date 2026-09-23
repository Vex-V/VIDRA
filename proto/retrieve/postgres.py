"""Ranking, in the database.

Ranking happens in Postgres via the `search_embeddings` RPC, which fuses a
vector ranking and a `ts_rank_cd` text ranking with RRF. Fetching every row to
rank in Python would move a video's whole index over the wire per query.

`db/supabase/install.sql` holds the schema and the RPC, and both are required:
a database without the RPC is a deployment that was never installed, and a
search there says so rather than answering worse.

**This is a reader, and that is why it is here rather than in `embed`.**
Writing the vectors is `shared/storage/supabase.write_embeddings`, called by
the pipeline; `embed` itself only writes `embedded.json`. There used to be a
`VectorIndex` protocol with two implementations that each did both halves,
which is what made `embed` depend on a database being reachable before it
would encode anything.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from falconvar.shared.storage import db
from falconvar.shared.storage.supabase import EMBEDDINGS

#: The whole-video half. Its own table because `embeddings` answers *which
#: twenty seconds* and this answers *which video* -- and a video is not a
#: moment you can play, so a shared table would return one beside real moments.
VIDEO_TABLE = "video_embeddings"


def search(vector: Sequence[float], query: str, embedder_key: str,
           limit: int = 20,
           video_ids: Optional[Sequence[str]] = None,
           sampler: Optional[str] = None,
           question: Optional[str] = None,
           strategy: Optional[str] = None,
           chunk_ids: Optional[Sequence[int]] = None,
           structured: Optional[dict[str, Any]] = None,
           api: Any = None) -> list[dict[str, Any]]:
    """Ranked units for one query. `video_ids=None` searches every video."""
    api = api or db.client(write=False)
    try:
        response = api.rpc("search_embeddings", {
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
            f"search_embeddings failed ({exc}). The ranking lives in that "
            "RPC; if it is missing, run db/supabase/install.sql") from None

    return [{
        "video_id": r.get("video_id", ""),
        "chunk_id": r["chunk_id"], "sampler_id": r["sampler_id"],
        "sampler": r.get("sampler", ""), "question": r.get("question", ""),
        "content": r.get("content", ""),
        "structured": r.get("structured", {}),
        "score": float(r.get("score", 0.0)),
        "dense_rank": r.get("vector_rank"),
        "text_rank": r.get("text_rank"),
        # The RPC already joins `chunks` for these. Dropping them meant the
        # caller re-read a local `timeline.json` to learn what the database
        # had just told it -- which made search fail outright on a deployment
        # that has the rows and no output directory.
        "start_ts": r.get("start_ts"), "end_ts": r.get("end_ts"),
    } for r in (response.data or [])]


def search_videos(vector: Sequence[float], embedder_key: str,
                  limit: int = 5, api: Any = None) -> list[dict[str, Any]]:
    """Which video is this about. Ranked in Python, deliberately.

    One row per video, so the whole table is a handful of vectors even on a
    large deployment -- the objection to ranking moments in Python (it moves a
    video's whole index over the wire) does not apply when the table IS one row
    per video.
    """
    import math

    rows = (api or db.client(write=False)).table(VIDEO_TABLE).select(
        "video_id,kind,content,embedding").eq("embedder", embedder_key
                                              ).execute().data or []

    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    scored = sorted(rows,
                    key=lambda r: -cosine(vector, db.as_vector(r.get("embedding"))))
    return [{"video_id": r["video_id"], "kind": r.get("kind", "summary"),
             "content": r.get("content", ""),
             "similarity": round(
                 cosine(vector, db.as_vector(r.get("embedding"))), 6)}
            for r in scored[:limit]]


__all__ = ["EMBEDDINGS", "VIDEO_TABLE", "search", "search_videos"]
