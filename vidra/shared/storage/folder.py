"""A database that is a folder of the pipeline's own output: search with no
server.

    db = Folder("data/out")                      # where video_rag wrote
    search("a man in a red cap", "test", embedder="local", database=db)

Reads `embedded.json` and `timeline.json` where they lie. Dense only: units
are ranked by cosine in the embedder's space. Writes copy only `timeline` and
`embedded` (a no-op when the root is the run's own folder). Aggregate vectors
are kept per embedder in `aggregate_units.json` in the source's folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from ..config import paths
from . import files
from .database import Database

#: What search reads, by artifact name.
KEPT = ("timeline", "embedded")
#: An aggregate source's row and its vectors, per embedder.
AGGREGATE_UNITS = "aggregate_units.json"


class Folder(Database):
    """A folder of video directories, `<root>/<video_id>/`, searched in place."""

    name = "folder"

    def __init__(self, root: Optional[str | Path] = None) -> None:
        #: None is the data root's `out/`, resolved on use.
        self._root = Path(root) if root is not None else None

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else paths.out_root()

    def __repr__(self) -> str:
        return f"Folder({str(self.root)!r})"

    def _file(self, video_id: str, artifact: str) -> Path:
        return self.root / paths.check_id(video_id) / paths.ARTIFACTS[artifact]

    def _read(self, video_id: str, artifact: str) -> Optional[dict[str, Any]]:
        where = self._file(video_id, artifact)
        return files.read_json(where) if where.exists() else None

    # --------------------------------------------------------------- writing

    def write(self, video_id: str, artifact: str, document: dict[str, Any]) -> bool:
        if artifact not in KEPT:
            return False
        if self._read(video_id, artifact) != document:
            files.write_json(self._file(video_id, artifact), document)
        return True

    def write_prompts(self, rows: list[dict[str, Any]]) -> int:
        # Prompt rows are not kept here.
        return 0

    def write_definitions(self, rows: list[dict[str, Any]]) -> int:
        return 0

    def _units_file(self, source_id: str) -> tuple[Path, dict[str, Any]]:
        where = self.root / paths.check_id(source_id) / AGGREGATE_UNITS
        held = files.read_json(where) if where.exists() else {}
        held.setdefault("source", {"source_id": source_id, "video_ids": [source_id]})
        held.setdefault("units", {})
        return where, held

    def write_source(self, source: dict[str, Any]) -> None:
        where, held = self._units_file(source["source_id"])
        held["source"] = source
        files.write_json(where, held)

    def write_answer(self, source_id: str, answer: dict[str, Any],
                     items: list[dict[str, Any]], mentions: list[dict[str, Any]]) -> None:
        # Answers are already files; only the vectors are kept.
        return None

    def write_aggregate_units(self, source_id: str, units: list[dict[str, Any]],
                              embedder_key: str) -> int:
        where, held = self._units_file(source_id)
        answers = {u["aggregate_id"] for u in units}
        kept = [u for u in held["units"].get(embedder_key, [])
                if u["aggregate_id"] not in answers]
        held["units"][embedder_key] = kept + [
            {k: u.get(k) for k in ("aggregate_id", "item_id", "level", "content",
                                   "text_hash", "start_ts", "end_ts", "vector")}
            for u in units if u.get("vector")]
        files.write_json(where, held)
        return sum(1 for u in units if u.get("vector"))

    # --------------------------------------------------------------- reading

    def search(self, vector: Sequence[float], query: str, embedder_key: str,
               limit: int = 20, video_ids: Optional[Sequence[str]] = None,
               sampler: Optional[str] = None, question: Optional[str] = None,
               strategy: Optional[str] = None,
               chunk_ids: Optional[Sequence[int]] = None,
               structured: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        import numpy as np

        wanted = set(chunk_ids) if chunk_ids else None
        rows, vectors = [], []
        for vid in (video_ids or self.video_ids()):
            index = self._read(vid, "embedded")
            if not index or index.get("embedder") != embedder_key:
                continue
            spans = self.spans(vid)
            for u in index["units"]:
                if not u.get("vector") \
                        or (sampler and u["sampler_id"] != sampler) \
                        or (question and u.get("question") != question) \
                        or (strategy and u.get("sampler") != strategy) \
                        or (wanted is not None and u["chunk_id"] not in wanted) \
                        or (structured and not _matches(u.get("structured") or {},
                                                        structured)):
                    continue
                start, end = (spans[u["chunk_id"]] if u["chunk_id"] < len(spans)
                              else (None, None))
                rows.append({"video_id": vid, "chunk_id": u["chunk_id"],
                             "sampler_id": u["sampler_id"],
                             "sampler": u.get("sampler", ""),
                             "question": u.get("question", ""),
                             "content": u.get("content", ""),
                             "structured": u.get("structured") or {},
                             "start_ts": start, "end_ts": end})
                vectors.append(u["vector"])
        if not rows:
            return []
        similarity = _cosine(np.asarray(vectors, dtype=np.float32),
                             np.asarray(vector, dtype=np.float32))
        order = np.argsort(-similarity, kind="stable")[:limit]
        # The shape a hybrid backend returns, with no lexical half.
        return [{**rows[i], "score": 1.0 / (60 + rank), "dense_rank": rank,
                 "text_rank": None, "similarity": float(similarity[i])}
                for rank, i in enumerate(order, start=1)]

    def search_aggregates(self, vector: Sequence[float], query: str, embedder_key: str,
                          level: str, limit: int = 5,
                          source_ids: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
        """Dense only, as `search` is here: cosine within one level."""
        import numpy as np

        found = []
        for where in sorted(self.root.glob(f"*/{AGGREGATE_UNITS}")):
            held = files.read_json(where)
            source = held.get("source") or {}
            if source_ids and source.get("source_id") not in source_ids:
                continue
            for u in held.get("units", {}).get(embedder_key, []):
                if u.get("level") == level and u.get("vector"):
                    found.append((source, u))
        if not found:
            return []
        similarity = _cosine(np.asarray([u["vector"] for _, u in found], dtype=np.float32),
                             np.asarray(vector, dtype=np.float32))
        order = np.argsort(-similarity, kind="stable")[:limit]
        return [{"source_id": found[i][0].get("source_id"),
                 "video_ids": found[i][0].get("video_ids") or [],
                 **{k: found[i][1].get(k) for k in ("aggregate_id", "item_id", "level",
                                                    "content", "start_ts", "end_ts")},
                 "score": 1.0 / (60 + rank), "dense_rank": rank, "text_rank": None,
                 "similarity": float(similarity[i])}
                for rank, i in enumerate(order, start=1)]

    def spans(self, video_id: str) -> list[tuple[float, float]]:
        timeline = self._read(video_id, "timeline")
        return ([(c["start_ts"], c["end_ts"]) for c in timeline["chunks"]]
                if timeline else [])

    def video_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(d.name for d in self.root.iterdir()
                      if d.is_dir() and not d.name.startswith(paths.RESERVED_PREFIX)
                      and (d / paths.ARTIFACTS["embedded"]).exists())


def _cosine(matrix: Any, vector: Any) -> Any:
    import numpy as np
    norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(vector) or 1.0)
    return (matrix @ vector) / np.where(norms == 0, 1.0, norms)


def _matches(have: dict[str, Any], want: dict[str, Any]) -> bool:
    """Every wanted field equal, or present in a list-valued field."""
    for field, value in want.items():
        got = have.get(field)
        if not (got == value or (isinstance(got, list) and value in got)):
            return False
    return True


__all__ = ["Folder", "KEPT", "AGGREGATE_UNITS"]
