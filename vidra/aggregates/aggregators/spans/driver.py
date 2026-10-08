"""The `spans` kind: contiguous ranges over the whole video. `chapters` is one.

Every chunk the input read is embedded, and `segment` puts a boundary where
similarity between neighbouring chunks dips well below the dips around it.
`min_span_s` (30 s) is the floor: a shorter span joins its more alike
neighbour. `max_spans` caps the count: while there are more, the two most
alike neighbouring spans merge. The model then names each span, one call per
span, with the definition's fields (`title`, `summary` for chapters). Spans
cover every chunk of the grid; boundaries are recorded with their similarity
and depth.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from vidra.shared.models.base import LLM, Embedder

from ... import definitions
from ...core.base import WINDOW, DefinitionRunner, schema
from ..fold import fold
from ...core.rendering import resolve_span
from .segment import segment

#: Texts per embedding request, as `embed` sends them.
BATCH = 64


class SpansAggregator(DefinitionRunner):
    def __init__(self, definition_id: str, llm: Optional[LLM] = None,
                 embedder: Optional[Embedder] = None,
                 max_spans: Optional[int] = None,
                 min_span_s: float = 30.0) -> None:
        super().__init__(definition_id, llm)
        if max_spans is not None and (type(max_spans) is not int or max_spans < 1):
            from vidra.shared.reporting.errors import Refused
            raise Refused(f"max_spans must be a whole number, 1 or more; "
                          f"got {max_spans!r}")
        if not isinstance(min_span_s, (int, float)) or min_span_s < 0:
            from vidra.shared.reporting.errors import Refused
            raise Refused(f"min_span_s must be 0 or more seconds; got {min_span_s!r}")
        from vidra.shared.models.base import require
        from vidra.shared.models.roles import resolve
        self.embedder = resolve("embedder", embedder)
        require("embedder", self.embedder)
        self.max_spans = max_spans
        self.min_span_s = float(min_span_s)

    @property
    def model_key(self) -> str:
        """The llm, the embedder and both caps: each changes the answer."""
        return (f"{self.llm.key}|{self.embedder.key}|"
                f"max={self.max_spans if self.max_spans is not None else 'none'}|"
                f"min={self.min_span_s:g}s")

    def _vectors(self, rows: list[Any]) -> list[list[float]]:
        """One vector per row: the mean of its parts, each embedded separately."""
        texts = [said for row in rows for _, said in row.parts]
        vectors = [v for at in range(0, len(texts), BATCH)
                   for v in self.embedder.embed(texts[at:at + BATCH])]
        out, at = [], 0
        for row in rows:
            mine = vectors[at:at + len(row.parts)]
            at += len(row.parts)
            out.append([sum(column) / len(mine) for column in zip(*mine)])
        return out

    async def _name(self, context: Any, rows: list[Any], index: int, total: int,
                    chunk_ids: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """One span's fields, from that span's rows alone."""
        units = [([r.chunk_id], r.line) for r in rows]
        layers: list[dict[str, Any]] = []
        if len(units) > WINDOW:
            # A span too long for one call is folded within itself.
            units, layers = await fold(context, self.llm, units, until=WINDOW)
        ask = definitions.kind_text("spans")["name"].format(
            number=index + 1, total=total, first=chunk_ids[0], last=chunk_ids[-1])
        answer = await self.llm.complete(
            f"{self.entry['instruction']} {ask}\n\n"
            + "\n".join(said for _, said in units),
            schema(self.name, self.properties()), definitions.system())
        return answer, layers

    async def _run(self, context: Any, read: Any) -> dict[str, Any]:
        key = self.entry.get("key") or "spans"
        fields = list(self.entry["fields"])
        rows = read.rows
        found = segment(self._vectors(rows), self.max_spans,
                        durations=[r.end - r.start for r in rows],
                        shortest=self.min_span_s)

        # Row positions -> chunk ids: each span runs from its first row's chunk to the
        # chunk before the next span's.
        grid = context.chunk_ids()
        starts = [rows[group[0]].chunk_id for group in found.groups]
        starts[0] = grid[0]
        bounds = starts[1:] + [grid[-1] + 1]
        spans_ids = [[c for c in grid if start <= c < end]
                     for start, end in zip(starts, bounds)]

        named = await asyncio.gather(*(
            self._name(context, [rows[i] for i in group], index, len(found.groups), ids)
            for index, (group, ids) in enumerate(zip(found.groups, spans_ids))))

        spans, layers = [], []
        for ids, (answer, folded) in zip(spans_ids, named):
            start, end = resolve_span(context, ids)
            spans.append({**{f: answer.get(f) for f in fields}, "chunk_ids": ids,
                          "start_ts": round(start, 3), "end_ts": round(end, 3)})
            layers += folded
        covered = {c for span in spans for c in span["chunk_ids"]}
        return {key: spans,
                "count": len(spans),
                "covers_all_chunks": covered >= set(grid),
                "uncovered": sorted(set(grid) - covered),
                "divided": "embeddings",
                "segmentation": {
                    "embedder": self.embedder.key,
                    "max_spans": self.max_spans,
                    "min_span_s": self.min_span_s,
                    "first_pass": found.first_pass,
                    "floored": found.floored,
                    "merged": found.merged,
                    # The first pass's boundaries: the chunk a span starts at, the similarity
                    # across the gap, and its depth.
                    "boundaries": [{"chunk_id": rows[b["row"]].chunk_id,
                                    "similarity": b["similarity"], "depth": b["depth"]}
                                   for b in found.boundaries]},
                **({"layers": layers} if layers else {})}


__all__ = ["SpansAggregator"]
