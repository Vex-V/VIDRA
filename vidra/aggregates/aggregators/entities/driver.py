"""The `link` kind: the same person or thing across chunks, and an account of each.

Who is who is decided by `linking` (no model). Then one call per linked
entity, concurrently, writes the profile's account from its observations.
With `check: flag` the same call names observations that contradict the
rest: they stay in the entity, marked, and the account is written from the
others. Also reported: how long each entity was in shot, and which were in
shot together.
"""

from __future__ import annotations

import asyncio
from itertools import combinations
from typing import Any, Optional

from vidra.shared.models.base import LLM, Embedder

from ... import definitions
from ...core import inputs
from .linking import Mentions, link_similar, mentions_of, similarity
from ...core.rendering import resolve_span
from ...core.base import DefinitionRunner, listing, schema


class EntitiesAggregator(DefinitionRunner):
    def __init__(self, definition_id: str, llm: Optional[LLM] = None,
                 embedder: Optional[Embedder] = None) -> None:
        super().__init__(definition_id, llm)
        # The same embedder resolution `embed` uses.
        from vidra.shared.models.base import require
        from vidra.shared.models.roles import resolve
        self.embedder = resolve("embedder", embedder)
        require("embedder", self.embedder)

    @property
    def model_key(self) -> str:
        """Who writes the accounts, which space identity is measured in, and the rule."""
        return (f"{self.llm.key}|{self.embedder.key}|{self.entry['rule']}/"
                f"{'mutual' if self.entry['mutual'] else 'any'}")

    def read(self, context: Any, one: inputs.Input) -> Mentions:
        chosen = definitions.selection(self.definition, one)
        return Mentions(chosen, mentions_of(context, chosen))

    async def _run(self, context: Any, read: Mentions) -> dict[str, Any]:
        entry, mentions = self.entry, read.items
        linking = {"embedder": self.embedder.key, "rule": entry["rule"],
                   "mutual": entry["mutual"], "field": read.selection.field,
                   "keys": list(read.selection.keys)}
        base = {"profile": self.definition, "check": entry["check"],
                "mentions": len(mentions)}
        if len(mentions) < 2:
            return {**base, "entities": [], "count": 0, "linked": 0, "narrated": 0,
                    "doubted": 0, "together": [], "linking": linking,
                    "note": "one mention; nothing to link"}

        sim, standardised = similarity(mentions, self.embedder, entry)
        linked = link_similar(mentions, sim, entry["rule"], entry["mutual"],
                              entry.get("threshold"))
        # Which scale the threshold is on.
        linking["measure"] = "z-blend" if standardised else "cosine"

        entities = []
        for group in linked.groups:
            ordered = sorted(group, key=lambda i: (mentions[i].chunk_id, mentions[i].sampler_id))
            chunk_ids = sorted({mentions[i].chunk_id for i in ordered})
            start, end = resolve_span(context, chunk_ids)
            entities.append({
                "field": mentions[ordered[0]].field,
                "label": max((mentions[i].signature for i in ordered), key=len),
                "appearances": len(chunk_ids),
                "chunk_ids": chunk_ids,
                "start_ts": round(start, 3), "end_ts": round(end, 3),
                "observed_s": round(sum(context.span_of(c)[1] - context.span_of(c)[0]
                                        for c in chunk_ids), 3),
                "mentions": [{"key": mentions[i].key, "chunk_id": mentions[i].chunk_id,
                              "sampler_id": mentions[i].sampler_id, **mentions[i].entry}
                             for i in ordered],
            })
        entities.sort(key=lambda e: (-e["appearances"], e["start_ts"]))
        for number, entity in enumerate(entities):
            entity["entity_id"] = f"e{number:03d}"

        present = [e for e in entities if e["appearances"] >= entry["min_appearances"]]
        narrate = present[:entry["max_narratives"]]
        accounts = await asyncio.gather(*(self._narrate(context, e, read) for e in narrate))
        for entity, account in zip(narrate, accounts):
            entity.update(account)

        together = sorted((
            {"a": a["entity_id"], "b": b["entity_id"], "chunk_ids": shared,
             "count": len(shared)}
            for a, b in combinations(present, 2)
            if (shared := sorted(set(a["chunk_ids"]) & set(b["chunk_ids"])))),
            key=lambda p: (-p["count"], p["a"], p["b"]))

        return {
            **base,
            "entities": entities,
            "count": len(entities),
            "linked": len(present),
            "narrated": len(narrate),
            "doubted": sum(len(e.get("doubts") or []) for e in entities),
            "together": together[:50],
            "linking": {**linking, "threshold": linked.threshold,
                        "calibration_pairs": linked.calibration_pairs,
                        "candidate_pairs": linked.candidate_pairs},
        }

    async def _narrate(self, context: Any, entity: dict[str, Any],
                       read: Mentions) -> dict[str, Any]:
        entry = self.entry
        keys = list(read.selection.keys)
        shown = keys + [k for k in entry["story"] if k not in keys]
        lines = []
        for number, mention in enumerate(entity["mentions"], 1):
            start, end = context.span_of(mention["chunk_id"])
            names = shown or sorted(k for k in mention
                                    if k not in ("key", "chunk_id", "sampler_id"))
            said = " | ".join(f"{k}: {mention[k]}" for k in names
                              if str(mention.get(k) or "").strip())
            lines.append(f"({number}) [{start:.0f}-{end:.0f}s] {said}")
        prompt = entry["instruction"]
        flag = entry["check"] == "flag"
        if flag:
            prompt += "\n\n" + definitions.kind_text("link")["check"]
        prompt += "\n\n" + "\n".join(lines)
        if entry["transcript"] and context.transcript is not None:
            spoken = [f"[{context.span_of(c)[0]:.0f}-{context.span_of(c)[1]:.0f}s] {said}"
                      for c in entity["chunk_ids"]
                      if (said := (context.transcript.text_of(c) or "").strip())]
            if spoken:
                prompt += "\n\nWhat was said during those segments:\n" + "\n".join(spoken)

        properties = self.properties()
        if flag:
            # Doubts first; the account is written without them.
            properties = {**listing("doubts", {"observation": {"type": "integer"},
                                               "reason": {"type": "string"}}),
                          **properties}
        answer = await self.llm.complete(prompt, schema(self.name, properties),
                                         definitions.system())

        doubts = []
        for doubt in (answer.get("doubts") or []) if flag else []:
            number = doubt.get("observation")
            if isinstance(number, int) and 1 <= number <= len(entity["mentions"]):
                mention = entity["mentions"][number - 1]
                if "doubt" not in mention:
                    mention["doubt"] = doubt.get("reason", "")
                    doubts.append({"key": mention["key"], "reason": mention["doubt"]})
        account = {f: answer.get(f) for f in entry["fields"]}
        if doubts and len(doubts) == len(entity["mentions"]):
            # Every observation disputed: no account.
            account = {f: None for f in entry["fields"]}
        return {"account": account, "doubts": doubts}


__all__ = ["EntitiesAggregator"]
