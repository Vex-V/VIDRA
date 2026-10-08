"""What an answer becomes in a database: its source, its items, and what is
embedded.

Reads the answers `aggregate` wrote and hands a `Database`:

    source   what the answers are about: one video, or several end to end
             (`video_ids`, and each member's chunk and time offsets)
    answer   one per aggregate, its payload whole
    items    one per thing an answer places in time -- a chapter, an event, a
             name `ner` found, a linked entity -- with its chunks and times;
             an entity's sightings as mentions
    units    the text that is embedded, at one of three levels:

                 source   the final summary      which video is this about
                 span     each chapter           which part of which video
                 entity   each linked entity     who, across every video

Events, names and counts are not embedded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from vidra.shared.contracts.documents import Timeline, fingerprint_of
from vidra.shared.contracts.units import render
from .. import definitions, kind_of
from ..core.inputs import definition_of

#: Unit levels, coarse to fine.
LEVELS = ("source", "span", "entity")

#: Texts per embedding request, as `embed` sends them.
BATCH = 64


def source_of(source_id: str, timeline: Timeline) -> dict[str, Any]:
    """The source row: which videos, and where each sits on the source's clock."""
    combined = (timeline.params or {}).get("combined") or []
    members = ([{"video_id": m["video_id"], "first_chunk": m["first_chunk"],
                 "chunk_count": m["chunks"], "offset_s": m["offset_s"]} for m in combined]
               or [{"video_id": source_id, "first_chunk": 0,
                    "chunk_count": len(timeline), "offset_s": 0.0}])
    return {"source_id": source_id,
            "video_ids": [m["video_id"] for m in members],
            "members": members,
            "duration_s": round(timeline.duration_s, 3),
            "chunk_count": len(timeline)}


def _span(timeline: Timeline, chunk_ids: Sequence[int]) -> tuple[Optional[float], Optional[float]]:
    valid = [c for c in chunk_ids if 0 <= c < len(timeline)]
    if not valid:
        return None, None
    return (round(min(timeline.bounds_of(c)[0] for c in valid), 3),
            round(max(timeline.bounds_of(c)[1] for c in valid), 3))


def _singular(key: str) -> str:
    return key[:-1] if key.endswith("s") else key


def _fields(name: str) -> list[str]:
    return list(definitions.get(*definitions.locate(name)).get("fields") or {})


def _entity_text(entity: dict[str, Any]) -> str:
    """A linked entity's text: its account, or the identity string it was linked on
    when it has none.
    """
    account = entity.get("account") or {}
    prose = " ".join(str(account[k]).strip() for k in ("description", "narrative")
                     if account.get(k))
    rest = {k: v for k, v in account.items() if k not in ("description", "narrative")}
    return render(prose, rest) if prose or rest else str(entity.get("label") or "")


def answer_row(document: dict[str, Any]) -> dict[str, Any]:
    """One answer as its row: the payload whole, and who made it from what."""
    stats = document.get("stats") or {}
    return {"aggregate_id": document["aggregate_id"],
            "aggregator": definition_of(document["aggregate_id"]),
            "tier": document.get("tier", "free"),
            "payload": document.get("payload") or {},
            "inputs": stats.get("inputs"), "model": stats.get("model"),
            "version": stats.get("version"),
            "inputs_fingerprint": document.get("inputs_fingerprint", "")}


def items_of(document: dict[str, Any], timeline: Timeline
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """An answer's items and, for a linked entity, its mentions. Empty for an answer
    that places nothing in time.
    """
    answer_id = document["aggregate_id"]
    name = definition_of(answer_id)
    payload = document.get("payload") or {}
    kind = kind_of(name) if name != "ner" else None
    items: list[dict[str, Any]] = []
    mentions: list[dict[str, Any]] = []

    if kind in ("spans", "items"):
        key = definitions.get(*definitions.locate(name)).get("key") or kind
        fields = _fields(name)
        for n, item in enumerate(payload.get(key) or []):
            ids = item.get("chunk_ids") or ([item["chunk_id"]] if "chunk_id" in item else [])
            start, end = _span(timeline, ids)
            own = {f: item.get(f) for f in fields}
            items.append({"item_id": f"{_singular(key)}-{n}", "item_kind": _singular(key),
                          "label": str(own.get(fields[0]) or "") if fields else "",
                          "text": render(str(own.get("summary") or ""),
                                         {f: v for f, v in own.items() if f != "summary"}),
                          "chunk_ids": ids, "start_ts": start, "end_ts": end,
                          "data": own})
    elif kind == "link":
        for entity in payload.get("entities") or []:
            ids = entity.get("chunk_ids") or []
            items.append({
                "item_id": entity["entity_id"], "item_kind": "entity",
                "label": str(entity.get("label") or ""), "text": _entity_text(entity),
                "chunk_ids": ids,
                "start_ts": entity.get("start_ts"), "end_ts": entity.get("end_ts"),
                "data": {k: entity.get(k) for k in ("field", "appearances", "observed_s",
                                                    "account", "doubts")}})
            for m in entity.get("mentions") or []:
                mentions.append({
                    "item_id": entity["entity_id"], "mention_key": m["key"],
                    "chunk_id": m["chunk_id"], "sampler_id": m["sampler_id"],
                    "entry": {k: v for k, v in m.items()
                              if k not in ("key", "chunk_id", "sampler_id", "doubt")},
                    "doubt": m.get("doubt")})
    elif name == "ner":
        for n, entity in enumerate(payload.get("entities") or []):
            ids = entity.get("chunk_ids") or []
            start, end = _span(timeline, ids)
            items.append({"item_id": f"name-{n}", "item_kind": "name",
                          "label": str(entity.get("label") or ""),
                          "text": str(entity.get("text") or ""),
                          "chunk_ids": ids, "start_ts": start, "end_ts": end,
                          "data": {"mentions": entity.get("mentions")}})
    return items, mentions


def units_of(document: dict[str, Any], items: list[dict[str, Any]]
             ) -> list[dict[str, Any]]:
    """What of an answer is embedded, as `{level, item_id, content, ...}`."""
    name = definition_of(document["aggregate_id"])
    kind = kind_of(name) if name != "ner" else None
    payload = document.get("payload") or {}
    units: list[dict[str, Any]] = []
    if kind == "fold" and str(payload.get("summary") or "").strip():
        # The final summary only, not its layers.
        units.append({"level": "source", "item_id": "",
                      "content": render(str(payload["summary"]),
                                        {k: payload[k] for k in ("topics", "setting", "notable")
                                         if payload.get(k)}),
                      "start_ts": None, "end_ts": None})
    elif kind in ("spans", "link"):
        level = "span" if kind == "spans" else "entity"
        units += [{"level": level, "item_id": item["item_id"], "content": item["text"],
                   "start_ts": item["start_ts"], "end_ts": item["end_ts"]}
                  for item in items if item["text"].strip()]
    for unit in units:
        unit.update(aggregate_id=document["aggregate_id"],
                    text_hash=fingerprint_of({"content": unit["content"]}))
    return units


def export(source_id: str, timeline: Timeline, answers: dict[str, str],
           definitions_used: dict[str, str], database: Any,
           embedder: Optional[Any] = None) -> tuple[list[str], int]:
    """Every answer file in `answers` into `database`. Returns (problems, units
    written). Failures are reported, not raised; the source goes first, and if
    it fails nothing else is tried. The units are embedded only when the
    database implements `write_aggregate_units`: that is a model call.
    """
    from vidra.shared.storage import files

    problems: list[str] = []
    try:
        database.write_source(source_of(source_id, timeline))
    except Exception as exc:                                  # noqa: BLE001
        return [f"source -> {database.name}: {exc}"], 0

    units: list[dict[str, Any]] = []
    for answer_id, where in sorted(answers.items()):
        try:
            document = files.read_json(Path(where))
            items, mentions = items_of(document, timeline)
            database.write_answer(source_id, answer_row(document), items, mentions)
            units += units_of(document, items)
        except Exception as exc:                              # noqa: BLE001
            problems.append(f"{answer_id} -> {database.name}: {exc}")
    try:
        from .. import definition_rows
        database.write_definitions(definition_rows(definitions_used))
    except Exception as exc:                                  # noqa: BLE001
        problems.append(f"definitions -> {database.name}: {exc}")

    written = 0
    if units and database.implements("write_aggregate_units"):
        try:
            from vidra.shared.models.base import require
            from vidra.shared.models.roles import resolve
            built = resolve("embedder", embedder)
            require("embedder", built)
            texts = [u["content"] for u in units]
            vectors = [v for at in range(0, len(texts), BATCH)
                       for v in built.embed(texts[at:at + BATCH])]
            for unit, vector in zip(units, vectors):
                unit["vector"] = vector
            database.write_aggregate_units(source_id, units, built.key)
            written = len(units)
        except Exception as exc:                              # noqa: BLE001
            problems.append(f"aggregate vectors -> {database.name}: {exc}")
    return problems, written


__all__ = ["LEVELS", "answer_row", "export", "items_of", "source_of", "units_of"]
