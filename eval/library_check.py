"""The whole library, checked: every component, every verb, both pipelines.

    python -m eval.library_check                      # free and offline
    python -m eval.library_check --local              # also whisper, pyannote, clip, yolo,
                                                      #   objects, text (free, a few minutes)
    python -m eval.library_check --llm                # also the paid models (a few cents)
    python -m eval.library_check --database supabase  # also export + search (writes)
    python -m eval.library_check --keep out/          # keep what it wrote

Every step checks what it claims, and the script exits non-zero if any check
fails. It writes under a scratch folder, never `data/out`. By default no paid
model runs: the transcriber and describer are stubs and the embedder,
`ner` and `sentiment` are local checkpoints.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import vidra
from vidra.shared.config import paths

PRIMARY = Path("samples/Chernobyl.mp4")        # 205 s, picture and narration
SECOND = Path("samples/fixtures/cuts.mp4")     # 60 s, picture only, real cuts

FREE = {"describer": "stub", "embedder": "local"}
failures: list[str] = []
passed = 0


# ----------------------------------------------------------------- harness

def check(ok: Any, what: str) -> None:
    """One claim, printed either way; failures are collected, not raised."""
    global passed
    if ok:
        passed += 1
        print(f"    ok    {what}")
    else:
        failures.append(what)
        print(f"    FAIL  {what}")


def refused(what: str, call: Callable[[], Any], fragment: str = "") -> None:
    """A call the library must refuse with a `VidraError`; `fragment`, when
    given, must be in the message.
    """
    try:
        call()
    except vidra.VidraError as exc:
        said = str(exc).splitlines()[0]
        check(fragment.lower() in said.lower(),
              f"refuses {what}: {type(exc).__name__}: {said[:110]}")
        return
    except Exception as exc:                                 # noqa: BLE001
        check(False, f"refuses {what} -- but with a bare {type(exc).__name__}: {exc}")
        return
    check(False, f"refuses {what} -- it did not")


class MemoryDatabase(vidra.Database):
    """A database that keeps what it is sent and searches it by cosine, so the
    export and search seam is checked without a server.
    """

    name = "memory"

    def __init__(self) -> None:
        self.documents: list[tuple[str, str, dict[str, Any]]] = []
        self.prompts: list[dict[str, Any]] = []
        self.definitions: list[dict[str, Any]] = []
        self.sources: dict[str, dict[str, Any]] = {}
        self.answers: list[tuple[str, dict[str, Any], list, list]] = []
        self.units: list[tuple[str, dict[str, Any], str]] = []
        self.searched: list[tuple[str, Any]] = []
        self.closed = False

    def latest(self, artifact: str) -> dict[str, dict[str, Any]]:
        return {vid: doc for vid, kind, doc in self.documents if kind == artifact}

    def write(self, video_id, artifact, document):
        self.documents.append((video_id, artifact, document))
        return True

    def write_prompts(self, rows):
        self.prompts += rows
        return len(rows)

    def write_definitions(self, rows):
        self.definitions += rows
        return len(rows)

    def write_source(self, source):
        self.sources[source["source_id"]] = source

    def write_answer(self, source_id, answer, items, mentions):
        self.answers.append((source_id, answer, items, mentions))

    def write_aggregate_units(self, source_id, units, embedder_key):
        self.units += [(source_id, u, embedder_key) for u in units if u.get("vector")]
        return sum(1 for u in units if u.get("vector"))

    def search(self, vector, query, embedder_key, limit=20, video_ids=None,
               sampler=None, question=None, strategy=None, chunk_ids=None,
               structured=None):
        self.searched.append((embedder_key, list(video_ids) if video_ids else None))
        scored = []
        for vid, doc in self.latest("embedded").items():
            if doc.get("embedder") != embedder_key or (video_ids and vid not in video_ids):
                continue
            for u in doc["units"]:
                if not u.get("vector") or (sampler and u["sampler_id"] != sampler) \
                        or (question and u.get("question") != question) \
                        or (strategy and u.get("sampler") != strategy) \
                        or (chunk_ids and u["chunk_id"] not in chunk_ids):
                    continue
                scored.append((cosine(vector, u["vector"]), vid, u))
        scored.sort(key=lambda s: -s[0])
        return [{"video_id": vid, "chunk_id": u["chunk_id"], "sampler_id": u["sampler_id"],
                 "sampler": u.get("sampler", ""), "question": u.get("question", ""),
                 "content": u.get("content", ""), "structured": u.get("structured", {}),
                 "score": 1 / (60 + rank), "dense_rank": rank, "text_rank": None,
                 "start_ts": None, "end_ts": None}
                for rank, (_, vid, u) in enumerate(scored[:limit], start=1)]

    def search_aggregates(self, vector, query, embedder_key, level, limit=5,
                          source_ids=None):
        ranked = sorted(((cosine(vector, u["vector"]), sid, u)
                         for sid, u, key in self.units
                         if key == embedder_key and u["level"] == level
                         and (not source_ids or sid in source_ids)),
                        key=lambda r: -r[0])
        return [{"source_id": sid, "item_id": u["item_id"], "level": u["level"],
                 "aggregate_id": u["aggregate_id"], "content": u["content"],
                 "start_ts": u.get("start_ts"), "end_ts": u.get("end_ts"),
                 "video_ids": self.sources.get(sid, {}).get("video_ids", []),
                 "similarity": score}
                for score, sid, u in ranked[:limit]]

    def spans(self, video_id):
        timeline = self.latest("timeline").get(video_id)
        return [(c["start_ts"], c["end_ts"]) for c in timeline["chunks"]] if timeline else []

    def video_ids(self):
        return sorted(self.latest("timeline"))

    def close(self):
        self.closed = True


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
    return dot / norm if norm else 0.0


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------ 1 · video_rag

def models_() -> None:
    """Models -- who answers each role, built once and checked when built."""
    import dataclasses
    import os

    from vidra import Models
    from vidra.shared.models.roles import unpack

    section("0 models")
    models = Models(**FREE)
    check(models.embedder == "local" and models.llm is None,
          f"Models(**FREE) holds what was given: {models}")
    check(models.resolved()["embedder"].startswith("local/"),
          f"resolved() names each role now: {models.resolved()['embedder']}")
    try:
        models.embedder = "openai"                           # type: ignore[misc]
        check(False, "a Models is frozen")
    except dataclasses.FrozenInstanceError:
        check(True, "a Models is frozen: it cannot change after it was checked")
    refused("an unknown provider", lambda: Models(describer="nope"), "nope")
    refused("a provider that cannot serve the role", lambda: Models(embedder="anthropic"))
    refused("an empty spec", lambda: Models(llm=""), "llm")
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        refused("a named role with no key", lambda: Models(llm="anthropic"), "ANTHROPIC")
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved
    check(unpack(models, embedder=None, llm="openai") == {"embedder": "local", "llm": "openai"},
          "unpack: Models fills what the call left None")
    refused("a role set on Models and as a keyword",
            lambda: unpack(models, embedder="openai"), "twice")
    refused("models that are not a Models", lambda: unpack({"embedder": "local"}, embedder=None),
            "Models")

    # Keys given in code: they outrank the environment and never print.
    from vidra.shared.models import providers
    openai = providers.get("openai")
    saved = {v: os.environ.pop(v) for v in openai.key_vars if v in os.environ}
    try:
        refused("an openai role with no key anywhere",
                lambda: Models(describer="openai"), "keys=")
        keyed = Models(describer="openai", keys={"openai": "sk-test-not-real"})
        check("sk-test" not in repr(keyed) and "sk-test" not in str(keyed.resolved()),
              f"Models(keys=...) builds, and its repr carries no key: {keyed!r}")
        with keyed:
            inside = providers.api_key(openai)
        try:
            providers.api_key(openai)
            outside = "found"
        except providers.ProviderUnavailable:
            outside = "refused"
        check(inside == "sk-test-not-real" and outside == "refused",
              "the key reaches a call inside `with models:` and nothing outside it")
        os.environ["OPENAI_API_KEY"] = "sk-from-environment"
        with keyed:
            check(providers.api_key(openai) == "sk-test-not-real",
                  "a key handed over outranks the environment's")
        os.environ.pop("OPENAI_API_KEY")
    finally:
        os.environ.update(saved)
    refused("keys for a provider nobody has",
            lambda: Models(keys={"opnai": "sk-x"}), "opnai")
    refused("an empty key", lambda: Models(keys={"openai": " "}), "non-empty")
    refused("an env_file that is not there",
            lambda: vidra.configure(env_file="nope/.env"), "does not exist")
    refused("an empty hf_token", lambda: vidra.configure(hf_token=""), "hf_token")


def media_(into: Path) -> tuple[Path, dict[str, Path]]:
    """media -- a file in, `media.json` in a folder it makes."""
    from vidra.video_rag import layout, media

    section("1 media")
    produced = media.media(PRIMARY, into)
    home = Path(produced.stats["home"])
    at = layout(home)
    check(at["media"].exists(), f"media(source, into) wrote {at['media'].name} in {home.name}/")
    described = media.load(at["media"])
    check(described.has_video and described.has_audio,
          f"load -> Media: {described.duration_s:.1f}s, both streams")

    probe = media.split(PRIMARY, video_id="probe")
    check(probe.source == described.source and not (into / "probe").exists(),
          "split describes the same file and writes nothing")

    # Only a *different* file is a conflict; the same file again reuses its id.
    check(media.media(PRIMARY, into).video_id == produced.video_id,
          "the same file again reuses its folder")
    clash = into.parent / "clash" / PRIMARY.name
    clash.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SECOND, clash)                     # another video, same filename
    minted = media.media(clash, into)
    check(minted.video_id != produced.video_id,
          f"a different file wanting '{produced.video_id}' is minted "
          f"'{minted.video_id}' (on_conflict='new')")
    refused("a taken id under on_conflict='refuse'",
            lambda: media.media(clash, into, produced.video_id, "refuse"), "")
    refused("a file that is not there", lambda: media.media("samples/nope.mp4", into))
    text = into.parent / "notes.txt"
    text.write_text("not a video", encoding="utf-8")
    refused("a file that is not media", lambda: media.media(text, into))
    refused("an id that climbs out of `into`",
            lambda: media.media(PRIMARY, into, "../../escaped"))
    refused("an id that hides itself", lambda: media.media(PRIMARY, into, "_hidden"))
    refused("a conflict rule nobody has",
            lambda: media.media(PRIMARY, into, None, "nope"), "on_conflict")
    shutil.rmtree(Path(minted.stats["home"]))

    # Name and recording time: defaults, given values, and a re-run keeping them.
    check(described.name == PRIMARY.name and described.recorded_at is None,
          f"name defaults to the filename ({described.name!r}); no creation "
          f"time in the file reads as None")
    stamped = into.parent / "stamped" / "stamped.mp4"
    _stamp(SECOND, stamped, "2024-03-15T09:30:00.000000Z")
    check(media.split(stamped).recorded_at == "2024-03-15T09:30:00+00:00",
          "recorded_at is read from the container's creation_time")
    named = stamped.parent / "named"                   # not `into`: it would keep them
    given = media.media(PRIMARY, named, name="Reactor night",
                        recorded_at="1986-04-26T01:23:40+04:00")
    check((given.stats["name"], given.stats["recorded_at"])
          == ("Reactor night", "1986-04-26T01:23:40+04:00"),
          "a given name and time replace the defaults")
    again = media.load(Path(media.media(PRIMARY, named).artifacts["media"]))
    check((again.name, again.recorded_at)
          == ("Reactor night", "1986-04-26T01:23:40+04:00"),
          "a re-run naming neither keeps what an earlier run was given")
    refused("an empty name", lambda: media.media(PRIMARY, named, name=" "), "name")
    refused("a year where a time was asked for",
            lambda: media.split(PRIMARY, recorded_at="2019"), "recorded_at")
    refused("a time that is not one",
            lambda: media.media(PRIMARY, named, recorded_at="yesterday"), "recorded_at")
    shutil.rmtree(stamped.parent)
    return home, at


def _stamp(source: Path, out: Path, creation_time: str) -> None:
    """A copy of `source` whose container records a creation time (remuxed)."""
    import av

    out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source)) as src, av.open(str(out), "w") as dst:
        dst.metadata["creation_time"] = creation_time
        streams = {s.index: dst.add_stream_from_template(s) for s in src.streams}
        for packet in src.demux():
            if packet.dts is None:
                continue
            packet.stream = streams[packet.stream.index]
            dst.mux(packet)


def audio_(at: dict[str, Path]) -> None:
    """audio -- media.json in, transcript.raw.json out."""
    from vidra.video_rag import audio, media

    section("2 audio")
    produced = audio.audio(at["media"], at["raw_transcript"],
                           transcriber="stub", diarizer="none")
    raw = audio.load(at["raw_transcript"])
    check(raw.words, f"audio(media, out) -> {len(raw.words)} words (stub transcriber)")
    again = audio.listen(media.load(at["media"]), transcriber="stub", diarizer="none")
    check(len(again.words) == len(raw.words), "listen(Media) is the same work, on objects")
    check(produced.stats.get("speakers", 0) == 0 and all(
        s.get("speaker") is None for s in raw.segments),
        "no diarizer -> speaker None, never an invented SPEAKER_00")
    refused("a setting no chosen backend takes",
            lambda: audio.audio(at["media"], at["raw_transcript"],
                                transcriber="stub", diarizer="none", language="en"),
            "language")


def boundaries_(at: dict[str, Path], scratch: Path) -> None:
    """boundaries -- evidence (a scene pass, or none), then the grid."""
    from vidra.video_rag import boundaries, media

    section("3 boundaries")
    found = boundaries.evidence(at["cuts"], "scene", media=at["media"])
    cuts = read(at["cuts"])
    check(at["cuts"].exists(), f"evidence('scene') scored the picture: "
                               f"{len(cuts.get('cuts', []))} cuts ({found.component})")
    boundaries.boundaries(at["media"], at["timeline"], "scene", cuts=at["cuts"],
                          min_s=5, max_s=20)
    grid = boundaries.load(at["timeline"])
    check(len(grid) >= 3 and all(e - s <= 20.0001 for s, e in grid.spans),
          f"boundaries(scene, max 20s) -> {len(grid)} chunks, none over 20s")

    started = time.perf_counter()
    boundaries.retune(at["cuts"], scratch / "retuned.json", threshold=45)
    check(time.perf_counter() - started < 1.0,
          "retune re-thresholds the cached scores without decoding")

    none = boundaries.evidence(scratch / "none.json", "uniform", media=at["media"])
    check("evidence" in none.skipped, "evidence('uniform') is a step that did nothing, said so")
    described = media.load(at["media"])
    check(boundaries.detect("uniform", media=described) is None,
          "detect('uniform') -> None: there are no cuts to find")
    rebuilt = boundaries.timeline(described, "scene", cuts=boundaries_cuts(at),
                                  min_s=5, max_s=20)
    check(rebuilt.fingerprint() == grid.fingerprint(),
          "timeline(Media, cuts) on objects == the grid on disk")
    refused("a floor above the ceiling",
            lambda: boundaries.boundaries(at["media"], scratch / "t.json", "scene",
                                          cuts=at["cuts"], min_s=30), "")
    refused("a setting the policy never reads (speaker + silence)",
            lambda: boundaries.evidence(scratch / "c.json", "speaker",
                                        raw_transcript=at["raw_transcript"],
                                        silence_s=2.0), "silence")


def boundaries_cuts(at: dict[str, Path]) -> Any:
    from vidra.shared.contracts.documents import Cuts
    from vidra.shared.storage import files
    return files.read(at["cuts"], Cuts)


def video_(at: dict[str, Path], scratch: Path) -> None:
    """video -- which frames each sampler keeps, into a store."""
    from vidra.video_rag import boundaries, media, video

    section("4 video")
    produced = video.video(at["media"], at["timeline"], at["manifest"],
                           store=at["store"], sampler="uniform:[overview,scene]",
                           every_n=5)
    manifest = video.load(at["manifest"])
    runs = manifest.config["samplers"]
    check(len(runs) == 1 and runs[0].get("prompts") == ["overview", "scene"],
          "uniform:[overview,scene] is one pass over the video answering two questions")
    stored = len(list(at["store"].glob("*.jpg")))
    check(stored == produced.stats["stored_frames"] > 0,
          f"{stored} frames in the store, as the receipt says")

    again = video.ingest(media.load(at["media"]), boundaries.load(at["timeline"]),
                         "uniform:[overview,scene]", every_n=5,
                         frames=video.FrameStore(scratch / "store2"))
    check(again.stats["frames_sampled"] == manifest.stats["frames_sampled"],
          "ingest(Media, Timeline) on objects picks the same frames")
    untouched = scratch / "never"
    refused("a question nobody defined, before anything is decoded",
            lambda: video.video(at["media"], at["timeline"], scratch / "m.json",
                                store=untouched, sampler="uniform:nope"), "nope")
    check(not untouched.exists(), "...and nothing was decoded or stored")
    refused("a setting no chosen sampler reads",
            lambda: video.video(at["media"], at["timeline"], scratch / "m.json",
                                sampler="uniform", confidence=0.5), "confidence")


def cut_(at: dict[str, Path]) -> None:
    """cut -- the transcript onto the grid."""
    from vidra.video_rag import audio, boundaries, cut

    section("5 cut")
    cut.cut(at["timeline"], at["raw_transcript"], at["transcript"])
    transcript = cut.load(at["transcript"])
    grid = boundaries.load(at["timeline"])
    check(len(transcript.chunks) == len(grid),
          "every chunk kept, silent ones with empty text: chunk ids stay shared")
    check(cut.apply(grid, audio.load(at["raw_transcript"])).as_dict()["chunks"]
          == transcript.as_dict()["chunks"], "apply(Timeline, RawTranscript) == the file")


def describe_(at: dict[str, Path]) -> None:
    """describe -- one answer per (chunk, sampler, question), resuming with
    `previous=`.
    """
    from vidra.video_rag import boundaries, describe, video

    section("6 describe")
    first = describe.describe(at["manifest"], at["timeline"], at["store"],
                              at["descriptions"], describer="stub")
    pairs = first.stats["described"]
    check(pairs > 0, f"describe -> {pairs} answers (stub describer)")
    blocks = describe.load(at["descriptions"]).chunks[0]["samplers"]
    check(set(blocks) == {"uniform:overview", "uniform:scene"},
          "one pass, two questions -> two answers per chunk, each its own id")
    again = describe.describe(at["manifest"], at["timeline"], at["store"],
                              at["descriptions"], previous=at["descriptions"],
                              describer="stub")
    check(again.stats["described"] == 0 and again.stats["skipped"] == pairs,
          f"previous= -> 0 described, {pairs} skipped")
    answered = describe.answer(video.load(at["manifest"]), boundaries.load(at["timeline"]),
                               video.FrameStore(at["store"]), describer="stub")
    check(answered.stats["described"] == pairs, "answer(Manifest, Timeline, frames) on objects")
    refused("limit=0, which reads as both 'none' and 'no limit'",
            lambda: describe.describe(at["manifest"], at["timeline"], at["store"],
                                      at["descriptions"], describer="stub", limit=0), "limit")
    refused("a describer nobody has",
            lambda: describe.describe(at["manifest"], at["timeline"], at["store"],
                                      at["descriptions"], describer="nope"), "nope")


def questions_(at: dict[str, Path], scratch: Path) -> None:
    """A custom question with its own shape, built from fields."""
    from vidra.video_rag import describe, video
    from vidra.video_rag.describe import library, prompts

    section("6b custom questions")
    added = describe.add_question(
        "hazards_demo", "These {n} frames span {span}. List every hazard.",
        fields={"hazards": {"type": "list", "about": "Each hazard, briefly."},
                "severity": {"type": "text", "about": "The worst one.",
                             "one_of": ["none", "low", "high"]},
                "people": {"type": "list", "about": "Who is exposed.",
                           "of": {"who": "by appearance", "risk": "how exposed"}}},
        about="hazards, for the check")
    check(added["fields"].keys() == {"hazards", "severity", "people"}
          and added["shape"] is None and not added["builtin"],
          f"add_question -> question(): its own fields {sorted(added['fields'])}")
    schema = prompts.schema_for("hazards_demo")
    check({"hazards", "severity", "people"} <= set(schema["properties"]),
          f"a custom shape compiles into its own schema: {sorted(schema['properties'])}")
    check(schema["properties"]["severity"].get("enum") == ["none", "low", "high"],
          "one_of becomes an enum -- a vocabulary a search can filter on")
    check("hazards_demo" in library.shapes() and "hazards_demo" in describe.questions(),
          "and survives a reload from disk")
    video.video(at["media"], at["timeline"], scratch / "q_manifest.json",
                store=at["store"], sampler="uniform:hazards_demo", every_n=5)
    asked = describe.describe(scratch / "q_manifest.json", at["timeline"], at["store"],
                              scratch / "q_descriptions.json", describer="stub")
    check(asked.stats["described"] > 0, "uniform:hazards_demo -> described with it")

    plain = describe.add_question("mood_demo", "Describe the mood in two sentences.")
    check(plain["shape"] == "prose" and plain["fields"] == {} and plain["summary"] == "brief",
          "no fields and no shape -> prose only, as `overview` answers")
    shared = describe.add_question("crew_demo", "These {n} frames: who is working?",
                                   shape="people")
    check(shared["shape"] == "people" and "people" in shared["fields"],
          "shape= shares a shipped shape")
    check(describe.question("yolo")["builtin"], "question() reads a built-in the same way")

    refused("redefining a built-in question",
            lambda: describe.add_question("yolo", "anything {n}"), "built-in")
    try:
        describe.add_question("yolo", "anything {n}")
    except describe.ProtectedPrompt:
        check(True, "a built-in is a ProtectedPrompt, catchable by name")
    refused("a placeholder nobody fills",
            lambda: describe.add_question("typo_demo", "These {frames} frames"), "frames")
    refused("a field type nobody has",
            lambda: describe.add_question("bad_demo", "{n} frames",
                                          fields={"x": {"type": "number", "about": "x"}}),
            "type")
    refused("a field with no `about`",
            lambda: describe.add_question("bad_demo", "{n} frames",
                                          fields={"x": {"type": "text"}}), "about")
    refused("fields and a shape at once",
            lambda: describe.add_question("bad_demo", "{n}", shape="people",
                                          fields={"x": {"type": "text", "about": "x"}}),
            "not both")
    refused("a summary length with a shipped shape",
            lambda: describe.add_question("bad_demo", "{n}", summary="brief"), "fields")
    refused("a name with a colon", lambda: describe.add_question("a:b", "{n}"), "colon")
    refused("a shape nobody ships",
            lambda: describe.add_question("bad_demo", "{n}", shape="nope"), "shipped")
    refused("another question's own shape, which dies with it",
            lambda: describe.add_question("bad_demo", "{n}", shape="hazards_demo"),
            "shipped")
    for name in ("hazards_demo", "mood_demo", "crew_demo"):
        describe.remove_question(name)
    check(not {"hazards_demo", "mood_demo", "crew_demo"} & set(describe.questions()),
          "remove_question takes them back out")
    refused("removing a built-in", lambda: describe.remove_question("overview"), "built in")


def embed_(at: dict[str, Path]) -> None:
    """embed -- text and vectors into embedded.json."""
    from vidra.video_rag import describe, embed

    section("7 embed")
    first = embed.embed(at["embedded"], descriptions=at["descriptions"],
                        transcript=at["transcript"], timeline=at["timeline"],
                        embedder="local")
    units = embed.load(at["embedded"]).units
    check(units and all(u.get("vector") for u in units),
          f"embed -> {len(units)} units, every one with a vector ({first.stats.get('embedder', 'local')})")
    again = embed.embed(at["embedded"], descriptions=at["descriptions"],
                        transcript=at["transcript"], previous=at["embedded"],
                        timeline=at["timeline"], embedder="local")
    check(again.stats.get("embedded", -1) == 0, "previous= -> 0 embedded, nothing paid twice")
    encoded = embed.encode(descriptions=describe.load(at["descriptions"]), embedder="local")
    check(len(encoded) == sum(1 for u in units if u["sampler_id"] != "transcript"),
          "encode(Descriptions) on objects: one unit per answer")
    refused("batch=0", lambda: embed.embed(at["embedded"], descriptions=at["descriptions"],
                                           embedder="local", batch=0), "batch")


def pipeline_(into: Path) -> Path:
    """video_rag() -- the eight components in order, over one folder."""
    from vidra.video_rag import Options, validate, video_rag

    section("8 video_rag() -- the pipeline")
    run = video_rag(SECOND, into, sampler="uniform", policy="scene", **FREE)
    check(run.video_id and "embedded" in run.artifacts(),
          f"{run.video_id}: {', '.join(sorted(run.artifacts()))}")
    check(run.skipped.get("audio", "").startswith("the file carries no audio"),
          f"a picture-only file: audio skipped -- {run.skipped.get('audio')}")
    again = video_rag(SECOND, into, sampler="uniform", policy="scene", **FREE)
    described = next(s for s in again.steps if s.component == "describe")
    check(described.stats["described"] == 0, "a second run resumes: 0 described")
    problems = validate(Options(source=SECOND, into=into, policy="vad", use_audio=False))
    check(any("soundtrack" in p for p in problems),
          f"validate: a policy from a stream it is not reading -> {problems[0][:70]}")
    problems = validate(Options(source=SECOND, into=into, sampler="yolo:overvew"))
    check(any("overvew" in p for p in problems), "validate: a typo'd question, before any work")
    return run.home


def database_(into: Path) -> MemoryDatabase:
    """A Database handed in: what a run exports to it, and search reading it back."""
    import os

    from vidra import Models, Supabase
    from vidra import aggregates
    from vidra.shared.storage import files
    from vidra.shared.storage.database import as_database
    from vidra.video_rag import Options, retrieve, validate, video_rag

    section("9 a database, handed in (held in memory: nothing is sent)")
    models = Models(**FREE)
    memory = MemoryDatabase()
    run = video_rag(PRIMARY, into, sampler="uniform:[overview,scene]",
                    models=models, database=memory)
    sent = {artifact for _, artifact, _ in memory.documents}
    check(not run.problems and sent >= {"media", "raw_transcript", "timeline", "manifest",
                                        "transcript", "descriptions", "embedded"},
          f"video_rag(models=, database=) exported {', '.join(sorted(sent))}")
    check("store" not in sent, "...but not the frame store, which is a directory")
    check({p["name"] for p in memory.prompts} >= {"overview", "scene"},
          f"...and the {len(memory.prompts)} questions describe asked")
    check(not memory.closed, "a run leaves the caller's database open")
    embedded = memory.latest("embedded")[run.video_id]
    check(embedded["embedder"].startswith("local"),
          f"the index was built by the Models embedder: {embedded['embedder']}")

    # A unit's own text should find its own chunk, timed by the database's grid.
    spoken = [u for u in embedded["units"] if u["sampler_id"] == "transcript"
              and len(u["content"]) > 80]
    unit = spoken[len(spoken) // 2]         # not chunk 0, which a broken rank also puts first
    found, notes = retrieve.search(unit["content"], run.video_id, models=models,
                                   database=memory)
    check(found and found[0].chunk_id == unit["chunk_id"] and found[0].end_ts > 0,
          f"search(a unit's own text) -> chunk {found[0].chunk_id if found else None} "
          f"first (wanted {unit['chunk_id']}), timed by the database's grid")
    check(memory.searched[-1] == (embedded["embedder"], [run.video_id]),
          "...asked in the embedder's space, scoped to that video")
    only, _ = retrieve.search("narration", run.video_id, models=models, database=memory,
                              sampler="transcript")
    check(only and all(h["sampler_id"] == "transcript" for m in only for h in m.hits),
          f"sampler='transcript' -> {len(only)} moments, every hit from the transcript")
    everywhere, notes = retrieve.search("a reactor", models=models, database=memory)
    check(everywhere and any("scope" in n for n in notes),
          "no video named -> every video the database holds")
    # The aggregates' copy: the free and local ones for real, then a summary and
    # chapters written by hand (no llm here) to check what is embedded and how
    # each level is searched.
    answers = run.home / "aggregates"
    done = aggregates.aggregate(run.home, answers, models=models, database=memory,
                                stats=True, ner=True)
    names = [i for _, a, items, _ in memory.answers if a["aggregator"] == "ner"
             for i in items]
    check(not done.stats["problems"] and run.video_id in memory.sources
          and memory.sources[run.video_id]["video_ids"] == [run.video_id],
          f"aggregate(database=) -> the source and {len(memory.answers)} answers")
    check(names and all(i["item_kind"] == "name" and i["end_ts"] for i in names),
          f"ner's {len(names)} names became items, each timed by the grid")
    check(done.stats["exported_units"] == 0,
          "stats and ner embed nothing: a count and a name are not vectors")
    grid = files.read_json(run.home / "timeline.json")["chunks"]
    last = len(grid) - 1
    files.write_json(answers / "summary.json", {
        "document": "aggregate", "version": 1, "video_id": run.video_id,
        "aggregate_id": "summary", "tier": "llm", "inputs_fingerprint": "x",
        "payload": {"summary": "A documentary about a nuclear accident at Chernobyl.",
                    "topics": ["reactor", "radiation"]}, "stats": {}})
    files.write_json(answers / "chapters.json", {
        "document": "aggregate", "version": 1, "video_id": run.video_id,
        "aggregate_id": "chapters", "tier": "llm", "inputs_fingerprint": "x",
        "payload": {"chapters": [
            {"title": "The map", "summary": "A map of Europe shows where fallout spread.",
             "chunk_ids": [0], "start_ts": grid[0]["start_ts"], "end_ts": grid[0]["end_ts"]},
            {"title": "The reactor", "summary": "Inside the reactor the core overheats.",
             "chunk_ids": list(range(1, last + 1)), "start_ts": grid[1]["start_ts"],
             "end_ts": grid[last]["end_ts"]}]}, "stats": {}})
    from vidra.aggregates.database.export import export
    from vidra.shared.contracts.documents import Timeline
    problems, units = export(run.video_id,
                             Timeline.from_dict(files.read_json(run.home / "timeline.json")),
                             {"summary": str(answers / "summary.json"),
                              "chapters": str(answers / "chapters.json")},
                             {}, memory, embedder="local")
    check(not problems and units == 3,
          f"export -> {units} vectors: the summary, and one per chapter")
    video = aggregates.search("the nuclear accident", level="source", models=models,
                              database=memory)
    part = aggregates.search("the core overheating", level="span", models=models,
                             database=memory)
    check(video and video[0]["source_id"] == run.video_id and video[0]["item_id"] == "",
          "search(level='source') -> the video, by its summary")
    check(part and part[0]["item_id"] == "chapter-1" and part[0]["start_ts"] is not None
          and all(r["level"] == "span" for r in part),
          f"search(level='span') -> the chapter, with its span: {part[0]['item_id'] if part else None}")
    refused("a level nobody has",
            lambda: aggregates.search("x", level="video", models=models, database=memory),
            "level")

    refused("an embedder on Models and as a keyword",
            lambda: retrieve.search("x", run.video_id, embedder="local", models=models,
                                    database=memory), "twice")
    refused("a describer on Models and as a keyword",
            lambda: video_rag(PRIMARY, into, describer="stub", models=models), "twice")
    refused("a database nobody has", lambda: as_database("mysql"), "mysql")
    check(any("database" in p for p in validate(Options(source=PRIMARY, into=into,
                                                         database="mysql"))),
          "validate names an unknown database before any work")
    with MemoryDatabase() as scoped:
        pass
    check(scoped.closed, "a Database is a context manager that closes")

    # The real backend, built but never connected.
    built = Supabase(url="https://example.supabase.co", key="sk-not-a-real-key")
    check("sk-not-a-real-key" not in repr(built), f"Supabase(...) connects on first use: {built!r}")
    saved = {k: os.environ.pop(k) for k in ("SUPABASE_URL",) if k in os.environ}
    try:
        refused("Supabase() with no URL anywhere", lambda: Supabase(), "URL")
    finally:
        os.environ.update(saved)
    return memory


def search_(home: Path, database: str) -> None:
    """retrieve.search -- ranked in Postgres, so only with --database."""
    from vidra.video_rag import layout, search, video_rag

    section("9b export + search, live")
    run = video_rag(PRIMARY, home.parent, sampler="uniform:[overview,scene]",
                    database=database, **FREE)
    check(not run.problems, f"every artifact exported to {database}: {run.problems or 'no problems'}")
    moments, notes = search("people in a shop", run.video_id, embedder="local",
                            grids={run.video_id: layout(run.home)["timeline"]})
    check(moments, f"search -> {len(moments)} moments; notes: {notes}")


# ----------------------------------------------------------- 2 · aggregates

def aggregate_components(home: Path, scratch: Path, llm: bool) -> None:
    """Every aggregator as a component: one input in, one answer out."""
    from vidra.aggregates import (Inapplicable, coverage, entities, ner, prompt,
                                      select, sentiment, speakers, stats)

    section("10 aggregates, one component at a time")
    out = scratch / "components"
    s = select.select(home, out / "in.json", "sentiment")
    excerpt = read(out / "in.json")
    check(s.stats["excerpt"] > 0 and excerpt["timeline"]["chunks"],
          f"select -> an excerpt: {s.stats['excerpt']} rows of "
          f"`{excerpt['selection']}`, with its grid inside")

    sentiment.sentiment(out / "in.json", out / "sentiment.json")
    tone = read(out / "sentiment.json")["payload"]
    check(len(tone["per_chunk"]) == s.stats["excerpt"],
          f"sentiment(excerpt) -> a tone per chunk, mean {tone['mean']}")
    ner.ner(out / "in.json", out / "ner.json", labels=("person", "object"))
    found = read(out / "ner.json")["payload"]
    check(found["labels"] == ["person", "object"],
          f"ner(excerpt, labels=...) -> {found['count']} entities; the labels reached the model")
    reused = ner.ner(out / "in.json", out / "ner2.json", previous=out / "ner.json",
                     labels=("person", "object"))
    check(reused.stats["reused"], "previous= -> the earlier answer, not recomputed")
    changed = ner.ner(out / "in.json", out / "ner3.json", previous=out / "ner.json",
                      labels=("person",))
    check(not changed.stats["reused"], "other labels -> recomputed: the label set is the version")

    for component, name in ((stats.stats, "stats"), (coverage.coverage, "coverage"),
                            (speakers.speakers, "speakers")):
        component(home, out / f"{name}.json")
        check((out / f"{name}.json").exists(), f"{name}(record) -> {name}.json")
    counted = read(out / "stats.json")["payload"]
    check(counted["words"] > 0 and counted["chunks"] == excerpt["timeline"]["chunk_count"],
          f"stats counts the record: {counted['chunks']} chunks, {counted['words']} words")

    # The stub describer names no people, so there is nobody to link.
    select.select(home, out / "people.json", "entities:people")
    try:
        entities.entities("people", out / "people.json", out / "entities.json",
                          embedder="local")
        check(False, "entities over no mentions says so")
    except Inapplicable as why:
        check(True, f"entities over no mentions is Inapplicable: {str(why)[:70]}")

    refused("ner handed a record", lambda: ner.ner(home, out / "x.json"), "select")
    refused("stats handed an excerpt file", lambda: stats.stats(out / "in.json",
                                                                out / "x.json"), "folder")
    refused("an excerpt named as a record's timeline",
            lambda: stats.stats({"timeline": out / "in.json"}, out / "x.json"),
            "holds `excerpt`, not `timeline`")
    refused("a selection for a counter",
            lambda: select.select(home, out / "x.json", "stats"), "whole record")
    refused("a link profile run as a prompt",
            lambda: prompt.prompt("entities:people", out / "people.json", out / "x.json"),
            "link profile")
    refused("an excerpt handed to entities",
            lambda: entities.entities("people", out / "in.json", out / "x.json"), "sightings")

    # A component takes `models=`; the refusal comes before any call.
    from vidra import Models
    keyed = Models(llm="openai", keys={"openai": "sk-test-not-real"})
    import inspect
    from vidra.aggregates import answer
    check(all("models" in inspect.signature(f).parameters
              for f in (prompt.prompt, entities.entities, answer)),
          "prompt(), entities() and answer() take models=")
    refused("an llm on models= and as llm=",
            lambda: prompt.prompt("summary", out / "in.json", out / "x.json",
                                  llm="anthropic", models=keyed), "twice")

    # Chapter boundaries from made-up vectors with three clear topics, then a cap.
    import random
    from vidra.aggregates import settings_of, uses_embedder, validate
    from vidra.aggregates.spans.segment import segment
    rng = random.Random(1)
    topics = [[[(1.0 if i == axis else 0.0) + rng.gauss(0, 0.15) for i in range(8)]
               for _ in range(size)] for axis, size in ((0, 5), (1, 4), (2, 6))]
    vectors = [v for topic in topics for v in topic]
    found = segment(vectors)
    check([g[0] for g in found.groups] == [0, 5, 9],
          f"segment finds the topic changes, at chunk precision: {[g[0] for g in found.groups]}")
    capped = segment(vectors, most=2)
    check(len(capped.groups) == 2 and capped.merged == 1
          and sum(len(g) for g in capped.groups) == len(vectors),
          "most=2 merges the two most alike neighbours, and every chunk stays in one span")
    check(settings_of("chapters") == ["max_spans", "min_span_s"] and uses_embedder("chapters")
          and not uses_embedder("summary"),
          "chapters takes max_spans, min_span_s and an embedder; summary takes none")
    refused("max_spans=0", lambda: prompt.prompt("chapters", out / "in.json", out / "x.json",
                                                 embedder="local", max_spans=0), "1 or more")
    check(any("max_spans" in p for p in validate(summary={"data": True, "max_spans": 2})),
          "max_spans on a prompt that is not spans is refused before anything runs")

    if llm:
        select.select(home, out / "summary_in.json", "summary")
        made = prompt.prompt("summary", out / "summary_in.json", out / "summary.json")
        words = read(out / "summary.json")["payload"].get("word_count")
        check(words, f"prompt('summary') -> {words} words by {made.stats.get('model')}")
        prompt.prompt("chapters", out / "summary_in.json", out / "chapters.json",
                      embedder="local", max_spans=2)
        chapters = read(out / "chapters.json")["payload"]
        check(chapters["count"] <= 2 and chapters["covers_all_chunks"],
              f"prompt('chapters', max_spans=2) -> {chapters['count']} chapters covering "
              f"every chunk, first pass {chapters['segmentation']['first_pass']}")


def aggregate_pipeline(home: Path, scratch: Path, llm: bool) -> None:
    """aggregate() -- runs what it is handed data for, and nothing else."""
    from vidra.aggregates import aggregate, validate

    section("11 aggregate() -- the pipeline")
    out = scratch / "answers"
    handed = {"stats": True, "coverage": True, "sentiment": True,
              "ner": {"data": "transcript+*", "labels": ["person", "object"]}}
    made = aggregate(home, out, **handed)
    check(set(made.stats["aggregates"]) == set(handed),
          f"handed four -> ran exactly those: {', '.join(made.stats['aggregates'])}")
    check(sorted(p.name for p in (out / "inputs").iterdir()) == ["ner.json", "sentiment.json"],
          "what each text aggregator read is kept in inputs/")
    again = aggregate(home, out, previous=out, **handed)
    check(again.stats["computed"] == 0 and again.stats["current"] == 4,
          "previous= -> 4 reused, 0 computed")

    two = aggregate(home, scratch / "two",
                    summary="transcript,uniform:overview" if llm else None,
                    sentiment="transcript,uniform:scene")
    check({"sentiment~transcript", "sentiment~uniform-scene"} <= set(two.stats["aggregates"]),
          f"`a,b` -> two answers: {', '.join(two.stats['aggregates'])}")
    from_file = aggregate(None, scratch / "from_file",
                          sentiment=str(out / "inputs" / "sentiment.json"))
    check(from_file.stats["aggregates"] == ["sentiment"],
          "handed an excerpt file instead of a selection, with no record at all")

    check(validate() and "nothing would run" in validate()[0], "handing nothing is refused")
    check(validate(stats="transcript"), "a selection for a counter is refused")
    check(validate(nerr=True), "an aggregator nobody has is refused")
    check(validate(ner={"data": True, "colour": "red"}), "a setting it does not take is refused")
    check(validate(summary="clip:nope"), "a question nobody defined is refused")
    if llm:
        people = Path("data/eval/people/out/test")
        if people.exists():
            linked = aggregate(people, scratch / "people", entities_people=True,
                               embedder="local")
            e = read(Path(linked.artifacts["entities:people"]))["payload"]
            check(e["count"], f"entities_people=True -> {e['count']} entities, "
                              f"{e['narrated']} with an account")


def custom_prompts(home: Path, scratch: Path, llm: bool) -> None:
    """add_prompt / add_profile: definitions of your own, run like the built-ins."""
    from vidra import aggregates
    from vidra.aggregates import DefinitionError, ProtectedDefinition

    section("12 custom aggregate prompts")
    fields = {"report": {"type": "text", "about": "what happened, in order"},
              "severity": {"type": "text", "about": "how serious",
                           "one_of": ["none", "minor", "major"]}}
    added = aggregates.add_prompt("incident_report",
                                  "Write an incident report for this video.",
                                  fields, about="an incident report")
    check(added["kind"] == "fold" and not added["builtin"]
          and "incident_report" in aggregates.available()
          and aggregates.tier_of("incident_report") == "llm",
          "add_prompt -> a fold prompt listed in available(), tier llm")
    check(not aggregates.validate(incident_report=True),
          "validate(incident_report=True) passes: it runs as a keyword like summary")
    steps = aggregates.add_prompt("steps_demo", "List each step of the procedure.",
                                  {"step": {"type": "text", "about": "the step"}},
                                  kind="items", key="steps")
    check(steps["kind"] == "items" and steps["key"] == "steps",
          "kind='items', key='steps' -> an items prompt")
    profile = aggregates.add_profile(
        "tools_demo", "objects",
        "Below are observations of what may be the same object. Say who used it.",
        {"use": {"type": "text", "about": "who used it, for what"}},
        identity=["object", "appearance"], story=["context"])
    check(profile["name"] == "entities:tools_demo" and profile["kind"] == "link"
          and profile["rule"] == "max",
          "add_profile -> entities:tools_demo, the built-in defaults filled in")
    check(aggregates.definition("summary")["builtin"]
          and aggregates.definition("entities:people")["kind"] == "link",
          "definition() reads a built-in prompt and profile too")

    one = {"a": {"type": "text", "about": "a"}}
    refused("a built-in's name", lambda: aggregates.add_prompt("summary", "x", one), "built-in")
    check(issubclass(ProtectedDefinition, DefinitionError),
          "ProtectedDefinition is a DefinitionError")
    refused("a code aggregator's name", lambda: aggregates.add_prompt("ner", "x", one),
            "reserved")
    refused("an unknown kind", lambda: aggregates.add_prompt("z", "x", one, kind="free"),
            "kind")
    refused("no fields", lambda: aggregates.add_prompt("z", "x", {}), "fields")
    refused("a key the kind writes",
            lambda: aggregates.add_prompt("z", "x", {"chunk_id": one["a"]}, kind="items"),
            "written by the kind")
    refused("an input naming no question",
            lambda: aggregates.add_prompt("z", "x", one, inputs="nope_q"), "nope_q")
    refused("a profile over a field no shape has",
            lambda: aggregates.add_profile("z", "nofield", "x", one, identity=["q"]),
            "nofield")

    if llm:
        made = aggregates.aggregate(home, scratch / "custom", incident_report=True)
        answer = read(Path(made.artifacts["incident_report"]))["payload"]
        check(answer.get("report") and answer.get("severity") in ("none", "minor", "major"),
              f"aggregate(incident_report=True) -> severity {answer.get('severity')!r}, "
              f"{len(str(answer.get('report')).split())} words of report")

    for name in ("incident_report", "steps_demo"):
        aggregates.remove_prompt(name)
    aggregates.remove_profile("tools_demo")
    check(not {"incident_report", "steps_demo", "entities:tools_demo"}
          & set(aggregates.available()),
          "remove_prompt / remove_profile take them back out")
    check(aggregates.validate(incident_report=True),
          "...and a run naming a removed prompt is refused")
    refused("removing a built-in", lambda: aggregates.remove_prompt("summary"), "built in")


def several(home: Path, second: Path, scratch: Path) -> None:
    """Two videos combined into one record, then aggregated."""
    from vidra.aggregates import aggregate, combine
    from vidra.aggregates.combination import origin
    from vidra.video_rag import boundaries

    section("13 several videos")
    both = combine([home, second], scratch / "both")
    grid = boundaries.load(scratch / "both" / "timeline.json")
    parts = [len(boundaries.load(h / "timeline.json")) for h in (home, second)]
    check(len(grid) == sum(parts), f"combine -> {len(grid)} chunks = {parts[0]} + {parts[1]}")
    where = origin(grid, parts[0])
    check(where["video_id"] == second.name and where["chunk_id"] == 0
          and where["start_ts"] == 0.0,
          f"origin(chunk {parts[0]}) -> {where['video_id']} chunk 0 at 0.0s")
    made = aggregate([home, second], scratch / "together", stats=True, coverage=True)
    counted = read(Path(made.artifacts["stats"]))["payload"]
    check(counted["chunks"] == len(grid) and counted["derived_from"] == "combined",
          f"aggregate([a, b]) combines first: {counted['chunks']} chunks, "
          f"record kept in together/record")
    refused("one video named twice", lambda: combine([home, home], scratch / "twice"),
            "more than once")
    _ = both


# ----------------------------------------------------------- 3 · whole run

def whole_run(scratch: Path) -> None:
    """workflow.process -- extract, then aggregate; and where it writes."""
    from vidra import Models, workflow

    section("14 workflow")
    memory = MemoryDatabase()
    options = workflow.Options(source=SECOND, into=scratch / "wf", tier="local",
                               use_audio=False, models=Models(**FREE), database=memory)
    run = workflow.process(options)
    ran = run.steps[-1].stats["aggregates"]
    check(set(ran) >= {"stats", "coverage", "sentiment", "ner"},
          f"tier='local' -> every free and local aggregator: {', '.join(ran)}")
    answers = memory.answers
    check(not run.problems and len(answers) == len(ran)
          and {"timeline", "embedded"} <= {kind for _, kind, _ in memory.documents},
          f"database= -> the extraction and all {len(answers)} answers exported")
    check(any("set twice" in p for p in workflow.validate(workflow.Options(
              source=SECOND, llm="openai", models=Models(llm="openai")))),
          "a role on Models and as a field is a validation problem")
    check(workflow.validate(workflow.Options(source=SECOND, tier="bogus")),
          "a tier nobody has is refused")

    # `into=None` is the data root.
    outside = paths.data_root()
    vidra.configure(data_root=scratch / "elsewhere")
    run = workflow.process(workflow.Options(source=SECOND, use_audio=False, **FREE))
    check(run.home.parent == paths.out_root() == scratch / "elsewhere" / "out",
          f"configure(): into=None wrote under {paths.out_root()}")
    vidra.configure(data_root=outside)
    check(paths.data_root() == outside, "...and configuring it back restores it")


# ------------------------------------------- 4 · extension points and helpers

def helpers(at: dict[str, Path], home: Path, memory: Any, scratch: Path) -> None:
    """A sampler of your own, the rate limits, progress callbacks, search's
    filters, and the helpers the registries publish.
    """
    from vidra.video_rag import boundaries, describe, embed, retrieve, video
    from vidra.video_rag.video import samplers
    from vidra.video_rag.video.samplers import Sampler, register

    section("15 extension points and helpers")
    out = scratch / "helpers"
    out.mkdir(parents=True, exist_ok=True)

    @register
    class ThirdDemo(Sampler):
        """Every third decimated frame: the smallest sampler there is."""
        name = "third_demo"

        def propose(self, frame: Any, chunk_local_index: int) -> bool:
            return chunk_local_index % 3 == 0

    check("third_demo" in samplers.available(), "register(cls) -> listed in available()")
    video.video(at["media"], at["timeline"], out / "third.json", store=at["store"],
                sampler="third_demo:overview")
    kept = [f["chunk_local_index"] for c in video.load(out / "third.json").chunks
            for f in c["samplers"]["third_demo"]["frames"]]
    check(kept and all(i % 3 == 0 for i in kept),
          f"video(sampler='third_demo:overview') runs it: {len(kept)} frames, "
          f"every one at a multiple of 3")
    check(samplers.build("uniform", every_n=2).every_n == 2, "build(name, **settings)")
    refused("building a sampler nobody has", lambda: samplers.build("nope"), "nope")

    # Rate limits sit in the base class, so they hold for every sampler.
    video.video(at["media"], at["timeline"], out / "capped.json", store=at["store"],
                sampler="uniform", max_per_chunk=2)
    counts = [c["samplers"]["uniform"]["frame_count"]
              for c in video.load(out / "capped.json").chunks]
    check(counts and max(counts) <= 2 and min(counts) >= 1,
          f"max_per_chunk=2 -> per chunk {counts}: at most 2, at least 1")
    video.video(at["media"], at["timeline"], out / "spaced.json", store=at["store"],
                sampler="uniform", min_interval_s=4.0)
    gaps = [b["media_ts"] - a["media_ts"]
            for c in video.load(out / "spaced.json").chunks
            for a, b in zip(c["samplers"]["uniform"]["frames"],
                            c["samplers"]["uniform"]["frames"][1:])]
    check(gaps and min(gaps) >= 4.0 - 1e-6,
          f"min_interval_s=4 -> no two frames of a chunk closer than {min(gaps):.2f}s")
    refused("max_per_chunk=0", lambda: video.video(at["media"], at["timeline"],
                                                   out / "x.json", max_per_chunk=0),
            "max_per_chunk")

    # Progress: one event per unit of work, counting up to the total.
    events: dict[str, list] = {}
    def record(p: Any) -> None:
        events.setdefault(p.component, []).append(p)
    video.video(at["media"], at["timeline"], out / "progress.json", store=at["store"],
                sampler="uniform", every_n=5, on_progress=record)
    describe.describe(out / "progress.json", at["timeline"], at["store"],
                      out / "progress_desc.json", describer="stub", on_progress=record)
    embed.embed(out / "progress_emb.json", descriptions=out / "progress_desc.json",
                embedder="local", on_progress=record)
    for name, seen in sorted(events.items()):
        check(seen and seen[-1].completed + seen[-1].skipped == seen[-1].total
              and seen[-1].fraction == 1.0,
              f"on_progress from {name}: {len(seen)} events, ending "
              f"{seen[-1].completed}/{seen[-1].total}")
    check(set(events) >= {"video", "describe", "embed"},
          f"video, describe and embed all report: {sorted(events)}")

    # Search's filters, against the database section 9 filled.
    from vidra import Models
    models = Models(**FREE)
    vid = memory.video_ids()[0]
    spans = memory.spans(vid)                  # the grid as the database holds it
    some = lambda **kw: retrieve.search("a reactor building", vid, models=models,  # noqa: E731
                                        database=memory, moments=20, **kw)
    only, _ = some(chunk_ids=[2])
    check(only and {m.chunk_id for m in only} == {2}, "chunk_ids=[2] -> chunk 2 only")
    near, _ = some(chunk_ids=[2], window=1)
    check(near and {m.chunk_id for m in near} <= {1, 2, 3} and len({m.chunk_id for m in near}) > 1,
          f"chunk_ids=[2], window=1 -> its neighbours too: {sorted({m.chunk_id for m in near})}")
    start, end = spans[1][0] + 0.5, spans[2][1] - 0.5
    timed, _ = some(after=start, before=end)
    check(timed and all(m.end_ts > start and m.start_ts < end for m in timed),
          f"after={start:.1f}, before={end:.1f} -> only chunks overlapping that window: "
          f"{sorted({m.chunk_id for m in timed})}")
    asked, notes = some(question="overview")
    check(asked and all(h["question"] == "overview" for m in asked for h in m.hits)
          and any("agreement" in n for n in notes),
          "question='overview' -> overview answers only, and a note on what filtering costs")
    nothing, notes = some(after=10_000)
    check(nothing == [] and any("window" in n for n in notes),
          "a window past the end -> no moments and a note saying why")

    # The aggregates' registry, as data.
    from vidra import aggregates
    from vidra.aggregates import combination, select
    check(set(aggregates.up_to("free")) == {"stats", "speakers", "coverage"},
          f"up_to('free') -> {sorted(aggregates.up_to('free'))}")
    local = set(aggregates.up_to("local"))
    check(local == {"stats", "speakers", "coverage", "ner", "sentiment"},
          f"up_to('local') adds the local models: {sorted(local)}")
    check((aggregates.kind_of("summary"), aggregates.kind_of("chapters"),
           aggregates.kind_of("events"), aggregates.kind_of("entities:people"),
           aggregates.kind_of("stats")) == ("fold", "spans", "items", "link", None),
          "kind_of: fold, spans, items, link -- and None for code")
    check(aggregates.takes_inputs("ner") and not aggregates.takes_inputs("stats"),
          "takes_inputs: ner reads a selection, stats the record")
    check(aggregates.tier_of("stats") == "free" and aggregates.tier_of("ner") == "local",
          "tier_of: what each costs")

    ctx = aggregates.context(at["timeline"], descriptions=at["descriptions"],
                             transcript=at["transcript"], manifest=at["manifest"])
    bare = aggregates.context(at["timeline"])
    why = aggregates.missing(aggregates.build("speakers"), bare)
    check(why and "transcript" in why and aggregates.missing(aggregates.build("speakers"), ctx) is None,
          f"missing(speakers): None with a transcript, and without one: {why}")
    excerpt = select.pick(ctx, "ner")
    check(len(excerpt.rows) > 0, f"pick(record, 'ner') -> an excerpt of {len(excerpt.rows)} rows")
    counted = aggregates.answer("stats", ctx)
    check(counted.payload["chunks"] == len(boundaries.load(at["timeline"])),
          "answer('stats', Context) on objects, nothing written")

    made = aggregates.aggregate(home, out / "answers", stats=True, coverage=True,
                                sentiment=True)
    loaded = aggregates.load_all(out / "answers")
    check(set(loaded) == set(made.stats["aggregates"]),
          f"load_all(folder) -> every answer by id: {sorted(loaded)}")
    check(aggregates.load_all(out / "nowhere") == {}, "load_all of an absent folder -> {}")
    taken = aggregates.load_input(out / "answers" / "inputs" / "sentiment.json")
    check(type(taken).__name__ == "Excerpt", "load_input -> an Excerpt, by what the file is")
    from vidra.aggregates import definitions
    version = definitions.version_of("prompts", "summary")
    rows = aggregates.definition_rows({"summary": version, "entities:people":
                                       definitions.version_of("profiles", "people")})
    check([(r["name"], r["kind"], r["builtin"]) for r in rows]
          == [("entities:people", "link", True), ("summary", "fold", True)]
          and rows[1]["version"] == version and "instruction" in rows[1]["definition"],
          "definition_rows -> what each definition said, as database rows")

    a, b = "Chernobyl", "cuts"
    check(combination.default_id([a, b]) == combination.default_id([a, b])
          != combination.default_id([b, a]),
          f"default_id is stable and ordered: {combination.default_id([a, b])}")
    from vidra.shared.contracts.documents import Descriptions, Manifest, Timeline, Transcript
    from vidra.shared.storage import files
    part = combination.Part(files.read(at["timeline"], Timeline),
                            files.read(at["descriptions"], Descriptions),
                            files.read(at["transcript"], Transcript),
                            files.read(at["manifest"], Manifest))
    other = scratch / "out"
    second = next(p for p in other.iterdir() if p.is_dir() and p != home
                  and (p / "embedded.json").exists())
    merged = combination.merge([part, combination.Part(
        files.read(second / "timeline.json", Timeline),
        files.read(second / "descriptions.json", Descriptions),
        None, files.read(second / "manifest.json", Manifest))])
    check(len(merged.timeline) == len(part.timeline) + len(boundaries.load(second / "timeline.json")),
          f"merge(Parts) on objects -> {len(merged.timeline)} chunks, nothing written")

    # Embedders, as functions.
    from vidra.shared.models import embedders
    hashed = embedders.build("hash")
    check(hashed.embed(["a reactor"]) == hashed.embed(["a reactor"]),
          f"build('hash') -> a deterministic, free embedder: {hashed.key}")
    query, document = embedders.prefixes_for("BAAI/bge-small-en-v1.5")
    check(query and not document, f"prefixes_for(bge): a query prefix {query[:30]!r}, none for passages")
    bge = embedders.build("local")
    q = embedders.query_vector(bge, "a reactor")
    d = bge.embed(["a reactor"])[0]
    check(len(q) == len(d) == 384 and q != d,
          "query_vector goes through the query side: same width, different vector")
    check(embedders.key_for("local", "m", 8, "", "") == "local:m:8"
          and embedders.key_for("local", "m", 8, "q: ", "") != "local:m:8",
          "key_for: name:model:dims, with a suffix when the prefixes are hand-set")


# --------------------------------------------- 5 · the models themselves

def local_models(at: dict[str, Path], scratch: Path) -> None:
    """Whisper and pyannote, the speech grids, and every model-backed sampler --
    free, but they need `[local]` and `[audio]` and take a few minutes.
    """
    from vidra import aggregates
    from vidra.video_rag import audio, boundaries, cut, media, video

    section("16 local models: whisper, pyannote, clip, yolo, objects, text")
    out = scratch / "local"
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    audio.audio(at["media"], out / "raw.json", transcriber="whisper", diarizer="pyannote")
    raw = audio.load(out / "raw.json")
    speakers = {s.get("speaker") for s in raw.segments} - {None}
    check(len(raw.words) > 300 and raw.turns and len(speakers) >= 1,
          f"audio(whisper, pyannote) -> {len(raw.words)} words, {len(raw.segments)} segments, "
          f"{len(raw.turns)} turns, speakers {sorted(speakers)} "
          f"in {time.perf_counter() - started:.0f}s")
    check(all(w.get("start") is not None for w in raw.words), "every word has a timestamp")

    # The device every torch model defaults to; Whisper alone never takes mps.
    from unittest import mock

    import torch

    from vidra.shared.models.devices import default_device
    expected = ("cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu")
    check(default_device() == expected, f"default_device() -> {default_device()}")
    with mock.patch.object(torch.cuda, "is_available", return_value=False), \
            mock.patch.object(torch.backends.mps, "is_available", return_value=True):
        check(default_device() == "mps", "no cuda but an Apple GPU -> mps")
    refused("whisper on mps",
            lambda: audio.audio(at["media"], out / "x.json", transcriber="whisper",
                                diarizer="none", device="mps"), "mps")

    duration = media.load(at["media"]).duration_s
    for policy in ("vad", "speaker"):
        boundaries.evidence(out / f"{policy}_cuts.json", policy, raw_transcript=out / "raw.json")
        boundaries.boundaries(at["media"], out / f"{policy}.json", policy,
                              cuts=out / f"{policy}_cuts.json", chunk_s=30)
        grid = boundaries.load(out / f"{policy}.json")
        spans = grid.spans
        tiled = spans[0][0] == 0 and all(abs(a[1] - b[0]) < 1e-6 for a, b in zip(spans, spans[1:]))
        check(tiled and abs(spans[-1][1] - duration) < 0.1 and max(e - s for s, e in spans) <= 30.0001,
              f"{policy} grid -> {len(grid)} chunks tiling {duration:.1f}s, none over 30s")

    cut.cut(out / "vad.json", out / "raw.json", out / "transcript.json")
    record = {"timeline": out / "vad.json", "transcript": out / "transcript.json"}
    aggregates.speakers.speakers(record, out / "speakers.json")
    who = read(out / "speakers.json")["payload"]
    check(who["speakers"] == len(speakers) and who["speech_ratio"] > 0.5,
          f"speakers(record) over real turns -> {who['speakers']} speaker(s), "
          f"{who['speech_ratio']:.0%} speech, monologue {who['monologue']}")

    # Every model-backed sampler, on a minute of a shop.
    shop = media.media(Path("samples/test2.mp4"), out / "videos")
    home = Path(shop.stats["home"])
    boundaries.boundaries(home / "media.json", home / "timeline.json", "uniform")
    started = time.perf_counter()
    made = video.video(home / "media.json", home / "timeline.json", home / "manifest.json",
                       store=home / "store", sampler="clip,yolo,objects",
                       vocabulary=["person", "shelf", "bottle", "basket"])
    manifest = video.load(home / "manifest.json")
    runs = [r["name"] for r in manifest.config["samplers"]]
    per = {r: [c["samplers"][r]["frame_count"] for c in manifest.chunks] for r in runs}
    check(runs == ["clip", "yolo", "objects"] and all(min(v) >= 1 for v in per.values()),
          f"clip, yolo, objects in one pass -> frames per chunk {per} "
          f"in {time.perf_counter() - started:.0f}s")
    check(made.stats["stored_frames"] <= made.stats["frames_sampled"],
          f"{made.stats['frames_sampled']} picks, {made.stats['stored_frames']} files: "
          f"a frame two samplers kept is stored once")

    from recovery.recreate import recreate, verify
    rebuilt = recreate(home / "manifest.json", out / "rebuilt")
    names = {f"{f['index']:07d}.jpg" for c in manifest.chunks
             for r in c["samplers"].values() for f in r["frames"]}
    same = verify(out / "rebuilt", home / "store", names)
    check(not rebuilt["missing"] and len(same["identical"]) == same["compared"] == len(names),
          f"recovery.recreate rebuilt the store {len(same['identical'])}/{len(names)} byte-identical")

    slides = media.media(Path("samples/fixtures/slides.mp4"), out / "videos")
    sh = Path(slides.stats["home"])
    boundaries.boundaries(sh / "media.json", sh / "timeline.json", "uniform")
    started = time.perf_counter()
    video.video(sh / "media.json", sh / "timeline.json", sh / "manifest.json",
                store=sh / "store", sampler="text", languages=["en"])
    texts = [c["samplers"]["text"]["frame_count"] for c in video.load(sh / "manifest.json").chunks]
    check(texts and min(texts) >= 1,
          f"text(languages=['en']) on slides -> frames per chunk {texts} "
          f"in {time.perf_counter() - started:.0f}s")
    return None


def paid_models(home: Path, scratch: Path) -> None:
    """A real describer and embedder, the items kind, and linking real people.
    Costs a few cents.
    """
    from vidra.aggregates import entities, prompt, select
    from vidra.video_rag import boundaries, describe, embed, media, video

    section("17 paid models: a real describer, embedder, events and linking")
    out = scratch / "paid"
    shop = media.media(Path("samples/test2.mp4"), out)
    h = Path(shop.stats["home"])
    boundaries.boundaries(h / "media.json", h / "timeline.json", "uniform")
    video.video(h / "media.json", h / "timeline.json", h / "manifest.json",
                store=h / "store", sampler="yolo,uniform:overview", every_n=5)
    started = time.perf_counter()
    made = describe.describe(h / "manifest.json", h / "timeline.json", h / "store",
                             h / "descriptions.json", describer="openai")
    answers = describe.load(h / "descriptions.json")
    people = [p for c in answers.chunks for p in
              (c["samplers"].get("yolo", {}).get("structured") or {}).get("people", [])]
    check(made.stats["described"] > 0 and people
          and all({"appearance", "clothing"} <= set(p) for p in people),
          f"describe(openai) -> {made.stats['described']} answers, {len(people)} people "
          f"as objects with appearance and clothing, in {time.perf_counter() - started:.0f}s")
    overview = [c["samplers"]["uniform:overview"] for c in answers.chunks
                if "uniform:overview" in c["samplers"]]
    check(overview and all(not o.get("structured") and len(o["description"].split()) > 30
                           for o in overview),
          "uniform:overview -> prose only, no structured fields")

    embed.embed(h / "embedded.json", descriptions=h / "descriptions.json",
                timeline=h / "timeline.json", embedder="openai")
    units = embed.load(h / "embedded.json").units
    check(units and all(len(u["vector"]) == 1536 for u in units),
          f"embed(openai) -> {len(units)} units of 1536 dimensions")

    select.select(h, out / "people.json", "entities:people")
    linked = entities.entities("people", out / "people.json", out / "entities.json",
                               embedder="local")
    e = read(out / "entities.json")["payload"]
    check(e["count"] and e["count"] <= len(people),
          f"entities('people') over {len(people)} sightings -> {e['count']} entities, "
          f"{e.get('narrated')} with an account ({linked.stats.get('model')})")

    select.select(h, out / "events_in.json", "events")
    prompt.prompt("events", out / "events_in.json", out / "events.json")
    ev = read(out / "events.json")["payload"]
    grid = read(out / "events_in.json")["timeline"]
    ids = {c["chunk_id"] for c in grid["chunks"]}
    check(ev["count"] and all(x["chunk_id"] in ids and x["end_ts"] > x["start_ts"]
                              for x in ev["events"]),
          f"prompt('events') -> {ev['count']} events, every one on a real chunk and timed by the grid")


# ------------------------------------------------------------------ main

def main() -> int:
    from vidra.shared.config import env
    env.load()        # an entry point reads .env; the library never does
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llm", action="store_true",
                    help="also run the paid models: a describer, an embedder, "
                         "summary, chapters, events and entities")
    ap.add_argument("--local", action="store_true",
                    help="also run the local models: whisper, pyannote, and the "
                         "clip, yolo, objects and text samplers (free, minutes)")
    ap.add_argument("--database", default=None,
                    help="also export to this database and search it -- WRITES to it")
    ap.add_argument("--keep", type=Path, default=None,
                    help="write here and keep it, rather than a temporary folder")
    args = ap.parse_args()
    for sample in (PRIMARY, SECOND):
        if not sample.exists():
            raise SystemExit(f"no such file: {sample} -- run from the checkout root")

    scratch = args.keep or Path(tempfile.mkdtemp(prefix="vidra-example-"))
    scratch.mkdir(parents=True, exist_ok=True)
    was = paths.data_root()
    vidra.configure(data_root=scratch / "root")
    print(f"writing under {scratch}")
    started = time.perf_counter()
    def step(call: Callable[..., Any], *arguments: Any) -> Any:
        """Run one section; a crash is recorded as a failure and the run continues."""
        try:
            return call(*arguments)
        except Exception as exc:                             # noqa: BLE001
            import traceback
            where = traceback.extract_tb(exc.__traceback__)[-1]
            check(False, f"{call.__name__} crashed: {type(exc).__name__}: "
                         f"{str(exc)[:150]}  ({Path(where.filename).name}:{where.lineno})")
            return None

    try:
        into = scratch / "out"
        step(models_)
        home, at = media_(into)          # everything after reads what this wrote
        step(audio_, at)
        step(boundaries_, at, scratch)
        step(video_, at, scratch)
        step(cut_, at)
        step(describe_, at)
        step(questions_, at, scratch)
        step(embed_, at)
        second = step(pipeline_, into)
        memory = step(database_, scratch / "db")
        if args.database:
            step(search_, home, args.database)
        step(aggregate_components, home, scratch, args.llm)
        step(aggregate_pipeline, home, scratch, args.llm)
        step(custom_prompts, home, scratch, args.llm)
        if second is not None:
            step(several, home, second, scratch)
        step(whole_run, scratch)
        if memory is not None:
            step(helpers, at, home, memory, scratch)
        if args.local:
            step(local_models, at, scratch)
        if args.llm:
            step(paid_models, home, scratch)
    finally:
        vidra.configure(data_root=was)
        if args.keep is None:
            shutil.rmtree(scratch, ignore_errors=True)

    section("result")
    print(f"  {passed} checks passed, {len(failures)} failed "
          f"in {time.perf_counter() - started:.0f}s")
    for what in failures:
        print(f"  FAILED  {what}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
