"""The whole library, exercised: every component, every verb, both pipelines.

    python example.py                      # free and offline: stubs + local models
    python example.py --llm                # also the paid aggregates (a few cents)
    python example.py --database supabase  # also export + search (writes to it)
    python example.py --keep out/          # keep what it wrote, to look at

This is a test as much as a tour: every step **checks** what it claims and the
script exits non-zero if any check fails. Nothing touches `data/out`: it writes
under a scratch folder and says where.

What it goes through, in the order a caller meets it:

    video_rag      media · audio · boundaries · video · cut · describe · embed
                   each as `component(paths...)`, then as its verb on objects,
                   then what it refuses -- and `video_rag()`, the pipeline
    aggregates     select · stats · coverage · speakers · ner · sentiment ·
                   prompt · entities, each as a component, then `aggregate()`,
                   the pipeline that runs only what it is handed
    several        two videos combined into one record and aggregated
    the whole run  `workflow.process`, and two data roots in one process

**Everything is addressed by path.** A component takes the files it reads and
the file it writes; `video_rag.layout(home)` names every file in a video's
folder from the library's own table, so nothing here spells `timeline.json`.

By default it runs no paid model: the transcriber and describer are stubs, the
embedder is local (bge-small), and `ner` / `sentiment` are local checkpoints.
Their first run downloads weights.
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

import falconvar
from falconvar.shared import paths

PRIMARY = Path("samples/Chernobyl.mp4")        # 205 s, picture and narration
SECOND = Path("samples/fixtures/cuts.mp4")     # 60 s, picture only, real cuts

FREE = {"describer": "stub", "embedder": "local"}
failures: list[str] = []
passed = 0


# ----------------------------------------------------------------- harness

def check(ok: Any, what: str) -> None:
    """One claim. Printed either way; a false one is collected, not raised, so
    one run reports every failure rather than the first."""
    global passed
    if ok:
        passed += 1
        print(f"    ok    {what}")
    else:
        failures.append(what)
        print(f"    FAIL  {what}")


def refused(what: str, call: Callable[[], Any], fragment: str = "") -> None:
    """A call the library must refuse -- with a `FalconvarError` that keeps a
    builtin base, so "anything the library refused" is one `except`, and with
    a message that says what to do (`fragment`, when given, must be in it)."""
    try:
        call()
    except falconvar.FalconvarError as exc:
        said = str(exc).splitlines()[0]
        check(fragment.lower() in said.lower(),
              f"refuses {what}: {type(exc).__name__}: {said[:110]}")
        return
    except Exception as exc:                                 # noqa: BLE001
        check(False, f"refuses {what} -- but with a bare {type(exc).__name__}: {exc}")
        return
    check(False, f"refuses {what} -- it did not")


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------ 1 · video_rag

def media_(into: Path) -> tuple[Path, dict[str, Path]]:
    """media -- a file in, `media.json` in a folder it makes. The one component
    that decides a folder: every later path is composed inside `home`."""
    from falconvar.video_rag import layout, media

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
    refused("a file that is not media", lambda: media.media("CLAUDE.md", into))
    refused("an id that climbs out of `into`",
            lambda: media.media(PRIMARY, into, "../../escaped"))
    refused("an id that hides itself", lambda: media.media(PRIMARY, into, "_hidden"))
    refused("a conflict rule nobody has",
            lambda: media.media(PRIMARY, into, None, "nope"), "on_conflict")
    shutil.rmtree(Path(minted.stats["home"]))
    return home, at


def audio_(at: dict[str, Path]) -> None:
    """audio -- media.json in, transcript.raw.json out. Scanned whole: speaker
    labels come from clustering the entire recording."""
    from falconvar.video_rag import audio, media

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
    from falconvar.video_rag import boundaries, media

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
    from falconvar.shared.contracts.documents import Cuts
    from falconvar.shared.storage import files
    return files.read(at["cuts"], Cuts)


def video_(at: dict[str, Path], scratch: Path) -> None:
    """video -- which frames are worth describing, into a store. One pass may
    answer several questions."""
    from falconvar.video_rag import boundaries, media, video

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
    """cut -- the transcript onto the grid. Free, so a new grid never re-runs
    Whisper."""
    from falconvar.video_rag import audio, boundaries, cut

    section("5 cut")
    cut.cut(at["timeline"], at["raw_transcript"], at["transcript"])
    transcript = cut.load(at["transcript"])
    grid = boundaries.load(at["timeline"])
    check(len(transcript.chunks) == len(grid),
          "every chunk kept, silent ones with empty text: chunk ids stay shared")
    check(cut.apply(grid, audio.load(at["raw_transcript"])).as_dict()["chunks"]
          == transcript.as_dict()["chunks"], "apply(Timeline, RawTranscript) == the file")


def describe_(at: dict[str, Path]) -> None:
    """describe -- one answer per (chunk, sampler run, question). Resume is
    `previous=`: a pair still current is not paid for twice."""
    from falconvar.video_rag import boundaries, describe, video

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
    """A custom question: an instruction and its own shape, built from fields.

    Stored under the data root (`prompts.json`), so it lands in the scratch
    root this script configured. Every custom shape on disk was once dropped
    in silence at load -- `compile_shape` called a function that did not
    exist -- and each question quietly answered in the fallback shape. This is
    the check that would have caught it."""
    from falconvar.video_rag import describe, video
    from falconvar.video_rag.describe import library, prompts

    section("6b custom questions")
    library.add("hazards_demo", "These {n} frames span {span}. List every hazard.",
                fields={"hazards": {"type": "list", "about": "Each hazard, briefly."},
                        "severity": {"type": "text", "about": "The worst one.",
                                     "one_of": ["none", "low", "high"]},
                        "people": {"type": "list", "about": "Who is exposed.",
                                   "of": {"who": "by appearance", "risk": "how exposed"}}})
    schema = prompts.schema_for("hazards_demo")
    check({"hazards", "severity", "people"} <= set(schema["properties"]),
          f"a custom shape compiles into its own schema: {sorted(schema['properties'])}")
    check(schema["properties"]["severity"].get("enum") == ["none", "low", "high"],
          "one_of becomes an enum -- a vocabulary a search can filter on")
    check("hazards_demo" in library.shapes(), "and survives a reload from disk")
    video.video(at["media"], at["timeline"], scratch / "q_manifest.json",
                store=at["store"], sampler="uniform:hazards_demo", every_n=5)
    asked = describe.describe(scratch / "q_manifest.json", at["timeline"], at["store"],
                              scratch / "q_descriptions.json", describer="stub")
    check(asked.stats["described"] > 0, "uniform:hazards_demo -> described with it")
    refused("redefining a built-in question",
            lambda: library.add("yolo", "anything {n}", shape="scene"), "built-in")
    refused("a placeholder nobody fills",
            lambda: library.add("typo_demo", "These {frames} frames", shape="scene"),
            "frames")
    refused("a field type nobody has",
            lambda: library.add("bad_demo", "{n} frames",
                                fields={"x": {"type": "number", "about": "x"}}), "type")
    library.remove("hazards_demo")
    check("hazards_demo" not in prompts.questions(), "remove takes it back out")


def embed_(at: dict[str, Path]) -> None:
    """embed -- text and vectors into embedded.json. No index: a database is
    the pipeline's business."""
    from falconvar.video_rag import describe, embed

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
    from falconvar.video_rag import Options, validate, video_rag

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


def search_(home: Path, database: str) -> None:
    """retrieve.search -- ranked in Postgres, so only with --database."""
    from falconvar.video_rag import layout, search, video_rag

    section("9 export + search")
    run = video_rag(PRIMARY, home.parent, sampler="uniform:[overview,scene]",
                    database=database, **FREE)
    check(not run.problems, f"every artifact exported to {database}: {run.problems or 'no problems'}")
    moments, notes = search("people in a shop", run.video_id, embedder="local",
                            grids={run.video_id: layout(run.home)["timeline"]})
    check(moments, f"search -> {len(moments)} moments; notes: {notes}")


# ----------------------------------------------------------- 2 · aggregates

def aggregate_components(home: Path, scratch: Path, llm: bool) -> None:
    """Every aggregator as a component: one input in, one answer out."""
    from falconvar.aggregates import (Inapplicable, coverage, entities, ner, prompt,
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

    # The stub describer's answers name no people, so there is nobody to link:
    # an answer, not a failure -- and raised, so a caller running it alone hears why.
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

    if llm:
        select.select(home, out / "summary_in.json", "summary")
        made = prompt.prompt("summary", out / "summary_in.json", out / "summary.json")
        words = read(out / "summary.json")["payload"].get("word_count")
        check(words, f"prompt('summary') -> {words} words by {made.stats.get('model')}")


def aggregate_pipeline(home: Path, scratch: Path, llm: bool) -> None:
    """aggregate() -- runs what it is handed data for, and nothing else."""
    from falconvar.aggregates import aggregate, validate

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


def several(home: Path, second: Path, scratch: Path) -> None:
    """Two videos, one record: laid end to end, then aggregated unchanged."""
    from falconvar.aggregates import aggregate, combine
    from falconvar.aggregates.combination import origin
    from falconvar.video_rag import boundaries

    section("12 several videos")
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
    from falconvar import workflow

    section("13 workflow")
    options = workflow.Options(source=SECOND, into=scratch / "wf", tier="local",
                               use_audio=False, **FREE)
    run = workflow.process(options)
    ran = run.steps[-1].stats["aggregates"]
    check(set(ran) >= {"stats", "coverage", "sentiment", "ner"},
          f"tier='local' -> every free and local aggregator: {', '.join(ran)}")
    check(workflow.validate(workflow.Options(source=SECOND, tier="bogus")),
          "a tier nobody has is refused")

    # `into=None` is the data root, which `configure()` decides for the process.
    outside = paths.data_root()
    falconvar.configure(data_root=scratch / "elsewhere")
    run = workflow.process(workflow.Options(source=SECOND, use_audio=False, **FREE))
    check(run.home.parent == paths.out_root() == scratch / "elsewhere" / "out",
          f"configure(): into=None wrote under {paths.out_root()}")
    falconvar.configure(data_root=outside)
    check(paths.data_root() == outside, "...and configuring it back restores it")


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llm", action="store_true",
                    help="also run the paid aggregates: a summary, and entities")
    ap.add_argument("--database", default=None,
                    help="also export to this database and search it -- WRITES to it")
    ap.add_argument("--keep", type=Path, default=None,
                    help="write here and keep it, rather than a temporary folder")
    args = ap.parse_args()
    for sample in (PRIMARY, SECOND):
        if not sample.exists():
            raise SystemExit(f"no such file: {sample} -- run from the checkout root")

    scratch = args.keep or Path(tempfile.mkdtemp(prefix="falconvar-example-"))
    scratch.mkdir(parents=True, exist_ok=True)
    was = paths.data_root()
    falconvar.configure(data_root=scratch / "root")
    print(f"writing under {scratch}")
    started = time.perf_counter()
    def step(call: Callable[..., Any], *arguments: Any) -> Any:
        """A section that crashes is a failure, not the end of the run: the
        sections after it still run where they can, so one pass reports every
        problem. Later sections that need its output fail on their own."""
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
        home, at = media_(into)          # everything after reads what this wrote
        step(audio_, at)
        step(boundaries_, at, scratch)
        step(video_, at, scratch)
        step(cut_, at)
        step(describe_, at)
        step(questions_, at, scratch)
        step(embed_, at)
        second = step(pipeline_, into)
        if args.database:
            step(search_, home, args.database)
        step(aggregate_components, home, scratch, args.llm)
        step(aggregate_pipeline, home, scratch, args.llm)
        if second is not None:
            step(several, home, second, scratch)
        step(whole_run, scratch)
    finally:
        falconvar.configure(data_root=was)
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
