"""The aggregates driver: one verb every aggregator shares, and the pipeline.

    answer(name, data)                   -> Aggregate   the work, on objects
    aggregate(out=..., **aggregators)    -> Produced    the pipeline

What each aggregator reads:

    stats · coverage · speakers     a record      `record(timeline=..., ...)`
    ner · sentiment · prompts       an Excerpt    `record.excerpt(...)`
    entities:<profile>              Sightings     `record.sightings(...)`

The pipeline runs exactly the aggregators it is handed an input for, cheapest
first, and nothing by default:

    aggregate(out="d/aggregates", models=models, database=db,
              summary=said_and_seen, ner=said, stats=video,
              sentiment={"spoken": said, "seen": seen},
              entities_people=people,
              settings={"ner": {"labels": ["person"]}})

A dict of inputs gives one answer each, stored as `sentiment~spoken`.
`entities:people` is `entities_people` as a keyword and `entities.people.json`
on disk. `previous=` is a folder of earlier answers, reused while current.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from ..shared.reporting import logs
from ..shared.contracts.documents import (Aggregate, Excerpt, Produced, Sightings,
                                          Timeline, Transcript, fingerprint_of)
from ..shared.reporting.errors import VidraError
from ..shared.models.roles import Models, keys_of, unpack
from ..shared.storage import files
from . import (available, build, definitions, kind_of, settings_of, takes_inputs,
               tier_of, uses_embedder)
from .core.base import TIERS, Context, missing
from .core.inputs import Input, Read, Row, answer_id, answer_of_file, filename, parse
from .core.record import Record, context


class AggregateError(VidraError, ValueError):
    """An aggregator handed the wrong kind of input, or asked for something that
    does not exist.
    """


class Inapplicable(VidraError, ValueError):
    """Nothing to answer here (`speakers` on a silent video, an input that found
    nothing). The pipeline reports it as skipped.
    """


# ------------------------------------------------------------------ the verb

#: The model assumed for an `llm` aggregate that does not record one.
LEGACY_MODEL = "openai:gpt-5.4-mini"


def made_by(document: Aggregate) -> Optional[str]:
    """`provider:model` for an aggregate a model wrote; None for arithmetic."""
    recorded = document.stats.get("model")
    if recorded is None and document.tier == "llm":
        return LEGACY_MODEL
    return recorded


def _read_of(excerpt: Excerpt) -> tuple[Input, Read]:
    """The `Read` an excerpt was written from, rebuilt exactly."""
    one = parse(excerpt.selection)[0]
    rows = [Row(r["chunk_id"], r["start_ts"], r["end_ts"],
                [tuple(part) for part in r["parts"]]) for r in excerpt.rows]
    return one, Read(one, rows, list(excerpt.answers))


def _mentions_of(sightings: Sightings) -> tuple[Input, Any]:
    """The `Mentions` sightings were written from, rebuilt exactly."""
    from .aggregators.entities.linking import Mention, Mentions
    from .core.inputs import Source as InputSource

    one = parse(sightings.selection)[0]
    chosen = definitions.Selection(tuple(InputSource(s.head) for s in one.sources),
                                   sightings.field_name, tuple(sightings.keys))
    return one, Mentions(chosen, [Mention(**m) for m in sightings.mentions])


def wants(name: str) -> str:
    """What an aggregator reads, in words: `a record`, `an excerpt` or
    `sightings`."""
    if not takes_inputs(name):
        return "a record"
    return "sightings" if kind_of(name) == "link" else "an excerpt"


#: How to build each kind of input, for refusals.
_HOW = {"a record": "aggregates.record(timeline=..., ...)",
        "an excerpt": "record.excerpt(transcript=True, answers={...})",
        "sightings": "record.sightings(profile=..., answers=[...])"}


def answer(name: str, data: Any, answer_id: Optional[str] = None,
           llm: Optional[str] = None, embedder: Optional[str] = None,
           previous: Optional[Aggregate] = None,
           settings: Optional[Mapping[str, Any]] = None,
           models: Optional[Models] = None) -> Aggregate:
    """One aggregator over one input. Reads and writes nothing.

    `data` is a `Record` (or its `Context`) for an aggregator that counts a
    whole record, an `Excerpt` for one that reads text, `Sightings` for a link
    profile. `previous` is an earlier answer, returned unchanged when it read
    the same text under the same version and model. `settings` go to the
    aggregator's constructor and may override `llm` / `embedder`. `models`
    carries the llm, the embedder and their keys; a role set there and as a
    keyword is refused. Raises `Inapplicable` when there is nothing to answer.
    """
    roles = unpack(models, llm=llm, embedder=embedder)
    with keys_of(models):
        return _answer(name, data, answer_id, roles["llm"], roles["embedder"],
                       previous, settings)


def _answer(name: str, data: Any, answer_id: Optional[str],
            llm: Optional[str], embedder: Optional[str],
            previous: Optional[Aggregate],
            settings: Optional[Mapping[str, Any]]) -> Aggregate:
    if name not in available():
        raise AggregateError(f"unknown aggregator {name!r}; "
                             f"known: {', '.join(available())}")
    reads = wants(name)
    own = dict(settings or {})
    llm, embedder = own.pop("llm", llm), own.pop("embedder", embedder)
    unknown = set(own) - set(settings_of(name)) - {"llm", "embedder"}
    if unknown:
        raise AggregateError(f"{name} has no setting {', '.join(sorted(unknown))}; "
                             f"it takes {', '.join(settings_of(name)) or 'none'}")
    read: Any = None
    one: Optional[Input] = None

    if isinstance(data, Record):
        data = data.context
    if isinstance(data, Context):
        if reads != "a record":
            raise AggregateError(f"{name} reads {reads}, not a record: build one "
                                 f"with {_HOW[reads]}")
        joined = data
    elif isinstance(data, Excerpt):
        if reads != "an excerpt":
            raise AggregateError(f"{name} reads {reads}, not an excerpt: {_HOW[reads]}")
        joined = Context(data.video_id, Timeline.from_dict(data.timeline))
        one, read = _read_of(data)
    elif isinstance(data, Sightings):
        if reads != "sightings":
            raise AggregateError(f"{name} reads {reads}, not sightings: {_HOW[reads]}")
        profile = definitions.locate(name)[1]
        if data.profile != profile:
            raise AggregateError(f"these sightings were taken for profile "
                                 f"{data.profile!r}, not {profile!r}: identity "
                                 f"keys are a profile's own")
        heard = (Transcript(data.video_id, chunks=[
            {"chunk_id": int(c), "text": t} for c, t in data.transcript.items()])
            if data.transcript else None)
        joined = Context(data.video_id, Timeline.from_dict(data.timeline),
                         transcript=heard)
        one, read = _mentions_of(data)
    else:
        raise AggregateError(f"{name} reads {reads}; got {type(data).__name__}")

    identity = answer_id or name
    if read is not None and read.empty:
        raise Inapplicable(f"{identity}: {read.why_empty}")

    # Built only once the input is known to be right and not empty.
    aggregator = build(name, llm, embedder, **own)
    if read is None:
        why = missing(aggregator, joined)
        if why is not None:
            raise Inapplicable(f"{name} {why}")
        expected = joined.inputs_fingerprint()
    else:
        # A hash of what was read and the definition version.
        expected = fingerprint_of({"timeline": joined.timeline.fingerprint(),
                                   "read": read.fingerprint(),
                                   "version": aggregator.version})

    # A different model makes a different answer.
    author = getattr(aggregator, "model_key", None)
    if (previous is not None and previous.inputs_fingerprint == expected
            and made_by(previous) == author):
        return previous

    payload = aggregator.run(joined) if read is None else aggregator.run(joined, read)
    stats: dict[str, Any] = {"about": aggregator.about,
                             **({"model": author} if author else {})}
    if read is not None:
        stats.update(inputs=str(one), version=aggregator.version,
                     read_chars=read.chars)
    return Aggregate(video_id=joined.video_id, aggregate_id=identity,
                     tier=aggregator.tier, payload=payload,
                     inputs_fingerprint=expected, stats=stats)


# ------------------------------------------------------------ reading files

def load(path: str | Path) -> Aggregate:
    """One answer's file, typed."""
    return files.read(path, Aggregate)


def answers(directory: str | Path) -> list[str]:
    """Every answer a folder holds, by id."""
    root = Path(directory)
    if not root.is_dir():
        return []
    return [answer_of_file(p.stem) for p in sorted(root.glob("*.json"))]


def load_all(directory: Optional[str | Path]) -> dict[str, Aggregate]:
    """Every answer in a folder, by id. An absent folder is no answers."""
    if directory is None:
        return {}
    return {name: load(Path(directory) / filename(name))
            for name in answers(directory)}


def load_input(path: str | Path) -> Union[Excerpt, Sightings]:
    """An excerpt or sightings file, as whichever it is."""
    where = Path(path)
    if not where.exists():
        raise files.MissingArtifact(f"no input at {where} -- record.excerpt(out=...) "
                                    f"and record.sightings(out=...) write them")
    if where.is_dir():
        raise AggregateError(f"{where} is a folder; this reads an excerpt or "
                             f"sightings file")
    kind = files.read_json(where).get("document")
    if kind not in ("excerpt", "sightings"):
        raise AggregateError(f"{where} is a {kind or 'document of no kind'}, not "
                             f"an excerpt or sightings")
    return files.read(where, Excerpt if kind == "excerpt" else Sightings)


# --------------------------------------------------------------- the pipeline

#: One input as the pipeline takes it: the object, or a file it was written to.
One = Union[Record, Excerpt, Sightings, str, Path]
#: What an aggregator is handed: one input, or several under labels.
Given = Union[One, Mapping[str, One], None]

#: A label, as `sentiment~spoken` stores it.
_LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


@dataclass
class Planned:
    """One aggregator the pipeline was handed inputs for."""

    name: str
    #: (label, input); the label is None for a single input.
    inputs: list[tuple[Optional[str], One]]
    settings: dict[str, Any] = field(default_factory=dict)


def name_of(key: str) -> Optional[str]:
    """A keyword as an aggregator id, or None: `entities_people` is
    `entities:people`.
    """
    known = set(available())
    candidates = [key, key.replace("_", "-")]
    if key.startswith("entities_"):
        candidates.insert(1, definitions.PROFILE_PREFIX + key[len("entities_"):])
    return next((c for c in candidates if c in known), None)


def _fits(name: str, one: Any) -> Optional[str]:
    """Why `one` is the wrong kind of input for `name`, or None."""
    reads = wants(name)
    if isinstance(one, (str, Path)):
        return None if reads != "a record" else (
            f"{name} counts a record, not a file: {_HOW[reads]}")
    kind = ("a record" if isinstance(one, Record) else
            "an excerpt" if isinstance(one, Excerpt) else
            "sightings" if isinstance(one, Sightings) else None)
    if kind is None:
        return (f"{name}: an input is a record, an excerpt or sightings; "
                f"got {type(one).__name__}")
    return None if kind == reads else f"{name} reads {reads}, not {kind}: {_HOW[reads]}"


def plan(settings: Optional[Mapping[str, Mapping[str, Any]]] = None,
         **aggregators: Given) -> tuple[list[Planned], list[str]]:
    """What the pipeline was handed, cheapest first, and every problem with it.
    None and False mean "do not run".
    """
    wanted: list[Planned] = []
    problems: list[str] = []
    for key, value in aggregators.items():
        if value is None or value is False:
            continue
        name = name_of(key)
        if name is None:
            problems.append(f"unknown aggregator {key!r}; known: {', '.join(available())}")
            continue
        if isinstance(value, Mapping):
            if not value:
                problems.append(f"{name}: an empty dict of inputs runs nothing")
                continue
            pairs = list(value.items())
            problems += [f"{name}: label {label!r} must match {_LABEL.pattern}"
                         for label, _ in pairs if not _LABEL.match(str(label))]
        else:
            pairs = [(None, value)]
        problems += [why for _, one in pairs if (why := _fits(name, one))]
        wanted.append(Planned(name, pairs))
    by_name = {p.name: p for p in wanted}
    for key, own in (settings or {}).items():
        name = name_of(key)
        if name is None or name not in by_name:
            problems.append(f"settings for {key!r}, which was not handed an input")
            continue
        by_name[name].settings = dict(own)
    wanted.sort(key=lambda p: TIERS.index(tier_of(p.name)))
    return wanted, problems


def validate(settings: Optional[Mapping[str, Mapping[str, Any]]] = None,
             llm: Optional[str] = None, embedder: Optional[str] = None,
             **aggregators: Given) -> list[str]:
    """Everything that stops a pipeline before it starts, as messages."""
    wanted, problems = plan(settings, **aggregators)
    for planned in wanted:
        name = planned.name
        allowed = set(settings_of(name)) \
            | ({"llm"} if tier_of(name) == "llm" else set()) \
            | ({"embedder"} if uses_embedder(name) else set())
        for key in planned.settings:
            if key not in allowed:
                problems.append(f"{name} has no setting {key!r}; it takes "
                                f"{', '.join(sorted(allowed)) or 'none'}")
        for _, one in planned.inputs:
            if isinstance(one, (str, Path)) and not Path(one).is_file():
                problems.append(f"{name}: no input file at {one}")
    if not wanted and not problems:
        problems.append("no aggregator was handed an input, so nothing would run: "
                        "name one, e.g. ner=record.excerpt(transcript=True)")
    if problems:
        return problems
    from ..shared.models import providers
    for spec in sorted({p.settings.get("llm", llm) or "" for p in wanted
                        if tier_of(p.name) == "llm"}):
        problems += providers.problems("llm", spec or None)
    for spec in sorted({p.settings.get("embedder", embedder) or "" for p in wanted
                        if uses_embedder(p.name)}):
        problems += providers.problems("embed", spec or None)
    return problems


def aggregate(*, out: str | Path,
              models: Optional[Models] = None,
              database: Optional[Any] = None,
              previous: Optional[str | Path] = None,
              settings: Optional[Mapping[str, Mapping[str, Any]]] = None,
              llm: Optional[str] = None, embedder: Optional[str] = None,
              **aggregators: Given) -> Produced:
    """Run the aggregators handed an input, and only those, cheapest first.

    Each aggregator is a keyword (`summary=`, `entities_people=`) whose value is
    its input -- a `record(...)` for the counters, an excerpt for a text
    aggregator, sightings for a link profile, or the file one was written to --
    or a dict of inputs under labels, one answer each. `settings` are per
    aggregator: `{"ner": {"labels": [...]}, "chapters": {"max_spans": 4}}`.

    Each answer is a file in `out`, and what it read a file in `out/inputs`.
    `previous` is a folder of earlier answers, each reused while it would be
    computed identically. `models` carries the llm, the embedder and their keys.
    `database` (a name or a built `Database`) also gets a copy: the source,
    every answer, its items, and the embedded summary, chapters and entities;
    failed writes are listed in `stats["problems"]`.
    """
    with keys_of(models):
        return _aggregate(out, models, database, previous, settings, llm,
                          embedder, aggregators)


def _aggregate(out: str | Path, models: Optional[Models], database: Optional[Any],
               previous: Optional[str | Path],
               settings: Optional[Mapping[str, Mapping[str, Any]]],
               llm: Optional[str], embedder: Optional[str],
               aggregators: Mapping[str, Given]) -> Produced:
    roles = unpack(models, llm=llm, embedder=embedder)
    llm, embedder = roles["llm"], roles["embedder"]
    problems = validate(settings, llm, embedder, **aggregators)
    if problems:
        raise AggregateError("; ".join(problems))
    wanted, _ = plan(settings, **aggregators)

    # Every input as an object, all from one video (or one combination).
    jobs: list[tuple[Planned, str, Any]] = []
    for planned in wanted:
        for label, one in planned.inputs:
            data = load_input(one) if isinstance(one, (str, Path)) else one
            jobs.append((planned, answer_id(planned.name, label), data))
    videos = sorted({data.video_id for _, _, data in jobs})
    if len(videos) > 1:
        raise AggregateError(
            f"these inputs are from different videos ({', '.join(videos)}); one run "
            f"answers one record -- combine them first with "
            f"aggregates.combine(records=[...], out=...)")
    first = jobs[0][2]
    grid = (first.timeline if isinstance(first, Record)
            else Timeline.from_dict(first.timeline))

    from ..shared.storage.database import as_database
    target = as_database(database)      # built before any work: a bad name fails here
    directory = Path(out)
    earlier = load_all(previous)
    written: dict[str, str] = {}
    selected: dict[str, str] = {}
    skipped: dict[str, str] = {}
    reused: list[str] = []
    authors: set[str] = set()
    used: dict[str, str] = {}

    with logs.timed("aggregate", videos[0]) as done:
        for planned, identity, data in jobs:
            name = planned.name
            if not isinstance(data, Record):
                selected[identity] = str(files.write(
                    directory / "inputs" / filename(identity), data))
            try:
                document = _answer(name, data, identity, llm, embedder,
                                   earlier.get(identity), planned.settings)
            except Inapplicable as why:
                skipped[identity] = str(why)
                continue
            if document is earlier.get(identity):
                reused.append(identity)
            if made_by(document):
                authors.add(made_by(document))
            if kind_of(name) is not None:
                used[name] = definitions.version_of(*definitions.locate(name))
            written[identity] = str(files.write_json(
                directory / filename(identity), document.as_dict()))
        done(ran=len(written), reused=len(reused), skipped=len(skipped))

    exported: list[str] = []
    units = 0
    if target is not None and written:
        from .database.export import export
        exported, units = export(videos[0], grid, written, used, target, embedder)
    return Produced(
        video_id=videos[0], component="aggregate", artifacts=written,
        stats={"ran": len(written), "current": len(reused),
               "computed": len(written) - len(reused),
               "aggregates": list(written), "models": sorted(authors),
               "skipped": skipped, "definitions": used,
               "inputs": selected, "out": str(directory),
               **({"exported_units": units, "problems": exported}
                  if target is not None else {})},
        skipped=sorted(skipped))


# ------------------------------------------------------------- for workflow

def definition_rows(used: dict[str, str]) -> list[dict[str, Any]]:
    """What each definition said at the version used, as rows for a database."""
    entries = []
    for name, version in sorted(used.items()):
        section, definition = definitions.locate(name)
        entry = definitions.get(section, definition)
        entries.append({"name": name, "version": version, "kind": kind_of(name),
                        "definition": {k: v for k, v in entry.items() if k != "builtin"},
                        "builtin": bool(entry.get("builtin"))})
    return entries


# ---------------------------------------------------------------------- CLI

def report(produced: Produced, as_json: bool = False) -> int:
    """How the CLI prints a receipt."""
    import json
    if as_json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0
    s = produced.stats
    print(f"{produced.video_id}   {produced.component}")
    for name, where in produced.artifacts.items():
        print(f"  {name:<24} -> {where}")
    for name, why in (s.get("skipped") or {}).items():
        print(f"  {name:<24} -- skipped: {why}")
    if "computed" in s:
        print(f"\n  {s['computed']} computed, {s['current']} reused")
    return 0


#: The keys a spec may have.
SPEC_KEYS = ("out", "previous", "llm", "embedder", "database", "record", "inputs",
             "run", "settings")


def from_spec(spec: Mapping[str, Any], base: Optional[Path] = None) -> Produced:
    """Run the pipeline a spec describes -- the CLI's input, as data:

        {"out": "d/aggregates", "previous": "d/aggregates",
         "llm": "openai", "embedder": "local",
         "record": {"timeline": "d/timeline.json", "transcript": "d/transcript.json",
                    "descriptions": "d/descriptions.json"},
         "inputs": {"said": {"excerpt": {"transcript": true}},
                    "people": {"sightings": {"profile": "people", "answers": ["yolo"]}}},
         "run": {"summary": "said", "ner": "said", "entities:people": "people",
                 "sentiment": {"spoken": "said"}, "stats": "record"},
         "settings": {"ner": {"labels": ["person"]}}}

    `run` names an input -- or `record` -- per aggregator. Relative paths are
    read from `base`.
    """
    from .core.record import record as open_record

    def at(where: Any) -> Any:
        return where if base is None or where is None else base / where

    unknown = set(spec) - set(SPEC_KEYS)
    if unknown:
        raise AggregateError(f"unknown spec key(s) {', '.join(sorted(unknown))}; "
                             f"a spec has {', '.join(SPEC_KEYS)}")
    video = open_record(**{k: at(v) for k, v in (spec.get("record") or {}).items()})
    built: dict[str, Any] = {"record": video}
    for label, how in (spec.get("inputs") or {}).items():
        if set(how) == {"excerpt"}:
            built[label] = video.excerpt(**how["excerpt"])
        elif set(how) == {"sightings"}:
            built[label] = video.sightings(**how["sightings"])
        else:
            raise AggregateError(f"input {label!r} is {{\"excerpt\": {{...}}}} or "
                                 f"{{\"sightings\": {{...}}}}")

    def resolve(label: str) -> Any:
        if label not in built:
            raise AggregateError(f"no input {label!r}; the spec defines "
                                 f"{', '.join(built)}")
        return built[label]

    handed = {key.replace(":", "_"):
              ({label: resolve(n) for label, n in value.items()}
               if isinstance(value, Mapping) else resolve(value))
              for key, value in (spec.get("run") or {}).items()}
    return aggregate(out=at(spec["out"]), previous=at(spec.get("previous")),
                     llm=spec.get("llm"), embedder=spec.get("embedder"),
                     database=spec.get("database"), settings=spec.get("settings"),
                     **handed)


def main(argv: Optional[list[str]] = None) -> int:
    from vidra.shared.config import env
    env.load()        # an entry point reads .env; the library never does
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Run the aggregators a JSON spec names, each on the input it "
                    "names. The format is in `vidra.aggregates.driver.from_spec`.")
    ap.add_argument("spec", nargs="?", help="a JSON spec file")
    ap.add_argument("--list", action="store_true",
                    help="every aggregator: its name, cost and what it reads")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        from . import about
        for name in available():
            print(f"{name:<22} {tier_of(name):<6} {wants(name):<12} {about(name)}")
        for where, found in definitions.load()["problems"].items():
            print(f"  ignored {where}: {'; '.join(found)}")
        return 0
    if not args.spec:
        ap.error("a spec file is required unless --list")
    where = Path(args.spec)
    try:
        produced = from_spec(json.loads(where.read_text(encoding="utf-8")),
                             base=where.parent)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1
    return report(produced, args.json)


__all__ = ["AggregateError", "Given", "Inapplicable", "One", "Planned", "SPEC_KEYS",
           "aggregate", "answer", "answers", "context", "definition_rows",
           "from_spec", "load", "load_all", "load_input", "made_by", "name_of",
           "plan", "report", "validate", "wants"]


if __name__ == "__main__":
    raise SystemExit(main())
