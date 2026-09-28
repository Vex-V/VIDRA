"""The aggregates driver: one verb every component shares, and the pipeline.

    answer(name, data)                -> Aggregate   the work, on objects
    aggregate(source, out, **named)   -> Produced    the pipeline over components

**Every aggregator is a component with one input**, and what that input is
depends only on what the aggregator does:

    stats · coverage · speakers     a record      they count whole documents
    ner · sentiment · prompts       an Excerpt    the rows a selection took
    entities:<profile>              Sightings     the entries a profile links

A record is a video's folder, or a combination's. An `Excerpt` or `Sightings`
is what `select` writes from one. So every component runs alone -- `select`,
then `ner.ner(excerpt, out)` -- and every input is a file a person can read,
keep, or hand to an aggregator in another run.

**The pipeline runs what it is handed data for, and nothing else.**

    aggregate("data/out/test", "data/out/test/aggregates",
              stats=True, ner="transcript", sentiment=True,
              summary="transcript+clip:activity", entities_people=True)

`True` is the aggregator's own default data; a string is a selection in the
`inputs` grammar (`a,b` makes two answers); a path ending `.json` is an excerpt
or sightings file written earlier; a dict is `{"data": ..., **settings}` for an
aggregator with settings of its own (`ner`'s `labels`, an `llm` for one
prompt). An aggregator not named does not run. There is no tier to reach any
more: naming an aggregator is the decision, and `up_to(tier)` builds the
everything-up-to-a-cost mapping `workflow` still offers. Several sources are
combined first (`combination.combine`) into a record folder kept beside the
answers: every chunk id in them is a combined id, and that folder is the only
way back to a video.

**`previous=` is the resume, and it is an argument** -- a folder of earlier
answers for the pipeline, an earlier answer's file for a component. An answer
is reused when it read the same text under the same version and model, and a
reused answer is still written: recompute and write are different questions.

**Nothing below this changed.** Every aggregator reads a `Context` and, if it
takes one, a `Read` or `Mentions`. `answer` rebuilds exactly those from the
file, so the fingerprint an answer is stored under is the one it always was.

A link profile's id carries a colon, `entities:people`; on disk it is
`entities.people.json`, as a keyword `entities_people`, as a flag
`--entities-people`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from ..shared import logs
from ..shared.contracts.documents import (Aggregate, Excerpt, Produced, Sightings,
                                          Timeline, Transcript, fingerprint_of)
from ..shared.errors import FalconvarError
from ..shared.storage import files
from . import available, build, definitions, kind_of, settings_of, takes_inputs, tier_of
from .base import TIERS, Context, missing
from .inputs import (Input, InputError, Read, Row, answer_id, answer_of_file,
                     check, filename, labels, parse)
from .inputs import DEFAULT as DEFAULT_INPUT
from .record import Source, context, open_record

#: What an aggregator may be handed by the pipeline.
Data = Union[bool, str, Path, Mapping[str, Any], None]


class AggregateError(FalconvarError, ValueError):
    """An aggregator handed the wrong kind of input, or a pipeline asked for
    something that does not exist."""


class Inapplicable(FalconvarError, ValueError):
    """Nothing to answer here: `speakers` on a silent video, a selection that
    found nothing. A reason, not a failure -- the pipeline reports it as
    skipped, and a component run alone raises it so the caller hears why."""


# ------------------------------------------------------------------ the verb

#: Who made an `llm` aggregate that does not say. Before providers existed
#: every one came from OpenAI's default, so reading a missing field as this is
#: a fact, not a guess -- and what stops the first run after that change paying
#: to rebuild every summary.
LEGACY_MODEL = "openai:gpt-5.4-mini"


def made_by(document: Aggregate) -> Optional[str]:
    """`provider:model` for an aggregate a model wrote; None for arithmetic."""
    recorded = document.stats.get("model")
    if recorded is None and document.tier == "llm":
        return LEGACY_MODEL
    return recorded


def default_selection(name: str) -> str:
    """What an aggregator reads when it is handed `True`."""
    return (DEFAULT_INPUT if kind_of(name) is None
            else definitions.default_selection(name))


def answer_id_of(name: str, one: Input) -> str:
    """The id an answer over one input is stored under: the aggregator's own,
    or `<id>~<label>` when the input carries a label."""
    return answer_id(name, one.label)


def _read_of(excerpt: Excerpt) -> tuple[Input, Read]:
    """The `Read` an excerpt was written from, rebuilt exactly -- the rows,
    their parts, the answers -- so its fingerprint is the original's."""
    one = parse(excerpt.selection)[0]
    rows = [Row(r["chunk_id"], r["start_ts"], r["end_ts"],
                [tuple(part) for part in r["parts"]]) for r in excerpt.rows]
    return one, Read(one, rows, list(excerpt.answers))


def _mentions_of(sightings: Sightings) -> tuple[Input, Any]:
    """The `Mentions` sightings were written from, rebuilt exactly."""
    from .entities.linking import Mention, Mentions
    from .inputs import Source as InputSource

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


def answer(name: str, data: Any, answer_id: Optional[str] = None,
           llm: Optional[str] = None, embedder: Optional[str] = None,
           previous: Optional[Aggregate] = None,
           settings: Optional[Mapping[str, Any]] = None) -> Aggregate:
    """One aggregator over one input. Reads and writes nothing.

    `data` is a `Context` for an aggregator that counts a whole record, an
    `Excerpt` for one that reads text, `Sightings` for a link profile -- and
    anything else is refused by name, never coerced. `previous` is an earlier
    answer, handed back as it is when it read the same text under the same
    version and model. `settings` reach the aggregator's own constructor, and
    may override `llm` and `embedder` for this one. Raises `Inapplicable`
    when there is nothing to answer.
    """
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

    if isinstance(data, Context):
        if reads != "a record":
            raise AggregateError(f"{name} reads {reads}, not a record: `select` "
                                 f"one from the record first")
        joined = data
    elif isinstance(data, Excerpt):
        if reads != "an excerpt":
            raise AggregateError(f"{name} reads {reads}, not an excerpt")
        joined = Context(data.video_id, Timeline.from_dict(data.timeline))
        one, read = _read_of(data)
    elif isinstance(data, Sightings):
        if reads != "sightings":
            raise AggregateError(f"{name} reads {reads}, not sightings")
        profile = definitions.locate(name)[1]
        if data.profile != profile:
            raise AggregateError(f"these sightings were selected for profile "
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

    identity = answer_id or (answer_id_of(name, one) if one is not None else name)
    if read is not None and read.empty:
        raise Inapplicable(f"{identity}: {read.why_empty}")

    # Built only once the input is known to be the right one and not empty:
    # a link profile's constructor builds its embedder, and a local aggregator
    # is the one that loads torch.
    aggregator = build(name, llm, embedder, **own)
    if read is None:
        why = missing(aggregator, joined)
        if why is not None:
            raise Inapplicable(f"{name} {why}")
        expected = joined.inputs_fingerprint()
    else:
        # What was read and what it was asked with -- never what was merely
        # available, so a summary of the transcript is not rebuilt because a
        # description changed.
        expected = fingerprint_of({"timeline": joined.timeline.fingerprint(),
                                   "read": read.fingerprint(),
                                   "version": aggregator.version})

    # The model is part of "current": same text, different model is a
    # different answer the caller asked for.
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
        raise files.MissingArtifact(f"no input at {where} -- `aggregates.select` "
                                    f"writes excerpts and sightings")
    if where.is_dir():
        # The likeliest mistake: a video's folder handed to a text aggregator,
        # which reads what was selected from it rather than the whole record.
        raise AggregateError(f"{where} is a folder -- a record. This aggregator "
                             f"reads an excerpt or sightings file: `select` one "
                             f"from the record first")
    kind = files.read_json(where).get("document")
    if kind not in ("excerpt", "sightings"):
        raise AggregateError(f"{where} is a {kind or 'document of no kind'}, not "
                             f"an excerpt or sightings -- `select` writes those")
    return files.read(where, Excerpt if kind == "excerpt" else Sightings)


# ------------------------------------------------------------ one component

def run_one(name: str, data: Union[Source, str, Path], out: str | Path,
            previous: Optional[str | Path] = None,
            llm: Optional[str] = None, embedder: Optional[str] = None,
            settings: Optional[Mapping[str, Any]] = None) -> Produced:
    """`answer` with a read at each end: what every component function is.

    `data` is a record (folder or mapping) for an aggregator that counts one,
    otherwise the path of an excerpt or sightings file. `previous` is an
    earlier answer's file to reuse if it is still current. Writes to `out`.
    """
    given = open_record(data) if wants(name) == "a record" else load_input(data)
    earlier = files.maybe(previous, Aggregate)
    with logs.timed(name, getattr(given, "video_id", None)) as done:
        document = answer(name, given, llm=llm, embedder=embedder,
                          previous=earlier, settings=settings)
        where = str(files.write_json(Path(out), document.as_dict()))
        reused = earlier is not None and document is earlier
        done(reused=reused)
    return Produced(video_id=document.video_id, component=name,
                    artifacts={document.aggregate_id: where},
                    stats={"aggregate": document.aggregate_id,
                           "tier": document.tier, "reused": reused,
                           "computed": int(not reused), "current": int(reused),
                           **({"model": made_by(document)} if made_by(document) else {})})


# --------------------------------------------------------------- the pipeline

@dataclass
class Planned:
    """One aggregator the pipeline was handed data for."""

    name: str
    #: True, a selection string, or an excerpt/sightings file.
    data: Union[bool, str, Path]
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def from_file(self) -> bool:
        return isinstance(self.data, Path) or (
            isinstance(self.data, str) and self.data.endswith(".json"))


def name_of(key: str) -> Optional[str]:
    """A keyword as an aggregator id, or None. A keyword cannot hold a colon or
    a hyphen, so `entities_people` is `entities:people` and `my_prompt` is
    `my-prompt` when only that one exists."""
    known = set(available())
    candidates = [key, key.replace("_", "-")]
    if key.startswith("entities_"):
        candidates.insert(1, definitions.PROFILE_PREFIX + key[len("entities_"):])
    return next((c for c in candidates if c in known), None)


def plan(aggregators: Optional[Mapping[str, Data]] = None,
         **named: Data) -> tuple[list[Planned], list[str]]:
    """What the pipeline was handed, cheapest first, and every problem with the
    handing itself. `None` and `False` mean "do not run"."""
    wanted: list[Planned] = []
    problems: list[str] = []
    for key, value in {**dict(aggregators or {}), **named}.items():
        if value is None or value is False:
            continue
        name = name_of(key)
        if name is None:
            problems.append(f"unknown aggregator {key!r}; known: {', '.join(available())}")
            continue
        settings: dict[str, Any] = {}
        if isinstance(value, Mapping):
            settings = {k: v for k, v in value.items() if k != "data"}
            value = value.get("data", True)
        if not isinstance(value, (bool, str, Path)):
            problems.append(f"{name}: data is True, a selection or an input file; "
                            f"got {type(value).__name__}")
            continue
        wanted.append(Planned(name, value, settings))
    wanted.sort(key=lambda p: TIERS.index(tier_of(p.name)))
    return wanted, problems


def _problems(planned: Planned, vocabulary_of: Any) -> list[str]:
    """Everything wrong with one aggregator's data, before anything runs."""
    name, data = planned.name, planned.data
    out: list[str] = []
    allowed = set(settings_of(name)) | ({"llm"} if tier_of(name) == "llm" else set()) \
        | ({"embedder"} if kind_of(name) == "link" else set())
    for key in planned.settings:
        if key not in allowed:
            out.append(f"{name} has no setting {key!r}; it takes "
                       f"{', '.join(sorted(allowed)) or 'none'}")
    if not takes_inputs(name):
        if data is not True:
            out.append(f"{name} counts the whole record; hand it True, not {data!r}")
        return out
    if planned.from_file:
        if not Path(data).exists():
            out.append(f"{name}: no input file at {data} -- `select` writes one")
        return out
    selection = default_selection(name) if data is True else str(data)
    try:
        parsed = parse(selection)
        labels(parsed)
    except InputError as exc:
        return out + [f"{name}: {exc}"]
    out += [f"{name}: {p}" for p in check(parsed, vocabulary_of())]
    if kind_of(name) == "link":
        for one in parsed:
            try:
                definitions.selection(definitions.locate(name)[1], one)
            except InputError as exc:
                out.append(f"{name}: {exc}")
    return out


def validate(aggregators: Optional[Mapping[str, Data]] = None,
             llm: Optional[str] = None, embedder: Optional[str] = None,
             **named: Data) -> list[str]:
    """What stops a pipeline before it starts, as messages -- every problem at
    once, so a CLI prints them all and a caller can show them together."""
    wanted, problems = plan(aggregators, **named)
    vocabulary: dict[str, Any] = {}

    def vocabulary_of() -> dict[str, Any]:
        # Asked of the other tier once, and only when a selection needs it.
        if not vocabulary:
            from ..video_rag import driver as video_rag
            vocabulary.update(video_rag.vocabulary())
        return vocabulary

    for planned in wanted:
        problems += _problems(planned, vocabulary_of)
    if not wanted and not problems:
        problems.append("no aggregator was handed any data, so nothing would run: "
                        "name one, e.g. stats=True or ner=\"transcript\"")
    if problems:
        return problems
    from ..shared.models import providers
    for spec in sorted({p.settings.get("llm", llm) or "" for p in wanted
                        if tier_of(p.name) == "llm"}):
        problems += providers.problems("llm", spec or None)
    for spec in sorted({p.settings.get("embedder", embedder) or "" for p in wanted
                        if kind_of(p.name) == "link"}):
        problems += providers.problems("embed", spec or None)
    return problems


def up_to(tier: str) -> dict[str, bool]:
    """Every aggregator whose cost is at most `tier`, each on its own default
    data: what "run the aggregates" meant before an aggregator had to be named."""
    if tier not in TIERS:
        raise AggregateError(f"tier must be one of {', '.join(TIERS)}")
    return {name: True for name in available()
            if TIERS.index(tier_of(name)) <= TIERS.index(tier)}


def aggregate(source: Union[Source, Sequence[Union[str, Path]], None],
              out: str | Path,
              previous: Optional[str | Path] = None,
              combined: Optional[str | Path] = None,
              llm: Optional[str] = None, embedder: Optional[str] = None,
              aggregators: Optional[Mapping[str, Data]] = None,
              **named: Data) -> Produced:
    """Run the aggregators handed data, and only those.

    `source` is a record -- a video's folder, a combination's, or a mapping of
    documents -- or a list of folders, combined first into `combined`
    (default `<out>/record`). It may be None when every aggregator is handed
    an input file. Each answer is a file in `out`, and the excerpts and
    sightings selected on the way are kept in `<out>/inputs`, so what an
    answer read can be looked at. `previous` is a folder of earlier answers.
    Aggregators are named as keywords, or in `aggregators` for ids no keyword
    can spell.
    """
    problems = validate(aggregators, llm, embedder, **named)
    if problems:
        raise AggregateError("; ".join(problems))
    wanted, _ = plan(aggregators, **named)
    directory = Path(out)

    record: Optional[Context] = None
    if any(not p.from_file for p in wanted):
        if source is None or (isinstance(source, (list, tuple)) and not source):
            raise AggregateError("a selection reads a record, and no source was given")
        if isinstance(source, (list, tuple)):
            if len(source) > 1:
                from .combination import combine
                home = Path(combined) if combined else directory / "record"
                combine(list(source), home)
                source = home
            else:
                source = source[0]
        record = open_record(source)

    earlier = load_all(previous)
    written: dict[str, str] = {}
    selected: dict[str, str] = {}
    skipped: dict[str, str] = {}
    reused: list[str] = []
    models: set[str] = set()
    used: dict[str, str] = {}

    with logs.timed("aggregate", record.video_id if record else None) as done:
        for planned in wanted:
            name = planned.name
            jobs: list[tuple[str, Any]] = []
            if not takes_inputs(name):
                jobs.append((name, record))
            elif planned.from_file:
                given = load_input(planned.data)
                jobs.append((answer_id_of(name, parse(given.selection)[0]), given))
            else:
                from .select import pick
                selection = (default_selection(name) if planned.data is True
                             else str(planned.data))
                parsed = parse(selection)
                for one, label in zip(parsed, labels(parsed)):
                    identity = answer_id(name, label)
                    picked = pick(record, name, one)
                    selected[identity] = files.write(
                        directory / "inputs" / filename(identity), picked)
                    jobs.append((identity, picked))

            for identity, data in jobs:
                try:
                    document = answer(name, data, identity, llm, embedder,
                                      earlier.get(identity), planned.settings)
                except Inapplicable as why:
                    skipped[identity] = str(why)
                    continue
                if document is earlier.get(identity):
                    reused.append(identity)
                if made_by(document):
                    models.add(made_by(document))
                if kind_of(name) is not None:
                    used[name] = definitions.version_of(*definitions.locate(name))
                written[identity] = str(files.write_json(
                    directory / filename(identity), document.as_dict()))
        done(ran=len(written), reused=len(reused), skipped=len(skipped))

    video_id = (record.video_id if record is not None
                else next((load(p).video_id for p in written.values()), ""))
    return Produced(
        video_id=video_id, component="aggregate", artifacts=written,
        stats={"ran": len(written), "current": len(reused),
               "computed": len(written) - len(reused),
               "aggregates": list(written), "models": sorted(models),
               "skipped": skipped, "definitions": used,
               "inputs": selected, "out": str(directory)},
        skipped=sorted(skipped))


# ------------------------------------------------------------- for workflow

def index_summary(video_id: str, payload: dict[str, Any],
                  embedder: Optional[str] = None) -> int:
    """Store the whole video as one vector in `video_embeddings`. Postgres only.

    **Its own table, never beside the moments.** `embeddings` answers *which
    twenty seconds*; a summary answers *which video*, and a video is not a
    moment you can play -- so the two never share a ranking, and `/search`
    reaches this one only as `level=video`. `chunk_id = -1` marks it as
    not-a-chunk; nothing keyed by chunk ever sees it.

    Only the final summary, never the intermediate layers: indexing those would
    return the same moment two or three times under different wordings.
    Best-effort: a video-level vector that fails to write must not fail the
    aggregates that already succeeded.
    """
    from ..shared.contracts.units import Unit, render
    from ..shared.models import embedders
    from ..shared.storage import supabase
    try:
        summary = (payload.get("summary") or "").strip()
        if not summary:
            return 0
        structured = {key: payload[key]
                      for key in ("topics", "setting", "notable")
                      if payload.get(key)}
        unit = Unit(video_id, -1, "summary", render(summary, structured),
                    structured, sampler="summary", question="summary")
        built = embedders.build(embedder)
        unit.vector = built.embed([unit.content])[0]
        return supabase.write_video_unit(unit, built.key)
    except Exception:                                    # noqa: BLE001
        return 0


def definition_rows(used: dict[str, str]) -> list[dict[str, Any]]:
    """Provenance for Postgres: what each definition said at the version used.

    Built here because the vocabulary is this tier's; written by whoever names
    a database.
    """
    entries = []
    for name, version in sorted(used.items()):
        section, definition = definitions.locate(name)
        entry = definitions.get(section, definition)
        entries.append({"name": name, "version": version, "kind": kind_of(name),
                        "definition": {k: v for k, v in entry.items() if k != "builtin"},
                        "builtin": bool(entry.get("builtin"))})
    return entries


# --------------------------------------------------------------------- CLIs

def report(produced: Produced, as_json: bool = False) -> int:
    """How every aggregates CLI prints its receipt."""
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


def flag_of(name: str) -> str:
    """An aggregator's CLI flag: `entities:people` is `--entities-people`."""
    return "--" + name.replace(":", "-")


def _typed(annotation: str, text: str) -> Any:
    """A CLI string as the constructor's annotation asks. Annotations are
    strings here (`from __future__ import annotations`), so they are read as
    words rather than evaluated."""
    if any(word in annotation for word in ("tuple", "list", "Sequence")):
        return tuple(part.strip() for part in text.split(",") if part.strip())
    if "float" in annotation:
        return float(text)
    if "int" in annotation:
        return int(text)
    return text


def component_main(argv: Optional[list[str]], description: str,
                   aggregator: Optional[str] = None, named: Optional[str] = None,
                   ) -> int:
    """The CLI of one aggregator component.

    `aggregator` fixes which one (`ner`); `named` instead takes it as the first
    argument, for the components that run a definition by name (`prompt`,
    `entities`) -- whose help names what it must be. Every setting the
    aggregator's constructor takes becomes a flag, read off its signature.
    """
    import argparse
    import inspect

    ap = argparse.ArgumentParser(description=description)
    if named:
        ap.add_argument("name", help=named)
    reads = wants(aggregator) if aggregator else ("sightings" if named and "profile" in named
                                                  else "an excerpt")
    ap.add_argument("data", help=f"the input: {reads}"
                                 + (" (a video's folder)" if reads == "a record"
                                    else " file, which `aggregates.select` writes"))
    ap.add_argument("out", help="where to write the answer")
    ap.add_argument("--previous", default=None,
                    help="an earlier answer's file; reused if it is still current")
    annotations: dict[str, str] = {}
    if aggregator:
        from . import _LAZY, _import
        if aggregator in _LAZY:
            signature = inspect.signature(_import(_LAZY[aggregator][0]).__init__)
            for setting in settings_of(aggregator):
                annotations[setting] = str(signature.parameters[setting].annotation)
                ap.add_argument(f"--{setting.replace('_', '-')}", dest=setting,
                                default=None, help=f"{annotations[setting]}")
    if named or (aggregator and tier_of(aggregator) == "llm"):
        ap.add_argument("--llm", default=None,
                        help="a provider or provider/model; default FALCONVAR_LLM, "
                             "then openai")
    if named and "profile" in named:
        ap.add_argument("--embedder", default=None,
                        help="a provider or provider/model; default "
                             "FALCONVAR_EMBEDDER, then openai")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    name = aggregator
    if named:
        given = args.name
        name = (given if given in available()
                else f"{definitions.PROFILE_PREFIX}{given}" if "profile" in named
                else given)
    settings = {k: _typed(annotations[k], getattr(args, k))
                for k in annotations if getattr(args, k) is not None}
    try:
        produced = run_one(name, args.data, args.out, args.previous,
                           getattr(args, "llm", None), getattr(args, "embedder", None),
                           settings)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1
    return report(produced, args.json)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Run the aggregators you hand data to, and only those. A "
                    "flag alone runs one on its default data; a flag with a "
                    "selection runs it on that. One not named does not run.")
    ap.add_argument("sources", nargs="*",
                    help="a video's folder -- or several, combined first")
    ap.add_argument("--out", help="the folder the answers are written to")
    ap.add_argument("--previous", default=None,
                    help="a folder of earlier answers; those still current are "
                         "reused. Omit to compute everything -- which costs "
                         "money for an llm aggregator. May be --out itself")
    ap.add_argument("--combined", default=None,
                    help="where several sources are combined; default <out>/record")
    ap.add_argument("--tier", default=None, choices=TIERS,
                    help="also run every aggregator up to this cost, on its default data")
    ap.add_argument("--llm", default=None,
                    help="who answers the llm aggregators: a provider or "
                         "provider/model; default FALCONVAR_LLM, then openai")
    ap.add_argument("--embedder", default=None,
                    help="who embeds for the link profiles; default "
                         "FALCONVAR_EMBEDDER, then openai")
    ap.add_argument("--list", action="store_true",
                    help="every aggregator: its flag, cost and default data")
    ap.add_argument("--json", action="store_true")
    names = available()
    for index, name in enumerate(names):
        ap.add_argument(flag_of(name), dest=f"_a{index}", nargs="?", const=True,
                        default=None, metavar="SELECTION",
                        help=f"{tier_of(name)}; reads {wants(name)}"
                             + (f", default {default_selection(name)}"
                                if takes_inputs(name) else ""))
    args = ap.parse_args(argv)

    if args.list:
        from . import about
        for name in names:
            reads = default_selection(name) if takes_inputs(name) else "the record"
            print(f"{flag_of(name):<24} {tier_of(name):<6} {reads:<16} {about(name)}")
        for where, found in definitions.load()["problems"].items():
            print(f"  ignored {where}: {'; '.join(found)}")
        return 0
    if not args.out:
        ap.error("--out is required unless --list")

    handed: dict[str, Data] = dict(up_to(args.tier)) if args.tier else {}
    for index, name in enumerate(names):
        value = getattr(args, f"_a{index}")
        if value is not None:
            handed[name] = value

    problems = validate(handed, args.llm, args.embedder)
    if problems:
        for problem in problems:
            print(f"error: {problem}")
        return 2
    try:
        produced = aggregate(args.sources or None, args.out, args.previous,
                             args.combined, args.llm, args.embedder, handed)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1
    return report(produced, args.json)


__all__ = ["AggregateError", "Data", "Inapplicable", "Planned", "aggregate",
           "answer", "answers", "component_main", "context", "default_selection",
           "definition_rows",
           "flag_of", "index_summary", "load", "load_all", "load_input", "made_by",
           "name_of", "plan", "report", "run_one", "up_to", "validate", "wants"]


if __name__ == "__main__":
    raise SystemExit(main())
