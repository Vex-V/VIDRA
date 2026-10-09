"""One aggregator at a time: each a function of keywords, one input in, one
answer file out.

    video = aggregates.record(timeline=..., transcript=..., descriptions=...)
    said = video.excerpt(transcript=True)

    aggregates.stats(record=video, out="answers/stats.json")
    aggregates.ner(input=said, out="answers/ner.json", labels=["person"])
    aggregates.summary(input=said, out="answers/summary.json", llm=OpenAI())
    aggregates.entities(profile="people", input=video.sightings(...), out=...)

What each reads:

    stats · coverage · speakers     record=   a `record(...)`
    ner · sentiment                 input=    an excerpt
    summary · chapters · events     input=    an excerpt
    custom(name=...)                input=    an excerpt, for a custom prompt
    entities(profile=...)           input=    sightings
    link(entries, profile=...)      entries   a list of dicts; returns `Linked`, writes nothing

`input=` is the object `excerpt()` / `sightings()` returned, or the path of
one they wrote with `out=`. `previous=` is an earlier answer's file, returned
unchanged when it read the same text with the same definition and model.
Every function returns the receipt (`Produced`); `aggregates.load(path)`
reads the answer back. Each takes the models it calls by role (`llm=`,
`embedder=`), None for the default; `aggregate()` takes a `Models`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from vidra.shared.contracts.documents import Aggregate, Excerpt, Produced, Sightings
from vidra.shared.models.base import Embedder, LLM
from vidra.shared.reporting import logs
from vidra.shared.storage import files
from . import available, definitions, kind_of, uses_embedder
from .core.record import PathLike, Record
from .driver import AggregateError, answer, load_input, made_by

#: What `input=` may be: the object, or a file one was written to.
InputLike = Excerpt | Sightings | str | Path


def _given(name: str, data: Any) -> Any:
    """`input=` as the object it names."""
    if isinstance(data, (str, Path)):
        return load_input(data)
    if isinstance(data, Record):
        raise AggregateError(
            f"{name} reads " + ("sightings" if kind_of(name) == "link" else "an excerpt")
            + " -- build one from the record: "
            + ("record.sightings(profile=..., answers=[...])" if kind_of(name) == "link"
               else "record.excerpt(transcript=True, answers={...})"))
    return data


def _run(name: str, data: Any, out: PathLike, previous: Optional[PathLike],
         llm: Optional[LLM] = None, embedder: Optional[Embedder] = None,
         **settings: Any) -> Produced:
    """`answer` with a write at the end: what every function here is."""
    own = {k: v for k, v in settings.items() if v is not None}
    earlier = files.maybe(previous, Aggregate)
    with logs.timed(name, getattr(data, "video_id", None)) as done:
        document = answer(name, data, llm=llm, embedder=embedder, previous=earlier,
                          settings=own)
        where = str(files.write_json(Path(out), document.as_dict()))
        reused = earlier is not None and document is earlier
        done(reused=reused)
    return Produced(video_id=document.video_id, component=name,
                    artifacts={document.aggregate_id: where},
                    stats={"aggregate": document.aggregate_id, "tier": document.tier,
                           "reused": reused, "computed": int(not reused),
                           "current": int(reused),
                           **({"model": made_by(document)} if made_by(document) else {})})


# ----------------------------------------------------- free: a record in

def stats(*, record: Record, out: PathLike,
          previous: Optional[PathLike] = None) -> Produced:
    """Durations, chunk lengths, words per minute, frames and answers counted."""
    return _run("stats", record, out, previous)


def coverage(*, record: Record, out: PathLike,
             previous: Optional[PathLike] = None) -> Produced:
    """How many chunks have picture, sound, both or neither."""
    return _run("coverage", record, out, previous)


def speakers(*, record: Record, out: PathLike,
             previous: Optional[PathLike] = None) -> Produced:
    """Who spoke, for how long, and how often the voice changed. Needs a
    transcript in the record."""
    return _run("speakers", record, out, previous)


# ----------------------------------------------------- local: an excerpt in

def ner(*, input: InputLike, out: PathLike, previous: Optional[PathLike] = None,
        labels: Optional[Sequence[str]] = None, threshold: Optional[float] = None,
        model: Optional[str] = None) -> Produced:
    """Named entities, and the chunks each appears in. `labels` are what to look
    for; the default is person, organisation, location, product, event, date."""
    return _run("ner", _given("ner", input), out, previous,
                labels=tuple(labels) if labels is not None else None,
                threshold=threshold, model=model)


def sentiment(*, input: InputLike, out: PathLike,
              previous: Optional[PathLike] = None,
              model: Optional[str] = None) -> Produced:
    """Tone per chunk, overall, and where it turns."""
    return _run("sentiment", _given("sentiment", input), out, previous,
                model=model)


# ----------------------------------------------------- llm: an excerpt in

def summary(*, input: InputLike, out: PathLike,
            previous: Optional[PathLike] = None, llm: Optional[LLM] = None) -> Produced:
    """One account of the whole video: a summary, topics, setting, notable."""
    return _run("summary", _given("summary", input), out, previous, llm)


def chapters(*, input: InputLike, out: PathLike,
             previous: Optional[PathLike] = None,
             llm: Optional[LLM] = None, embedder: Optional[Embedder] = None,
             max_spans: Optional[int] = None,
             min_span_s: Optional[float] = None) -> Produced:
    """Consecutive chapters: the embedder places the boundaries, the llm names
    them. `max_spans` caps the count; `min_span_s` is the shortest chapter."""
    return _run("chapters", _given("chapters", input), out, previous, llm,
                embedder, max_spans=max_spans, min_span_s=min_span_s)


def events(*, input: InputLike, out: PathLike,
           previous: Optional[PathLike] = None, llm: Optional[LLM] = None) -> Produced:
    """Discrete things that happened, each tied to the chunk it happened in."""
    return _run("events", _given("events", input), out, previous, llm)


def custom(*, name: str, input: InputLike, out: PathLike,
           previous: Optional[PathLike] = None,
           llm: Optional[LLM] = None, embedder: Optional[Embedder] = None,
           **settings: Any) -> Produced:
    """Run an aggregator defined by a prompt, by name: one added with
    `add_prompt`, or a built-in (`summary`, `chapters`, `events`). A `spans`
    prompt also takes `embedder`, `max_spans` and `min_span_s`."""
    prompts = [n for n in definitions.ids() if kind_of(n) != "link"]
    if name not in prompts:
        hint = (" -- it is a link profile; run it with `entities(profile=...)`"
                if name in available() and kind_of(name) == "link" else "")
        raise AggregateError(f"no prompt {name!r}{hint}; known: {', '.join(prompts)}")
    return _run(name, _given(name, input), out, previous, llm,
                embedder if uses_embedder(name) else None, **settings)


# ----------------------------------------------------- linking: sightings in

def entities(*, profile: str, input: InputLike, out: PathLike,
             previous: Optional[PathLike] = None,
             llm: Optional[LLM] = None, embedder: Optional[Embedder] = None) -> Produced:
    """The same person, object or text linked across chunks, with an account of
    each. The embedder decides who is who; the llm writes the accounts."""
    name = definitions.profile_id(profile)
    if name not in available():
        raise AggregateError(f"no link profile {profile!r}; known: "
                             f"{', '.join(sorted(definitions.load()['profiles']))}")
    return _run(name, _given(name, input), out, previous, llm, embedder)


def link(entries: Sequence[dict[str, Any]], *, profile: str = "people",
         different: Any = None, threshold: Optional[float] = None,
         embedder: Optional[Embedder] = None) -> Any:
    """Which of `entries` are the same subject, by a link profile's rules: no
    record, no file and no model call but the embedder's. Returns `Linked`,
    whose `groups` are lists of indexes into `entries`.

    Each entry is a dict carrying the profile's identity keys (`appearance`,
    `clothing` for `people`). `different` marks pairs known to be different
    subjects -- an N x N mask, or a function of two indexes -- and the bar is
    read off them. Without it, a `threshold` is needed (or the profile's own),
    on the profile's measure: a z-score when the profile has `weights` or
    `attributes` and `different` marks a pair, else a cosine.
    """
    from vidra.shared.models.base import require
    from vidra.shared.models.roles import resolve
    from vidra.shared.reporting.errors import Refused
    from .aggregators.entities.linking import Linked, Mention, link_similar, similarity

    name = definitions.profile_id(profile)[len(definitions.PROFILE_PREFIX):]
    entry = definitions.get("profiles", name)
    identity = entry["identity"]
    mentions = []
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise Refused(f"entry {index} is a {type(item).__name__}, not a dict")
        signature = "; ".join(str(item[k]).strip() for k in identity
                              if str(item.get(k) or "").strip())
        if not signature:
            raise Refused(f"entry {index} has none of {', '.join(identity)}, "
                          f"which the {name} profile links on")
        mentions.append(Mention(index, "", entry["field"], 0, signature, dict(item)))

    count = len(mentions)
    mask = ([[i != j and bool(different(i, j)) for j in range(count)]
             for i in range(count)] if callable(different) else different)
    bar = threshold if threshold is not None else entry.get("threshold")
    if mask is None and bar is None:
        raise Refused("nothing says which entries differ: pass `different` (pairs "
                      "known to be different, such as two people seen at once) or "
                      "a `threshold`")
    if not mentions:
        return Linked([], bar, 0, 0)

    chosen = resolve("embedder", embedder)
    require("embedder", chosen)
    sim, _ = similarity(mentions, chosen, entry, mask)
    return link_similar(mentions, sim, entry["rule"], entry["mutual"], bar, mask)


__all__ = ["chapters", "coverage", "custom", "entities", "events", "link", "ner",
           "sentiment", "speakers", "stats", "summary"]
