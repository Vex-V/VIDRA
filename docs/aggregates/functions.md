# aggregates: functions

How the aggregators work is in [how-it-works.md](how-it-works.md). Models
and databases are the same objects `video_rag` takes; see
[the video_rag functions](../video_rag/functions.md#models).

- [1. A record](#1-a-record)
- [2. Inputs](#2-inputs)
- [3. One aggregator](#3-one-aggregator)
- [4. Several at once](#4-several-at-once)
- [Reading answers](#reading-answers)
- [Several videos](#several-videos)
- [Custom prompts and profiles](#custom-prompts-and-profiles)
- [Search](#search)
- [Databases](#databases)

```python
from pathlib import Path
from vidra import LocalEmbedder, Models, OpenAI, Supabase, aggregates

models = Models(llm=OpenAI("gpt-5.4-mini"), embedder=LocalEmbedder())
db = Supabase()                              # optional; only for a database copy
D = Path("data/out/shop")                    # a folder video_rag wrote
```

## 1. A record

```python
video = aggregates.record(
    timeline=D / "timeline.json",            # required
    transcript=D / "transcript.json",        # optional
    descriptions=D / "descriptions.json",    # optional
    manifest=D / "manifest.json",            # optional; only stats reads it
)

video.answer_ids()     # ["clip", "clip:checkout", "yolo"] -- the answers it holds
```

## 2. Inputs

An **excerpt** is text per chunk. Name the transcript, and/or answers with
the fields to read from each; `summary` is an answer's prose.

```python
said = video.excerpt(transcript=True)
seen = video.excerpt(answers={"clip": ["summary"],
                              "clip:checkout": ["summary", "payment", "people"]})
both = video.excerpt(transcript=True, answers={"clip": ["summary"]})
```

**Sightings** are the entries of a list field, for linking. The profile says
which field (`people` for `entities:people`):

```python
people = video.sightings(profile="people", answers=["yolo"])
by_clothes = video.sightings(profile="people", answers=["yolo"], keys=["clothing"])
```

`keys=` narrows which of the profile's fields decide who is who.

An answer, field or profile the record does not have is refused here,
before anything runs. `out=` also saves the input, and an aggregator accepts
that path in place of the object:

```python
video.excerpt(transcript=True, out=D / "inputs" / "said.json")
aggregates.summary(input=D / "inputs" / "said.json", out=D / "summary.json", models=models)
```

## 3. One aggregator

Each writes one answer to `out` and returns a receipt (`.artifacts`,
`.stats`).

```python
# free: a record in
aggregates.stats(record=video, out=D / "answers" / "stats.json")
aggregates.coverage(record=video, out=D / "answers" / "coverage.json")
aggregates.speakers(record=video, out=D / "answers" / "speakers.json")

# local: an excerpt in
aggregates.ner(input=said, out=D / "answers" / "ner.json",
               labels=["person", "product"], threshold=0.5)
aggregates.sentiment(input=said, out=D / "answers" / "sentiment.json")

# llm: an excerpt in
aggregates.summary(input=both, out=D / "answers" / "summary.json", models=models)
aggregates.chapters(input=both, out=D / "answers" / "chapters.json", models=models,
                    max_spans=5, min_span_s=30)
aggregates.events(input=seen, out=D / "answers" / "events.json", models=models)

# linking: sightings in
aggregates.entities(profile="people", input=people,
                    out=D / "answers" / "people.json", models=models)
```

- `previous=` is an earlier answer file: it is returned unchanged if it would
  be computed the same way.
- `llm=` and `embedder=` can be given instead of `models=`.
- `chapters` and `entities` use the embedder as well as the LLM.
- An input with nothing to answer (`speakers` on a silent video) raises
  `aggregates.Inapplicable`, with the reason.

## 4. Several at once

`aggregate` runs every aggregator it is handed, and only those, cheapest
first. Each keyword is an aggregator, and its value is that aggregator's
input:

```python
done = aggregates.aggregate(
    out=D / "aggregates",
    models=models,
    previous=D / "aggregates",                   # reuse answers still current
    database=db,                                 # optional copy
    stats=video,
    ner=said,
    summary=both,
    chapters=both,
    events=seen,
    entities_people=people,                      # entities:people
    sentiment={"spoken": said, "seen": seen},    # two answers: sentiment~spoken, sentiment~seen
    settings={"ner": {"labels": ["person"]}, "chapters": {"max_spans": 4}},
)

done.artifacts              # {"summary": ".../summary.json", ...}
done.stats["computed"]      # how many were run, not reused
done.stats["skipped"]       # {"speakers": "needs transcript, ..."}
done.stats["problems"]      # database writes that failed
```

What each aggregator read is saved in `out/inputs/`.
`aggregates.validate(...)`, given the same aggregator keywords and
`settings`, lists every problem without running anything.

### Both tiers in one call

```python
from vidra import workflow

run = workflow.process(workflow.Options(
    source=Path("shop.mp4"), sampler="clip,yolo", tier="llm",
    models=models, database=db))
```

`workflow` extracts with `video_rag`, then runs every aggregator up to
`tier` (`free`, `local` or `llm`) on everything extraction produced: the
transcript and every answer's prose for text, every answer for linking.

## Reading answers

```python
answer = aggregates.load(D / "answers" / "summary.json")
answer.payload["summary"]
answer.payload["topics"]
answer.stats["model"]                    # "openai:gpt-5.4-mini"
```

Main payload keys:

| Aggregator | Payload |
|---|---|
| `stats` | `duration_s`, `chunks`, `chunk_s`, `words`, `words_per_minute`, `frames_sampled` |
| `coverage` | `both`, `picture_only`, `sound_only`, `neither` |
| `speakers` | `speakers`, `turns`, `handovers`, `speech_ratio`, `dominant` |
| `ner` | `entities` (each with `text`, `label`, `chunk_ids`), `by_label` |
| `sentiment` | `per_chunk`, `mean`, `turning_points` |
| `summary` | `summary`, `topics`, `setting`, `notable` |
| `chapters` | `chapters`, each with `title`, `summary`, `chunk_ids`, `start_ts`, `end_ts` |
| `events` | `events`, each with `what`, `kind`, `chunk_id`, `start_ts`, `end_ts` |
| `entities:*` | `entities`, each with `label`, `chunk_ids`, `mentions`, `account`, `doubts` |

Without writing anything, `aggregates.answer` runs one aggregator on an
input in memory and returns the answer:

```python
answer = aggregates.answer("summary", both, models=models)
```

## Several videos

```python
monday = aggregates.record(timeline="data/out/monday/timeline.json",
                           transcript="data/out/monday/transcript.json")
tuesday = aggregates.record(timeline="data/out/tuesday/timeline.json",
                            transcript="data/out/tuesday/transcript.json")

week = aggregates.combine(records=[monday, tuesday], out="data/out/week")
aggregates.summary(input=week.excerpt(transcript=True),
                   out="data/out/week/summary.json", models=models)
```

The combination is a record like any other.

## Custom prompts and profiles

A custom prompt is an instruction and the fields to fill. It is saved in
`data/aggregates.json` and run with `custom`, or as a keyword of `aggregate`:

```python
aggregates.add_prompt(
    "incident_report",
    "Write an incident report for this video.",
    fields={
        "report": {"type": "text", "about": "what happened, in order"},
        "severity": {"type": "text", "about": "how serious",
                     "one_of": ["none", "minor", "major"]},
    },
    kind="fold",           # fold: one answer; spans: chapters; items: cited things
)

aggregates.custom(name="incident_report", input=seen,
                  out=D / "answers" / "incident.json", models=models)
aggregates.aggregate(out=D / "aggregates", models=models, incident_report=seen)
```

`custom` runs any aggregator defined by a prompt, by name, including the
built-ins: `custom(name="summary", ...)` is `summary(...)`. It does not run
the code aggregators (`stats`, `ner`, ...) or linking profiles, which have
their own functions.

A linking profile links the entries of a list field and writes an account of
each. It runs as `entities:<name>`:

```python
aggregates.add_profile(
    "tools", "objects",                                   # name, the list field
    "These may be the same object. Say what it is and who used it.",
    fields={"use": {"type": "text", "about": "who used it, for what"}},
    identity=["object", "appearance"],                    # what decides who is who
    story=["context"],                                    # what the account also reads
)

tools = video.sightings(profile="tools", answers=["objects"])
aggregates.entities(profile="tools", input=tools, out=D / "answers" / "tools.json",
                    models=models)
```

```python
aggregates.available()                   # every aggregator id
aggregates.definition("incident_report") # a prompt or profile, as stored
aggregates.remove_prompt("incident_report")
aggregates.remove_profile("tools")
```

Built-in prompts and profiles cannot be changed.

## Search

Searches the summaries, chapters and entity accounts a run exported to a
database, one level at a time:

```python
videos = aggregates.search("a busy checkout", level="source", models=models, database=db)
parts = aggregates.search("customers paying", level="span", models=models, database=db)
people = aggregates.search("a man in a red cap", level="entity", models=models, database=db)

parts[0]["source_id"], parts[0]["item_id"], parts[0]["start_ts"], parts[0]["content"]
```

The embedder must be the one the aggregates were exported with.

## Databases

A custom `Database` (see [the video_rag functions](../video_rag/functions.md#your-own-database))
receives the aggregates through these methods:

| Method | Called with |
|---|---|
| `write_source(source)` | `source_id`, `video_ids`, and each video's offsets on the combined clock |
| `write_answer(source_id, answer, items, mentions)` | the answer row; the things it places in time; an entity's sightings |
| `write_definitions(rows)` | each prompt or profile used, at its version |
| `write_aggregate_units(source_id, units, embedder_key)` | the summary, chapters and entity accounts, each with `aggregate_id`, `item_id`, `level`, `content`, `start_ts`, `end_ts`, `vector`. They are embedded only if this method is defined |

For `aggregates.search`, also define:

| Method | Returns |
|---|---|
| `search_aggregates(vector, query, embedder_key, level, limit, source_ids)` | the best units of that level, best first, as dicts with `source_id`, `video_ids`, `aggregate_id`, `item_id`, `level`, `content`, `start_ts`, `end_ts`, `score` |
