# aggregates: how it works

`aggregates` answers questions about a whole video that a search for moments
cannot: how long, who spoke most, what it is about, where the chapters are,
what happened, who appears more than once. It reads only the files
`video_rag` wrote, never the video.

The functions are in [functions.md](functions.md).

## From files to an answer

```text
files  ->  record  ->  input  ->  aggregator  ->  answer file
```

1. **A record** is a video's documents, named by path: `timeline.json`
   (required), and any of `transcript.json`, `descriptions.json`,
   `manifest.json`. Documents from different videos or different grids are
   refused.
2. **An input** is what one aggregator reads, built from the record. Nothing
   is read by default: you choose the transcript, which answers, and which of
   their fields.
3. **An aggregator** reads one input and writes one answer file.

There are three kinds of input:

| Input | Is | Read by |
|---|---|---|
| record | the documents themselves | `stats`, `coverage`, `speakers` |
| excerpt | text per chunk: the transcript and/or chosen fields of chosen answers | `ner`, `sentiment`, `summary`, `chapters`, `events`, custom prompts |
| sightings | one entry per thing a list field mentions, e.g. each person in `yolo`'s `people` | `entities` |

An excerpt or sightings carries its own grid, so an aggregator never needs a
second file. Each can be saved to a file and handed over later.

## The aggregators

Each has a cost tier. The pipeline runs whatever it was given cheapest first,
so a run that fails partway still has the free answers.

| Tier | Aggregator | Reads | Answers |
|---|---|---|---|
| free | `stats` | record | duration, chunk lengths, words, words per minute, frames and answers counted |
| free | `coverage` | record | how many chunks have picture, sound, both or neither |
| free | `speakers` | record | speakers, turns, handovers, speech ratio, dominant speaker. Needs a transcript |
| local | `ner` | excerpt | named entities (GLiNER) and the chunks each appears in |
| local | `sentiment` | excerpt | tone per chunk (DistilBERT), overall, and where it turns |
| llm | `summary` | excerpt | a summary, topics, setting and notable moments |
| llm | `chapters` | excerpt | consecutive chapters, each with a title, summary and time span |
| llm | `events` | excerpt | discrete events, each tied to a chunk |
| llm | `entities:people`, `entities:objects`, `entities:text` | sightings | the same person, object or text linked across chunks, with an account of each |

`free` runs no model; `local` runs a small model on this machine; `llm` calls
the LLM.

## How the llm aggregators work

The LLM ones are defined as data (an instruction and the fields to fill), and
each follows one of four patterns:

- **fold** (`summary`): the rows are summarised in groups, the group
  summaries are summarised again, until one answer is left. Nothing is cut to
  fit a context window.
- **spans** (`chapters`): every chunk is embedded, and chapters break where
  neighbouring chunks are least alike. Chapters shorter than `min_span_s`
  (default 30 s) are merged into their more similar neighbour, and
  `max_spans` caps the count. The LLM then names and summarises each chapter.
  Boundaries come from the grid, so a chapter's times are always real.
- **items** (`events`): the rows are read in windows of 100; each item the
  LLM returns cites a chunk id, and its times are read from the grid.
- **link** (`entities:*`): each sighting's identifying fields (for people,
  `appearance` and `clothing`) are embedded, and sightings are linked by
  rules, not by the model. Two entries in one answer are known to be
  different things, so their similarity sets the bar for linking in this
  video. The LLM then writes one account per linked entity, and marks
  observations that contradict the rest as `doubts` instead of dropping them.

Custom prompts (`add_prompt`) and linking profiles (`add_profile`) use the
same four patterns.

## Answers

Each answer is a JSON file holding:

| Field | Is |
|---|---|
| `aggregate_id` | `summary`, `entities:people`, or with a label, `sentiment~spoken` |
| `payload` | the answer itself, its keys depending on the aggregator |
| `inputs_fingerprint` | a hash of the text the aggregator read |
| `version` | a hash of the definition: instruction, fields, settings |
| `stats.model` | which model wrote it, by key |

`entities:people` is stored as `entities.people.json`, since `:` cannot be in
a Windows filename.

## Reuse

Given `previous=`, an earlier answer is reused when the text read, the
definition's version and the model key are all unchanged. Changing the
input, a prompt's wording, a setting such as `max_spans`, or the model
recomputes it.

## Several videos

`combine` lays several records end to end on one clock: chunks are numbered
on after the previous video's, times are shifted, and speakers are prefixed
with their video (`SPEAKER_00` in two recordings is two people). The result is
a record, and every aggregator reads it unchanged.

## Databases and search

With a `database`, the pipeline also writes, for each run:

| Method | What |
|---|---|
| `write_source` | what the answers are about: the video, or the videos combined |
| `write_answer` | each answer, plus the things it places in time (chapters, events, names, entities) and an entity's sightings |
| `write_definitions` | the wording of each prompt used, at its version |
| `write_aggregate_units` | the summary, each chapter and each entity's account, embedded |

`aggregates.search` searches those embedded units one level at a time:

| Level | Finds |
|---|---|
| `source` | which video, by its summary |
| `span` | which part, by its chapter (with a time span) |
| `entity` | who or what, by its account |

Levels are never mixed in one ranking: a long summary would crowd out every
chapter.
