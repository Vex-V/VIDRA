# VIDRA

**Video Description, Retrieval & Aggregation.** A Python library that
turns a video into a searchable index of moments.

The video is split into time-based chunks. For each chunk it keeps a set of
frames, describes them with a vision model, transcribes the audio, and
embeds the text. A query then returns the chunks that match, with their
start and end times. A second tier answers questions about the whole video:
counts, summaries, chapters, events, named entities, and which people or
objects recur across chunks.

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [One stage at a time](#one-stage-at-a-time)
- [How it works](#how-it-works)
- [Samplers and questions](#samplers-and-questions)
- [Aggregates](#aggregates)
- [Search and storage](#search-and-storage)
- [Models](#models)
- [Hardware](#hardware)
- [Checking an install](#checking-an-install)
- [Project layout](#project-layout)
- [Limitations](#limitations)

## Requirements

- Python 3.11 to 3.14.
- No system FFmpeg: decoding uses PyAV, which bundles it.
- An API key for a hosted model, unless every role runs locally
  (see [Models](#models)).
- Optional: an NVIDIA GPU. Everything also runs on the CPU, more slowly.

Tested on Windows 11 with an NVIDIA GPU (CUDA), and on the CPU. Linux and
macOS have not been run.

## Install

```bash
git clone https://github.com/Vex-V/FalCONvar.git
cd FalCONvar
pip install -r requirements.txt
```

On an NVIDIA GPU, add PyTorch's CUDA index; pip's default PyTorch build on
Windows is CPU-only:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

`requirements.txt` installs every dependency, pinned to the versions the
library was developed against. `vidra` itself is not installed, so run your
scripts from the repository root, or add the root to `PYTHONPATH`.

## Configuration

**Keys.** The library reads API keys from environment variables and never
reads a `.env` file on its own. Copy `.env.example` to `.env` and fill in
the keys you use. Then either call

```python
import vidra
vidra.configure(env_file=".env")
```

or pass a key to the model that uses it, which takes precedence over the
environment:

```python
from vidra import OpenAI
vlm = OpenAI("gpt-5.4-mini", api_key="sk-...")
```

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | `OpenAI` and `OpenAIEmbedder`, when no `api_key=` is given |
| `ANTHROPIC_API_KEY` | `Anthropic`, when no `api_key=` is given |
| `HF_TOKEN` | the first download of pyannote's gated model |
| `SUPABASE_URL`, `SUPABASE_SECRET_KEY`, `SUPABASE_PUBLISHABLE_KEY` | the Supabase backend (writes, searches) |

**Where output goes.** In order of precedence:
`vidra.configure(data_root=...)`, then the `VIDRA_DATA` variable,
then `data/` in a checkout, then `~/.vidra`. Model weights follow the
same rule with `weights=` and `VIDRA_WEIGHTS`.

## Quick start

```python
import vidra
from vidra import LocalEmbedder, Models, OpenAI, Supabase, aggregates
from vidra.video_rag import search, video_rag

vidra.configure(env_file=".env")
models = Models(vlm=OpenAI("gpt-5.4-mini"),      # describes frames
                llm=OpenAI("gpt-5.4-mini"),      # summaries, chapters, events
                embedder=LocalEmbedder())        # vectors, on this machine

# Supabase: url and key default to SUPABASE_URL and SUPABASE_SECRET_KEY.
# Nothing connects until the first call. For no server at all, use
# Folder("data/out") instead (vector search only).
db = Supabase()                      # or Supabase(url="https://...", key="...")

# Extract: media -> audio -> grid -> frames -> transcript -> descriptions -> vectors.
# Every artifact is written to data/out/<video_id>/ and copied to the database.
run = video_rag("video.mp4", "data/out", policy="scene", sampler="clip,yolo",
                models=models, database=db)

# Search: vector and full-text, fused
moments, notes = search("a customer paying in cash", run.video_id,
                        models=models, database=db)
for m in moments:
    print(m.chunk_id, m.start_ts, m.end_ts, m.hits[0]["content"][:80])

# Whole-video answers: name the files, choose what each aggregator reads, run.
d = run.folder
video = aggregates.record(timeline=d / "timeline.json",
                          transcript=d / "transcript.json",
                          descriptions=d / "descriptions.json")
said = video.excerpt(transcript=True)

done = aggregates.aggregate(out=d / "aggregates", models=models, database=db,
                            summary=said, chapters=said)
print(aggregates.load(done.artifacts["summary"]).payload["summary"])

# Search the summaries and chapters
chapters = aggregates.search("the checkout gets busy", level="span",
                             models=models, database=db)
```

Before the first run, create the tables: see [Search and storage](#search-and-storage).

## One stage at a time

Each stage of `video_rag` is also a function that reads the files it needs and
writes one, so a stage can be re-run or tuned alone. `D` is a video's folder:

```python
from vidra.video_rag import boundaries, describe, video

boundaries.evidence(out=D / "cuts.json", policy="scene", media=D / "media.json")
boundaries.calibrate(D / "cuts.json")         # cuts per threshold; nothing is decoded
boundaries.boundaries(media=D / "media.json", out=D / "timeline.json",
                      policy="scene", cuts=D / "cuts.json")

video.video(media=D / "media.json", timeline=D / "timeline.json",
            out=D / "manifest.json", store=D / "store", sampler="clip:[text,scene]")
describe.describe(manifest=D / "manifest.json", timeline=D / "timeline.json",
                  store=D / "store", out=D / "descriptions.json", limit=5)
```

`vidra.workflow.process(...)` runs both tiers in one call.

## How it works

There are two tiers. `video_rag` reads the video and builds the index.
`aggregates` reads only what `video_rag` wrote. `workflow` runs one, then
the other.

`video_rag` is eight stages. Each reads files and writes one, and each also
has a function that works on in-memory objects instead:

| # | Stage | Reads | Writes | Object function |
|---|---|---|---|---|
| 1 | `media` | the video | `media.json` | `split` |
| 2 | `audio` | `media.json` | `transcript.raw.json` | `listen` |
| 3 | `boundaries.evidence` | `media.json` or the raw transcript | `cuts.json` | `detect` |
| 4 | `boundaries` | `media.json`, `cuts.json` | `timeline.json` | `timeline` |
| 5 | `video` | `media.json`, `timeline.json` | `manifest.json`, `store/` | `ingest` |
| 6 | `cut` | raw transcript, `timeline.json` | `transcript.json` | `apply` |
| 7 | `describe` | `manifest.json`, `timeline.json`, `store/` | `descriptions.json` | `answer` |
| 8 | `embed` | `descriptions.json`, `transcript.json` | `embedded.json` | `encode` |

`timeline.json` is the chunk grid. The picture and the transcript are both cut
onto it, so a chunk id means the same span everywhere. The `policy` decides
where its boundaries fall:

| Policy | Boundaries at | Needs |
|---|---|---|
| `uniform` | fixed intervals (`chunk_s`, default 20 s) | nothing |
| `scene` | picture changes | a scene-detection pass |
| `vad` | pauses in speech | the transcript |
| `speaker` | changes of speaker | the transcript with diarization |

Chunks are kept between `min_s` and `max_s` seconds; a short final chunk is
merged into the one before it.

A video's output lives in one folder, `<into>/<video_id>/`: `media.json`,
`transcript.raw.json`, `cuts.json`, `timeline.json`, `manifest.json`,
`store/` (the kept frames as JPEGs), `transcript.json`, `descriptions.json`,
`embedded.json` and `aggregates/`. Running the pipeline again reuses what is
already done: a description or vector is recomputed only when its input,
model or question changed. A stage run on its own resumes when given
`previous=`.

## Samplers and questions

A sampler picks which frames are kept from each chunk; a question decides
what the vision model is asked about them. Any sampler pairs with any
question as `sampler:question`. With no question, the sampler's own name
is used.

| Sampler | Keeps a frame when |
|---|---|
| `uniform` | every Nth frame (`every_n`) |
| `clip` | the scene changes (CLIP embedding) |
| `yolo` | the people in view change (YOLO) |
| `objects` | detected objects change (YOLO-World, `vocabulary=`) |
| `text` | on-screen text changes (EasyOCR, `languages=`) |

Every chunk keeps at least one frame. `max_per_chunk` and `min_interval_s`
cap how many are kept.

```text
uniform:text            every Nth frame, asked to read the text on screen
yolo:overview           frames where people changed, answered as prose
clip:[text,scene]       one pass over the video, two questions
```

The built-in questions are `scene`, `overview` (prose only), `text`, and one
named after each sampler (`clip`, `uniform`, `yolo` for people, `objects`). A custom question is an instruction plus the fields to fill;
`one_of` fixes a field's values so search can filter on it exactly:

```python
from vidra.video_rag import describe

describe.add_question(
    "safety", "These {n} frames span {span}. List every hazard.",
    fields={"hazards": {"type": "list", "about": "Each hazard."},
            "severity": {"type": "text", "about": "The worst one.",
                         "one_of": ["none", "low", "high"]}})
# then: sampler="clip:safety"
```

Custom questions are stored in `data/prompts.json`. Built-in questions cannot
be redefined.

## Aggregates

An aggregator answers a question about the whole video and writes one answer
file. Nothing runs, and nothing is read, unless you name it.

| Cost | Aggregator | Answers |
|---|---|---|
| free | `stats` | duration, chunk lengths, words, words per minute, frames sampled |
| free | `speakers` | speaker count, turns, speech ratio, dominant speaker |
| free | `coverage` | how many chunks have picture, sound, both or neither |
| local | `ner` | named entities (GLiNER) and the chunks they appear in |
| local | `sentiment` | tone per chunk, overall, and where it turns |
| llm | `summary` | a summary of the whole video |
| llm | `chapters` | chapters, with boundaries placed by embedding similarity |
| llm | `events` | notable events, each tied to a chunk |
| llm | `entities:people`, `entities:objects`, `entities:text` | the same person, object or text linked across chunks, with an account of each |

Linking in `entities:*` is done by rules over embeddings; the model only
writes the account of each linked entity.

### 1. Name the files: a record

```python
d = run.folder                                   # data/out/<video_id>
video = aggregates.record(
    timeline=d / "timeline.json",              # required: the chunk grid
    transcript=d / "transcript.json",          # optional
    descriptions=d / "descriptions.json",      # optional
    manifest=d / "manifest.json",              # optional; only stats reads it
)
```

Documents from another video, or cut on a different grid, are refused.

### 2. Choose what each aggregator reads

| Input | Built with | Read by |
|---|---|---|
| record | `aggregates.record(...)` | `stats`, `coverage`, `speakers` |
| excerpt | `video.excerpt(transcript=..., answers=...)` | `ner`, `sentiment`, `summary`, `chapters`, `events`, custom prompts |
| sightings | `video.sightings(profile=..., answers=[...])` | `entities` |

```python
said = video.excerpt(transcript=True)
seen = video.excerpt(answers={"clip:activity": ["summary", "actors"]})
said_and_seen = video.excerpt(transcript=True,
                              answers={"clip:activity": ["summary", "actors"]})
people = video.sightings(profile="people", answers=["yolo"])
```

- An **excerpt** is text per chunk: the transcript, and/or chosen fields of
  chosen answers. A key of `answers` is an answer id as `descriptions.json`
  stores it (`clip:activity`, or `yolo` for a sampler asked its own question),
  and its value lists the fields to read; `summary` is the answer's prose.
- **Sightings** are one entry per thing a list field mentions, such as each
  person in `yolo`'s `people`, for linking. `keys=["clothing"]` narrows what
  identifies an entry.
- An answer, field or profile the record does not have is refused at this
  step, before anything is paid for. `out=` also writes the input to a file.

### 3a. Run one aggregator

```python
aggregates.stats(record=video, out="answers/stats.json")
aggregates.coverage(record=video, out="answers/coverage.json")
aggregates.speakers(record=video, out="answers/speakers.json")

aggregates.ner(input=said, out="answers/ner.json", labels=["person", "product"])
aggregates.sentiment(input=said, out="answers/sentiment.json")

aggregates.summary(input=said_and_seen, out="answers/summary.json", models=models)
aggregates.chapters(input=said, out="answers/chapters.json", models=models, max_spans=4)
aggregates.events(input=said_and_seen, out="answers/events.json", models=models)

aggregates.entities(profile="people", input=people, out="answers/people.json",
                    models=models)
```

`input=` also accepts the path of an input written with `out=`. `previous=`
is an earlier answer file, reused unchanged if it would be computed the same
way. Each call returns a receipt; `aggregates.load(path)` reads an answer.

### 3b. Run several: the pipeline

```python
done = aggregates.aggregate(
    out=d / "aggregates",
    models=models,
    database=db,                                 # optional copy
    previous=d / "aggregates",                   # reuse answers still current
    summary=said_and_seen,
    chapters=said,
    ner=said,
    sentiment={"spoken": said, "seen": seen},    # two answers: sentiment~spoken, ~seen
    entities_people=people,
    stats=video,
    settings={"ner": {"labels": ["person"]}, "chapters": {"max_spans": 4}},
)
```

It runs exactly the aggregators named, cheapest first, and keeps what each
one read in `out/inputs/`.

### Several videos

```python
both = aggregates.combine(records=[monday, tuesday], out="data/out/both")
```

This lays the videos end to end on one clock and returns a record, used like
any other.

### Custom prompts

A custom prompt is stored in `data/aggregates.json`:

```python
aggregates.add_prompt(
    "incident_report", "Write an incident report for this video.",
    fields={"report": {"type": "text", "about": "What happened, in order."},
            "severity": {"type": "text", "about": "How serious.",
                         "one_of": ["none", "minor", "major"]}})

aggregates.prompt(name="incident_report", input=seen, out="answers/incident.json",
                  models=models)
# or, in the pipeline: aggregates.aggregate(out=..., incident_report=seen)
```

`kind="fold"` (the default) gives one answer, `"spans"` a set of time spans,
and `"items"` a list where each item cites a chunk. `aggregates.add_profile`
adds a linking profile the same way.

## Search and storage

Every result is written to files. A database copy is optional:

| Backend | Search | Setup |
|---|---|---|
| `Folder(path)` | vector similarity only | none; reads the output folder in place |
| `Supabase()` | vector similarity and Postgres full-text, fused by rank (RRF) | run the SQL files below |

With no server, search the output folder in place:

```python
from vidra import Folder
moments, notes = search("the reactor explodes", run.video_id, models=models,
                        database=Folder("data/out"))
```

Supabase setup: run `db/supabase/video_rag.sql`, then
`db/supabase/aggregates.sql`, in the SQL editor, and add `vidra` under
Settings → API → Exposed schemas.

### A database of your own

`Database` is a base class with one method per thing a run saves. Subclass it
and fill in only the ones you want; the rest do nothing:

```python
from vidra import Database

class VectorsOnly(Database):
    name = "vectors"

    def write_embedded(self, video_id, document):
        for unit in document["units"]:
            my_index.upsert(f"{video_id}/{unit['chunk_id']}/{unit['sampler_id']}",
                            unit["vector"], unit)

video_rag("shop.mp4", "data/out", models=models, database=VectorsOnly())
```

Each hook receives a document as a dict, exactly what its JSON file holds:

| Hook | Called with |
|---|---|
| `write_media(video_id, document)` | `media.json` |
| `write_raw_transcript(video_id, document)` | `transcript.raw.json` |
| `write_cuts(video_id, document)` | `cuts.json` |
| `write_timeline(video_id, document)` | `timeline.json` |
| `write_manifest(video_id, document)` | `manifest.json` |
| `write_transcript(video_id, document)` | `transcript.json` |
| `write_descriptions(video_id, document)` | `descriptions.json` |
| `write_embedded(video_id, document)` | `embedded.json`: `embedder` and `units`, each with its `vector` |
| `write_prompts(rows)` | the questions describe asked |
| `write_source(source)`, `write_answer(source_id, answer, items, mentions)`, `write_definitions(rows)` | aggregate answers, as rows |
| `write_aggregate_units(source_id, units, embedder_key)` | embedded summaries, chapters and entities; they are embedded only if this is defined |

To search through it, also define `search`, `spans` and `video_ids` (and
`search_aggregates` for `aggregates.search`). A read that is not defined
raises `Unsupported` instead of returning nothing. `Supabase` in
`vidra/shared/storage/supabase.py` defines every hook and is the reference.

`search` filters by `sampler`, `question`, `chunk_ids`, a time window
(`after`, `before`), neighbouring chunks (`window`) and exact field values
(`structured={"severity": "high"}`). `aggregates.search` searches summaries,
chapters and linked entities, one level at a time.

A search must use the embedder that built the index; passing the same
`Models` to both ensures it.

## Models

Three roles, each filled by a model object:

| Role | Base class | Does | Default |
|---|---|---|---|
| `vlm` | `VLM` | frames to structured answers | `OpenAI("gpt-5.4-mini")` |
| `llm` | `LLM` | the llm aggregates and entity accounts | `OpenAI("gpt-5.4-mini")` |
| `embedder` | `Embedder` | text to vectors, for the index, search and linking | `OpenAIEmbedder("text-embedding-3-small")` |

A role left out of `Models` uses its default. Any combination works: the VLM,
the LLM and the embedder can each come from a different place.

### Ready-made models

| Class | Serves | Key |
|---|---|---|
| `OpenAI(model)` | VLM and LLM, through OpenAI's Responses API | `api_key=` or `OPENAI_API_KEY` |
| `Anthropic(model)` | VLM and LLM, through the Messages API | `api_key=` or `ANTHROPIC_API_KEY` |
| `Chat(model, base_url=, name=)` | VLM and LLM, for any server speaking Chat Completions | `api_key=`, optional |
| `OpenAIEmbedder(model)` | embedder, for OpenAI or any server's `/embeddings` | `api_key=` or `OPENAI_API_KEY` without `base_url` |
| `LocalEmbedder(model)` | embedder, a Hugging Face model in this process | none |
| `Stub()` | VLM and LLM that answer with placeholder text, for tests | none |
| `HashEmbedder()` | embedder from word hashes, for tests | none |

`Chat` covers Ollama, LM Studio, llama.cpp, vLLM, Gemini, Mistral, Groq,
OpenRouter and other OpenAI-compatible servers:

```python
import os
from vidra import Chat, OpenAIEmbedder

vlm = Chat("qwen2.5vl:7b", base_url="http://localhost:11434/v1", name="ollama",
           concurrency=1)                         # Ollama answers one call at a time
llm = Chat("gemini-2.5-flash", name="gemini", api_key=os.environ["GEMINI_API_KEY"],
           base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
embedder = OpenAIEmbedder("nomic-embed-text", base_url="http://localhost:11434/v1",
                          name="ollama")
```

For a server without JSON Schema support, pass `structured="json_object"` or
`structured="prompt"` (the schema goes into the prompt and the answer is
parsed).

Only `OpenAI`, `OpenAIEmbedder` and `LocalEmbedder` have been run against the
real service. The other request formats are checked against a mock server.

### A model of your own

Subclass `VLM`, `LLM` or `Embedder` and fill in its one method:

```python
from vidra import LLM, VLM, Embedder, Models

class MyVLM(VLM):
    key = "mylab:vision-v2"
    concurrency = 4                               # calls in flight at once

    async def generate(self, parts, schema=None, system=None, max_output_tokens=4000):
        # parts, in order: the instruction, then a label and an image per frame:
        #   {"type": "text", "text": "..."}
        #   {"type": "image", "data": b"<jpeg>", "mime": "image/jpeg"}
        # schema: {"name": ..., "schema": <JSON Schema>}. Return a dict matching
        # it (or a JSON string). With no schema, return text.
        ...

class MyLLM(LLM):
    key = "mylab:text-v1"

    def complete(self, prompt, schema=None, system=None, max_output_tokens=4000):
        ...                                       # a plain def runs in a thread

class MyEmbedder(Embedder):
    key = "mylab:embed-v1:768"

    def embed(self, texts):
        ...                                       # one vector per text, same width

models = Models(vlm=MyVLM(), llm=MyLLM(), embedder=MyEmbedder())
```

`key` is required. It is recorded with every answer and vector, and decides
whether stored work is still current, so **change the key whenever the model
behind it changes.** For an embedder it names the vector space: vectors under
different keys are never compared, so a search must use the embedder that
built the index. Override `embed_query` for a model that embeds queries
differently from passages, and `problems()` to report a missing API key before
a run starts.

A failure inside a method is reported as `ModelFailed`, naming the model.

## Hardware

Each PyTorch model uses CUDA if available, then Apple's `mps`, then the CPU.
Whisper uses CUDA or the CPU, since its backend (CTranslate2) has no `mps`
support. The `mps` path has not been run on a Mac. Intel Macs cannot install
the PyTorch in `requirements.txt`: there are no Intel Mac builds after 2.2.

Measured on an RTX 4060 laptop, and on the same machine with the GPU hidden:

| Step | GPU | CPU |
|---|---|---|
| Whisper + pyannote, 205 s of narration | 21 s | 316 s |
| `clip`, `yolo` and `objects` together, 60 s video | 25 s | 52 s |
| `text` sampler, 60 s of slides | 13 s | 212 s |

Results were the same on both: the same frames kept and the same 428 words.

## Checking an install

On 2026-10-05 the library passed its full check (every stage, the local
models and the paid models) on Python 3.11, 3.12, 3.13 and 3.14.

A frame store can be rebuilt from its manifest and the video, and compared
byte for byte with the original. A store that does not rebuild identically
means the manifest is missing something:

```python
from vidra.video_rag import video

done = video.recreate(manifest=D / "manifest.json", video="shop.mp4",
                      out="rebuilt", verify=D / "store")
done["verified"]["identical"]       # True: every frame named is byte-identical
```

`recreate` refuses a video that does not match the manifest (size, frame
rate, time base, frame count); `force=True` rebuilds anyway and lists the
differences.

From a checkout, `python -m vidra.shared.contracts.schemas --check` confirms
the JSON Schemas in `db/json/` match the document types.

## Project layout

```text
vidra/
  workflow.py     both tiers in one call
  video_rag/      tier 1: media, audio, boundaries, video, cut, describe, embed, retrieve
  aggregates/     tier 2: record and inputs, the aggregator functions, the pipeline,
                  combination, definitions
  shared/         config, errors and logging, document types, storage backends, models
db/               Supabase SQL, generated JSON Schemas, a wipe script
data/             default output location; not in git
```

## Limitations

- No test suite in the repository. The schema check and `video.recreate`
  are the checks that ship.
- Only OpenAI and the local embedder have been run against real services.
- Not run on Linux or macOS. The `mps` path is untested.
- Recorded files only; no live streams.
- People linking reads descriptions, not pixels, so it depends on how
  consistently the VLM words the same person.
- No licence file yet.
