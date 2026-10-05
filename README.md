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
- [Command line](#command-line)
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
- An API key for at least one model provider, unless every role runs locally
  (see [Models](#models)).
- Optional: an NVIDIA GPU. Everything also runs on the CPU, more slowly.

Tested on Windows 11 with an NVIDIA GPU (CUDA), and on the CPU. Linux and
macOS have not been run.

## Install

```bash
pip install vidra              # core: uniform sampling, API models
pip install "vidra[local]"     # clip, yolo, objects, text samplers; local embedder, NER, sentiment
pip install "vidra[audio]"     # Whisper transcription, pyannote speaker diarization
pip install "vidra[all]"       # both
```

On an NVIDIA GPU, install PyTorch's CUDA build: pip's default build on Windows
is CPU-only.

```bash
pip install "vidra[all]" --extra-index-url https://download.pytorch.org/whl/cu130
```

From a checkout, `pip install -e ".[all]"` installs it editable;
`requirements.txt` pins the exact versions it was developed against.

## Configuration

**Keys.** The library reads API keys from environment variables and never
reads a `.env` file on its own. Copy `.env.example` to `.env` and fill in
the providers you use. The command-line tools load `.env` themselves; in your
own code, either call

```python
import vidra
vidra.configure(env_file=".env")
```

or pass keys directly, which takes precedence over the environment:

```python
from vidra import Models
models = Models(describer="openai", keys={"openai": "sk-..."})
```

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ... | model calls, per provider |
| `HF_TOKEN` | the first download of pyannote's gated model |
| `SUPABASE_URL`, `SUPABASE_SECRET_KEY`, `SUPABASE_PUBLISHABLE_KEY` | the Supabase backend (writes, searches) |
| `VIDRA_DESCRIBER`, `VIDRA_LLM`, `VIDRA_EMBEDDER` | default model per role |

**Where output goes.** In order of precedence:
`vidra.configure(data_root=...)`, then the `VIDRA_DATA` variable,
then `data/` in a checkout, then `~/.vidra`. Model weights follow the
same rule with `weights=` and `VIDRA_WEIGHTS`.

## Quick start

```python
import vidra
from vidra import Models, Supabase, aggregates
from vidra.video_rag import search, video_rag

vidra.configure(env_file=".env")
models = Models(describer="openai", embedder="local", llm="openai")

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

# Whole-video answers, also exported to the database
done = aggregates.aggregate(run.home, run.home / "aggregates",
                            models=models, database=db, summary=True, chapters=True)
print(aggregates.load(done.artifacts["summary"]).payload["summary"])

# Search the summaries and chapters
chapters = aggregates.search("the checkout gets busy", level="span",
                             models=models, database=db)
```

Before the first run, create the tables: see [Search and storage](#search-and-storage).

## Command line

Every module runs with `python -m`. Each one's `--help` lists its options.

```bash
# Everything: extract, then aggregate up to a cost tier
python -m vidra.workflow video.mp4 --sampler clip,yolo --tier llm

# Extraction only
python -m vidra.video_rag video.mp4 data/out --sampler clip

# Search
python -m vidra.video_rag.retrieve "a customer paying" <video_id> --embedder local --database folder

# Aggregates
python -m vidra.aggregates data/out/<video_id> --out data/out/<video_id>/aggregates --summary --ner
python -m vidra.aggregates --list
```

Each stage also has its own command taking the files it reads and the file it
writes, so a stage can be re-run or tuned alone. For example
(`D` is a video's output folder):

```bash
python -m vidra.video_rag.boundaries D/media.json D/cuts.json --policy scene --evidence
python -m vidra.video_rag.boundaries D/media.json D/timeline.json --policy scene --cuts D/cuts.json
python -m vidra.video_rag.video D/media.json D/timeline.json D/manifest.json --sampler "clip:[text,scene]"
python -m vidra.video_rag.describe D/manifest.json D/timeline.json D/store D/descriptions.json --limit 5
```

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

| Sampler | Keeps a frame when | Extra |
|---|---|---|
| `uniform` | every Nth frame (`every_n`) | core |
| `clip` | the scene changes (CLIP embedding) | `[local]` |
| `yolo` | the people in view change (YOLO) | `[local]` |
| `objects` | detected objects change (YOLO-World, `vocabulary=`) | `[local]` |
| `text` | on-screen text changes (EasyOCR, `languages=`) | `[local]` |

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

Each aggregator writes one answer file and runs only when named.

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

Text aggregators read a selection, by default `transcript+*` (the transcript
and every description):

| Selection | Means |
|---|---|
| `transcript` | the transcript only |
| `clip:safety` | one sampler's answers to one question |
| `clip:safety[severity,hazards]` | two fields of those answers |
| `a+b` | `a` and `b` read as one input |
| `a,b` | two separate answers |

```python
aggregates.aggregate(run.home, run.home / "aggregates", models=models,
                     summary=True, ner="transcript", sentiment=True)
```

Several videos can be aggregated as one: `aggregates.aggregate([home_a, home_b], out, ...)`
places them on one clock first.

A custom aggregate prompt is stored in `data/aggregates.json` and runs like
the built-ins:

```python
aggregates.add_prompt(
    "incident_report", "Write an incident report for this video.",
    fields={"report": {"type": "text", "about": "What happened, in order."},
            "severity": {"type": "text", "about": "How serious.",
                         "one_of": ["none", "minor", "major"]}},
    inputs="clip:safety")
aggregates.aggregate(run.home, run.home / "aggregates", models=models,
                     incident_report=True)
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

`search` filters by `sampler`, `question`, `chunk_ids`, a time window
(`after`, `before`), neighbouring chunks (`window`) and exact field values
(`structured={"severity": "high"}`). `aggregates.search` searches summaries,
chapters and linked entities, one level at a time.

A search must use the embedder that built the index; passing the same
`Models` to both ensures it.

## Models

Three roles, each set to a provider or `provider/model`:

| Role | Does | Default |
|---|---|---|
| `describer` | frames to structured answers (needs a vision model) | `openai` (`gpt-5.4-mini`) |
| `embedder` | text to vectors, for the index, search and linking | `openai` (`text-embedding-3-small`) |
| `llm` | the llm aggregates and entity accounts | `openai` (`gpt-5.4-mini`) |

Built-in providers: `openai`, `anthropic`, `gemini`, `mistral`, `deepseek`,
`openrouter`, `groq`, `xai`, `together`, `voyage`, `ollama`, `lmstudio`,
`llamacpp`, and `local` (an in-process embedder, `BAAI/bge-small-en-v1.5`,
no key). Any other OpenAI-compatible server is added in `data/providers.json`,
which names the key's environment variable and never holds a key.

Only `openai` and `local` have been run against the real service. The others
are checked against a mock server that records requests.

## Hardware

Each PyTorch model uses CUDA if available, then Apple's `mps`, then the CPU.
Whisper uses CUDA or the CPU, since its backend (CTranslate2) has no `mps`
support. The `mps` path has not been run on a Mac. Intel Macs cannot install
`[local]`: PyTorch publishes no Intel Mac builds after 2.2.

Measured with `eval.library_check --local` on an RTX 4060 laptop, and on the
same machine with the GPU hidden:

| Step | GPU | CPU |
|---|---|---|
| Whisper + pyannote, 205 s of narration | 21 s | 316 s |
| `clip`, `yolo` and `objects` together, 60 s video | 25 s | 52 s |
| `text` sampler, 60 s of slides | 13 s | 212 s |

Results were the same on both: the same frames kept and the same 428 words.

## Checking an install

```bash
python -m eval.library_check             # every stage and both tiers; free, offline
python -m eval.library_check --local     # + Whisper, pyannote and the model-backed samplers
python -m eval.library_check --llm       # + a real describer, embedder and llm aggregates (paid)
```

It exits non-zero if any check fails. On 2026-10-05, an installed `[all]` wheel
passed every check on Python 3.11, 3.12, 3.13 and 3.14.

Two further checks run from a checkout:

```bash
python -m vidra.shared.contracts.schemas --check     # JSON Schemas match the dataclasses
python -m recovery.recreate D/manifest.json --verify D/store   # rebuild the frame store, byte-compare
```

## Project layout

```text
vidra/
  workflow.py     both tiers in one call
  video_rag/      tier 1: media, audio, boundaries, video, cut, describe, embed, retrieve
  aggregates/     tier 2: select, one folder per aggregator, combination, definitions
  shared/         config, errors and logging, document types, storage backends, model providers
eval/             library_check, and benchmarks for linking and retrieval
recovery/         rebuilds a frame store from a manifest and the video; imports nothing from vidra
db/               Supabase SQL, generated JSON Schemas, a wipe script
data/             default output location; not in git
```

## Limitations

- No unit test suite. Verification is `eval.library_check`, the schema check
  and the frame-store rebuild.
- Only OpenAI and the local embedder have been run against real services.
- Not run on Linux or macOS. The `mps` path is untested.
- Recorded files only; no live streams.
- People linking reads descriptions, not pixels, so it depends on how
  consistently the describer words the same person.
- No licence file yet.
