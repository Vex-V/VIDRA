# FalCONvar

Video RAG ingestion, as a library. A video goes in; both the picture and the
soundtrack are read onto **one chunk grid**, described, embedded, and a
searchable index of moments comes out. A second tier answers questions over the
whole video: summaries, chapters, and who is who across chunks.

`CLAUDE.md` is the reasoning behind every decision here, and records what is
measured and what is not. `db/json/` holds every document's JSON Schema,
generated from the dataclasses.

## Install

```bash
pip install -r requirements.txt   # working on it: all of it, nothing falls back
pip install -e .
cp .env.example .env              # provider keys; Supabase and HF tokens if used
```

As a dependency, the heavy halves are opt-in:

```bash
pip install falconvar            # core: uniform sampling + an API model
pip install "falconvar[local]"   # clip/yolo/objects/text samplers, local models
pip install "falconvar[audio]"   # whisper + pyannote
pip install "falconvar[all]"
```

An extra you did not install fails with a plain `ModuleNotFoundError` where it
is needed.

## Use

```python
from falconvar import Models, Supabase, aggregates
from falconvar.video_rag import layout, search, video_rag

# who answers each role, chosen once and checked when built
models = Models(describer="openai", embedder="local", llm="openai")

run = video_rag("x.mp4", "data/out", policy="scene", sampler="clip,yolo",
                models=models)                      # media -> ... -> embed
at = layout(run.home)                               # every file in its folder

done = aggregates.aggregate(run.home, at["aggregates"], previous=at["aggregates"],
                            models=models, summary=True, entities_people=True)
summary = aggregates.load(done.artifacts["summary"]).payload
```

`example.py` is that, runnable. A copy in a database is one more argument, and
search reads it back:

```python
db = Supabase()                                     # SUPABASE_* from .env
run = video_rag("x.mp4", "data/out", models=models, database=db)
moments, notes = search("the moment the reactor exploded", run.video_id,
                        models=models, database=db)
```

For Supabase, run `db/supabase/video_rag.sql` and then `db/supabase/aggregates.sql`
in the SQL editor, and add `falconvar` to **Settings → API → Exposed schemas**.
video_rag's tables start `vr_`, the aggregates' `ag_`. Both files are idempotent
and are how a schema change is applied; `reset.sql` drops everything first.

A role can still be a string per call (`embedder="local"`), and a database a
name (`database="supabase"`). Setting a role both ways is refused. The one rule
that matters: **search with the embedder that built the index**, which a single
`Models` makes the default.

## The pipeline

Two tiers. `video_rag` extracts, and is a complete RAG engine on its own.
`aggregates` answers over what it extracted and never reads the video.
`workflow` runs one, then the other.

Every component takes the files it reads and the file it writes, so each runs
alone, and each also has a verb that works on objects with no filesystem.

| # | component | reads | writes | verb |
|---|---|---|---|---|
| 1 | `media` | the video | `media.json` | `split` |
| 2 | `audio` | `media.json` | `transcript.raw.json` | `listen` |
| 3 | `boundaries.evidence` | `media.json` *or* the raw transcript | `cuts.json` | `detect` |
| 4 | `boundaries` | `media.json` + `cuts.json` | **`timeline.json`** | `timeline` |
| 5 | `video` | `media.json` + `timeline.json` | `manifest.json`, `store/` | `ingest` |
| 6 | `cut` | the raw transcript + `timeline.json` | `transcript.json` | `apply` |
| 7 | `describe` | `manifest.json` + `timeline.json` + `store/` | `descriptions.json` | `answer` |
| 8 | `embed` | `descriptions.json` + `transcript.json` | `embedded.json` (text and vectors) | `encode` |

`--policy` decides where the grid's boundaries come from: `uniform`
(arithmetic, needs nothing), `scene` (picture changes), `vad` (silences) or
`speaker` (voice changes).

```bash
python -m falconvar.workflow samples/x.mp4 --sampler clip,yolo --tier llm
python -m falconvar.video_rag samples/x.mp4 data/out --sampler clip   # tier 1 only
python -m falconvar.aggregates data/out/<id> --out data/out/<id>/aggregates --summary --ner
python -m falconvar.aggregates --list                                 # every aggregator

# one component at a time; per-stage tuning lives here, not on workflow
python -m falconvar.video_rag.media samples/x.mp4 data/out
python -m falconvar.video_rag.boundaries D/media.json D/cuts.json --policy scene --evidence
python -m falconvar.video_rag.boundaries D/media.json D/timeline.json --policy scene --cuts D/cuts.json
python -m falconvar.video_rag.video D/media.json D/timeline.json D/manifest.json --sampler "clip:[text,scene]"
python -m falconvar.video_rag.describe D/manifest.json D/timeline.json D/store D/descriptions.json
python -m falconvar.video_rag.embed D/embedded.json --descriptions D/descriptions.json --embedder local
python -m falconvar.video_rag.retrieve "..." <id> --question text --embedder local
```

Each CLI's `--help` names its files.

## Samplers and questions

A **sampler** decides which frames get described; a **question** decides what
is asked about them. Pair any of each as `name:question`.

| sampler | keeps a frame when |
|---|---|
| `clip` | the scene changes |
| `yolo` | the people change |
| `objects` | an open-vocabulary detection changes (`--vocabulary`) |
| `text` | the writing on screen changes |
| `uniform` | every Nth decimated frame |

```bash
--sampler uniform:text          # read the screen on a stride
--sampler yolo:overview         # frames where people changed, asked for prose
--sampler "clip:[text,scene]"   # ONE pass over the video, two questions
```

A custom question is an instruction and a shape, stored in `data/prompts.json`:

```python
from falconvar.video_rag import describe
describe.add_question("safety", "These {n} frames span {span}. List every hazard.",
                      fields={"hazards": {"type": "list", "about": "Each hazard."},
                              "severity": {"type": "text", "about": "The worst one.",
                                           "one_of": ["none", "low", "high"]}})
```

Editing a question re-describes only the answers that used it.

## Aggregates

One file per answer, run cheapest first.

| cost | aggregators |
|---|---|
| free | `stats`, `speakers`, `coverage` |
| local | `ner`, `sentiment` |
| llm | `summary`, `chapters`, `events`, `entities:people`, `entities:objects`, `entities:text` |

What a text aggregator reads is a selection, defaulting to `transcript+*`:
`clip:activity` is one pairing, `clip:hazards[severity,hazards]` two fields as
one input, `a+b` joins, and `a,b` gives two answers.
`entities:<profile>` links the same person or thing across chunks by rules,
not by a model, then writes one account per entity.

A custom aggregate prompt is stored in `data/aggregates.json` and runs like the
built-ins. `kind` is `fold` (one answer), `spans` (chapters) or `items` (each
citing a chunk); `add_profile` adds a link profile the same way:

```python
from falconvar import aggregates
aggregates.add_prompt("incident_report", "Write an incident report for this video.",
                      fields={"report": {"type": "text", "about": "What happened, in order."},
                              "severity": {"type": "text", "about": "How serious.",
                                           "one_of": ["none", "minor", "major"]}},
                      inputs="clip:safety")
aggregates.aggregate("data/out/x", "data/out/x/aggregates", incident_report=True)
```

## Models

Three roles, each one string (a provider, or `provider/model`):

| role | does |
|---|---|
| `describer` | frames → structured answers (a vision model) |
| `embedder` | text → vectors, for `embed`, search and the linker |
| `llm` | text → the llm aggregates and entity accounts |

Providers: `openai`, `anthropic`, `gemini`, `mistral`, `deepseek`, `voyage`,
`openrouter`, `groq`, `xai`, `together`, `ollama`, `lmstudio`, `llamacpp`, and
`local` (in-process `BAAI/bge-small-en-v1.5`, no key). Keys go in `.env` under
each provider's usual name; any other OpenAI-compatible server goes in
`data/providers.json`, which names a key's variable and never holds one.

## Checking it

```bash
python -m eval.library_check              # every component and both pipelines, free
python -m eval.library_check --llm        # + the paid aggregates
python -m eval.linkers                    # people linking against hand labels
python -m falconvar.shared.contracts.schemas --check
python -m recovery.recreate data/out/<id>/manifest.json --verify data/out/<id>/store
```

## Layout

```
falconvar/
  workflow.py      the whole run
  shared/          paths · errors · contracts/ · storage/ (files, database,
                   supabase) · models/ (providers, llm, embedders, roles)
  video_rag/       tier 1: media audio boundaries video cut describe embed
                   retrieve, and driver.py (the pipeline)
  aggregates/      tier 2: select, one folder per aggregator, driver.py
example.py         the pipeline, then a summary and the people linker
eval/              library_check · linkers + attributes · harness
recovery/          STANDALONE: rebuild a frame store from a manifest + the video
db/                supabase/ video_rag.sql · aggregates.sql · reset.sql · wipe.py · json/ schemas
data/              everything a run writes; gitignored
```
