# video_rag: functions

How the stages fit together is in [how-it-works.md](how-it-works.md).

- [The whole pipeline](#the-whole-pipeline)
- [Search](#search)
- [One stage at a time](#one-stage-at-a-time)
- [Questions](#questions)
- [Rebuilding the frame store](#rebuilding-the-frame-store)
- [Models](#models)
- [Databases](#databases)
- [Errors](#errors)

## The whole pipeline

```python
from vidra import LocalEmbedder, Models, OpenAI, Supabase
from vidra.video_rag import video_rag

models = Models(vlm=OpenAI("gpt-5.4-mini"), embedder=LocalEmbedder())

run = video_rag(
    "shop.mp4", "data/out",          # the video, and the folder that holds one folder per video
    policy="scene",                  # chunk at picture cuts (uniform, scene, vad, speaker)
    sampler="clip,yolo:overview",    # which frames to keep, and what to ask about them
    models=models,
    database=Supabase(),             # optional copy of every artifact
)

run.video_id        # "shop"
run.folder          # Path("data/out/shop")
run.artifacts()     # {"media": ".../media.json", "timeline": ..., ...}
run.skipped         # {"audio": "the file carries no audio"} -- stages that did not run, and why
run.problems        # database writes that failed; the files were still written
```

Other arguments:

| Argument | Default | Meaning |
|---|---|---|
| `video_id` | the file name | the folder name under `into` |
| `on_conflict` | `"new"` | when another file has that id: `"new"` (mint `shop-2`), `"replace"`, `"refuse"` |
| `name`, `recorded_at` | from the file | the video's display name and recording time |
| `use_video`, `use_audio` | `True` | read the picture, the soundtrack |
| `vlm`, `embedder` | OpenAI | models, instead of `models=` |
| `resume` | `True` | reuse descriptions and vectors that are still current |
| `on_step` | none | `on_step(component, produced)`, called before (with `None`) and after each stage |

`video_rag.validate(Options(...))` returns every problem with a request
before anything runs.

## Search

```python
from vidra.video_rag import search

moments, notes = search("a customer paying in cash", "shop",
                        models=models, database=Supabase())

for m in moments:
    print(m.chunk_id, m.start_ts, m.end_ts, m.score)
    for hit in m.hits:                         # the answers that matched in this chunk
        print("   ", hit["sampler_id"], hit["content"][:80])
```

`models` (or `embedder=`) must hold the embedder that built the index. Read
`notes` when nothing comes back: they say why.

| Argument | Example | Meaning |
|---|---|---|
| `video_id` | `"shop"`, `["shop", "lobby"]`, omitted | which videos; omitted searches all |
| `moments` | `5` | how many moments to return |
| `question` | `"text"` | only answers to this question, under any sampler |
| `sampler` | `"clip:text"` | only this answer id |
| `strategy` | `"clip"` | only this sampler, any question |
| `chunk_ids` | `[3, 4, 5]` | only these chunks |
| `after`, `before` | `120.0`, `300.0` | only chunks overlapping this window, in seconds |
| `window` | `1` | also return each match's neighbouring chunks |
| `structured` | `{"payment": "cash"}` | exact field values |
| `database` | `Folder("data/out")` | where to search; default `"supabase"` |

## One stage at a time

Each stage is a function that reads its input files and writes one output
file. `D` is a video's folder.

```python
from pathlib import Path
from vidra.video_rag import audio, boundaries, cut, describe, embed, media, video

made = media.media("shop.mp4", "data/out")          # makes the folder
D = Path(made.stats["folder"])

audio.audio(D / "media.json", D / "transcript.raw.json",
            transcriber="whisper", diarizer="pyannote", language="en")

boundaries.evidence(D / "cuts.json", "scene", media=D / "media.json", threshold=27.0)
boundaries.boundaries(D / "media.json", D / "timeline.json", policy="scene",
                      cuts=D / "cuts.json", min_s=10, max_s=40)

video.video(D / "media.json", D / "timeline.json", D / "manifest.json",
            store=D / "store", sampler="objects", vocabulary=["cart", "basket"])

cut.cut(D / "timeline.json", D / "transcript.raw.json", D / "transcript.json")

describe.describe(D / "manifest.json", D / "timeline.json", D / "store",
                  D / "descriptions.json", previous=D / "descriptions.json",
                  vlm=OpenAI("gpt-5.4-mini"), limit=10)

embed.embed(D / "embedded.json", descriptions=D / "descriptions.json",
            transcript=D / "transcript.json", previous=D / "embedded.json",
            timeline=D / "timeline.json", embedder=LocalEmbedder())
```

Every stage returns a receipt (`Produced`) with `.artifacts` (what it wrote)
and `.stats` (its numbers). `previous=` is the stage's earlier output: what is
still current is reused.

Reading a result back, typed:

```python
media.load(D / "media.json")              # Media
audio.load(D / "transcript.raw.json")     # RawTranscript
boundaries.load(D / "timeline.json")      # Timeline
video.load(D / "manifest.json")           # Manifest
cut.load(D / "transcript.json")           # Transcript
describe.load(D / "descriptions.json")    # Descriptions
embed.load(D / "embedded.json")           # Embedded
```

Each stage also has a function that works on objects and writes nothing:

```python
m = media.split("shop.mp4")                                  # Media
raw = audio.listen(m)                                        # RawTranscript
cuts = boundaries.detect("scene", media=m)                   # Cuts
grid = boundaries.timeline(m, "scene", cuts=cuts)            # Timeline
plan = video.ingest(m, grid, "clip", frames=video.FrameStore("store"))   # Manifest
spoken = cut.apply(grid, raw)                                # Transcript
said = describe.answer(plan, grid, video.FrameStore("store"))   # Descriptions
units = embed.encode(descriptions=said, transcript=spoken)   # list of Unit
```

Trying scene thresholds without decoding again:

```python
boundaries.calibrate(D / "cuts.json")    # [{"threshold", "cuts", "rate", "median_gap_s"}, ...]
boundaries.retune(D / "cuts.json", D / "cuts.json", threshold=35.0)   # rewrites the cuts
boundaries.boundaries(D / "media.json", D / "timeline.json", policy="scene",
                      cuts=D / "cuts.json")                           # then the grid
```

## Questions

```python
from vidra.video_rag import describe

describe.add_question(
    "checkout",
    "These {n} frames span {span}. Describe what happens at the checkout.",
    fields={
        "payment": {"type": "text", "about": "how the customer pays",
                    "one_of": ["cash", "card", "phone", "none visible"]},
        "items": {"type": "list", "about": "each item scanned or bagged"},
        "people": {"type": "list", "about": "each person at the checkout",
                   "of": {"role": "cashier or customer", "doing": "what they do"}},
    })

video_rag("shop.mp4", "data/out", sampler="clip,clip:checkout", models=models)
search("paying", "shop", question="checkout", structured={"payment": "cash"},
       models=models)
```

- A field is `{"type": "text"}` or `{"type": "list"}`, with an `about` that
  tells the model what to put in it. `of` makes a list of objects; `one_of`
  fixes the values.
- With no `fields`, the answer is prose only. `shape="people"` reuses a
  built-in shape instead.
- `{n}` and `{span}` in the instruction are filled with the frame count and
  the chunk's time span.

```python
describe.questions()                  # every question name
describe.question("checkout")         # its instruction, fields and shape
describe.remove_question("checkout")  # answers already stored are kept
```

Custom questions are saved in `data/prompts.json`. Built-in questions cannot
be changed.

## Rebuilding the frame store

```python
from vidra.video_rag import video

done = video.recreate(manifest=D / "manifest.json", video="shop.mp4",
                      out="rebuilt", verify=D / "store")
done["verified"]["identical"]      # True when every frame matches byte for byte
```

A video that does not match the manifest is refused (`video.Mismatch`);
`force=True` rebuilds anyway and lists the differences in `done["problems"]`.

## Models

Three roles, each a model object. A role left out uses its default.

| Role | Used by | Default |
|---|---|---|
| `vlm` | `describe` | `OpenAI("gpt-5.4-mini")` |
| `llm` | the llm aggregators | `OpenAI("gpt-5.4-mini")` |
| `embedder` | `embed`, `search`, chapters, entity linking | `OpenAIEmbedder("text-embedding-3-small")` |

```python
from vidra import Anthropic, Chat, LocalEmbedder, Models, OpenAI, OpenAIEmbedder

models = Models(
    vlm=Chat("qwen2.5vl:7b", base_url="http://localhost:11434/v1",
             name="ollama", concurrency=1),
    llm=Anthropic("claude-haiku-4-5"),
    embedder=LocalEmbedder("BAAI/bge-small-en-v1.5"),
)
```

| Class | Role | Notes |
|---|---|---|
| `OpenAI(model)` | VLM, LLM | `api_key=` or `OPENAI_API_KEY` |
| `Anthropic(model)` | VLM, LLM | `api_key=` or `ANTHROPIC_API_KEY` |
| `Chat(model, base_url=, name=)` | VLM, LLM | any Chat Completions server: Ollama, LM Studio, vLLM, Gemini, Groq, ...; `api_key=` optional; `structured="json_object"` or `"prompt"` for a server without JSON Schema support |
| `OpenAIEmbedder(model)` | embedder | OpenAI, or any `/embeddings` server with `base_url=` |
| `LocalEmbedder(model)` | embedder | a Hugging Face model in this process; no key |
| `Stub()` | VLM, LLM | placeholder answers, for tests |
| `HashEmbedder()` | embedder | word-hash vectors, for tests |

`concurrency` (default 8) is how many calls run at once.

### Your own model

Subclass `VLM`, `LLM` or `Embedder` and fill in one method. The method may be
`async def` or a plain `def`.

```python
from vidra import Embedder, LLM, VLM

class MyVLM(VLM):
    key = "mylab:vision-v2"

    async def generate(self, parts, schema=None, system=None, max_output_tokens=4000):
        # parts: [{"type": "text", "text": ...},
        #         {"type": "image", "data": b"<jpeg>", "mime": "image/jpeg"}, ...]
        # schema: {"name": ..., "schema": <JSON Schema>}
        return {"summary": "...", "setting": "..."}   # a dict, or its JSON text

class MyLLM(LLM):
    key = "mylab:text-v1"

    def complete(self, prompt, schema=None, system=None, max_output_tokens=4000):
        return "..."            # text; or a dict when a schema is given

class MyEmbedder(Embedder):
    key = "mylab:embed-v1:768"

    def embed(self, texts):
        return [[0.0] * 768 for _ in texts]   # one vector per text, same width
```

- `key` is required. It is stored with every answer and vector and decides
  whether they are still current: **change it when the model changes.**
- Override `embed_query(text)` if queries are embedded differently from
  passages, and `problems()` to report, say, a missing API key before a run.
- An exception inside a method is reported as `ModelFailed`, naming the key.

## Databases

Results are always written to files. A database is an optional copy, and is
what `search` reads.

| Class | Search |
|---|---|
| `Supabase(url=, key=)` | vector and full-text, fused. Run `db/supabase/video_rag.sql` and `aggregates.sql` first, and expose the `vidra` schema |
| `Folder("data/out")` | vector only, reading the output folder in place; no server |

### Your own database

Subclass `Database` and fill in the methods you want. A write method left out
does nothing; a read method left out raises `Unsupported`.

```python
from vidra import Database

class MyStore(Database):
    name = "mystore"

    def write_timeline(self, video_id, document):
        self.grids[video_id] = [(c["start_ts"], c["end_ts"]) for c in document["chunks"]]

    def write_embedded(self, video_id, document):
        for u in document["units"]:
            my_index.upsert(id=f"{video_id}/{u['chunk_id']}/{u['sampler_id']}",
                            vector=u["vector"],
                            meta={**u, "video_id": video_id, "embedder": document["embedder"]})

    def spans(self, video_id):
        return self.grids.get(video_id, [])

    def video_ids(self):
        return list(self.grids)

    def search(self, vector, query, embedder_key, limit=20, video_ids=None,
               sampler=None, question=None, strategy=None, chunk_ids=None,
               structured=None):
        found = my_index.query(vector, top_k=limit, where={"embedder": embedder_key})
        return [{**f.meta, "score": f.score, "dense_rank": i, "text_rank": None}
                for i, f in enumerate(found, start=1)]
```

Write methods. Each receives the document as a dict, as its JSON file holds it:

| Method | Called with |
|---|---|
| `write_media(video_id, document)` | `media.json` |
| `write_raw_transcript(video_id, document)` | `transcript.raw.json` |
| `write_cuts(video_id, document)` | `cuts.json` |
| `write_timeline(video_id, document)` | `timeline.json`: `chunks`, each with `chunk_id`, `start_ts`, `end_ts` |
| `write_manifest(video_id, document)` | `manifest.json` |
| `write_transcript(video_id, document)` | `transcript.json` |
| `write_descriptions(video_id, document)` | `descriptions.json` |
| `write_embedded(video_id, document)` | `embedded.json`: `embedder` (the key) and `units`, each with `chunk_id`, `sampler_id`, `sampler`, `question`, `content`, `structured`, `vector` |
| `write_prompts(rows)` | the wording of each question a describe run asked |
| `write_source`, `write_answer`, `write_definitions`, `write_aggregate_units` | aggregate answers; see [the aggregates functions](../aggregates/functions.md#databases) |

Read methods, needed only to search through the database:

| Method | Returns |
|---|---|
| `spans(video_id)` | the chunk grid: `[(start_ts, end_ts), ...]`, index = chunk id |
| `video_ids()` | every video id held |
| `search(vector, query, embedder_key, limit, video_ids, sampler, question, strategy, chunk_ids, structured)` | the best units, best first, as dicts with `video_id`, `chunk_id`, `sampler_id`, `sampler`, `question`, `content`, `structured`, `score`, `dense_rank`, `text_rank` |

`search` must only compare against units whose `embedder` equals
`embedder_key`. `score` only has to sort correctly; `text_rank` may be `None`.
`Supabase` (`vidra/shared/storage/supabase.py`) implements every method and is
the reference.

## Errors

Every error the library raises on purpose is a `vidra.VidraError`:

| Class | Means |
|---|---|
| `vidra.Unavailable` | something is missing: a package, weights, an API key, a server. Retrying will not help |
| `vidra.shared.reporting.errors.Refused` | the request itself is invalid, e.g. `min_s` above `max_s` (also a `ValueError`) |
| `vidra.shared.models.base.ModelFailed` | a model call failed |
