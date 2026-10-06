# VIDRA

**Video Description, Retrieval & Aggregation.** A Python library that turns a
video into a searchable index of moments, and answers questions about the
whole video.

- **`video_rag`** splits a video into time chunks, keeps a few frames per
  chunk, describes them with a vision model, transcribes the audio, and embeds
  the text. A query returns the matching chunks with their start and end times.
- **`aggregates`** reads what `video_rag` wrote and answers questions about the
  whole video: counts, speakers, a summary, chapters, events, named entities,
  and which people or objects recur.

Models and databases are pluggable: use the ready-made ones (OpenAI,
Anthropic, any OpenAI-compatible server, local embedders; Supabase or a local
folder), or subclass a base class to bring your own.

## Documentation

The full docs are HTML pages in [`docs/`](docs/): open
[`docs/index.html`](docs/index.html) in a browser. They work offline, straight
from disk, with search across every page.

| Page | Covers |
|---|---|
| Start here | install, keys, a quickstart |
| Concepts | chunks, the video's folder, answer ids, reuse |
| Recipes | complete scripts, also in [`docs/pages/examples/`](docs/pages/examples/) |
| The pipeline, Stages | `video_rag()`, `workflow`, and each stage with every setting |
| Samplers, Questions, Search | which frames are kept, what the vision model is asked, finding moments |
| Aggregates, Custom prompts | whole-video answers, and adding your own |
| Models, Databases | the ready-made ones, and writing your own |
| Reference | every file's fields, every error, hardware |

`python docs/pages/build.py --check` confirms the docs mention every public function
and parameter.

## Requirements

- Python 3.11 to 3.14.
- No system FFmpeg: decoding uses PyAV, which bundles it.
- An API key for a hosted model, unless every model runs locally.
- Optional: an NVIDIA GPU. Everything also runs on the CPU, more slowly.

Tested on Windows 11, with an NVIDIA GPU and on the CPU. Linux and macOS have
not been run.

## Install

```bash
git clone https://github.com/Vex-V/FalCONvar.git
cd FalCONvar
pip install -r requirements.txt
```

On an NVIDIA GPU, add PyTorch's CUDA index (pip's default PyTorch build on
Windows is CPU-only):

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

`vidra` itself is not installed: run your scripts from the repository root,
or add the root to `PYTHONPATH`.

## Configuration

The library reads API keys from the environment and never reads a `.env`
file on its own. Copy `.env.example` to `.env`, fill in the keys you use, and
load it:

```python
import vidra
vidra.configure(env_file=".env")
```

or pass keys in code: `OpenAI(api_key=...)`, `Supabase(url=..., key=...)`.

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | `OpenAI`, `OpenAIEmbedder` |
| `ANTHROPIC_API_KEY` | `Anthropic` |
| `HF_TOKEN` | the first download of pyannote's speaker model |
| `SUPABASE_URL`, `SUPABASE_SECRET_KEY`, `SUPABASE_PUBLISHABLE_KEY` | the Supabase database |

Output goes to `data/` in the checkout by default. `vidra.configure(data_root=...)`
or the `VIDRA_DATA` variable moves it.

## Quick start

```python
import vidra
from vidra import LocalEmbedder, Models, OpenAI, Supabase, aggregates
from vidra.video_rag import search, video_rag

vidra.configure(env_file=".env")
models = Models(vlm=OpenAI("gpt-5.4-mini"),      # describes frames
                llm=OpenAI("gpt-5.4-mini"),      # summaries, chapters, events
                embedder=LocalEmbedder())        # text to vectors, on this machine
db = Supabase()                                  # or Folder("data/out"): no server

# Index a video: every artifact goes to data/out/<video_id>/, and a copy to db.
run = video_rag("shop.mp4", "data/out", policy="scene", sampler="clip,yolo",
                models=models, database=db)

# Find moments.
moments, notes = search("a customer paying in cash", run.video_id,
                        models=models, database=db)
for m in moments:
    print(m.chunk_id, m.start_ts, m.end_ts, m.hits[0]["content"][:80])

# Answer questions about the whole video.
d = run.folder
video = aggregates.record(timeline=d / "timeline.json",
                          transcript=d / "transcript.json",
                          descriptions=d / "descriptions.json")
said = video.excerpt(transcript=True)
done = aggregates.aggregate(out=d / "aggregates", models=models, database=db,
                            summary=said, chapters=said, stats=video)
print(aggregates.load(done.artifacts["summary"]).payload["summary"])
```

For Supabase, run `db/supabase/video_rag.sql` and then
`db/supabase/aggregates.sql` once, and add `vidra` under Settings → API →
Exposed schemas.

## Hardware

PyTorch models use CUDA if available, then Apple's `mps`, then the CPU.
Whisper uses CUDA or the CPU. Measured on an RTX 4060 laptop:

| Step | GPU | CPU |
|---|---|---|
| Whisper + pyannote, 205 s of narration | 21 s | 316 s |
| `clip`, `yolo` and `objects` samplers, 60 s video | 25 s | 52 s |
| `text` sampler, 60 s of slides | 13 s | 212 s |

The `mps` path has not been run on a Mac. Intel Macs cannot install the
PyTorch in `requirements.txt`.

## Checking an install

- `python -m vidra.shared.contracts.schemas --check` confirms the JSON
  Schemas in `db/json/` match the document types.
- `video.recreate(...)` rebuilds a video's frames from its manifest and
  compares them byte for byte.
- `python docs/pages/build.py --run a.mp4 b.mp4` runs every docs example on two
  videos, with free stand-ins for the hosted models.

The full library check passed on Python 3.11, 3.12, 3.13 and 3.14.

## Project layout

```text
vidra/
  workflow.py     both tiers in one call
  video_rag/      media, audio, boundaries, video, cut, describe, embed, retrieve
  aggregates/     records and inputs, the aggregators, the pipeline, combining videos
  shared/         config, errors, document types, storage, models
docs/             index.html to open; pages/ holds the rest, examples and build.py
db/               Supabase SQL, generated JSON Schemas
data/             default output location; not in git
```

## Limitations

- No test suite in the repository.
- Only OpenAI and the local embedder have been run against real services;
  Anthropic and the Chat Completions servers are checked against a mock.
- Not run on Linux or macOS.
- Recorded files only; no live streams.
- People linking reads descriptions, not pixels, so it depends on how
  consistently the vision model describes the same person.
- No licence file yet.
