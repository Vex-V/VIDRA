# VIDRA

VIDRA (Video Description, Retrieval & Aggregation) is a Python library that
turns videos into a searchable index of moments and answers questions about
whole videos.

- **`video_rag`** splits a video into time chunks, keeps a few frames from each,
  has a vision model describe them, transcribes the audio, and embeds the
  text. A search returns the matching chunks with their start and end times.
  It also works on live streams.
- **`aggregates`** reads what `video_rag` produced and answers questions about
  the whole video: statistics, speakers, a summary, chapters, events, named
  entities, and the people or objects that recur.

Models and databases are pluggable. VIDRA includes classes for OpenAI,
Anthropic, any OpenAI-compatible server, local embedders, Supabase and a local
folder, and you can subclass a base class to add your own.

## Installation

VIDRA requires Python 3.11 or later. It is not on PyPI yet; install it from a
clone of the repository:

```bash
git clone https://github.com/Vex-V/FalCONvar.git
cd FalCONvar
pip install ".[all]"
```

| Extra | Adds |
|---|---|
| none | decoding, the `uniform` and `text` samplers, hosted models, Supabase and folder storage |
| `[local]` | PyTorch models: the `clip`, `yolo` and `objects` samplers, local embedders, `ner`, `sentiment` |
| `[audio]` | transcription (faster-whisper) and speaker identification (pyannote) |
| `[all]` | both of the above |

On Windows, pip installs a CPU-only PyTorch by default. To use an NVIDIA GPU,
add `--extra-index-url https://download.pytorch.org/whl/cu130`.
`requirements.txt` pins the exact versions VIDRA is tested with.

## Quick start

```python
import vidra
from vidra import Folder, LocalEmbedder, Models, OpenAI, aggregates
from vidra.video_rag import search, video_rag
from vidra.video_rag.samplers import Clip, Yolo

vidra.configure(env_file=".env")                 # reads OPENAI_API_KEY
models = Models(vlm=OpenAI("gpt-6-luna"),      # describes frames
                llm=OpenAI("gpt-6-luna"),      # writes summaries and chapters
                embedder=LocalEmbedder())        # embeds text on this machine
db = Folder("data/out")                          # search the output folder directly

# Index a video. Every result is written to data/out/shop/.
run = video_rag("shop.mp4", "data/out", policy="scene", sampler=[Clip(), Yolo()],
                models=models, database=db)

# Find moments.
moments, notes = search("a customer paying in cash", run.video_id,
                        models=models, database=db)
for m in moments:
    print(f"{m.start_ts:.0f}-{m.end_ts:.0f}s", m.hits[0]["content"][:80])

# Summarise the whole video.
d = run.folder
video = aggregates.record(timeline=d / "timeline.json",
                          transcript=d / "transcript.json",
                          descriptions=d / "descriptions.json")
told = video.excerpt(transcript=True, answers={"clip": ["summary"]})
aggregates.summary(input=told, out=d / "answers" / "summary.json", llm=models.llm)
print(aggregates.load(d / "answers" / "summary.json").payload["summary"])
```

## Documentation

The documentation is a set of HTML pages in [`docs/`](docs/). Open
[`docs/index.html`](docs/index.html) in a browser; the pages work offline and
include search. They cover installation and configuration, the pipeline and
each of its stages, samplers and questions, search, aggregates, writing your
own models, databases and samplers, and a reference for every file format and
error. [`docs/pages/examples/`](docs/pages/examples/) has complete scripts.

## Configuration

VIDRA reads API keys from environment variables. It does not read a `.env`
file unless you call `vidra.configure(env_file=".env")`. You can also pass
keys directly, for example `OpenAI(api_key=...)`.

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | `OpenAI`, `OpenAIEmbedder` |
| `ANTHROPIC_API_KEY` | `Anthropic` |
| `HF_TOKEN` | the first download of pyannote's speaker model |
| `SUPABASE_URL`, `SUPABASE_SECRET_KEY`, `SUPABASE_PUBLISHABLE_KEY` | `Supabase` |
| `VIDRA_DATA`, `VIDRA_WEIGHTS` | where output and model weights are stored |

To use Supabase, run `db/supabase/video_rag.sql` and then
`db/supabase/aggregates.sql` in its SQL editor, and add `vidra` under
Settings → API → Exposed schemas.

## Hardware

PyTorch models use CUDA if available, then Apple's `mps`, then the CPU.
Approximate processing time per minute of video on a mid-range laptop:

| Step | NVIDIA GPU | CPU only |
|---|---|---|
| Transcription and speakers (Whisper `small` + pyannote) | ~6 s | ~1.5 min |
| `clip`, `yolo` and `objects` samplers | ~25 s | ~50 s |
| `text` sampler | ~12 s | ~12 s |

## Status

VIDRA has been tested on Windows 11, with and without an NVIDIA GPU, on
Python 3.11 to 3.14. It has not yet been run on Linux or macOS. The `OpenAI`
and `OpenAIEmbedder` classes have been run against OpenAI; `Anthropic` and
`Chat` with servers other than OpenAI have only been tested against a mock
server.

## Development

```bash
pip install -r requirements.txt
python -m eval.library_check              # checks the whole library; exits 1 on failure
python -m eval.library_check --local      # also runs the local models
python docs/pages/build.py --check        # docs cover every public name and parameter
python -m vidra.shared.contracts.schemas --check   # JSON Schemas match the document types
```

## License

MIT. See [LICENSE](LICENSE).
