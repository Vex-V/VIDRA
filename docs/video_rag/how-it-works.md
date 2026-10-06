# video_rag: how it works

`video_rag` turns one video into a searchable index of moments. It reads the
picture and the soundtrack onto one grid of time chunks, describes each chunk
with a vision model, transcribes it, embeds the text, and answers queries with
the chunks that match.

The functions are in [functions.md](functions.md).

## Stages

Eight stages run in order. Each reads files and writes one, into the video's
folder (`<into>/<video_id>/`):

| # | Stage | Reads | Writes |
|---|---|---|---|
| 1 | `media` | the video file | `media.json`: container, streams, duration |
| 2 | `audio` | `media.json` | `transcript.raw.json`: words, segments, speakers |
| 3 | `boundaries.evidence` | `media.json` or the raw transcript | `cuts.json`: where the content changes |
| 4 | `boundaries` | `media.json`, `cuts.json` | `timeline.json`: the chunk grid |
| 5 | `video` | `media.json`, `timeline.json` | `manifest.json` and `store/`: the frames kept |
| 6 | `cut` | raw transcript, `timeline.json` | `transcript.json`: the transcript per chunk |
| 7 | `describe` | `manifest.json`, `timeline.json`, `store/` | `descriptions.json`: one answer per chunk and question |
| 8 | `embed` | `descriptions.json`, `transcript.json` | `embedded.json`: text and vectors |

A stage that does not apply is skipped with a reason: `audio` on a file with
no sound, `describe` when the picture is not read.

## The chunk grid

`timeline.json` divides the video into chunks. The frames, the transcript,
the descriptions and the vectors are all keyed by chunk id, so a chunk id
means the same span of time everywhere, and a search result is a chunk.

The `policy` decides where chunk boundaries fall:

| Policy | Boundaries at | Needs |
|---|---|---|
| `uniform` | every `chunk_s` seconds (default 20) | nothing |
| `scene` | cuts in the picture | a scene-detection pass over the video |
| `vad` | pauses in speech | the transcript |
| `speaker` | changes of speaker | the transcript, with speakers |

Every chunk is kept between `min_s` (default 5) and `max_s` (default
`chunk_s`) seconds: short chunks are merged, long ones split evenly. A short
final chunk is merged into the one before it. A video with no detected cuts
still gets chunks of `max_s`.

Scene detection stores a score per frame in `cuts.json`, so a different
threshold can be tried (`boundaries.retune`, `boundaries.calibrate`) without
decoding the video again.

## Frames: samplers

The video is decoded once. Frames are offered at `per_second` (default 1) and
a **sampler** decides which to keep:

| Sampler | Keeps a frame when | Settings |
|---|---|---|
| `uniform` | every `every_n`-th offered frame | `every_n` |
| `clip` | the scene changes (CLIP embedding) | `threshold` |
| `yolo` | the people in view change (YOLO) | `threshold` |
| `objects` | the detected objects change (YOLO-World) | `threshold`, `vocabulary`, `confidence` |
| `text` | the text on screen changes (EasyOCR) | `threshold`, `languages` |

Every chunk keeps at least one frame. `max_per_chunk` and `min_interval_s` cap
how many are kept. Kept frames are written to `store/` as JPEGs, and
`manifest.json` records which frame each sampler kept in each chunk.

## Descriptions: questions

A **question** decides what the vision model is asked about the frames a
sampler kept. A sampler and a question pair as `sampler:question`:

```text
clip                  clip's frames, asked clip's own question
uniform:text          every Nth frame, asked to read the screen
yolo:overview         frames where people changed, answered as prose
clip:[text,scene]     one pass over the video, two questions
clip,yolo             two samplers
```

Each question has a shape: the fields its answer fills. Every answer has a
`summary` (prose) besides:

| Question | Fields |
|---|---|
| `scene`, `clip`, `uniform` | `setting`, `people`, `objects`, `visible_text`, `actions`, `changes`, `tags` |
| `yolo` | `people`: one object per person, with `appearance`, `clothing`, `role`, `action` and more |
| `objects` | `objects`: `object`, `appearance`, `context` |
| `text` | `visible_text`: `text`, `context` |
| `overview` | none: prose only |

Custom questions bring their own fields (`describe.add_question`). A field
with `one_of` has a fixed set of values, so search can filter on it exactly.

Each answer in `descriptions.json` is stored under its **answer id**: the
sampler and question (`clip:text`), or the bare name when the question is the
sampler's own (`yolo`).

## Embedding

Each answer becomes one **unit**: its summary plus its fields, rendered as
text. Each chunk's transcript is one more unit, with the answer id
`transcript`. Every unit is embedded, and `embedded.json` holds the text and
the vectors.

## Search

1. The query is embedded with the same embedder that built the index.
2. The database ranks units. `Supabase` fuses vector similarity with
   Postgres full-text search by rank (RRF); `Folder` uses vector similarity
   only.
3. Units are grouped into one **moment** per chunk. A chunk scores by its
   best unit plus half its second best, so a chunk is not ranked higher just
   for having more answers.

Filters narrow the units before ranking: a sampler, a question, a set of
chunks, a time window, or exact field values.

Vectors from different embedders are never compared. Every unit records the
embedder's key, and a search only reads units under its own embedder's key.

## Reuse

Running the pipeline again on the same file reuses the same folder and redoes
only what changed:

- A description is reused while its frames, its VLM (key and settings) and
  its question's wording are unchanged.
- A vector is reused while its text and its embedder's key are unchanged.

A different file that would get the same video id gets a new id (`clip-2`)
unless `on_conflict` says otherwise.

## Output and databases

Every artifact is a file in the video's folder. A database is an optional
copy: after each stage, the pipeline hands each new document to the
database's `write_<artifact>` method. A failed write is listed in
`run.problems` and does not stop the run.

## The frame store

`video.recreate` rebuilds `store/` from `manifest.json` and the video, and can
compare the result byte for byte with an existing store. The manifest alone is
enough to rebuild the frames, which makes it a check on the frame encoding.
