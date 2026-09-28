# FalCONvar — working notes

Video RAG ingestion. A video goes in; both the picture and the soundtrack are
read onto **one chunk grid**, and a searchable index of moments comes out.

This file is the *why*. File headers say what a module does; the reasoning, the
measurements and the traps live here.

---

## Layout

```
falconvar/
  __init__.py      configure() · __version__ · the error base ·
                   the NullHandler. Imports almost nothing
  py.typed         PEP 561: without it a consumer's checker sees no annotations
  workflow.py      the whole run: video_rag's driver, then aggregates'
  shared/          paths · env · errors · logs · progress   both tiers import these
    contracts/     documents · schemas   what components hand each other
                   units (one embeddable thing + render) · fields (the builder)
                   reference: TYPES.txt, generated from the dataclasses
    storage/       files · db · supabase   where a document goes, and
                   separately whether a copy also goes to a database
    models/        providers · llm · embedders/   who answers a model call
                   errors imports nothing; paths imports only errors
  video_rag/       TIER 1: the video in, a searchable index out, and the search
    driver.py        Options · validate · process · video_rag() · the CLI
                     and vocabulary(), the one thing aggregates asks
    media/         1 split: what streams the file carries
    audio/         2 source · reader · models
      backends/      whisper · pyannote · cuda
    boundaries/   3+4 scenes (picture) · speech (soundtrack) · grid
    video/         5 reader · decimate · pipeline (store -> ../frames)
      samplers/      base · uniform · scene · people · objects · ocr
        perception/  detectors · descriptors · embedders. Every model weight
                     lives below this line and none above it.
    cut/           6 the transcript, onto the grid
    describe/      7 prompts (logic) · library (the vocabulary) · frames · reader
      prompts.json   BUILT-IN questions and shapes; shipped, read-only
      backends/      stub · model (every provider, through shared/models/llm)
    embed/         8 units · embedders · remote · local · readable
                     writes embedded.json -- text AND vectors. No index
    retrieve/        search: a query to ranked moments
                     postgres: the search_embeddings RPC, the one reader
  aggregates/      TIER 2: answers over what video_rag extracted; never the video
    driver.py        answer() · aggregate(), the pipeline · the summary vector
    record           a record (a video's folder) opened as a Context
    base             the protocols · Context · DefinitionRunner
    inputs (the selection grammar) · rendering (how rows read to a model)
    definitions/     prompts and link profiles, as data
      definitions.json  BUILT-IN summary · chapters · events · people · objects · text
    select/          record + selection -> an Excerpt or Sightings file
    one component per aggregator, each one input -> one answer:
    stats/ speakers/ coverage/        free:  a record in
    ner/ sentiment/                   local: an excerpt in
    prompt/                           llm:   an excerpt in, any prompt by name
      fold/ spans/ items/               the runner per prompt kind
    entities/      sightings in · linking (who is who, no model)
    combination      several videos' documents laid end to end, one record
api/               main (routes) · service (dispatch) · jobs (one worker)
                   browse (read-only queries over the rows a run wrote)
web/               the client at /app. No build step: index.html · app.js ·
                   app.css. Every form generated from /capabilities. Pages:
                   video rag · aggregates · prompts · search · data
recovery/          STANDALONE: recreate.py, imports nothing from the pipeline
db/
  supabase/        install.sql · reset.sql
  wipe.py          delete every video, locally and in Supabase
  json/            document schemas, generated from the dataclasses
data/              everything a run writes; gitignored
  out/<id>/        media, transcript.raw, cuts, timeline, manifest, store,
                   transcript, descriptions, embedded, aggregates/
  uploads/         what the API parked until a run read it
  prompts.json     custom questions, added through the API
  providers.json   model endpoints added or overridden; names key variables, never keys
weights/           detector and embedder checkpoints; a cache, not output
docs/ROUTES.md     the HTTP surface
```

126 Python files, ~17.0k lines.

## Commands

```bash
python -m falconvar.workflow samples/x.mp4 --policy vad --sampler clip,yolo:overview
python -m falconvar.workflow samples/x.mp4 --no-audio --sampler uniform:text
python -m falconvar.workflow samples/x.mp4 --tier llm --database supabase

# one tier at a time
python -m falconvar.video_rag samples/x.mp4 --sampler clip   # extract: media -> embed
python -m falconvar.aggregates data/out/<id> --out data/out/<id>/aggregates \
    --previous data/out/<id>/aggregates --ner --summary transcript+clip:activity   # only those two
python -m falconvar.aggregates data/out/<id> --out D --tier llm   # everything up to a cost
python -m falconvar.aggregates data/out/a data/out/b --out D --stats --entities-people   # combined first
python -m falconvar.aggregates --list                            # every flag, cost, default data

# one component at a time; per-stage tuning lives on these, not on workflow
python -m falconvar.video_rag.media samples/x.mp4
python -m falconvar.video_rag.media samples/x.mp4 --on-conflict replace   # not clip-2
python -m falconvar.video_rag.audio <id> --transcriber whisper --diarizer pyannote
python -m falconvar.video_rag.boundaries <id> --policy scene --evidence --stride 5 --threshold 27
python -m falconvar.video_rag.boundaries <id> --evidence --detect-width 160   # cheaper pass
python -m falconvar.video_rag.boundaries <id> --calibrate        # sweep, no decode
python -m falconvar.video_rag.boundaries <id> --retune 45        # rethreshold cached scores
python -m falconvar.video_rag.boundaries <id> --policy scene --chunk-duration 30
python -m falconvar.video_rag.video <id> --sampler "clip:[text,scene]"   # one pass, two questions
python -m falconvar.video_rag.video <id> --sampler clip:text+scene       # same, no brackets
python -m falconvar.video_rag.video <id> --sampler yolo --per-second 4 --min-interval 3
python -m falconvar.video_rag.video <id> --sampler objects --vocabulary "crate,pallet"
python -m falconvar.video_rag.video <id> --sampler objects --confidence 0.55   # detector, not change
python -m falconvar.video_rag.video <id> --sampler text --languages en,de
python -m falconvar.video_rag.audio <id> --no-vad-filter --compute-type int8
python -m falconvar.video_rag.audio <id> --overlaps      # keep overlapping speech
python -m falconvar.video_rag.describe <id> --max-tokens 4000   # re-describes everything
python -m falconvar.video_rag.video <id> --prune-store           # irreversible, opt-in
python -m falconvar.video_rag.cut <id>
python -m falconvar.video_rag.describe <id> --describer openai --limit 5   # costs money
python -m falconvar.video_rag.describe <id> --describer ollama/gemma3:4b   # any provider, provider/model
python -m falconvar.aggregates data/out/<id> --out D --summary --llm anthropic
python -m falconvar.video_rag.embed <id> --embedder local   # no key needed
python -m falconvar.video_rag.video <id> --sampler uniform:safety   # a custom question
python -m falconvar.video_rag.retrieve "..." <id> --sampler clip:text   # one pairing
python -m falconvar.video_rag.retrieve "..." <id> --question text       # across samplers
# the aggregates one component at a time: select what to read, then answer it
python -m falconvar.aggregates.select data/out/<id> in.json ner --selection transcript
python -m falconvar.aggregates.ner in.json ner.json --labels person,product
python -m falconvar.aggregates.prompt summary in.json summary.json --llm anthropic
python -m falconvar.aggregates.select data/out/<id> people.json entities:people
python -m falconvar.aggregates.entities people people.json entities.people.json
python -m falconvar.aggregates.stats data/out/<id> stats.json   # a record in, no selection
python -m falconvar.aggregates.combination data/out/a data/out/b --out data/out/ab   # several videos, one record
python -m eval.entities                         # grade linking against hand labels
python -m eval.attributes                       # prototype: people by attributes (test.md)
python -m eval.tracking --errors 3              # prototype: tracking veto / merge variants

# as a library, not a CLI
pip install falconvar            # core: uniform + an API model
pip install "falconvar[local]"   # clip/yolo/objects/text + in-process models
pip install "falconvar[audio]"   # whisper + pyannote
pip install "falconvar[all]"
python -m pip wheel --no-deps -w dist .      # build it

python example.py                # the whole library, checked: exits 1 on any failure
python example.py --llm           # + the paid aggregates (a summary, entities)
python example.py --database supabase   # + export and search -- WRITES to it
python example.py --keep out/     # keep what it wrote

python -m falconvar.shared.contracts.schemas --check     # CI: are the schemas stale
python -m falconvar.shared.contracts.reference           # regenerate TYPES.txt
python -m falconvar.shared.contracts.reference --check   # CI: is TYPES.txt stale
python -m falconvar.shared.contracts.reference --undocumented   # fields with no `#:`
python -m recovery.recreate data/out/<id>/manifest.json --verify data/out/<id>/store
#      ^ from the checkout root: recovery/ is not an installed package, on purpose
python -m uvicorn api.main:app --port 8000     # the app at /, /docs for the schema
```

**Everything a run writes lives under `data/out/<video-id>/`.** Grouped by
video rather than by artifact type, so one video's whole output is one thing to
inspect, copy or delete, and a later component adds to it without a new
top-level directory.

---

## Architecture — the load-bearing decisions

**Two tiers, each a driver over the components in its folder.** `video_rag`
extracts: the file, both modalities onto one grid, descriptions, vectors -- and
answers queries over them, so it is a complete RAG engine on its own.
`aggregates` answers higher-level questions over what video_rag extracted, and
never touches the video. `workflow.py` calls the two drivers and nothing below
them; each driver calls its own components -- the shape the pipeline always
had, one level up.

The dependency runs one way, and it is **one function wide**. `aggregates`
calls `video_rag.driver.vocabulary()` and nothing else: what a question or a
sampler may be named, so an input can be checked before a job is queued.

It was seven. The other six were never video_rag features -- they were shared
concerns misfiled there because video_rag was written first, which their own
imports gave away. `embed/embedders.py` imported nothing but `shared.errors`
and `shared.models.providers`; `units.render` imported nothing at all;
`library.check_fields` carried a docstring already saying *"a describe shape is
not the only answer built from fields: an aggregate prompt's is too"*. They now
live where both tiers can reach them without an edge:

    shared/contracts/units    Unit, render        embed makes one per chunk,
                                                  aggregates one per video
    shared/contracts/fields   the field builder   a describe shape and an
                                                  aggregate prompt both compile
    shared/models/embedders/  the registry        identity must be measured in
                                                  the space the index was built
    shared/storage/rows       write_video_unit    a row write, like every other
    (nothing)                 documents()         `paths` + a dataclass; the
                                                  other tier just reads the files

**`vocabulary()` is the one that cannot move**, and the reasons are worth
keeping. `shared` holding it would mean `shared` importing a tier -- a cycle,
and the one rule that holds today without exception. `aggregates` holding it
would be the same dependency spelled *wider*: two component imports instead of
one call, coupled to `shape_of`'s internal return shape rather than flat data.
And it cannot be read off a video's output, because `aggregates.validate` takes
no `video_id` -- a request is checked before any video is named, which is what
makes a typo a 422 at submit time rather than a job that runs and finds
nothing.

Verified by the move itself: renaming every component's `run`, renaming
`reader.describe` to `answer` and moving `linking.py` into `entities/` touched
**no line of `aggregates`**. And `embed` re-run against an already-embedded
video reported **0 embedded, 15 unchanged** -- the embedder key and every text
hash identical after the registry moved, so nothing already stored became
unreachable.

video_rag never reads anything aggregates wrote: the whole-video vector used to
be made by `embed` reading `aggregates/summary.json`, which also meant a first
run never made one, since embed runs before aggregate. `aggregates.index_summary`
writes it now, through `shared`, so the handoff is not a call in either
direction.

**The grid is a component, not a side effect.** Everything that needs
boundaries reads `timeline.json`; nothing derives them as a byproduct. There is
no ordering rule anywhere in `video_rag/driver.py`, because the answer falls out of
what the policy depends on, and `boundaries.POLICIES` is that table as data:

    uniform    nothing.  arithmetic over a duration `media.json` already has
    scene      the picture.  a scene pass decodes and scores it
    vad        a transcript. audio must finish first
    speaker    a transcript. audio must finish first

`uniform` is the case worth noticing: it needs neither modality to have run.

**Each modality has a precursor and a consumer**, and the symmetry is the
point:

    audio  ──┐              ┌── cut
             ├──► boundaries ┤
    scenes ──┘              └── video

`speech.py` derives vad/speaker cuts and lives in `boundaries/`, not `audio/`,
so that package never learns chunking exists. It reads `transcript.raw.json` as
a **file**, never by importing `audio` — which is what keeps the import graph
acyclic while the run order flips between policies.

**Ingest never edits a boundary.** It asks `timeline.nearest(ts)` and nothing
else. No streaming chunker, no `observe()` at native rate, no open-ended
`bounds_of`, no final-`end_ts` correction, no tail merge — all of which exist
only when a pass produces the grid it is simultaneously consuming, so decisions
get made before the information to make them exists and have to be patched.

**A component's signature is its whole surface, so everything under it has to
be reachable from there.** `/capabilities` reads the signature and the client
builds a form from that, so a setting the `run()` does not take is a setting
nobody outside the package has. Audited across video_rag once and it had
drifted in five places: the diarizer was constructed with **no arguments at
all**, so pyannote's `exclusive` -- which decides whether overlapping speech is
resolved, and the grid rests on the resolved form -- could not be set; whisper's
`vad_filter` likewise, and on the reference video it is the difference between
**10 segments and 0**; describe pinned `max_output_tokens` at 2000, the number
the `people` schema already truncated at 700; the text sampler was English-only
because `languages` stopped at the sampler constructor; and `embed` took a
`sink` it never read, which `/capabilities` then published as a working
control. A parameter accepted and ignored is worse than one missing: the form
offers it, the run reports success, and nothing says the setting did nothing.

Still not reachable, deliberately: per-sampler internals (`clip.mode`,
`yolo.crop_pad`, `objects.class_aware`/`metric`, `ocr.grid`, EasyOCR's
`canvas_size`), because one CLI has one `--threshold` and one `--vocabulary`,
and per-sampler configuration needs a spec syntax before it needs parameters.
`FrameStore.quality` and the `rotation` override are simply not wired.

**Components exchange files, never objects.** Every one is
`run(video_id, ...) -> Produced`, addressed by video id and a backend rather
than by assembled paths. That is what makes each independently runnable,
retryable and testable, and what lets one API route serve all of them.

**`documents.py` imports nothing, and that is load-bearing.** If a document's
dataclass lived in the component that produces it, `cut` would import
`boundaries` to read a timeline — growing exactly the edges the file handoff
removes.

**The raw transcript is stored before it is cut.** `listen` writes words,
segments and turns with no chunk ids; `cut` applies a grid. Re-cutting is free
and transcription is not, so a grid change never re-runs Whisper. That
asymmetry — audio conforms cheaply, a sampler's decisions cannot be revisited —
is what lets either modality own the grid.

**Cascade follows cost.** Rebuildable in seconds (`chunk_samplers`,
`transcript_chunks`) gets a foreign key with `on delete cascade`; anything that
cost inference or a paid call (`descriptions`, `embeddings`) gets **none** and
carries a fingerprint instead. Re-ingesting costs seconds where describing
costs money, so a cascade from the grid into `descriptions` would mean retuning
a scene threshold silently destroying everything a VLM was paid to produce.

**The frames are written before the row that claims them.** There is no
transaction across two REST writes, so ordering is the only guard. `_manifest`
wrote `manifests` and then `chunk_samplers`; when the second was refused the
first had landed, leaving a manifest claiming a run with no sampled frames —
indistinguishable from a run whose samplers kept nothing, which happens. Written
the other way round, the same failure leaves rows nobody points at and no
manifest claiming them, so a reader is told the truth: not ingested here yet.

**A component writes one file and stops; the pipeline decides the rest.**
Every component took a `sink` fanning its document out to a file and to
Postgres, and `embed` took an `index_name` naming a vector store. Both are
gone. `shared/storage/files.py` writes the document; `shared/storage/supabase.py`
is one public `write_<artifact>` per document; and `video_rag.driver.export`
reads each artifact back and calls it when a run names a `database`.

That one seam had produced the same class of bug over and over. `embed`
accepted a `sink` it never read, and `/capabilities` published it as a working
control. `sinks.support()` had to answer `any` rather than `all`, because
`video` writes a manifest *and* a directory of JPEGs that never reaches
`write`. The best-effort-versus-raise question was documented one way and
coded the other, and no component passed the `on_problem` that would have made
the docs true. `describe` and `aggregate` each wrote their own provenance rows
from inside the component, gated on a string nobody else read.

The layering is what makes a second database cheap: a `sql.py` offering the
same `write_<artifact>` names is a sibling file, where a new sink was a branch
in the fan-out, a row in `BACKENDS`, and a matching case in the `support()`
derivation. It also makes best-effort an *informed* choice -- the pipeline's,
recorded in `Run.problems` -- rather than a default nobody picked.

The re-read is the price, and it is small: the document is on disk because the
next component is about to read it anyway.

**Recompute and write are different questions.** The fingerprint governs
whether to recompute; the write happens regardless. Conflating them means a
run that adds a destination has nothing to recompute, writes nothing, and
reports success — found twice, in `embed` across two indexes and in
`aggregate` across two sinks. Both of those destinations are gone, and the
rule is what survives them: `embed` still rewrites `embedded.json` whole on a
run that embeds nothing, and a pipeline that adds `database=supabase` still
pushes every artifact even though none was recomputed.

**`recovery/` imports nothing from the pipeline.** Hand someone those files, a
manifest and the video, and they rebuild the store byte for byte with only
`av`, `opencv-python`, `numpy`. If recovery imported the pipeline it could lean
on a default living in code rather than in the manifest, and the manifest's
claim to be authoritative would go untested. Nothing enforces this
automatically any more — the old AST check went with `imports.py`.

**Recreate is the end-to-end oracle.** Any change to encoding, addressing or
the manifest is verified by rebuilding a store and byte-comparing. Verified at
13/13, 60/60, 83/83, 206/206, 76/76. If a change makes recreate non-identical,
the change is wrong.

**`--verify` compares the frames the manifest names, not the output
directory.** An output directory accumulates across runs for exactly the reason
a frame store does — which the orphan rule already tolerates in the other
direction. `rebuilt/` holding 206 frames from an earlier manifest made a
correct 76-frame run report **FAIL, 130 absent from the store**, when every
named frame was present and byte-identical. A false FAIL is the worst answer
the oracle can give, because a real one means the change is wrong and is
supposed to stop everything.

**Paths are anchored to the checkout, found by marker.** `shared/paths.py`
searches upward for `pyproject.toml` rather than counting parents — a parent
count is a fact about a file's depth in the tree, which is exactly what a
reorganisation changes. It broke twice this way: `WEIGHTS_DIR` at
`parents[3]/"weights"` resolved to a directory that never existed, so weights
downloaded wherever ultralytics decided and nothing reported it; and moving
`paths.py` one level down silently redirected every artifact.

**A marker is only ours if the package under it is ours.** Searching upward
finds *a* `pyproject.toml`, not necessarily this project's. Installed into a
venv inside someone else's repo, the search walked out of `site-packages` and
returned **their** root: the library imported fine and resolved `DATA_ROOT` and
`WEIGHTS` into their tree, so a run would have written every artifact and
338 MB of checkpoints there with nothing reporting it. Worse than the bare
case, which at least raised `RuntimeError` from `import falconvar` and so could
not be shipped at all. A candidate counts only when
`root/falconvar/shared/paths.py` resolves to that very file.

**The roots resolve on first use, not at import.** As module constants they
were computed before a caller could say where its data should go — an
installed copy raised from the import itself. `paths.configure(data_root=...)`
now works after the import that triggers it, and PEP 562 keeps every reader
spelling it `paths.OUT_ROOT` while the value is computed per access. Four
module-level captures had to go with them (`SCHEMA_DIR`, the two weights
caches, the API's uploads directory): captured at import, they ignore a later
`configure()` and put the checkpoints in the old place.

Precedence: `configure()`, then `FALCONVAR_DATA` / `FALCONVAR_WEIGHTS`, then
the checkout, then `~/.falconvar`. `configure()` wins because an embedding
application must be able to guarantee where it writes; the variables are for
when you do not control the calling code. Neither is a `.env` key: `.env` is
read when a key is needed, which can be earlier or later than a path is, and a
root that moved depending on which module was touched first would be worse than
one that cannot live there at all. `.env` itself is the checkout's from a
checkout and the working directory's from an install — the only file a
consumer would expect to be read.

**A leading underscore under `data/out/` marks a directory that is not a
video.** Written for the embedded Qdrant store, which lived at `_qdrant`
beside the videos rather than inside one; without the rule it was listed as a
video with no artifacts, by `paths.videos()` and so by `GET /videos`. That
store is gone and the rule is not, because the next directory that is not a
video would repeat it exactly.

**A video id becomes a directory name, so it is checked and not cleaned.**
`out_root() / video_id` is a join, not a containment, and a library takes its
id from whoever calls it -- an HTTP path segment, a filename stem, a job
record. `media.run(f, video_id="../../escaped")` wrote `media.json` **two
levels above the data root** while `Produced.video_id` read back unchanged, so
nothing anywhere reported that a run had left its own tree. The quieter one was
the underscore: the rule above was enforced on *read* only, so `_hidden` wrote
a complete, correct output directory that `videos()` -- and so `GET /videos`,
and so the whole client -- would never list again.

`paths.check_id` is on `home()`, the one line where both tiers join an id to a
root, so `artifact`, `exists`, `present` and every component's `load` arrive
through it. Cleaning instead of refusing would be worse: two ids differing only
in a refused character would land in one directory and silently overwrite each
other's artifacts. The alphabet is deliberately `api/main.safe_id`'s, so
whatever that cleaning produces is guaranteed to be accepted here -- verified
across `../../escaped.mp4`, `_hidden.mp4`, `café vidéo.mp4`, `...` and
`C:/x/y/My Video!.MP4`. One `@app.exception_handler` turns it into a 422 on all
six routes that take an id as a path segment, rather than a guard on each.

Still not refused: Windows device names (`con`, `nul`, `com1`), which cannot
name a directory there. That failure is a loud `OSError` at write time rather
than a silent one, so it is a gap rather than a trap.

**An id defaults to the filename stem, so two videos can want one directory --
and only a *different* file is a conflict.** `monday/clip.mp4` and
`tuesday/clip.mp4` both asked for `clip`, and the second silently won. Worse
than losing a document: only `media.json` was replaced, so the grid, the frame
store and the descriptions of the *first* video survived, each still
well-formed. Measured -- a directory whose `timeline.json` covered 205.28 s in
11 chunks beside a `media.json` reading 60.333 s. Nothing downstream compares a
grid against the media it was cut from, and `Timeline.fingerprint` does not
catch it: that guards documents cut on different *grids*. Over the API the
upload itself was overwritten too, so the first video's bytes were gone.

`media.on_conflict` is `new` (mint `clip-2`), `replace` or `refuse`, and `new`
is the default because nothing is lost by it. **`replace` deletes the whole
directory, not `media.json`** -- taking the id while leaving the old artifacts
is precisely the corruption above, with a flag on it.

The rule cannot be "the id is taken", because the ordinary case is the same
file again: `video_rag()` opens with `media.run` on every pass, and minting
there would orphan the manifest, the descriptions and the vectors, so a re-run
would re-pay for everything already done. `Media.source` is what tells them
apart -- `fingerprint_of(size, duration, container, both streams)`, which is
**free**, since every part of it was read to describe the streams anyway. The
same bytes at a different path fingerprint the same, which is wanted: a file
that moved is still that video. Two genuinely different videos agreeing on all
of it would read as one; hashing the bytes instead costs ~1.9 ms/MB (177 ms for
95 MB, so ~7 s on a three-hour original) and nothing else would change.

A `media.json` written before `source` existed carries None and reads as the
same video, so an existing checkout keeps resuming instead of minting a
duplicate of everything it has; the fingerprint is written on the way past. One
that exists but will not parse is a conflict, because the directory is occupied
either way.

Nothing below `media` changed, and that is the test of the design:
`video_rag()` already did `run.video_id = first.video_id`, reading the id out of
the receipt rather than trusting what it passed in. Verified end to end -- a
second `clip.mp4` ran the whole pipeline under `clip-2` while `clip` kept its
own artifacts.

Not done, deliberately: the API still writes uploads to `uploads/{id}{suffix}`
before `media` can say what the id is, so the *file* is still overwritten on a
collision, and `jobs.submit` records the pre-run id. And `videos` in Postgres
has no `source` column -- that needs a section 10 migration, and a replaced id
leaves its old rows behind, which `db/wipe.py` is the tool for.

**A missing input names the component that makes it.** Components hand off
through files, so getting the order wrong is the ordinary mistake on the
component path -- and seven of the eight answered it with the interpreter's
`[Errno 2] No such file or directory: 'C:\...\timeline.json'`, which asks the
caller to already know who writes that file. Worse, "that video does not exist"
and "you skipped a step" were the same sentence.

`paths.PRODUCED_BY` is artifact -> producing component, beside `ARTIFACTS`
because it answers the other half of the same question, and `paths.require`
is the one line all eleven reads pass through:

    V has no timeline; `boundaries` writes it. Present: media
    V has no raw_transcript; `audio` writes it. Present: media
    no video 'ghost' under <root>\out -- `media(path)` is what creates one

`MissingArtifact` keeps `FileNotFoundError` as a base, so every `except
FileNotFoundError` already written still fires -- including the one in each
driver's `main`, which is what makes a CLI print `error: ...` rather than a
traceback. `boundaries` keeps its own hand-written version, because it can say
more: which *policy* wants the cuts and which half of `evidence` produces them.

**A requirement is required.** Nothing falls back when a package in
`requirements.txt` is missing or older, or when `install.sql` was never run:
no hand-parsed `.env` without `python-dotenv`, no plain Supabase client for an
old SDK, no Python ranking when the search RPC is absent, no `transformers`
stand-in for `sentence-transformers`. Each of those was a second code path that
answered *differently* while looking the same -- the dense-only Supabase
ranking had never ranked anything, and nobody could tell. Heavy imports stay
function-local so `import falconvar` is light; a missing one fails with the
interpreter's own `ModuleNotFoundError`.

---

## Sampling and questions

**Which frames, and what to ask, are independent.** Every sampler takes a
`prompt` — it lives on the base class — so any strategy pairs with any question
as `name:prompt`. `uniform:text` reads the screen on a stride; `yolo:overview`
keeps frames where the people changed and asks for prose instead of the
structured people call. Unpaired, the question is the sampler's own name.

The point is cost. `text` fires *when the writing changes*, and finding that
out costs EasyOCR on every decimated frame — 98.1% of that sampler's total.
"Read the screen every so often" wants none of that: measured on Chernobyl,
`uniform:text` ran **no model at ingest** and the VLM still transcribed
`RadioFreeEurope RadioLiberty`, `BYELORUSSIAN S.S.R.`, `REACTOR 1`, `1977`
correctly. Two different questions, not two settings of one.

**A detector's settings are not the sampler's threshold.** `threshold` is how
much the frame must have changed to keep it; `confidence` is how sure the
detector must be that a box is a box at all. Different quantities on different
scales, so they are separate parameters landing on separate samplers --
`confidence` on `objects`, `languages` on `text` -- rather than more readings of
one number. `languages` is the one with no workaround: without it EasyOCR is
asked for English and there is nothing to say otherwise.

**A value with two readings is refused, and so is one the interpreter would
explain.** `describe.run(limit=0)` described nothing and reported success:
`limit` reads as a ceiling to a caller, and no ceiling is what `None` means, so
zero is the one value meaning two things -- and the run that does nothing looks
exactly like the run with nothing to do. `embed.run(batch=0)` reached
`range()`, which answered *"arg 3 must not be zero"*: the interpreter talking
about a loop the caller cannot see, where every sibling parameter already had
`per_second must be positive`. Both are named refusals now, and both are
checked before an embedder is built or an index opened.

**And a setting no chosen sampler reads is refused.** `--sampler uniform
--confidence 0.55` used to return `Produced` with the sampler built from *no
config at all*, and say nothing — the rule `audio.run` already applied to its
two backends, not applied here when `confidence` and `languages` were added.
`video.SAMPLER_SETTINGS` is setting → the samplers that read it, and
`build_samplers` refuses anything that lands nowhere, naming it and what does
read it. `boundaries.EVIDENCE_SETTINGS` is the same for the precursor passes,
keyed by **policy and not by precursor**: `speech.detect` takes `silence_s`
but only `vad` reads it — `speaker_cuts` has no such argument — so grouping
both speech policies together would let `--policy speaker --silence 2.0`
through to be ignored, which is the failure the table exists to stop.

Both are compared against the published default rather than a `None` sentinel,
so `/capabilities` keeps showing `stride: 5` to a form; passing a default
unchanged is a no-op either way. And `service.conditions()` now *derives* its
`when` from these two tables instead of restating them — a form that hid a
different set from the one the component enforces would offer a field whose
value is then rejected.

**A sampler runs once and answers a list of questions.** `clip:[text,scene]`
is one pass over the video answering two questions about the frames it kept;
`clip:text+scene` is the same thing without brackets, for shells that glob
them. Selecting frames is the expensive half -- CLIP or YOLO on every decimated
frame, EasyOCR at 98% of the text sampler's cost -- and a second question about
frames already chosen costs one more describe call.

**Specs naming the same sampler merge into one run.** `clip:text,clip:scene`
means exactly `clip:[text,scene]`, so brackets are the explicit spelling of
something that happens anyway rather than the only way to avoid paying twice.
Measured before this existed: `uniform:text` and `uniform:reactor` produced
**identical frame lists on all 14 chunks**, each having walked the video
separately. Merging is by name because one CLI has one `--threshold` and one
`--vocabulary`, so every spec shares a configuration.

**The manifest is keyed by run; everything downstream is keyed by answer.**
`chunks[].samplers` holds `clip` once, with its frames and `prompts:
[text, scene]` in the config. `descriptions` and `embeddings` hold `clip:text`
and `clip:scene` separately, because they are different units to a search even
though one pass produced both. `answer_id` is the bare name when the question
is the strategy's own, so an unpaired sampler keeps the id it always had and
nothing already indexed becomes unreachable.

`chunk_samplers.questions` is a `text[]` for the same reason: one row per run,
carrying the list.

**Every (sampler, question) pairing is independent.** `schema_for` is a
function of the question alone: two questions that share a field both answer
it, and both answers are kept under their own sampler id. Overlap is a choice
the user made at the `--sampler` line, and the answers are genuinely different
— measured on one chunk, `clip` gave `visible_text` as
`['RadioFreeEurope', 'RadioLiberty']`, `uniform:text` gave six objects each
with a `context` and an explicit `[unreadable]`, and a custom `reactor`
question gave five plain strings including `RADIATION` and `115,000`. All three
are stored, embedded and separately searchable.

**Sibling narrowing was tried and removed.** A call's schema used to depend on
which other questions were asked about the same chunk: the fallback shape gave
up any key a specialist owned. It produced three silent faults — sampler ids
passed where questions were meant; `yolo:overview` taking a key from a call
whose schema answered none; and two fallback questions overlapping on all seven
keys with no rule for which won, so a chunk rollup kept whichever sorted first.

Each fault left a well-formed document. That is the signature of a seam in the
wrong place rather than three unrelated mistakes, and the payoff did not
justify it: narrowing only ever fired for **one of four** pairings (fallback +
specialist, for three specific keys) and saved 2.3% of output on the case
measured, because the summary is half the output and was never narrowed.
Extending it instead would have meant answering "when two questions both want
`people`, who wins" — and there is no non-arbitrary answer, because the user
asked both.

Gone with it: `OWNER`, `owned_by`, `questions_on`, the `siblings` argument and
the `chunk_questions` context key, across five files.

**There is no chunk-level rollup.** `descriptions.chunks[].structured` used to
flatten every sampler's answer into one record, which is what forced a winner
for a shared key. Nothing read it — units, both index writers, `rows.py` and
every aggregator work from the per-sampler blocks — so it was deleted rather
than given a tie-break rule.

**An unknown question is rejected, not fallen through.** `question_for` falls
back to the scene question by design, which makes `yolo:overvew` a run that
completes, costs money and answers something nobody asked. `prompts.QUESTIONS`
is the vocabulary, and `workflow.validate` checks **both halves** of every
`name:question` pair against it and the sampler registry.

It has to be there rather than only in `describe`, which is where it used to
be: that check reads the finished manifest, so `yolo:overvew` was a 202 that
ran media, audio, boundaries and a whole video decode before failing on a
typo -- the late failure `validate` exists to prevent.

**And it has to be on `video.run` as well, because `validate` guards only one
of the two public levels.** `build_samplers` has taken a `questions`
vocabulary since it was written and *nothing ever passed one*, so a caller
driving the components itself -- half of what the library is for, and every
`POST /videos/{id}/run/{component}` besides -- walked straight past the check.
Measured: `video.run(sampler="uniform:nope")` completed in **1.09 s** having
decoded the video and stored 61 frames (26.68 MB), and then every later
`describe` refused the manifest it wrote, *including one naming only good
samplers*. The only way out was re-running `video`, and nothing said so. Now
0 ms, nothing decoded.

The vocabulary is resolved in `video/driver.py`, which is a composition root
like `workflow.validate` and the tier driver. **Ingest still does not depend on
describe**: a sampler records the question as an opaque string and
`samplers/base.py` never reads it. The samplers are also built before any
artifact is read, so an argument error never costs an artifact load.

**Ingest does not depend on describe.** A sampler records the question as an
opaque string and `base.py` never reads it. Only the drivers import `prompts`,
function-locally: they are composition roots, and validating a typo is the one
thing they want the vocabulary for.

**A question is an instruction and a shape, and the shape carries the schema.**
`prompts.json` holds both; `prompts.py` is logic over it. The built-ins are
expressed in the same terms a custom question uses -- `yolo` is not a special
case in the code, it is the `people` shape -- so there is one mechanism rather
than a shipped set and an extension point beside it. A shape is a set of
fields; `prose` is the degenerate one with none, which is what `overview`
answers in. The shape marked `fallback` is what an unrecognised question
resolves to, and that is its only privilege.

**A custom question names a shape or brings its own, and a custom shape is
built rather than accepted.** `fields` is a builder -- `text` or `list`, plus
`of` for a list of objects and `one_of` for a fixed vocabulary -- and the JSON
Schema is generated from it. Raw JSON Schema over HTTP is refused because the
call goes out with `strict: true`, whose subset is narrow: a schema the API
rejects would fail *after* the frames are read, with the request about to be
paid for, which is the same late failure `check()` exists to prevent for a
`{typo}` placeholder. The builder spans exactly what the shipped shapes use --
verified by rebuilding all five of them from their own field lists, identical.

Shipped shapes stay shipped. A custom shape is stored under its question's
name, dies with it, may not take a built-in shape's name (`people` and `prose`
are shape names that are *not* question names, so the question-level guard
misses them), and can never claim `fallback`. So `yolo` still means the same
thing in every deployment; what is deployment-local is a custom question, which
it already was. Caps -- 12 fields, 8 nested keys, 24 enum values -- are refused
at write time, because structured answers already run ~3x longer than prose and
the `people` schema truncated into unparseable JSON at 700 output tokens.

The earlier shipped-shapes-only rule was verified against the previous
hard-coded module over **448 (question, siblings) combinations** at the time it
was extracted -- every schema, every instruction, the owner map and `merge`
identical.

**Built-ins ship in the package; custom questions live in `data/prompts.json`.**
A custom entry may not shadow a built-in, refused at write time as a 409 and
dropped at load time as well, because the file is hand-editable and a shadowed
built-in is the one failure that would change a shipped question's meaning
silently. The alternative is a deployment whose `yolo` means something other
than every other deployment's, with nothing in the repo saying so.

**A single shared schema was tried and is wrong.** Every field being present
means the model may fill any of them, and it does — asked about people it
returned a paragraph about the room, paid for twice, leaving two `setting`
values with no rule for which wins. A prompt saying "focus on X" is a request;
a schema with no Y field is a guarantee.

**A specialist returns objects, never parallel lists.** One entry per person
carrying `appearance`, `clothing`, `role`, `action` — not a list of people
beside a list of actions, which does not say who did what and cannot be made to
afterwards.

**`overview` is a question with no fields at all**, and only a question. Summary
only, 4–5 sentences, overriding the ">= 150 words" instruction every other
prompt carries. Its shape has no fields at all, so it costs only a summary.
Measured: 84 and 86 words, 4 sentences, zero structured keys.

**Structured answers are ~3x longer than prose.** At `max_output_tokens=700`
the people schema truncated mid-string and came back as unparseable JSON. The
default is 2000, and truncation is detected from the response's own
`status`/`incomplete_details` rather than inferred from a JSON error further
down.

**`uniform` counts decimated frames, not seconds.** `every_n` is a stride over
the stream the sampler was actually offered, read off `chunk_local_index` — the
one thing about position a sampler is handed. The cadence in seconds is a
consequence of decimation: `every_n=3` is one frame every 3 s at
`per_second=1` and one every 0.75 s at 4. A cadence in seconds regardless is
still expressible through `min_interval_s`, enforced in the base class before
the strategy runs. Default 1: every decimated frame.

**`max_output_tokens` is a setting, and it is part of the resume key.**
`ModelDescriber.config()` reports it, so raising it re-describes everything
already stored, at cost. That is correct rather than unfortunate: a truncated
answer and a whole one are different answers, and a stored one cannot say which
it was. `None` means the backend's 2000 and produces the config byte for byte
as before, so nothing already described went stale when the parameter appeared.

**Resume is keyed on the manifest, the describer *and* the prompts.** Without
the model check, describing with the stub and then switching to a real one
skips every pair and reports success having done nothing — the most expensive
kind of silent no-op, since the output looks complete. Editing a prompt changes
the output but not the model id, so the `model` block carries prompt hashes
too.

**Those hashes are per question, not one over the vocabulary.** A single hash
meant that *adding* a question — which cannot change what any existing answer
should say — invalidated every description of every video, and the next run
silently paid to rebuild them all. `model.prompts` is `{question: hash}`, and a
stored pair is current when its own question still hashes the same. Measured on
Chernobyl with `clip,uniform:safety`: editing only `safety` gave **14
described, 14 skipped**, and adding an unrelated third question gave **0
described, 28 skipped in 0.0 s**. Under the single hash both would have been 28.

A stored `prompts` that is a bare string is the pre-map format; it cannot be
reduced to a per-question map, so every pair is re-described once rather than
kept on a provenance nothing can check.

**Describing reads the frame store and nothing else.** No seek-the-video
fallback: the store exists so this stage has its frames in hand, and a fallback
would quietly do its job while leaving it broken — silently, since the output
is identical and only ~40x slower. A short frame list is never returned either:
a description covering 8 of the 9 frames it claims is indistinguishable from a
correct one once written down.

---

## Cost — where the time actually goes

**Convert lazily. This is the largest single factor.** Decoding a frame costs
**0.40 ms**; converting it to a full-resolution BGR array costs **6.2 ms**.
Ingest needs pixels only for frames that survive decimation — 4% of them at 1/s
from 25 fps — and the decimator answers from `media_ts` alone, so its verdict
is asked *before* the conversion.

| 3 h of 720p25 | |
|---|---|
| convert every frame | 3.55 ms/frame → **16.0 min** |
| convert only decimated | 0.50 ms/frame → **2.2 min** |

**7.2x for identical output.** This is only possible because scene detection
left the ingest pass; while it was there, `observe()` needed pixels at native
rate and the reader was the only thing holding them.

**A separate scene pass costs ~3 minutes per three hours, not more.** Measured
optimally on both sides: fused 11.97 min against separate 14.86, the difference
being one bare decode at 1.82 min. So the split is architectural, not a
performance trade — and the earlier "10–15 min" estimate was measuring a
lazy-conversion bug, not the architecture.

**Stride the scene detector, and recalibrate when you do.** It scores the
difference between *consecutive frames it was given*, so at stride 5 it
compares moments 200 ms apart rather than 40 ms.

| stride | ms/frame | 3 h | cuts found | real cuts kept |
|---|---|---|---|---|
| 1 | 12.89 | 58.0 min | 2 | — |
| 5 | 2.97 | **13.4 min** | 6 | 2/2 |
| 25 | 0.94 | 4.2 min | 11 | 2/2 |

Every real cut survives every stride; what rises is false positives, because
threshold 27 is calibrated for adjacent frames. `min_s` absorbs some — measured,
a strided pass double-fired at 27.0 s and 27.2 s on one scene change and the
guard merged both away.

**Cache the scores, not the cuts.** `cuts.json` records the per-frame series,
so re-thresholding is arithmetic over a cached array. Verified **identical**
cut lists to a full re-run at thresholds 45 and 20, in 0.12 ms against 5.9 s —
40,000x. Fused, every threshold change costs a full re-ingest with CLIP and
YOLO loaded.

**A frame kept by two samplers is one file, and must be counted once.**
`uniform:text` at stride 1 offers every decimated frame and `clip` picks from
that same set, so the second pick names a file the first already wrote. The
store is addressed by read index, so the write was harmless -- the accounting
was not: 266 frames and 52.75 MB reported against 206 files and 42.18 MB on
disk, a 25% overstatement of the figure someone would use to plan capacity, on
a two-sampler run. `frames_sampled` is picks and `stored_frames` is files, and
they are different numbers. Deduplicated per run, not per directory, so a file
an earlier run left behind is still rewritten.

**A store accumulates across runs and is never pruned automatically.**
Ingesting with `uniform` then with `clip` left 206 files where the manifest
named 83. `--prune-store` is opt-in because deleting frames is the one
irreversible thing ingest can do.

---

## Retrieval — measured

These figures predate the current tree and were taken on the previous
pipeline's corpus. The **orderings** are the finding; treat the decimals as
provenance rather than current fact, since nothing here has been re-measured
against a corpus larger than one video.

**Embed the summary *and* the structured fields.** 22 disjoint query pairs,
dense MRR (random 0.457):

| embedded from | literal | paraphrase |
|---|---|---|
| summary | 0.528 | 0.522 |
| structured | 0.636 | 0.586 |
| both | **0.705** | **0.608** |

Every summary repeats the same setting; the fields do not, so they carry far
more distinctive content per token while a bound object per entity keeps
who-did-what intact.

**Each half of the hybrid is strong exactly where the other fails.** 22 query
pairs with zero shared content words:

| | literal top-1 | literal MRR | paraphrase top-1 | paraphrase MRR |
|---|---|---|---|---|
| dense | 23% | 0.528 | 23% | 0.522 |
| BM25 | **59%** | **0.752** | 18% | 0.468 |

Neither half knows which kind of query it was handed, and a search box gives no
signal — which is the whole argument for fusing rather than choosing. Withheld
lexically on the shipped path: literal **+0.140, CI [+0.020, +0.260]**;
paraphrase **+0.000, CI [0, 0]** — not degraded, *identical*, because BM25
matched nothing at all.

**RRF twice, never a weighted score.** Cosine distance and `ts_rank_cd` have no
common scale, and any weight between them would be invented; RRF reads only the
orderings, so it needs no calibration.

**A chunk scores as its best unit plus a discounted second, never a sum.**
`score = 1/(k+best) + 0.5/(k+second)` at k=10. Summing over every unit a chunk
contributed applies RRF to the wrong problem: it fuses several rankings of the
*same* items, where the term count is constant, while a chunk contributes one
term per sampler that described it. At k=60 over ~20 candidates `1/(k+rank)`
spans only 1.31x, so count overwhelmed rank — a chunk whose best description
ranked 13th beat one whose best ranked 1st, and the shipped ranking got the
*video* right 57.7% of the time.

| aggregation | video ok | literal MRR | paraphrase MRR |
|---|---|---|---|
| sum, k=60 | 0.577 | 0.421 | 0.341 |
| max, k=60 | 1.000 | 0.668 | 0.442 |
| **max + 0.5·second, k=10** | **1.000** | **0.682** | 0.446 |

Invisible on a single video with a uniform sampler set, where every chunk
contributes the same number of terms and the bias cancels. It appears the
moment an index holds more than one video.

**One vector space per embedder, never per sampler.** A space is defined by the
model, not by which prompt produced the text. The sampler is payload and
querying one is a *filter* — which gives up the agreement signal: a chunk
contributes fewer terms, so scores fall.

**A unit carries both halves of its id as fields, not just the id.** `sampler`
and `question` are separate columns in `embeddings`, so three filters are
each one equality: this **pairing**
(`sampler_id = clip:text`), this **question** wherever asked
(`question = text`), and this sampler's whole output (`sampler = clip`).

Filtering by question is the query a person actually makes — "the text on
screen", not "what the CLIP sampler said" — and it is **not** a suffix match on
the id, because a bare id like `clip` means the question *is* the strategy
name. Measured with `clip:[text,scene],uniform:text`: `question=text` returns a
chunk carrying `clip:text` **and** `uniform:text` at 0.1222 where
`sampler=clip:text` returns one unit at 0.0909, and unfiltered returns all four
units at 0.1294. The three answers are different, which is the point.

Neither column cost a re-embedding: `text_hash` is over the content, so
existing rows were backfilled from `sampler_id` with an `update` (since removed
from `install.sql`: 0 of 49 rows still needed it). Qdrant was the exception,
and the reason is worth keeping for any store that behaves like it: payload is
written only on upsert, so points predating a field need a forced re-index
where a table can be backfilled in place.

**What gets embedded must not depend on which copy it was read from.** `jsonb`
preserves array order but not object key order, so a description read back from
Postgres hands its keys back in a different order from the file. Joining
`item.values()` in iteration order made the same person into different text, a
different `text_hash` and a different vector depending on its source — cosine
0.995 between them, close enough that no ranking ever looked wrong. Keys are
sorted at **every** level; determinism had been handled one level deep and not
two.

**A structured field is only filterable if its values are a vocabulary.** `role`
is free text, so one video produced `cashier`, `customer`, `cashier or customer
near checkout`, `child customer` — and a filter for "cashier" matches all of
them. The mechanism works; what is missing is an `enum`.

**Generating paraphrase test queries needs verification.** Asked to "share no
content words", the model kept a median 50% of them, and BM25 appeared to win
on paraphrases as a result.

**Cross-modal agreement is the point, and it shows.** Chernobyl, one `vad`
grid: `"the moment the reactor exploded"` returns the chunk where the visual
pass describes "a severe explosion and fire in one section of the reactor
building" and the audio pass "At 1.23 a.m., reactor 4 exploded" — two
independent accounts of the same 16 seconds, neither pass having seen the
other's output.

### The two lexical halves were not equivalent, and this is why one is left

Measured on 41 identical units with identical OpenAI vectors. **Dense halves
identical on 6/6 queries**; both silent on a query sharing no content word with
the corpus. The lexical halves differed, and `supabase` was the better one --
which is what settled the removal when the sink work made it a choice:

- **No stemming in `indexes.tokenize`.** Postgres stems via
  `to_tsvector('english')`, so "lived" matches "live". On one query Postgres
  matched 21 of 41 units and Qdrant 13.
- **No length normalisation or TF saturation in the sparse vector.**
  `Modifier.IDF` supplies the IDF half of BM25 and expects the client to send
  weighted values; the client sends raw counts. `ts_rank_cd` normalises by
  cover density, so the shortest unit in the corpus tops Qdrant's half where
  Postgres ranks it third.

Neither was fixed and neither needs to be now: both are faults of the
client-side sparse vector, and Postgres computes its half in the database.
What went with Qdrant is the only configuration that searched without a
server -- `--embedder local --index qdrant` needed no key and no Postgres --
and nothing replaces that.

**`websearch_to_tsquery` ANDs its terms**, so one word absent from the corpus
silences the whole lexical half — `"reactor exploded"` ranked 2 rows and
`"the moment the reactor exploded"` ranked **0**, because "moment" appears
nowhere. The RPC replaces `&` with `|` in the rendered tsquery, keeping the
parser's stemming and stopword removal and only loosening the conjunction;
`ts_rank_cd` then does the work AND was doing badly. After the fix those
queries rank 7 rows each, and a query sharing no content word still ranks 0.

**ANY-term over-corrects, so it has a floor.** It fires on a single stem
collision: measured, a query sharing no content word with the corpus still
matched, promoted an unrelated chunk to second and pushed the right answer to
third, where Qdrant's half stayed silent and ranked better for it. With two or
more query lexemes a row must share at least two; a one-word query needs one.
"the moment the reactor exploded" shares `reactor` and `explod`, so it still
ranks.

**RRF ties are broken by the vector rank.** Ties are exact and common -- dense
1 / text 2 and dense 2 / text 1 are both 1/61 + 1/62 -- and `order by score`
alone left the winner to the planner. The dense half has an opinion on every
query where the lexical one may not, and the two backends now agree rather
than flipping a coin opposite ways.

**Qdrant did sparse and hybrid, and the note that said otherwise was
wrong.** Sparse vectors since 1.7, native `Fusion.RRF` since 1.10; calling it
dense-only was a claim about the implementation dressed as one about the
database. The measurement that outlives it is about **depth**: prefetching
only `limit` truncates each ranking before fusion, and the lexical half fired
on 12/20 rows where Postgres, ranking `greatest(limit*4, 40)`, fired on 20/20.
`candidates` is that knob on the one backend left.

---

## Audio

**Audio is scanned whole; it cannot be chunked first.** Whisper carries context
across an utterance and detects language from the opening seconds. Diarization
is worse: speaker labels come from clustering embeddings over the *entire*
recording, so `SPEAKER_00` in one window bears no relation to `SPEAKER_00` in
the next — chunk first and the speakers are not misaligned, they are
unnameable.

**A setting no chosen backend takes is refused, not dropped.** `stub` has no
`language` and `none` has no `exclusive`, and the two halves are configured
independently -- both call their checkpoint `model`, which is why the component
spells them `model` and `diarizer_model`. `models.settings()` reads each
backend's constructor, `audio.run` routes each setting to the half that takes
it (`device` to both, being a fact about the machine), and names any that
landed nowhere. Quietly ignoring one would mean a run reporting success having
transcribed under settings nobody asked for -- the same silence the named
defaults in `audio/driver.py` exist to prevent. The client never builds such a
request anyway: `conditions()` hides a setting whose backend is not selected.

**Transcription and diarization stay separate passes, joined by `align`.**
Whisper does not know who spoke and pyannote does not know what was said. A
word is attributed by its **midpoint**, because the two models estimate edges
independently and word spans routinely straddle a turn boundary; a segment
takes the speaker who spoke most of it by duration. With no diarizer, every
segment keeps `speaker=None` — truthful, where labelling everything
`SPEAKER_00` is not.

**Silence is answered without a model.** CCTV with a live but empty microphone
sits at RMS 0.000221, peak 0.0291, against 0.1796 for narration — three orders
of magnitude, so the 1e-3 threshold sits in a wide gap rather than on a cliff.

**Chunks with no speech are kept, with empty text.** The grid is shared, so
`chunk_id` must mean the same thing in the manifest and the transcript;
dropping the quiet ones renumbers everything after them.

**Only `speakers` goes into `structured`, never `turns`.** `turns[].text` *is*
the transcript, so rendering it appended the whole chunk a second time
interleaved with timestamps read as numbers — 337 characters where 151 was
right.

**Measured, RTX 4060:** decode 205 s of AAC in 0.30 s (~700x realtime), Whisper
`small` float16 37.6x, pyannote 3.1 30.1x. Chernobyl: 34 segments, 428 word
timestamps, 1 speaker over 18 turns covering 89% of duration.

---

## The grid

**One chunk grid per run, and either modality may decide it.** `chunk_id` is
simultaneously the unit a describer summarises, a transcript is cut into, and
retrieval returns — so if the two halves disagreed about a boundary, one query
would return two different clips with no honest way to say which is the answer.

**Content-derived boundaries need both guards or they are unusable.** Voice
activity cuts on every pause — every second or two on conversational audio,
which would shred the video into chunks too short to describe. A monologue
gives the opposite failure: zero cuts, one chunk covering the file. Measured on
Chernobyl, `speaker` on single-narrator audio found no speaker changes and
`enforce` divided the file into 7 even chunks of 29.3 s, which is the honest
outcome rather than an invented one.

**A floor above the ceiling is refused, not resolved.** `enforce` merges up to
`min_s` and *then* splits at `max_s`, so the split runs last and wins — and
`--max-chunk` defaults to `--chunk-duration`, which defaults to 20. So
`--min-chunk 30` on its own asked for "at least 30, at most 20" and produced a
grid whose shortest span was **18.07 s**, reported as success, because a scene
grid of 18-second chunks is an entirely ordinary thing to see. There is no
reading of that request to honour. With a coherent pair the same video gives 4
chunks of 39.8-55.6 s.

**`max_s` splits evenly, not into fixed bites.** Taking 30 s bites off a 62.5 s
span leaves a 2.5 s remainder, so the guard against short chunks would create
one. It becomes three of 20.8 s.

**A short final chunk is merged into the one before it, under every policy.** A
grid divides the media wherever it happens to end, so the tail is uniformly
distributed over the chunk length: 97.99 s at 20 s leaves a usable 17.99 s, but
100.4 s leaves 0.40 s. That stub costs a describer call *per sampler*, keeps a
frame because every chunk keeps one, and is a moment retrieval can return that
nobody can play. `MIN_TAIL_FRACTION = 0.25` is a fraction rather than a number
of seconds so it holds at any chunk length.

**The grid spans the longer stream.** A file is as long as its longest one, and
the streams differ — 205.264 s of audio against 205.280 s of video on Chernobyl
— so a grid built from the audio alone leaves the last video frames outside
every chunk. `Timeline.nearest` clamps anything past the end to the last chunk
rather than dropping the frame.

**`Timeline.fingerprint` is recorded by everything cut on it.** If two
documents disagree, they were cut on different grids and `chunk_id` means two
different things — a comparison a reader can make, rather than drift nothing
reports.

**`media_ts` is the only clock a decision may use.** Never wall time, never
frame counts. Decimation buckets on media time, never "every Nth frame":
identical on a clean file, self-correcting on a lossy one.

**Samplers reset at every chunk boundary** and every chunk keeps at least one
frame. Rate limits are enforced in the base class *before* the strategy runs,
so a rate-limited frame costs no inference.

**Pixels are borrowed.** `frame.release()` runs every iteration; anything that
outlives the loop must copy.

---

## Aggregates

**Aggregates answer what retrieval cannot.** Embeddings cannot count, so "the
busiest moment", "who dominated", "how much of this is speech" are exact
questions similarity answers approximately. Measured on Chernobyl: 89.4% speech
ratio, 125.1 words per minute, `monologue: true` with 0 handovers, and chapters
that tile the whole video with the explosion at 94.4 s.

**One folder per aggregator, and a tier is not a folder.** They were grouped by
tier -- `statistics/`, `model/`, `llm/` -- which put a *cost* in the directory
tree and left `linking.py`, 230 lines read by one runner, three levels away from
it at the package root. A tier is already a class attribute, and the ladder is
`TIERS` in `base`, so the directory said nothing the code did not. `model/` was
also the worst name available: everywhere else here a model is a provider, and
that folder held NER and sentiment.

So each aggregator is a folder with its own `driver.py`, as a video_rag
component is. Most hold one file today, which is the point -- `entities/` needed
two the day it was written, and the next aggregator that needs three has
somewhere to put them. What every aggregator shares stays at the root, the way
video_rag's components share `falconvar/shared/`: `base` (the protocols,
`Context` and `DefinitionRunner`), `inputs` and `rendering`.

`DefinitionRunner` moved out of `llm/__init__.py` for the reason `Sampler` is in
`samplers/base.py`: a base class inside a package's `__init__` is reached by
importing the package, so every kind that subclasses it drags in its siblings.
Measured after the move -- importing `falconvar.aggregates` loads the three free
aggregators and nothing else; `ner`, `sentiment` and all four runners stay
unimported until asked for by name.

Nothing stored was invalidated. A worktree at the previous commit, run against
the aggregates the new layout had just written, read **4 of 4 as current** --
same fingerprints, same `version` hashes, same model keys -- and every id, tier,
kind, default input and definition entry compares byte-identical.

**Every aggregator is a component with one input, and there is a pipeline
over them** -- video_rag's shape. What the input is depends only on what the
aggregator does:

    stats · coverage · speakers     a record       they count whole documents
    ner · sentiment · prompt        an Excerpt     the rows a selection took
    entities                        Sightings      the entries a profile links

A record is a video's folder (or a combination's, or a mapping of document
paths). `select` is the component that turns a record and one selection into
an `Excerpt` or `Sightings` file -- choosing what to read is a step of its own,
which is what lets every aggregator after it take one file. Each input embeds
its grid, so an aggregator never needs a second file: descriptions and
transcripts carry chunk ids but no times, and "one file plus one field" per
aggregator would have broken on exactly that, on the cross-modal default
`transcript+*`, and on selections like `clip:hazards[severity,hazards]` that no
single field names. One excerpt feeds any text aggregator: selected for
`sentiment`, it was read by `ner` too.

`answer(name, data)` is the one verb: a `Context`, an `Excerpt` or `Sightings`
in, an `Aggregate` out, anything else refused by name (`ner reads an excerpt,
not a record: select one from the record first`). Each component --
`stats.stats`, `ner.ner`, `prompt.prompt(name, ...)`, `entities.entities(profile,
...)` -- is that verb with a read at each end, and its own CLI. An
aggregator's settings are its constructor's parameters, read off the signature
(`settings_of`): `ner`'s `labels`, `threshold` and `model` were unreachable
while `build()` always constructed defaults, and are flags now.

**The pipeline runs what it is handed data for, and nothing else.**
`aggregate(record, out, ner="transcript", sentiment=True)` runs those two:
`True` is the default selection, a string a selection (`a,b` makes two
answers), a `.json` path an input file written earlier, a dict
`{"data": ..., **settings}`. Keywords spell `entities:people` as
`entities_people`. There is no tier to reach any more -- `--only summary`
with the default `--tier free` used to run nothing and say nothing, and naming
is now the whole decision; `up_to(tier)` builds the everything-up-to-a-cost
mapping `workflow` still offers. Handing nothing is refused, as is a selection
for an aggregator that counts the record. Selected inputs are kept in
`<out>/inputs/`, so what an answer read can be looked at.

**`previous=` is the resume, and it is an argument** -- a folder of earlier
answers for the pipeline, one answer's file for a component. It used to be the
output folder itself, read whenever it existed, with `force` to override. A
reused answer is still written: recompute and write are different questions.

**Nothing below the driver changed.** `answer` rebuilds the `Read` or
`Mentions` an aggregator always took from the file, so the fingerprint an
answer is stored under is the one it always was. Verified: pointed at answers
the old id-addressed driver wrote on three folders -- free tier, and
`entities:people` at `--tier llm`, round-tripped through a sightings file --
the pipeline **reused 7 of 7 and every payload compared identical**. `ner` and
`sentiment` gave identical answers as components and through the pipeline;
`previous=` reused a component's answer; `prompt` answered a real summary
(258 words); two sources combined to an 18-chunk record first; a workflow
re-run reused 2 of 2; `11/11 schemas current`, `TYPES.txt current`.

**Several videos are one record first, and no aggregator knows it.**
`combination.combine([folder, ...], out)` lays the videos' timelines,
descriptions, transcripts and manifests end to end in the same dataclasses and
filenames -- chunks renumbered after the previous video's, times shifted onto
one clock -- and the aggregates run over that folder unchanged. Provenance is
`Timeline.params["combined"]`, once, and part of the grid's fingerprint;
`combination.origin(timeline, chunk_id)` turns a combined chunk back into a
video, a chunk and a local time. Speakers are prefixed with their video,
because `SPEAKER_00` in two recordings is two people. One video combined
alone answers `coverage` identically and `stats` differing only in
`derived_from: combined`.

Linking over a combination pools every mention, and it links lookalikes across
videos as readily as across chunks: test + test2 share nobody, and
`entities:people` still made 4 entities spanning both -- the red-cap man
merged with test2's man in a black cap and black shorts. Measured before this
existed, a two-stage link (each video alone, then entities matched one-to-one
across videos) made 0-2 wrong links on those pairs.

The module is `combination` and not `combine` for `boundaries.grid`'s reason:
a package attribute and a submodule of one name are one slot, and the first
version recursed forever in `__getattr__`.

**A tier is a cost, and the pipeline runs cheapest first.** A tier no longer
selects anything -- naming an aggregator does -- but whatever was named runs
free, then local, then llm, so a run that dies partway has produced the free
results rather than none. `up_to(tier)` is where a tier is still a ceiling.

**`depends_on` drops rather than fails.** `speakers` on silent CCTV is not an
error, it is a question that does not apply, and it is reported as skipped
*with the reason* — because "speakers did not run" is only useful beside why.
`chapters` used to depend on `summary` and never read it; no aggregator depends
on another now.

**What an aggregate reads is an input, never a guess from sampler names.**
`rendering.pick_sources` listed `transcript`, `clip`, `uniform` by preference and
fell back to everything, so what a summary read depended on what the samplers
happened to be called -- test1's `clip:topic` and `clip:activity` matched none
of them. `aggregates/inputs.py` is a grammar with three operators:

    ,        separate inputs -- one answer each
    +        sources joined into one input
    [a,b]    fields of one answer joined into one input

`clip:hazards[severity,hazards]` is one summary over both fields;
`clip:hazards[severity],clip:hazards[hazards]` is two, stored as
`summary~severity` and `summary~hazards`. **A bare name is a question, never a
sampler** -- `text` is both, and "the text on screen" is what a person means --
so a sampler's whole output is `text:*`. Fields render through the index's own
`units.render`, so a field reads to an aggregate as it reads to a search.

**Every account of a chunk is read by default, not the first one found.** The
default input is `transcript+*`. Taking only the best-ranked source throws the
other modality away. Measured the hard way: a run picked stub `clip` text over
428 words of real narration, and the model correctly reported that it had been
given nothing to summarise.

**A typo in an input is a 422; an input with nothing to read is a skip.**
`aggregates.validate` parses every selection and checks each question, field and
entry key against the vocabulary before a job is queued. Whether *this* video
was asked a question is not in the request, so an input that finds nothing is
skipped at run time with the reason.

**Prompts are data; the kinds are the only code.** `summary`, `chapters` and
`events` are one entry each of kind `fold`, `spans` and `items` in
`aggregates/definitions/definitions.json`; a custom prompt is another entry in
`data/aggregates.json`, not a class. The answer's fields use describe's builder
and the schema is generated from it, for the reason a custom question's is.
Every word a model is told -- the kinds' own citation and check lines included
-- is in the JSON, so `version_of` covers it and editing any of it rebuilds what
it wrote. No free-form kind: an answer without citations is a fold. Verified: a
custom `incident_report` fold over `clip:hazards[severity,hazards]` on Chernobyl,
reused on the second run, deleted with its answer left in place.

**A colon cannot be in a Windows filename.** `entities:people` is stored as
`entities.people.json`, and the API's aggregate URLs use the stem. No
definition name or label contains a `.`, so the mapping reverses.

**Nothing is cut to fit.** `chunk_rows` stopped at 400 rows and `sentiment`
scored a chunk's first 480 characters as its tone. `items` now asks per
100-line window, concurrently. `spans` folds chunks into parts until they fit
one call and cites part ids, because dividing windows separately would force a
chapter break at every window edge -- exercised on Chernobyl with the window
shrunk to 4: 3 chapters over folded parts, every chunk covered. The local models
cut text at sentence ends into pieces each reads whole, and sentiment weights
them by length. On Chernobyl NER went from 27 entities to 55 and mean sentiment
from -0.37 to -0.67 -- but the default input widened at the same time, so
neither is a like-for-like measurement of the pieces.

**Ask for a word count, not "several sentences".** Once structured fields
arrived the model sized the summary as one field among many: 105 median words
against 363 in the prose-only era. Saying "at least 150 words" took it back to
246 median. The summary is the only text that gets embedded, so its length is a
retrieval parameter.

**Every summary layer is recorded; none of them is embedded.** A leaf summary
covers a real span and is the only description at that granularity, between one
chunk and the whole file, so it is kept. Indexing them would return the same
moment two or three times over under different wordings — the count-bias
failure the moment aggregation guards against, one level up.

The final summary goes somewhere else: `workflow` asks
`aggregates.index_summary` for it when a run names a `database`, and one unit
per video lands in `video_embeddings` rather than `embeddings`. `embeddings`
answers *which twenty seconds*, a summary answers *which video*, and a video
is not a moment you can play -- so the two never share a ranking, and
`/search` reaches the second only as `level=video`. Postgres only, and
best-effort: a vector that fails to write does not fail the aggregates.
Verified after the tier split: Chernobyl, `--tier llm`, `video_units: 1`.

It used to be gated on an `index` parameter of `aggregates.run`, which meant
the tier could not be run at all without naming a destination -- the same
component-owns-the-database shape the sink removal undid everywhere else.
Making the *vector* still the tier's work and the *write* the pipeline's is
the split: `index_summary` embeds and upserts, and only `workflow` calls it.

**Spans are resolved through the timeline, never trusted from the model.** It
is asked for chunk ids, which it can copy; times it would invent.

**`inputs_fingerprint` is a hash of the chunk text actually read.** A summary of
descriptions since rewritten reads perfectly, which is precisely why staleness
cannot be left to a reader to notice.

**Entities: who is who is decided by rules; the model only writes the
account.** `aggregates/entities/linking.py` embeds each mention's identity keys and
merges under constraints; the `link` runner then asks for one account per
linked entity (concurrently, capped at 12). v0's flow, cluster then narrate.
Tried the other way first: gpt-5.4-mini, given test1's 39 actor entries, linked
31, broke the same-answer rule twice after being told it, and merged an older
woman with an older man.

**Identity belongs to a link profile, not to a shape.** A profile names a list
field, the entry keys that identify one (`people`: `appearance`, `clothing`),
the keys its account reads besides, and the account's instruction and fields.
It used to be declared beside the describe shapes, which allowed one way to
link a given answer; a profile can link the same `people` answers by clothing
alone, and a run can narrow a profile with `yolo[people.clothing]` -- narrow
only, since a key the profile does not name is a different profile. Moving it
re-described nothing, because identity was never in a prompt hash. test1's
labels are on the custom `activity` question, so `data/aggregates.json` carries
an `actors` profile: after the move test1 linked into **the same groups**, and
`eval.entities` printed the same table as below.

**Whole values need a threshold.** A profile over a text field, the prose or the
transcript yields one mention per answer, so nothing in the video is provably
different and no threshold can be read off it. Such a profile must set one; it
is refused at write time otherwise.

**Cannot-link is per answer, and the threshold is read off it.** Two entries in
one answer are different by the question's own wording ("one entry per
distinct person"), so their similarities are a calibration set for this video
and this embedder: link only above the most similar provably-different pair,
and only mutual best matches. Per answer rather than v0's per chunk, because
two questions about one chunk may describe the same person. v0's fixed 0.88 was
a fact about one embedder:

| F1, test1 / test2 | text-embedding-3-small | bge-small |
|---|---|---|
| v0: 0.88 fixed, cannot-link per chunk | 0.64 / 0.29 | 0.77 / 0.80 |
| **`max` + mutual (default)** | **0.94 / 0.91** | 0.88 / 0.91 |
| `q95` + mutual | 0.90 / 0.91 | 0.97 / 0.91 |

The default holds precision 1.00 on both videos under both embedders. `q95` is
better on bge and makes wrong merges on OpenAI, and the default has to hold
under whatever embedder a deployment runs. v0's genericness filter, tried as an
outlier test on mean similarity, changed no result anywhere and was dropped.

**That table is `actors`, and it does not transfer to `people`.** On the
`people` shape -- test re-described as `yolo,clip:yolo` under `data/eval/people`,
12 people and 125 of 128 mentions labelled by *watching the video*
(`eval/people_labels_test.json`) -- one cosine over `appearance; clothing` under
`max` found **71 groups for 12 people**: B-cubed 0.35 on OpenAI, 0.19 on bge.
The two cashiers both read "grey top, hair in a bun, dark pants", so the most
alike provably-different pair outscored most same-person pairs and nothing
cleared the bar. On test1 the `max` is set instead by an answer listing one
person twice. Either way one pair decides it.

So `people` measures more than one embedding, as profile data -- `weights`
(each identity key embedded apart), `attributes` + `near` (agreement over the
shape's closed vocabularies), `shared` (a shared accessory or shared
distinguishing words) -- each a z-score against the provably different pairs,
averaged, under `q95`. `link()`'s rules are unchanged. Worst case over test and
test1 under both embedders, `python -m eval.linkers`:

| B-cubed / precision, worst of 4 | |
|---|---|
| one cosine, `max` (before) | 0.19 / 0.84 |
| + `q95` | 0.52 / 0.76 |
| + identity keys embedded apart | 0.82 / 0.84 |
| + attributes, untuned | 0.82 / 0.86 |
| **+ attributes as tuned on test1 (shipped)** | **0.87 / 0.91** |

The attribute weights and near pairs are `eval/attributes.py`'s, set on test1
before test was labelled. Only `people` sets the new keys, and they are absent
rather than defaulted, so `objects`, `text` and `actors` hash exactly as before
-- verified against HEAD -- and nothing they stored went stale.

Tried and not shipped, all in the bench: average and complete link; Hungarian
tracking + merge (worst B3 0.88 but precision 0.90); **pairing a chunk's two
answers first** (precision 1.00 on both videos, but only with place words read
from `action`, which helped OpenAI and hurt bge); a lower second bar for
stragglers; and **a model merging whole rule-built groups**
(`eval/llm_merge.py`), which moved B3 -0.02 to +0.02 run to run and proposed
4-9 merges per run between people it had been told were on screen together.

**test2 was not part of the choice, and it is the weakest of the three.** 60 s,
7 people, 31 of 32 mentions labelled (`eval/people_labels_test2.json`). The
shipped profile scores B3 0.76 / 0.69 (OpenAI / bge) against 0.41 / 0.40
before, with one wrong pair -- but recall is 0.56 / 0.39, because most of what
is missed is a partial view at the frame edge ("only a sliver of the body is
visible"), and two people split into four singletons each. Over all three
videos the same similarity under Hungarian tracking + merge + straggler attach
holds worst B3 0.75 at precision 0.90 against the shipped 0.69 at 0.91: the
straggler pass is what those edge views need. Not switched: that grouping is
new code in `linking`, and 31 mentions is a direction, not a result.

**The describer moves the score as much as the linker does.** `test3.mp4` is a
byte-identical copy of test2, so describing it again is a second draw over the
same frames: shipped B3 went 0.76 -> **0.60** on OpenAI (precision 0.97 ->
0.65) and 0.69 -> 0.74 on bge. The bar held (z 1.05 vs 1.03); what changed is
that 7 cross-answer pairs of different people cleared it instead of 4 --
three of the seven people wear all black, and this draw said so more vaguely
-- and greedy single link chained them into one group of four people. No
config tried rescues that draw (best worst case over four videos: 0.66). On a
video this small, ~0.15 of B3 is describer noise, which is the argument for
more readings per chunk or for identity from pixels rather than a better rule.

What is left is not a similarity problem. The right cashier is "grey T-shirt,
printed back" from behind and "white shirt, dark apron" from the front, and a
customer beside her reads like the first: end to end, 9 of 10 narrated
entities are one person, and the tenth is those two. The check found that
split -- it flagged all 8 customer mentions -- and also flagged 10 of 26
mentions of the *correctly* linked left cashier, on `role` alone, because the
describer called her a customer in those chunks.

**The check flags; it never drops.** A model check inside the account call --
"which of these observations is not the same subject" -- was measured as a
filter that removed what it named:

| | test1 F1 | test2 F1 |
|---|---|---|
| rules only | **0.94** | **0.91** |
| check drops, run 1 | 0.91 | 0.80 |
| check drops, run 2 | 0.81 | 0.80 |

It caught the known bad merge, a cream coat inside the dark-puffy-coat woman,
and it rejected true matches -- three of four the same way: a detail present in
one observation and absent from another read as a contradiction, despite the
instruction saying it is not one. The same prompt rejected three on one run and
four on the next. Verifying candidate pairs instead of clusters made the same
mistake. So `check: flag` keeps every member, marks the disputed ones (`doubt`
on the mention, `doubts` on the entity) and writes the account from the rest;
an entity whose every observation is disputed gets no account. On test1 after
the change: 3 doubts, the cream coat among them, 0 members lost.

**A score that skips doubtful labels hides exactly the wrong merges.** The
first real run scored precision 1.00 while merging a dark puffy coat and a
cream coat into the woman in the gray top -- every one of those mentions was
labelled unsure, so no pair of them was scored. `eval/entities.py` now counts
links touching unsure mentions as `unchecked`, and `different` labels rule a
mention out of a group. Still linked, unscored: the dark-coat and cream-coat
women (now flagged by the check), and two women linked on "entering" (not
flagged). `activity`'s `actor` field carries position and behaviour; the
`people` profile's `clothing` would not.

**The labels are tiny and were written by the builder**: 20 scored mentions on
test1 and 9 on test2, read from the descriptions after seeing one run. test2 is
the check test1 was not tuned on. A direction, not a result.

**Attribute tracking is a prototype in `eval/`, not the linker.** test.md's
proposal -- the `people` shape answering in fixed vocabularies, a weighted
distance over them, Hungarian assignment over answers in time order -- lives in
`eval/attributes.py` and `eval/tracking.py`, reading test1 re-described under
`data/eval/people` so the real test1's `activity` labels were untouched.
Attributes separate same from different people at AUC 0.970 against 0.862 for
word overlap. The one fix that helped is a **merge pass** joining tracks that
never share an answer: F1 0.94 / 4 wrong against 0.93 / 7. Vetoes hurt -- age
0.93 -> 0.81, gender 0.94 -> 0.92 -- because a refused mention lands in another
person's track; matching against the whole track did not beat the last two
appearances. 62 labelled mentions on one video, not yet checked on test2.

---

## Models and providers

**A stage names a provider and a model; `shared/models/providers.py` knows the rest.**
Three roles -- `describe`, `llm`, `embed` -- and four protocols, because the
wire format is the only thing that really differs between vendors:

    openai     Responses + /embeddings
    chat       Chat Completions + /embeddings: Ollama, LM Studio, llama.cpp,
               vLLM, Gemini, Mistral, Groq, OpenRouter, Together, DeepSeek,
               xAI, Voyage
    anthropic  Messages, with the schema as a forced tool's input
    local      a Hugging Face model in this process. Vectors only

One `ModelDescriber` and one `llm.Model` serve every provider, so adding one is
a row in `_BUILTIN` or an entry in `data/providers.json`, never a class. A
describer and an embedder per vendor would be the per-stage copies of
connect-and-complain that `db.py` and `llm.py` each exist to prevent.

**OpenAI stayed on Responses rather than joining Chat Completions.** A
describer's `config()` is half of describe's resume key, and every description
in the corpus was paid for on the Responses path; folding OpenAI into the
generic path would have been one branch fewer and a reason to re-describe
everything. Verified: `ModelDescriber().config()` is dict-equal to the old
describer's, and **40/40** stored pairs across four videos still resolve as
current. The generic path was then run against OpenAI itself, as a custom
`chat` provider: the same pair in the same shape, and embeddings identical to
the native path's (cosine 1.000000).

**A default is resolved when a call is made, never captured.** `Options`'
model fields are `None`; `providers.choose` reads the call, then
`FALCONVAR_DESCRIBER` / `_LLM` / `_EMBEDDER`, then `openai`. A constant would be
read at import, before `.env` -- the trap that keeps `FALCONVAR_DATA` a process
variable. `/capabilities.defaults` calls `providers.defaults()` for the same
reason: a form defaulting to the dataclass's `None` shows nothing where the
answer is `openai`.

**A model choice is one string, and only the stages that call a model take
it.** `describe`, `embed`/`retrieve` and `aggregates` each take one
`provider/model` argument; media, audio, boundaries, video and cut never see
one. There was a separate `model` field beside every provider field -- six on
`Options` and the upload form, a `--model` on four CLIs, two env variables per
role -- and it bought nothing `ollama/gemma3:4b` does not already say, while
needing a rule for which of the pair wins (an env model applying only to the
env's provider). Collapsed: three fields, three variables, no precedence rule.

**`provider/model` splits on the first slash, and only after a known
provider.** Model ids carry slashes (`BAAI/bge-small-en-v1.5`), so a provider
name may not. An unknown head leaves the spec whole, so the error names what
was typed rather than half of it.

**A missing key is a 422, not a failed job.** `workflow.validate` checks every
role the run will use -- describe only when reading the picture, llm only at
`--tier llm` -- because `describe` finding no `ANTHROPIC_API_KEY` happens after
the whole video has been decoded. It does not ping local servers: validation
stays synchronous and offline.

**The same check is on the components, for the reason the question check is.**
`validate` returns a list, because a request is checked all at once and a
caller wants every problem; `providers.require` raises the first, because a
component has one role to check and nowhere to put a list. Without it
`describe.run(describer="anthropic")` read every frame before discovering there
was no key -- the exact late failure, reachable from the library's other public
level and from the per-component route. It runs before the manifest, the grid
or a single frame: measured at 2 ms against a whole decode.

**Keys never enter `providers.json`.** A field with `key` in its name is
refused; the file names variables in `key_vars`. A bad entry is dropped and
listed under `/capabilities.models.problems` rather than raised, because one
typo in a hand-edited file taking down `/capabilities` takes every generated
form with it.

**Switching `--llm` rebuilds the llm aggregates.** `inputs_fingerprint` said
nothing about who wrote an answer, so a switch reused the stored summary and
reported success -- describe's silent no-op, one stage later. Aggregates now
record `stats.model` and reuse needs it to match. A stored llm aggregate with
no `model` reads as `openai:gpt-5.4-mini`, which is a fact rather than a guess:
before this, `aggregate.run` had no way to name another. That reading is also
what stopped the change rebuilding every summary -- verified, `--tier llm` on
defaults computed **0**.

**An embedder key carries its width, known before the index opens.** OpenAI's
are tabled; any other remote model is probed with one short string, once per
process, and every later batch is checked against it -- a server swapping the
model behind a name would otherwise write a new width into the old space.

**Query and document are embedded differently where the model says so.** e5,
nomic, bge and mxbai were trained with prefixes, and a search embedded as a
passage loses recall with no error anywhere. `embedders.PREFIXES` looks them up
by model id and `retrieve` goes through `query_vector`. A hand-set prefix adds
`:p<hash>` to the key, since its vectors are not comparable to the defaults';
nothing is added otherwise, so every key already written is unchanged.

**`local` is `sentence-transformers`, and nothing else.** It applies the
model's whole module list -- the pooling it declares (bge is CLS, not mean) and
any Dense layer after it. A hand-rolled `transformers` path for when it was not
installed was removed: it was a second function that had to agree with the
first under one key, and a model with a Dense layer could not agree at all.
Verified before removing it: on bge-small the two gave **identical** vectors
(max abs diff 0.0), so the index it built stands. Weights land in
`weights/embedders/`; a model is loaded once per process, because `/search`
runs in the server and a reload is seconds.

**Measured: a 33M-parameter local embedder is level with OpenAI here.**
`BAAI/bge-small-en-v1.5` (384-d, CUDA) against `text-embedding-3-small`, the
same hybrid, `eval/harness.py`:

| embedder | MRR | top-1 | recall@5 | median query |
|---|---|---|---|---|
| openai | 0.7315 | 0.611 | 0.833 | 0.619 s |
| local bge-small | 0.7241 | 0.611 | **0.917** | **0.051 s** |

By band: low overlap 0.222 vs 0.333 (n=3), mixed 0.694 vs 0.806 (n=6), high
overlap 0.911 vs 0.815 (n=9). Eighteen cases is a direction, and the direction
is "not obviously worse, and free". The 12x latency is the network round trip.
Indexing all 49 units took 2.0 s after a 34.5 s first load, download included.

**Supabase's vector columns were `vector(1536)`** -- OpenAI's width written into
the schema, refusing any other embedder at the first upsert. `install.sql`
alters both to unconstrained `vector` and drops the HNSW index, which needs a
fixed width. Every query filters on `embedder`, whose key carries the width,
before a distance is taken, so two widths never meet. On a corpus this size the
exact scan is milliseconds; one space with millions of rows would want a
partial expression index back. `SupabaseIndex` turns "expected 1536
dimensions" into "re-run install.sql".

**Verified as far as this machine reaches.** Real: OpenAI on both protocols
(including gpt-5.4-mini refusing `max_tokens` on Chat Completions and the retry
answering), and the local embedder end to end. A mock server that records
requests covered every other shape: images as data URIs, json_schema /
json_object / prompt modes, fenced JSON, truncation, an unreachable local
server, and Anthropic's base64 image blocks, forced tool, 401 and `max_tokens`
stop. **Not run against** the real Anthropic, Gemini, Mistral, Groq,
OpenRouter, Together, DeepSeek, xAI or Voyage APIs, nor a real Ollama or LM
Studio. Their default model ids are reasonable picks, not measured ones, and a
server's schema support is exactly what `structured` exists to downgrade.

**Model calls are async, and the provider says how many at once.**
`llm.Model.generate` is a coroutine behind every provider, so concurrency was
added once. `describe` plans every call in manifest order and reserves its slot
in the document, then gathers one task per (chunk, sampler run) under
`describer.concurrency`; a run's frames are read inside that gate, so memory is
bounded by the cap rather than the video, and still read once per run. The
summary's folds within a layer are gathered too. `Provider.concurrency` is 8
for cloud APIs and 1 for Ollama, LM Studio and llama.cpp, which answer one at a
time; `providers.json` can set it. Each stage runs its own `asyncio.run`, so a
client is opened per call -- an async client belongs to the loop that opened it
and fails when a second loop reuses it. The Anthropic path retries 429 and 5xx
twice, as the OpenAI SDK already does inside a call. Embeddings stay
synchronous: one request carries 64 texts, which is a whole test video.

Measured on Chernobyl with `gpt-5.4-mini`, nothing written:

| | concurrency 1 | concurrency 8 | |
|---|---|---|---|
| describe, 18 calls / 108 images | 76.0 s | **12.8 s** | 5.9x |
| summary, `batch=2`: 10 folds over 3 levels + final | 28.9 s | **16.7 s** | 1.7x |

Describe's wall is now its slowest single call (12.7 s). The summary gains
less because its layers, and the final call, are sequential by construction.
**A failed call still discards the run's answers** -- as it did sequentially,
but now with more paid calls already in flight when it happens.

---

## As a library

The product is the library; `api/` and `web/` are how it gets exercised. Two
levels are public, and a third exists because the first two need it.

```python
from falconvar.video_rag import boundaries, describe, embed   # one component
embed.run(video_id, embedder="local")               # writes embedded.json
boundaries.evidence(video_id, "scene", stride=5)     # not everything is `run`
descriptions = describe.load(video_id)               # read a result back

from falconvar.video_rag import video_rag            # the whole pipeline
video_rag("x.mp4", policy="scene", sampler="clip:[text,scene]")

import falconvar
falconvar.configure(data_root="/var/lib/falconvar")  # before anything runs
```

**A component publishes its entry points, its errors and its return types, and
nothing else.** It published 72 names across eight components, where
`run`/`load` plus the four genuine extra ways in account for 22. The rest was
machinery -- `enforce`, `merge_tail`, `from_cuts`, `ingest`, `build_samplers`,
`collect`, `from_descriptions`, `query_vector`, `to_chunks`, `to_moments`,
`Frame`, `FrameStore`, `Describer`, `Description`, `Unit` -- reached by nothing
outside its own package, and every one of them a line in a user's
autocomplete asking to be understood. 72 became 29, and then 41 when each
component published the verb that does its work on objects:

    media       run load split       UnusableMedia VideoIdTaken
    audio       run load listen
    boundaries  run load detect timeline evidence retune
                                     POLICIES EVIDENCE_SETTINGS
    video       run load ingest      SAMPLER_SETTINGS UnreadableSource
                                     Frames FrameStore MemoryFrames
    cut         run load apply
    describe    run load answer      available DescriberUnavailable
                                     StoreUnavailable
    embed       run      encode      available EmbedderUnavailable  Unit
    retrieve    search videos        Moment

Three tests for staying: a second way *in* that no naming collapses into `run`
(`evidence`, `retune`, `search`, and now each component's verb), an exception a
caller has to catch by name, or a type they would annotate (`Moment` is what
`search` returns, `Unit` what `encode` does, `Frames` what `ingest` takes). The
registries stay because `/capabilities` is built on them and "what policies
are there" is a fair question to ask the library. Nothing else does.

**`main` was in all eight and nothing ever imported it from the package** --
every `__main__.py` reaches it as `from .driver import main`. Eight exports
advertising an argparse entry point as a library function.

Machinery is not hidden, only unadvertised: it stays reachable through the
module it lives in, so `media.split` is `from ...media.split import split` and
`DEFAULT_INDEX` is `embed.driver.DEFAULT_INDEX`. `falconvar.workflow` had no
`__all__` at all and so published its own imports -- `Callable`, `Optional`,
`Path`, `dataclass`, `annotations` -- as though they were surface.

**A missing key raises `ProviderUnavailable`, not the role's own class.** The
check lives in `providers.require`, which knows the role and not the component,
so `except embed.EmbedderUnavailable` does not catch it. That is why
`Unavailable` is the branch to catch: every "not fixed by retrying" failure
across both tiers is one, verified across all four of the exported
`*Unavailable` classes plus `ProviderUnavailable`. The specific classes stay
for their own causes -- no package, no weights, no server.

**The module is the unit: `embed.run`, never a bare `embed`.** The function in
`embed/driver.py` *is* named `embed`, with `run = embed` beneath it, and that
is worth keeping -- eight components all raising from a frame called `run` put
zero information in the one line a user pastes into a bug report, and
`help(embed)` said `run(...)`. But only `run` is exported, because a bare-name
style does not survive contact with the rest of the surface:

    load        in 6 of 8 components
    build       in 3       boundaries · describe · embed
    available   in 2       describe · embed

So `from ...boundaries import load` and `from ...describe import load` collide
immediately, and a caller ends up writing `load as load_timeline` -- which is
exactly what the components already do to each other internally. And "more than
`run`" is the *common* case, not the rare one: reading results is half of what
a library is for. Measured on the example script written the other way, three
of seven components had to be imported twice, once as the function and once as
the module.

`boundaries` is the clearest case, with three ways in and only one of them
`run`: `evidence` decodes and scores (14.0 s on the test video), `run` turns
the cached cuts into a grid (0.03 s), and `retune` re-thresholds the stored
score series without decoding at all (0.004 s, **3,500x**). No naming collapses
that into one function.

Renaming the functions cost two collisions, both worth fixing -- `media/driver.py`
had a local `media`, and `describe/` had a *second* function called `describe`
in `reader.py`, renamed `answer`, which is what the rest of the tree calls one.
Left unfixed the second was infinite recursion.

**A module's public surface is `run` and `load`.** Two directions, not two
steps: `run` does the work and writes, `load` reads the result back, typed.
Nothing in a hand-written pipeline calls `load` to make the pipeline work --
each `run` resolves its own inputs through other components' `load`, which is
where the data flow went when it left the driver. `load` is public because it
is simultaneously a component's input mechanism and a user's way to read
results.

Two components honestly break the pair, and both are worth knowing. `embed`
has `run` and no `load`, which is now a smaller break than it was: it wrote
*vectors* into an index and `embedded.json` beside them was the readable half,
so there was no document to load. The index is gone and `embedded.json` is the
whole output, vectors included -- but the reader stayed `readable.load`, so the
name is the only thing still odd about it. `retrieve` has neither `run` nor
`load` -- it is `search(query, ...)`, because a query is not a video id and
what comes back is a ranking, not an artifact.

**Exports are lazy where eagerness would cost an import.**
`from falconvar.video_rag import video_rag` binds `process`, which imports
every component: 53 modules, 352 ms and `av`, for someone who wanted one of
them. PEP 562 `__getattr__` defers it, and `import falconvar.video_rag` stays
at 1.3 ms. The same trick holds `__version__`, because
`importlib.metadata.version` scans the environment's distributions -- measured
**~120 ms**, a tenth of a second on every `import falconvar` to compute a
string almost nobody reads.

**Every deliberate failure is a `FalconvarError`, and keeps its builtin
base.** Removing `sinks` proved that rule needs enforcing rather than
asserting: `UnknownBackend` had been the `FalconvarError, ValueError` covering
"a value outside a fixed vocabulary", and deleting it left `media.on_conflict`
and the pipeline's `database` raising a *bare* `ValueError` that "anything the
library refused" could not catch. `shared.errors.UnknownOption` is that hole
filled, and deliberately unexported -- nobody catches a bad literal by name,
and `except ValueError` already fires. Found by `example.py`, which prints the
base of every refusal it demonstrates and so showed a `(!)` where every other
line says `(FalconvarError)`.

Twenty classes across ten modules had no common ancestor, so an embedding
application could not say "anything the library refused" without listing them,
and a list drifts. `class UnknownBackend(FalconvarError, ValueError)` mixes the
base in rather than substituting it, so every `except ValueError` already
written still fires. `Unavailable` is the branch worth catching on its own --
no package, no weights, no key, no server, none of it fixed by retrying.
Two names collided: `ModelUnavailable` existed twice with *different* bases,
so `except ModelUnavailable` silently covered half of what it looked like it
covered (now one class in `shared/errors.py`), and `Protected` is now
`ProtectedDefinition` and `ProtectedPrompt`.

**The rule was asserted, not held, until `example.py` checked it.** Its first
run found seven of the refusals a caller meets first -- a setting no backend
takes, a floor above the ceiling, `limit=0`, `batch=0`, an unknown question --
raising a bare `ValueError`, and a sweep found 63 deliberate builtin raises
across the package (49 `ValueError`, 9 `KeyError`, 2 `TypeError`, 2
`RuntimeError`, 1 `FileNotFoundError`).
`shared.errors.Refused(FalconvarError, ValueError)` is a request refused as
given; `UnknownOption` now carries **both** builtin bases, `ValueError` and
`KeyError`, because the registries (policies, samplers, audio backends) raised
`KeyError` for the same mistake the rest raised `ValueError` for -- and it
renders as a sentence rather than a quoted key. Deployment faults are
`Unavailable` and keep `RuntimeError` (`paths.NotACheckout`,
`db.SchemaOutOfDate`). `documents.py` now imports `shared.errors`: a leaf that
imports nothing, so the no-edges reason for the rule holds. What stays a
builtin is protocol, not refusal: `KeyError` from a `__getitem__`,
`AttributeError` from a PEP 562 `__getattr__`, `OSError` from a failed disk
write.

**A `try` that drops a bad input silently hid a broken function for every
input.** `library.compile_shape` called `_compile_field`, which does not exist,
and `load()` compiles each custom shape inside `except Exception: continue` --
so all nine custom shapes in `data/prompts.json` were dropped at every load
and each of their questions answered in the fallback `scene` shape, reporting
nothing. Found by pyflakes (undefined name), not by any run. It calls
`compile_fields` now, a dropped shape logs a WARNING with the reason, and
`example.py` adds a custom question and checks its schema. Found the same
way: `fields.check_fields` still checked an `identity` key nothing defines,
a leftover from when identity lived on shapes -- a `NameError` for any custom
field with `of`.

**A component has two ways in, and one of them needs no filesystem.** Every
component was addressable only by video id, so "use one component" and "adopt
this pipeline's directory layout" were the same decision -- and the second is
not something a library gets to require. Each now also publishes the verb that
does its work on objects handed in:

    media       split(path)                              -> Media
    audio       listen(media, ...)                       -> RawTranscript
    boundaries  detect(policy, media=, transcript=, ...) -> Cuts | None
                timeline(media, policy, cuts=, ...)      -> Timeline
    video       ingest(media, timeline, sampler, frames=)-> Manifest
    cut         apply(timeline, raw)                     -> Transcript
    describe    answer(manifest, timeline, frames, ...)  -> Descriptions
    embed       encode(descriptions=, transcript=, ...)  -> list[Unit]

They pass the same test the surface trim already applied to `split`, `evidence`
and `retune`: a second way *in* that no naming collapses into `run`. `media`
needed nothing -- `split` was already this, and is the model the rest follow.

**Almost none of this was a refactor.** The work was already factored out of
every driver: `scenes.detect`, `speech.detect`, `grid.build`, `pipeline.ingest`,
`cutter.to_chunks`, `reader.answer` and `units.from_descriptions` all took
objects and returned objects before any of this. A driver was already
load -> compute -> write. What changed is that the middle is published, and
`run` is *defined as* the verb with a read at each end, so there is one
implementation of the work and one set of checks.

**The checks moved down, and that is the point rather than a tidy-up.** The
refuse-don't-drop tables (`SAMPLER_SETTINGS`, `EVIDENCE_SETTINGS`, audio's
settings routing), the question vocabulary, `limit < 1`, `batch < 1`,
`providers.require`, `min_s` above the ceiling -- every one now lives in the
verb, so both ways in get it. This is the level-asymmetric-validation trap
generalised: `workflow.validate` guarded the pipeline level and
`sampler="uniform:nope"` walked past it into a whole video decode. A check on
one of two public paths is a check half the callers never reach.

**Three names collided, and none was renamed away.** `audio.listen`,
`video.ingest` and `describe.answer` each already name a function one module
down -- but those take *built models* and *a store*, where the driver's take
setting names and a spec string. Two public functions of one name in one
package is the infinite recursion `describe` had to be rescued from, so the
lower one is imported under a private alias (`from .reader import listen as
_pass`) rather than renamed: the rest of the tree calls it `listen`, and it
should go on doing so.

**`boundaries.grid` is refused, which is why the verb is `timeline`.**
`boundaries/grid.py` is a submodule, and a package attribute of the same name
shadows it -- `import falconvar.video_rag.boundaries.grid` would stop
resolving. `timeline` also pairs with `load() -> Timeline`, so the return type
is legible from the call.

**`detect` returns None for `uniform`; `evidence` still returns a `Produced`.**
Those look contradictory and are not. `evidence` answers "what happened in this
step", and a step that did nothing is still a step -- which is the claim the
uniform signature and one HTTP route rest on. `detect` answers "what are the
cuts", and for `uniform` the honest answer is that there are none. Two
questions, two types, each answered on its own terms.

**`encode` does not resume, and says so.** Resume needs `stored_hashes()` from
an index, and the verb has none -- so it embeds everything handed to it while
`run` keeps the per-index diff. Collapsing them would mean either the verb
opening an index or `run` losing its resume, and the second is the expensive
silent no-op this tree has been bitten by twice.

**Pixels are the one thing that is not a document, so they got a protocol.**
`FrameStore` was built from `paths.artifact(id, "store")` inside `video`'s
driver and the path resolved a second time inside `describe`'s -- so describe
could only ever read frames this layout had put there. `video_rag/frames.py`
holds `Frames`, `FrameStore` (moved up from `video/store.py`, which is now a
shim) and `MemoryFrames`, and `FrameSource` takes whichever it is handed.
`describe.store_of` is the one line that still derives a store path from a
video id.

`MemoryFrames` is a separate class and not a flag, because the cost is the
caller's to accept: measured, a 5-minute video at 1 fps held 299 frames and
**170 MB**. Frames are byte-identical between the two, verified 299/299.

It also found a real bug on the way in: `pipeline.ingest` tested `if store`,
and `MemoryFrames` is sized, so an empty one is falsy -- and it is always empty
at the line that records `frame_store` in the manifest config. Truthiness wrote
`frame_store: null` into every in-memory manifest. `is not None` throughout.

**There is no per-thread data root any more, and that is deliberate.** A
`Workspace` -- a `ContextVar` that outranked `configure()` -- was written when
every component found its files by video id under the data root, so "two
tenants in one process" needed two roots. Addressing by path took that away:
every component writes where it is told, and its headline spelling,
`ws.media.run("talk.mp4")`, stopped existing along with `media.run`. What still
reads the root is deployment configuration -- `prompts.json`,
`aggregates.json`, `providers.json`, the weights cache -- and `workflow`'s
default `into`. Nothing served two of those from one process, and API keys were
never isolated per workspace anyway (one process, one `os.environ`). Removed:
precedence is `configure()`, then the variables, then the checkout, then
`~/.falconvar`. If per-tenant vocabularies are ever wanted, the path-addressed
answer is to pass the vocabulary files, not to hide a root in thread state.

The caches stay keyed by the data root: `describe/library.py` and
`aggregates/definitions/` both cache a merged vocabulary whose custom half lives
under it, and `configure()` can still move it mid-process.

**Progress is a callback, not a generator.** `describe` gathers under the
provider's concurrency -- 5.9x on the measured case -- so a synchronous
generator would have to serialise that or buffer through a queue, and buffering
loses the stop-part-way that was the only reason to want one. A callback that
raises stops at the next await, and `limit=` already expresses "this many, and
resume later". `pipeline.ingest` had taken an `on_chunk` since it was written
and nothing ever passed one.

Two things stated in the docstring rather than left to be discovered: the events
arrive **out of manifest order**, because completion order is the network's; and
the callback runs on the event loop, so a slow one serialises the gather it is
reporting on. `skipped` is counted apart from `completed` because "nothing to
do" and "everything done" look identical otherwise -- the same failure `limit=0`
had.

At level 2, `on_step` gains a third argument *if the callback takes one*, read
off its signature. Announcing it instead would break every two-argument callback
already written, the API's job runner included.

**Logging warns about nothing that works.** plan.txt proposed WARNING when a
setting resolves by fallback -- "the embedder defaulted to openai". That is not
a fallback here: `providers.choose` resolving to openai is the documented answer
and `/capabilities.defaults` publishes it, and warning on a working default
teaches people to filter warnings out. Those are DEBUG. WARNING is the
reported-but-continued case, which is the best-effort Supabase write.

The records go through two choke points rather than into every caller:
`paths.require` is the one line every artifact read passes through, and
`files.write` the one every document write does -- and `PRODUCED_BY` names the
component, so a record says who wrote what is being read. A database write is
a third, in `video_rag.driver.export`, and it is the only one that logs at
WARNING. Each `run` adds one
INFO in and one INFO out with its headline numbers, which is the timing
breakdown that previously existed only in CLI printing.

`logs._extra` takes a dict and not `**kwargs`, and renames any key a
`LogRecord` already owns. A component's stats are its own vocabulary:
`done(video_id=...)` is a reasonable thing to write, and as `**kwargs` it
collided with the function's own parameter and took the run down from inside a
logging call. A logging call must never be the thing that fails a stage that has
already done its work.

**TYPES.txt is generated, and `--check` fails on a stale copy.** The argument is
`schemas.py`'s, for the same reason: a restated field list drifts, and plan.txt
proved it by naming `Media.fingerprint` (it is `source`), `Cuts.cut_points` (it
is `cuts`) and a `Manifest.frames` that does not exist. Python keeps no
per-attribute docstring at runtime, so `reference.py` parses the module with
`ast` and reads both spellings already used here -- `#:` above a field and a
bare string below one. `--undocumented` listed 78 fields; it lists none now.

**The sink table was derived, and then there was no table.** `sinks.support()`
read `BACKENDS`, `rows.WRITERS` and each component's signature, because written
by hand it would drift from `write`. Deriving it was right and the thing it
derived should not have existed: every component is file-only now, and
`supabase.WRITERS` answers the only remaining question, which is whether a
document has a row representation at all.

Two findings came out of writing it down, and both are why the removal
happened. It had to be `any` and not `all` over a component's artifacts --
`video` produces a manifest *and* a `store`, and a directory of JPEGs never
reaches `write`, so `all` reported `video` as file-only. And `sinks.write` was
documented as best-effort for Postgres unconditionally when it only was if
given an `on_problem`, which no component passed. A destination concern inside
a component could be wrong in both directions at once.

**The two oracles for all of this.** Recreate is still the end-to-end one for
encoding and addressing; these two are for the factoring:

    EQUIVALENCE   one video through `run` and through the verbs, comparing
                  every document byte for byte with wall-clock stats lifted
                  out. If the two disagree, the factoring is wrong.
    NO FILESYSTEM the whole chain into a `MemoryFrames` with `paths.out_root`
                  replaced by something that raises, against an empty data
                  root that must still be empty afterwards.

Both pass, along with 299/299 frames identical between disk and memory, the
`9/9 schemas current`,
`recovery.recreate 299/299`, all 110 modules importing and every CLI running.

**Heavy dependencies are extras, and that does not weaken "a requirement is
required".** That rule forbids a second code path when a package is absent.
An extra that is not installed still fails at the function-local import with
the interpreter's own `ModuleNotFoundError`, and nothing falls back. Extras
change what gets *installed*, never what happens when something is not --
and `pip install falconvar` pulling several GB of CUDA wheels for someone
running `uniform` against an API model is a worse default than `[local]`.

**Packages are found, not listed.** The explicit list drifted the moment a
folder was added, and a package left off a wheel is not a build error -- it is
an `ImportError` for whoever installs it. `recovery/` stays excluded and
`__init__`-less on purpose.

---

## API

**Every component has the same signature, so one route runs any of them.**
`POST /videos/{id}/run/{component}` — `service.COMPONENTS` is a dispatch table,
and adding a component adds a row. A route and a handler per stage is what the
uniform signature removes. `params` is passed through as keyword arguments, so
**every component setting is reachable over HTTP** without the route knowing
any of them.

**`POST /videos` runs the whole pipeline; `run=false` registers and stops.**
The workflow deliberately carries no per-stage tuning — a scene grid with a
30 s floor, a uniform stride of 5, an `objects` vocabulary — so a caller
wanting those drives the components itself. Without `run=false` it had to run
the pipeline once on defaults first, paying for a describe it was about to
redo. `run=false` runs `media` and nothing else, answered rather than queued
because it is a container probe, and returns **201** with the id. `media` is
the one component the run route cannot reach: until it has run there is no id
to address.

**`/capabilities` publishes each component's parameters**, read off the
signature — name, type, default, required. Introspected for the reason
`defaults` is read off `workflow.Options`: a restated list drifts, and a
drifted one offers a parameter the component does not take or hides one it
does. That is the contract a configuration UI builds against.

**A signature says what a component accepts, not when a setting means
anything.** `boundaries.evidence` takes `stride` and `silence_s` together, and a
form built from the signature alone offered `silence_s` beside `policy: scene`,
where it is silently ignored. `service.conditions()` adds a `when` to each such
parameter -- another parameter and the values under which this one applies,
or for a sampler spec the sampler names that read it -- with the values read off
`POLICIES`, the sampler registry and `TRANSCRIBERS`. The client hides what does
not apply and never sends a hidden value, so a `stride` typed before switching
to `vad` does not ride along.

**An annotation is the published type, so `Any` publishes nothing.**
`retrieve.search` took `video_id: Any` -- true, since it accepts a string or a
sequence -- and a generated form had nothing to build a widget from. It is
`Optional[str | Sequence[str]]` now, which is what it always meant.

**A published type is the annotation as written, and the client reads its
shape.** `/capabilities` restates nothing, so a parameter's type arrives as
`Optional[float]` or `str | Sequence[str]`. The client matched those strings
*exactly* to pick a widget, so every `Optional[...]` field fell through to a
text box and was sent as text: `threshold` reached a sampler as `"0.9"` and
raised on its range check, and `vocabulary` arrived as a string that `list()`
turned into one entry per letter. `kindOf` strips the wrapper, takes a union's
first member -- which is why `str | Sequence[str]` stays one text box holding a
spec the component parses itself -- and a sequence is split on commas for every
such parameter rather than for `samplers` alone. Nine parameters were affected
before the two `Optional[bool]` ones existed to make it visible.

**The API calls components, never drivers.** A driver is argparse; importing
one to reach the work behind it would make a server depend on a CLI.

**Slow work is queued, one job at a time.** Every heavy stage contends for the
same 8 GiB GPU: CLIP and YOLO in the video pass, Whisper and pyannote in the
audio one. Two videos at once doubles the resident weights and invites an
allocator failure halfway through the more expensive one.

**Validation is synchronous even though the work is not.** `workflow.validate`
returns problems as a list, so a contradiction is a 422 the caller sees at once
rather than a job that fails a minute later. What cannot be known without
opening the file — whether a track carries speech — still fails inside the job,
because that is a property of the media rather than of the request.

**Job records die with the process; artifacts do not.** `GET /jobs` says so.
`GET /videos` reads the directory rather than remembering.

**`stage` is what is running; `history` is what has finished.** The workflow
announces each component twice -- once by name before it runs, once with its
`Produced` after -- because a `stage` set only on completion names the
*previous* component throughout the longest stage of the run. Observed: a
poller read `stage: "cut"` for the whole of describe, four minutes of OpenAI
calls, which is indistinguishable from being stuck on cut. The before-name is
the caller's word for the step, since `boundaries.evidence` only reveals
itself as `boundaries.audio` or `boundaries.scenes` once it has run.

**A single-component job has no progress callback**, because components do not
take one — they are one step, and a step emitting its own progress would be
reporting to itself. The runner records the returned `Produced` instead, so a
one-component job ends with the same shape a workflow job builds up.

**Adding a prompt runs nothing, so it is not queued.** `POST /prompts` writes
one file and returns 201; the queue exists for work that contends for the GPU.
Deleting a question does not touch the descriptions it produced — a description
cost a paid call and records the question it was asked, so removing the question
does not make the answer untrue, it only stops new runs asking it.

**A placeholder is checked at write time, not at call time.** `{frames}` in an
instruction would raise inside `str.format` — after the frames are read, with
the request about to be paid for. `check()` parses the instruction against
`{n}`, `{span}`, `{vocabulary}` when it is submitted.

**The export surface lists what exists, not what could exist.** An audio-only
video advertises no manifest rather than offering a link that 404s — a broken
link reads as breakage, not as a stage that never ran.

**A form built from a registry defaults to whatever sorts first**, and that was
`stub`: an audio-only run completed in 10.6 s, reported 42 segments and 205
words, and wrote a transcript of `[stub0.0][stub0.1]`. Nothing was wrong enough
to report. `/capabilities` publishes `defaults` read off `workflow.Options`, so
the default lives in the dataclass the pipeline actually uses -- and, for the
three model roles, `providers.defaults()`, since those fields stay `None` until
a run resolves them.

---

## Building a client

Nothing about the pipeline needs to be hardcoded in a UI. `GET /capabilities`
publishes every registry, the defaults, and each component's parameters, so a
form is generated from it rather than kept in step with it.

**Two ways to run a video, and the choice is about tuning.**

    POST /videos                          upload + the whole pipeline.  202
      policy sampler use_video use_audio describer embedder tier database
      -> workflow defaults for everything per-stage

    POST /videos  run=false               upload, probe, stop.          201
      -> then POST /videos/{id}/run/{component} with `params`, in order:
         audio · boundaries.evidence · boundaries · video · cut ·
         describe · embed · aggregate

The second is the one a configuration UI wants: `min_s`, `every_n`,
`threshold`, `vocabulary`, `per_second` and the rest live on the component that
owns them, and `/capabilities.parameters` names them with their types and
defaults. `boundaries.evidence` returns a `Produced` with `skipped: [evidence]`
when the policy needs none, rather than nothing -- **from the component
itself**, not from a wrapper. It used to return `None` and `api/service.py`
assembled the `Produced`, so one call answered differently depending on whether
it arrived over HTTP or from an `import`, and the library's half was the one a
caller had to special-case. A policy needing no precursor is an answer, not an
absence, and every component returns `Produced` at both levels now -- which is
the claim the uniform signature rests on. The wrapper is gone.

**Order is the caller's responsibility on that path.** `workflow.py` is the
reference for it; the dependencies are real — `boundaries` under `vad` or
`speaker` needs `audio` to have run, under `scene` needs
`boundaries.evidence`, and `cut`/`video` need the grid.

**Progress.** `stage` is what is running, `history` what has finished, and a
job's `detail` is the last `Produced`. Records die with the process; artifacts
do not, so a restarted server still lists every video from disk.

**Displaying results.** A search moment carries `descriptions` and `questions`
keyed by answer id, so grouping by sampler or by question needs no id parsing.
`GET /videos/{id}` lists only the artifacts that exist. Frames are
`GET /videos/{id}/frames/{index}` by the read index a manifest names.

**A score is a rank fusion, not a similarity.** `1/(k+best) + 0.5/(k+second)`
at k=10, so 0.1326 is the ceiling for a chunk contributing two units and means
"best ranked first, second ranked second" — never "this matched well". Measured:
a nonsense query scores identically to the best real one, because dense always
returns nearest neighbours and there is no relevance floor. If a UI shows a
number, show the ranks beside it — the CLI prints `clip(v7,t1)
uniform:text(v9,tNone)`, and `tNone` is how a reader sees the lexical half was
silent. The API carries them as `moments[].ranks`.

**Prompts are editable at runtime.** `GET/POST/DELETE /prompts`; a custom
question names a shape or brings its own, built-ins refuse edits with 409, and
adding a question invalidates nothing already described.

**`/search` narrows nine ways, and `/capabilities.search` publishes them** --
`sampler` (a pairing), `question`, `strategy` (one sampler's whole output),
`chunk_ids`, `window`, `after`/`before`, `structured`, `candidates`. None costs
a re-embedding: `text_hash` is over content alone, so every one of them is a
query-layer change over payload the unit already carries. Measured across both
backends, all six filters tried select **identical chunk sets**; rank order
differs on two, which is the lexical halves diverging.

**Time is resolved to chunk ids through the grid, never stored beside a
vector.** It needs no column and no migration, so nothing already stored
becomes unreachable.

**`structured` is only useful with `one_of`.**
`/capabilities.search.structured_fields` reads the shapes and lists exactly the
fields whose values are a vocabulary, so a form offers `severity: severe` and
not a free-text box that would match three different things.

**A moment carries the ranks.** `ranks[sampler_id] = {dense, text}`, `text:
null` meaning the lexical half was silent. Measured on the four-video corpus: a
nonsense query scores **0.1136** against a real one's **0.1294**, so there is no
relevance floor and the number alone says nothing. The ranks are the signal, and
the CLI printed them long before the API did.

**Scope is a set of videos, and that is one endpoint.** `video_ids` names them,
omitting it searches every one, and `video_id` is the one-element shorthand.
Searching one video, three, or all is the same question over a different set, so
a second route for "all" was a distinction the data never had.

**A search's notes are top-level, not only on its moments.** They rode on each
moment, so an empty result -- the case "nothing matched those filters" exists
for -- had nowhere to put them and reached the caller as a bare `[]`. Found by
searching with an embedder that had never indexed the video: a silent 200. The
note now names the embedder key, because that is exactly how such a search
comes back empty.

**Moments are keyed by `(video_id, chunk_id)`.** A chunk id indexes *one*
video's grid. Grouping on the id alone fused chunk 0 of two videos into one
moment with two unrelated accounts, and the agreement bonus scored that
collision above either real answer -- invisible while the scope was one video,
which is why it was written that way.

**`level` picks granularity and never mixes the two.** `moment` ranks chunks;
`video` ranks whole videos by their summary out of `video_embeddings`, which
`aggregates` fills from its `summary` (Postgres only; 4/4 on the test
corpus, winner 0.40-0.50 against runners-up of 0.05-0.32). One endpoint, but
never one ranking: a whole-video "moment" beside real ones is not something you
can play. At `level=video` the moment filters are reported under `ignored`
rather than silently dropped, because they narrow *inside* a video.

---

## Measured facts — do not re-derive

**Thresholds do not transfer between videos.** Same sampler, same domain,
`clip 0.96`:

| video | keep rate | median 1-second similarity |
|---|---|---|
| test.mp4 | 13.4% | 0.9868 |
| test2.mp4 | 18.0% | 0.9895 |
| test1.mp4 | 59.2% | 0.9605 |

That spread is the sampler working. **If cost is too high use
`min_interval_s` / `max_per_chunk`, not a lower threshold**: they keep the
most-changed frames and even out the per-chunk yield, where lowering the
threshold just keeps fewer and leaves the distribution lopsided.

**Do not solve for a fixed keep rate.** Forcing 15% turns a change sampler into
a worse uniform one. On a frozen video, solving for 15% produced threshold
1.000 which sampled *100%* of frames — encoder noise, since one PNG looped for
60 s H.264-encodes to 60 *different* frames.

**Detection samplers shifted 12–15% when the reader moved OpenCV → PyAV**
(yolo 40→45, objects 48→55; clip unchanged). Sub-LSB colour differences flip
detections near the confidence boundary.

**EasyOCR is 98.1% of the text sampler's cost** (129.6 ms of 132.0). The
descriptor is 2.5 ms. `canvas_size` is the only real lever and it is **not**
free: at 736 it is 3x faster but covers only 70% of the ink 1280 finds.

**PaddleOCR and craft-text-detector were evaluated and rejected.** EasyOCR's
detector *is* CRAFT, so the standalone package is the same model in an
unmaintained wrapper. PaddleOCR will not import here: `WinError 127` on
`cudnn_cnn64_9.dll` despite exactly-matching pinned versions.

**A difference hash is not usable for text change detection.** On a static
slide, dHash scored 0.86 agreement against *itself* frame to frame while two
*different* slides scored 0.88 — pure noise. dHash compares adjacent pixels and
most of a text region is flat background, so the sign it records comes from
sensor noise. Averaging down with `INTER_AREA`, centring and normalising gives
1.000 for a static slide and at most 0.80 between different ones.

**`objects` vocabulary is the highest-value setting**, and has no useful
default. A mismatched list found 2.4 detections/frame and labelled wire baskets
"shopping bag"; a matched one found 5.1.

---

## Environment traps

**`cublas64_12.dll` is not found, on a machine where it is present.**
CTranslate2 asks Windows for it *by name* at the first encode, not at import.
`nvidia-cublas-cu12` installs it to `site-packages/nvidia/cublas/bin`, which is
on no search path: since Python 3.8 an extension's dependencies do not resolve
from `PATH`, and `os.add_dll_directory` does not help because the load happens
lazily inside an already-initialised C++ extension. `audio/backends/cuda.py`
loads each by absolute path with `ctypes.WinDLL` first. The version is a
contract — this project's torch is cu130 and ships `cublas64_13.dll`, which is
not a substitute.

**pyannote 4.x returns `DiarizeOutput`, not `Annotation`.** The 3.x recipe
`pipeline(audio).itertracks(yield_label=True)` raises `AttributeError`. The
annotation is `.speaker_diarization`; `.exclusive_speaker_diarization` has
overlaps resolved, which is what this uses — a word cannot belong to two
speakers.

**`create table if not exists` never changes a table.** `install.sql` is re-run
against live databases, so a column added, dropped or retyped only inside the
`create` is simply not applied on every deployment that already had the table
-- and the first statement to reference it fails, or worse, a writer sends a
column PostgREST does not know. This bit three times: `structured`, then
`chunk_samplers.questions`, then `embeddings.sampler`/`question`.

So the `create`s hold the current shape and **section 10 holds the change**
for a live database -- an `alter`, a `drop ... if exists` -- each a no-op once
applied, and removed once every deployment has run it. Those migrations had
accumulated to about a third of the file before being cleared out: dropped
columns long gone, a backfill 0 rows needed, three superseded RPC signatures.

An audit is cheap and worth running after editing the file: parse the `create
table` bodies, diff them against PostgREST's deployed column list, and check
that every difference is covered in section 10.

**A generated column is not repaired by `add column if not exists`.** It is a
no-op when the column *exists*, so an `fts` built by an earlier expression
survived a re-run and the lexical half quietly stopped indexing the terms it is
best at. Changing `fts` means `drop column if exists fts` then the add, in
section 10; the column is generated, so nothing is lost.

**A `vector` column comes back from PostgREST as a *string*.** `vector(1536)`
arrives as the text `"[-0.0342,0.0450,...]"`, not a list. A cosine written
against a list of floats returned the not-comparable sentinel for every row and
`sorted` fell through to whatever order the rows arrived in, so it had never
ranked anything. Found by `video_embeddings` returning -1.0000 for four rows at
once; on the moment path (a dense-only fallback, since removed) the wrongness
was invisible, because table order is roughly chunk order and scores a
plausible MRR. `as_vector()` parses either form.

**PostgREST's cached schema is the fastest way to see what is really
deployed.** `GET /rest/v1/` with `Accept: application/openapi+json` lists every
column and RPC parameter it knows. That is how a table-name collision with the
previous pipeline was found before it silently ate writes.

**Exposing a Postgres schema is a dashboard setting, not SQL.** Until the
schema is in Settings → API → Exposed schemas, every request returns
`PGRST106`, which reads as a missing table rather than a missing setting.

**Embedded Qdrant took an exclusive lock** on its storage folder, and the
shape of that bug is worth keeping even though the backend is gone. It was not
a *two-process* problem, which is how it was filed until a server reproduced
it alone: the client had no `close`, no `__del__` and no context manager, so
one opened for a search never gave the lock back. A CLI run never notices --
the process exits and the lock goes with it. A long-lived API leaked it on the
first search and failed every later one with `Storage folder ... is already
accessed by another instance`, on a route that worked a minute earlier, and
the traceback frame of the first failure pinned the client alive so it never
recovered.

The fix was `close()` from a `finally` in both `retrieve.search` and
`embed.run`, because the not-indexed raise in the middle of search is exactly
the exit that leaked it. **The general lesson outlived the backend**: a
component that must reach a server to finish can fail a run whose paid calls
already succeeded. That is why `embed` now writes a file and the pipeline
pushes -- there is nothing left in the expensive path that holds a handle.

**`weights/clip/ViT-B-32.pt` (338 MB) is NOT stale.** YOLO-World embeds its
vocabulary with OpenAI CLIP. That is a *different* CLIP from the one the scene
sampler loads through HuggingFace — different library, format and job.

**opencv variants shadow each other.** `opencv-python`,
`opencv-contrib-python` and `opencv-python-headless` all install `cv2`;
whichever wins depends on install order and nothing warns you. Uninstalling one
**breaks the others** — repair with `pip install --force-reinstall --no-deps
opencv-python==5.0.0.93`.

**Installing `gliner` downgraded transformers 5.15.1 → 5.13.1.** An import
check would not have caught it: it proves the import works, not that CLIP still
embeds. Verify by actually embedding — 512 dims, L2 norm 1.0.

**`PYTHONIOENCODING=utf-8`** is needed for third-party libraries that print
non-ASCII on Windows' cp1252 console.

**`stream.thread_type = "AUTO"`** is mandatory in PyAV, not an optimisation:
7.15 ms/frame without it against 3.97 with.

**Background processes started with `&` survive `pkill -f` on Windows.** Six
uvicorn servers from an earlier session held the Qdrant lock and blocked a
directory delete. `Get-CimInstance Win32_Process` and `Stop-Process` is what
actually finds and kills them.

---

## Current state

The pipeline runs end to end on real models — Whisper `small`, pyannote 3.1,
CLIP, YOLO, `gpt-5.4-mini`, `text-embedding-3-small`, GLiNER, DistilBERT —
writing every document to a file, and a copy to Postgres when a run names a
`database`.

**The API figures below predate the sink and Qdrant removal**, and were taken
when every component took a `sink` and `embed` wrote to a vector store. The
run itself has been re-verified since, locally and on the component path
(`data root -> 8 components -> embedded.json`, resume 0/8 on a second pass,
299/299 on recreate, all three oracles); what has *not* been re-run since is
the whole thing through HTTP against a live Postgres.

Verified end to end **through the API**, from wiped local, Qdrant and Postgres
state, driving every stage with its own settings rather than the workflow's
defaults: `run=false` to register (201, no job), then `audio`,
`boundaries.evidence`, `boundaries` (scene, `min_s=30`, `max_s=60`), `video`
(`uniform:overview` at `every_n=5`, `clip:[mood,motion]` — two custom prompts
added over HTTP), `cut`, `describe`, `embed`, `aggregate --tier llm`.

Produced 4 chunks of 39.8-55.6 s on real scene cuts; `uniform` kept 11 frames
of a 54 s chunk, which is exactly every 5 s at 1/sec decimation; `clip` kept 51
on content change. 12 descriptions, 16 units under `clip:mood`, `clip:motion`,
`transcript` and `uniform:overview`, 8 aggregates. Every Postgres table exact
under both the secret and the publishable key — 4 chunks, 8 `chunk_samplers`
(one row per run, `questions` an array), 12 descriptions, 16 embeddings, 8
aggregates. `recovery.recreate` rebuilt the store **76/76 byte-identical**.

Both search filters verified against the RPC: `p_question` alone, `p_sampler`
alone, both intersecting, and a contradictory pair returning 0 rows. Lexical
firing by query type on a 55-unit corpus: literal 14/20 rows, paraphrase 20/20,
narration 20/20, nonsense **0/20**.

`shared.contracts.schemas --check` proves the dataclasses and the generated JSON
Schema still agree. The API serves **26 routes**; `docs/ROUTES.md` is the
reasoning and `/docs` the authority on shapes. The web client at `/app` drives
every route a run or a question needs, generating each form from
`/capabilities` rather than restating it.

**The live deployment, as of 2026-09-15** -- facts about one database, not the
code, and worth re-checking before trusting:

- `install.sql` is current, including the aggregates redesign. Verified on
  test1 with the secret key: 8 `aggregates` rows carrying `inputs`, `version`
  and `stats.model`; `entities:actors` as 23 `entities` and 39
  `entity_mentions` (3 with a `doubt`); 4 `aggregate_definitions`. A planted
  stale entity and mention were deleted by the next write, and "who is in
  chunk 2" is one equality on `entity_mentions`.
- The publishable key in `.env` is refused with a 401 while the secret key
  works, so every read that goes through `db.client(write=False)` -- the
  `/db/*` routes, `video_embeddings`, the grid fallback in `retrieve` -- fails
  there. The code has not changed; the key needs re-copying from Settings >
  API. The "exact under both keys" result above predates it.

## Not built

- **Tests.** No suite. Verification is recreate's byte-comparison, the schema
  check, `reference --check`, and ad-hoc scripts that are not kept -- which
  now includes the two oracles this factoring rests on, the `run`-vs-verb
  equivalence and the no-filesystem chain. Those two are worth keeping and are
  not kept.
- **An async surface.** `describe` is already async inside, with a
  provider-driven concurrency gate, but nothing is awaitable from outside: a
  caller wanting two videos at once needs threads. `arun` per component is
  additive and was deliberately left out of the decoupling work.
- **A formal sampler spec grammar.** `parse_spec` handles `name`, `name:q`,
  `name:[a,b]` and `name:a+b` and there is no escaping, so a question cannot
  contain a comma, colon or bracket. Nothing needs one yet, and the grammar is
  documented only by example.
- **An import checker.** The previous tree had one that AST-enforced the
  recovery invariant and proved every module loads after an install. Nothing
  enforces the recovery rule automatically now.
- **A bigger corpus.** `eval/harness.py` exists and runs, but 18 cases over 4
  videos is a direction, not a result -- the low-overlap band is n=3.
- **Sampler threshold calibration.** Scene thresholds sweep from cached scores;
  sampler thresholds (`clip 0.96`, `yolo 0.83`) do not.
- **The other providers, run.** Anthropic, Gemini, Mistral, Groq, OpenRouter,
  Together, DeepSeek, xAI, Voyage, Ollama and LM Studio are verified against a
  mock server's recording of the request, not against the real thing.
- **An answering model in-process.** Local answers go through a server --
  Ollama, LM Studio, llama.cpp, vLLM. Nothing loads a VLM into this process.
- **Supabase at any width, run.** `install.sql` drops the fixed width and the
  live database has had it, but no non-OpenAI embedder has written there yet.
- **Spans and items on a video longer than one window.** Both paths were
  exercised with the window shrunk to 4 chunks, never on a video that needs it.
- **Keeping answers when a describe call fails.** One failed call raises and
  the run's other answers -- already paid for, and with concurrency more of
  them in flight -- are discarded. Writing the successes before raising would
  let a re-run pay only for the failures.
- **A second backend, of any kind.** Postgres is the only one. The file is
  the portable half -- `embedded.json` carries the vectors, so a deployment
  can load them anywhere -- but nothing in this tree reads them back out into
  another store. `shared/storage/supabase.py` is the shape a `sql.py` would
  copy.
- **Reranking, query expansion, fusion tuning.** All plausible; none measured.
- **Live sources.** `Frame` carries no `gap_before`/`discontinuity` seams, so
  that is a retrofit through every stage rather than a field already there.
- **Entity linking from pixels, and labels worth the name.** `entities` reads
  descriptions only, so wording decides identity. `yolo` computes a CLIP
  embedding per person crop and discards it; keeping crops, and asking the
  describer to cite numbered boxes, would let appearance decide instead. The
  labelled set is 29 scored mentions.
- **The database export, run against a live Postgres.** `video_rag.driver.export`
  and `workflow._export_aggregates` are exercised against a recording double --
  every artifact pushed, a failing table reported rather than raised -- and
  not against the real thing since the rewrite.
