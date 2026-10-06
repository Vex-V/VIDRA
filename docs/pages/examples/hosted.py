"""The whole library with hosted models: OpenAI describes and summarises, a
local embedder makes the vectors, Supabase keeps a copy and serves search.

Needs: vidra[all], OPENAI_API_KEY, the Supabase variables (see Databases),
and HF_TOKEN for pyannote's first download.

    python hosted.py path/to/video.mp4
"""

import sys

import vidra
from vidra import LocalEmbedder, Models, OpenAI, Supabase, aggregates
from vidra.video_rag import search, video_rag

vidra.configure(env_file=".env")     # keys from .env; nothing reads it otherwise
source = sys.argv[1]

models = Models(
    vlm=OpenAI("gpt-5.4-mini"),      # describes the frames
    llm=OpenAI("gpt-5.4-mini"),      # summary, chapters, events, entity accounts
    embedder=LocalEmbedder(),        # BAAI/bge-small-en-v1.5, on this machine
)
database = Supabase()                # or Folder("data/out") for no server

# Chunk at picture cuts; keep frames when the scene or the people change.
run = video_rag(source, "data/out", policy="scene", sampler="clip,yolo",
                models=models, database=database)
print(run.video_id, "->", run.folder)
if run.problems:
    print("database writes that failed:", run.problems)

for query in ("someone paying at the till", "a person in a red jacket"):
    moments, notes = search(query, run.video_id, models=models, database=database)
    print(f"\n{query!r}")
    for m in moments:
        best = m.hits[0]
        print(f"  {m.start_ts:6.1f}-{m.end_ts:6.1f}s  {best['sampler_id']:<6} "
              f"{best['content'][:90]}")

# Whole-video answers: everything the run read, and who appears.
d = run.folder
video = aggregates.record(timeline=d / "timeline.json",
                          transcript=d / "transcript.json",
                          descriptions=d / "descriptions.json",
                          manifest=d / "manifest.json")
told = video.excerpt(transcript=True,
                     answers={"clip": ["summary"], "yolo": ["summary"]})
done = aggregates.aggregate(
    out=d / "aggregates", previous=d / "aggregates",
    models=models, database=database,
    stats=video, summary=told, chapters=told, events=told,
    entities_people=video.sightings(profile="people", answers=["yolo"]),
)
print("\nskipped:", done.stats["skipped"])

summary = aggregates.load(done.artifacts["summary"]).payload
print("\n" + summary["summary"])
for chapter in aggregates.load(done.artifacts["chapters"]).payload["chapters"]:
    print(f"  {chapter['start_ts']:6.1f}s  {chapter['title']}")
