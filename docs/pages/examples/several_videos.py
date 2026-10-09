"""Several videos: index each, search across all of them, and answer about
them as one.

    python several_videos.py monday.mp4 tuesday.mp4
"""

import sys

import vidra
from vidra import Folder, LocalEmbedder, Models, OpenAI, aggregates
from vidra.video_rag import search, video_rag

vidra.configure(env_file=".env")
models = Models(vlm=OpenAI("gpt-5.4-mini"), llm=OpenAI("gpt-5.4-mini"),
                embedder=LocalEmbedder())
database = Folder("data/out")

runs = [video_rag(path, "data/out", sampler="clip", models=models, database=database)
        for path in sys.argv[1:]]
ids = [run.video_id for run in runs]

# Leave video_id out to search every video the database holds, or name some.
moments, notes = search("a person carrying a box", ids, moments=5,
                        models=models, database=database)
for m in moments:
    print(f"{m.video_id:<12} chunk {m.chunk_id:>3}  {m.start_ts:6.1f}-{m.end_ts:6.1f}s")

# One record per video, then laid end to end on one clock.
records = [aggregates.record(timeline=run.folder / "timeline.json",
                             transcript=run.folder / "transcript.json",
                             descriptions=run.folder / "descriptions.json")
           for run in runs]
together = aggregates.combine(records=records, out="data/out/combined")
aggregates.summary(input=together.excerpt(transcript=True, answers={"clip": ["summary"]}),
                   out="data/out/combined/summary.json", llm=models.llm)
print(aggregates.load("data/out/combined/summary.json").payload["summary"])
