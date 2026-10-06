"""Every aggregator, on a folder video_rag wrote: one at a time, then all at
once. Run hosted.py (or video_rag with sampler="clip,yolo") first.

    python aggregates_tour.py data/out/<video_id>
"""

import sys
from pathlib import Path

import vidra
from vidra import LocalEmbedder, Models, OpenAI, aggregates

vidra.configure(env_file=".env")
D = Path(sys.argv[1])
models = Models(llm=OpenAI("gpt-5.4-mini"), embedder=LocalEmbedder())
out = D / "answers"

# A record: the documents, named by path. Only the timeline is required.
video = aggregates.record(timeline=D / "timeline.json",
                          transcript=D / "transcript.json",
                          descriptions=D / "descriptions.json",
                          manifest=D / "manifest.json")
print("answers in this record:", video.answer_ids())

# Inputs, built from the record. Nothing is read unless named.
said = video.excerpt(transcript=True)                            # what was said
seen = video.excerpt(answers={"clip": ["summary"]})              # what was seen
both = video.excerpt(transcript=True, answers={"clip": ["summary", "actions"]})
people = video.sightings(profile="people", answers=["yolo"])     # each person entry

# free: count the record.
aggregates.stats(record=video, out=out / "stats.json")
aggregates.coverage(record=video, out=out / "coverage.json")

# local: small models on this machine (vidra[local]).
aggregates.ner(input=both, out=out / "ner.json", labels=["person", "product"])
aggregates.sentiment(input=both, out=out / "sentiment.json")

# llm: model calls.
aggregates.summary(input=both, out=out / "summary.json", models=models)
aggregates.chapters(input=both, out=out / "chapters.json", models=models,
                    max_spans=5, min_span_s=30)
aggregates.events(input=seen, out=out / "events.json", models=models)
try:
    aggregates.entities(profile="people", input=people, out=out / "people.json",
                        models=models)
except aggregates.Inapplicable as why:      # e.g. nobody was described
    print("entities skipped:", why)

for name in ("stats", "summary", "chapters", "people"):
    if (out / f"{name}.json").exists():
        payload = aggregates.load(out / f"{name}.json").payload
        print(f"\n{name}:", {k: payload[k] for k in list(payload)[:3]})

# The same, as one pipeline: cheapest first, reusing what is still current.
done = aggregates.aggregate(
    out=D / "aggregates", previous=D / "aggregates", models=models,
    stats=video, coverage=video, speakers=video,
    ner=both, summary=both, chapters=both, events=seen,
    sentiment={"spoken": said, "seen": seen},          # two answers, labelled
    entities_people=people,
    settings={"ner": {"labels": ["person"]}, "chapters": {"max_spans": 4}},
)
print("\nwrote:", sorted(done.artifacts))
print("computed:", done.stats["computed"], " skipped:", done.stats["skipped"])
