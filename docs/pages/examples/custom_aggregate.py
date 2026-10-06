"""Aggregators of your own: a prompt run over the whole video, and a linking
profile that follows one kind of thing across chunks.

    python custom_aggregate.py data/out/<video_id>
"""

import sys
from pathlib import Path

import vidra
from vidra import LocalEmbedder, Models, OpenAI, aggregates

vidra.configure(env_file=".env")
D = Path(sys.argv[1])
models = Models(llm=OpenAI("gpt-5.4-mini"), embedder=LocalEmbedder())
video = aggregates.record(timeline=D / "timeline.json",
                          transcript=D / "transcript.json",
                          descriptions=D / "descriptions.json")
told = video.excerpt(transcript=True, answers={"clip": ["summary"]})

# fold: one answer over the whole video.
aggregates.add_prompt(
    "incident_report",
    "Write an incident report for this video.",
    fields={
        "report": {"type": "text", "about": "what happened, in order"},
        "severity": {"type": "text", "about": "how serious it was",
                     "one_of": ["none", "minor", "major"]},
        "people_involved": {"type": "list", "about": "each person involved, by appearance"},
    },
    kind="fold",
    about="an incident report with a severity",
)

# items: discrete things, each citing the chunk it is in.
aggregates.add_prompt(
    "handovers",
    "List every moment something is handed from one person to another.",
    fields={
        "what": {"type": "text", "about": "what was handed over"},
        "between": {"type": "text", "about": "who gave it to whom, by appearance"},
    },
    kind="items",
    key="handovers",
)

aggregates.custom(name="incident_report", input=told,
                  out=D / "answers" / "incident_report.json", models=models)
report = aggregates.load(D / "answers" / "incident_report.json").payload
print(report["severity"], "-", report["report"][:200])

# Linking: the same object across chunks, from the `objects` field of the
# `objects` sampler's answers (video_rag with sampler="objects").
aggregates.add_profile(
    "tools", "objects",
    "These may be the same object, seen in different chunks. Say what it is "
    "and who used it for what.",
    fields={"use": {"type": "text", "about": "who used it, and for what"}},
    identity=["object", "appearance"],     # what decides that two entries are one
    story=["context"],                     # what the account reads besides
)

# All of them run in the pipeline too, each a keyword named after it
# (`entities:tools` is spelled `entities_tools`).
inputs = {"incident_report": told, "handovers": told}
if "objects" in video.answer_ids():
    inputs["entities_tools"] = video.sightings(profile="tools", answers=["objects"])
done = aggregates.aggregate(out=D / "aggregates", previous=D / "aggregates",
                            models=models, **inputs)
print("wrote:", sorted(done.artifacts))
print(aggregates.definition("handovers"))
