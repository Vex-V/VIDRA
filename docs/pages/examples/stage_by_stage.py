"""Every video_rag stage called on its own, with its settings.

This is what `video_rag()` does, with the knobs it leaves at their defaults
exposed. Each stage reads files and writes one; D is the video's folder.

    python stage_by_stage.py path/to/video.mp4
"""

import sys
from pathlib import Path

import vidra
from vidra import LocalEmbedder, OpenAI
from vidra.video_rag import audio, boundaries, cut, describe, embed, media, samplers, video

vidra.configure(env_file=".env")
source = sys.argv[1]

# 1. media: what the file holds. Makes the video's folder.
made = media.media(source, "data/out", on_conflict="new")
D = Path(made.stats["folder"])
print("video id:", made.video_id)

# 2. audio: the whole soundtrack, transcribed and diarized.
audio.audio(D / "media.json", D / "transcript.raw.json",
            transcriber="whisper", model="small", language="en",
            diarizer="pyannote", exclusive=True)

# 3. boundaries.evidence: where the picture changes (scene scores per frame).
boundaries.evidence(D / "cuts.json", "scene", media=D / "media.json",
                    stride=5, threshold=27.0)

# How many cuts other thresholds would give, from the cached scores.
for row in boundaries.calibrate(D / "cuts.json"):
    print(f"threshold {row['threshold']:>5}: {row['cuts']} cuts")
boundaries.retune(D / "cuts.json", D / "cuts.json", threshold=30.0)

# 4. boundaries: the chunk grid, 10 to 40 seconds a chunk.
boundaries.boundaries(D / "media.json", D / "timeline.json", policy="scene",
                      cuts=D / "cuts.json", min_s=10, max_s=40)
print("chunks:", len(boundaries.load(D / "timeline.json")))

# 5. video: which frames to keep. Two samplers in one decode, each with its own
#    settings: uniform keeps every 5th frame (one per 5 s at per_second=1) and is
#    asked to read the screen; clip keeps a frame when the scene changes.
video.video(D / "media.json", D / "timeline.json", D / "manifest.json",
            store=D / "store", per_second=1.0,
            sampler=[samplers.build("uniform", every_n=5, max_per_chunk=6, prompts=["text"]),
                     samplers.build("clip", threshold=0.95, max_per_chunk=6)])

# 6. cut: the transcript onto the grid.
cut.cut(D / "timeline.json", D / "transcript.raw.json", D / "transcript.json")

# 7. describe: one model call per (chunk, sampler, question).
def progress(event):
    print(f"  describe {event.completed}/{event.total} ({event.skipped} reused)")

describe.describe(D / "manifest.json", D / "timeline.json", D / "store",
                  D / "descriptions.json", previous=D / "descriptions.json",
                  vlm=OpenAI("gpt-5.4-mini"), max_output_tokens=2000,
                  on_progress=progress)

# 8. embed: every answer and every chunk's transcript, as vectors.
embed.embed(D / "embedded.json", descriptions=D / "descriptions.json",
            transcript=D / "transcript.json", previous=D / "embedded.json",
            timeline=D / "timeline.json", embedder=LocalEmbedder())

# Read any result back, typed.
said = describe.load(D / "descriptions.json")
first = said.chunks[0]["samplers"]
for answer_id, answer in first.items():
    print(answer_id, "->", answer["description"][:80])
