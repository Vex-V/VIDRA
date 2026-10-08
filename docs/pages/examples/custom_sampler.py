"""A sampler of your own: keep a frame when enough of the picture moved.

Cheap -- no model, a downscaled difference -- and for fixed cameras only. On
shop CCTV, threshold 0.10 kept 10-18% of decimated frames (CLIP at its default
keeps 13-18% there) and ignored sensor noise, which 0.05 and below did not. A
moving camera or an edited film changes nearly every pixel every second: it
kept 30-76% even at 0.45. Use `clip` for those.

    python custom_sampler.py path/to/video.mp4
"""

import sys

import numpy as np

import vidra
from vidra import Folder, LocalEmbedder, Models, OpenAI, Sampler
from vidra.video_rag import samplers, video, video_rag, video_rag_live

vidra.configure(env_file=".env")


class Motion(Sampler):
    """Keep a frame when the share of pixels that changed since the last kept
    frame reaches `threshold`."""

    name = "motion"                     # written in specs and stored with answers

    def __init__(self, threshold=0.10, scale=8, min_interval_s=0.0,
                 max_per_chunk=None, sampler_id=None, prompts=None):
        super().__init__(min_interval_s, max_per_chunk, sampler_id, prompts)
        self.threshold, self.scale = threshold, scale
        self._last_score = None

    def on_reset(self, chunk_id):
        self.reference = None           # every chunk starts afresh

    def propose(self, frame, chunk_local_index):
        # frame.image is BGR and borrowed: copy anything kept past this call.
        small = frame.image[::self.scale, ::self.scale].mean(axis=2)
        if self.reference is None:
            self.reference, self._last_score = small.copy(), None
            return True                 # the first frame of a chunk is kept anyway
        moved = float((np.abs(small - self.reference) > 25).mean())
        self._last_score = moved
        if moved >= self.threshold:
            self.reference = small.copy()
            return True
        return False

    def last_score(self):
        return self._last_score         # recorded beside each kept frame

    def config(self):
        # What the manifest records about the run: plain JSON only.
        return {**self._base_config(), "threshold": self.threshold, "scale": self.scale}


source = sys.argv[1]
models = Models(vlm=OpenAI("gpt-5.4-mini"), embedder=LocalEmbedder())

# An object carries its own settings and questions; a built-in is configured
# the same way through samplers.build.
run = video_rag(source, "data/out", use_audio=False, models=models,
                sampler=[Motion(threshold=0.10, prompts=["overview"]),
                         samplers.build("uniform", every_n=10, prompts=["overview"])],
                database=Folder("data/out"))
manifest = video.load(run.folder / "manifest.json")
for sid in ("motion", "uniform"):
    print(sid, "kept per chunk:",
          [c["samplers"].get(sid, {}).get("frame_count", 0) for c in manifest.chunks])

# Registered, it can be named in a spec string too, with any question.
samplers.register(Motion)
print("samplers:", samplers.available())
live = video_rag_live(source, "data/out", video_id="motion-live",
                      sampler="motion:overview", models=models)
print(live.observations, "answers from motion:overview, lag", live.lag_s)
