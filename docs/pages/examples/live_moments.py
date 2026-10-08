"""Log the moments a live stream shows that match ones you describe, as they
arrive: a Supabase subclass that looks at each answer before storing it.

    python live_moments.py rtsp://camera/stream     # or a file, read as a stream
"""

import json
import sys
import threading

import numpy as np

import vidra
from vidra import LocalEmbedder, Models, OpenAI, Supabase
from vidra.video_rag import describe, video_rag_live

vidra.configure(env_file=".env")

# A question whose answers are easy to check: one fixed-choice field per rule,
# each with a way to say "cannot tell" (without one, the model guesses), and a
# list of hazards, each a short phrase, for everything else.
SEEN = ["every member of staff wears one", "a member of staff without one",
        "cannot tell", "no staff visible"]
describe.add_question(
    "ppe",
    "These {n} frames span {span}. Describe the staff at work (not customers or "
    "visitors) and any safety hazards.",
    fields={
        "hard_hats": {"type": "text", "about": "hard hats on the staff", "one_of": SEEN},
        "hi_vis": {"type": "text", "about": "high-visibility vests or clothing on "
                   "the staff", "one_of": SEEN},
        "hazards": {"type": "list", "about": "each hazard visible, in a few words"},
    },
)
RULES = {"hard_hats": "no hard hat", "hi_vis": "no high-visibility clothing"}

# What to look for among the hazards, matched by similarity.
TARGETS = {"spill": "liquid spilled on the floor",
           "blocked_exit": "boxes or a pallet blocking a doorway",
           "ladder": "a person working on a step ladder"}


class Moments(Supabase):
    """Supabase, plus: every answer that breaks a rule or names a hazard close to
    a target is logged as a moment before it is stored."""

    def __init__(self, targets, embedder, threshold=0.85, log="moments.jsonl", **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder
        self.names, self.targets = list(targets), self._unit(list(targets.values()))
        self.threshold, self.log = threshold, log
        self.lock, self.found = threading.Lock(), 0

    def _unit(self, texts):
        vectors = np.array(self.embedder.embed(texts))
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def write_observations(self, video_id, rows):
        try:
            self.record([m for row in rows for m in self.matches(video_id, row)])
        except Exception as exc:         # a bug here must not stop the storing
            print("matching failed:", exc)
        super().write_observations(video_id, rows)

    def matches(self, video_id, row):
        moment = {"video_id": video_id, "frame_index": row["frame_index"],
                  "media_ts": row["media_ts"], "seen_at": row["seen_at"],
                  "description": row["description"]}
        for field, target in RULES.items():
            if row["structured"].get(field) == "a member of staff without one":
                yield {**moment, "target": target, "matched": field, "similarity": None}
        # Each hazard on its own: a whole answer is a paragraph about everything
        # in shot, and one detail in it barely moves its vector.
        hazards = row["structured"].get("hazards") or []
        if hazards:
            similar = self._unit(hazards) @ self.targets.T          # hazards x targets
            for h, t in zip(*np.nonzero(similar >= self.threshold)):
                yield {**moment, "target": self.names[t], "matched": hazards[h],
                       "similarity": round(float(similar[h, t]), 3)}

    def record(self, moments):
        # Several runs may share one database, each writing from its own thread.
        with self.lock, open(self.log, "a", encoding="utf-8") as out:
            for m in moments:
                out.write(json.dumps(m) + "\n")
                print(f"{m['seen_at']}  {m['video_id']}  t={m['media_ts']:.1f}s  "
                      f"{m['target']}: {m['matched']}")
            self.found += len(moments)


embedder = LocalEmbedder()
database = Moments(TARGETS, embedder)
run = video_rag_live(sys.argv[1], "data/out", sampler="clip:ppe", context=(1, 0),
                     models=Models(vlm=OpenAI("gpt-5.4-mini"), embedder=embedder),
                     database=database)
print(f"{run.observations} answers, {database.found} moments, "
      f"{run.written} stored as they arrived, lag {run.lag_s}")
