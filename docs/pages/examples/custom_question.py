"""A question of your own: what the vision model is asked about each chunk,
and the fields its answer must fill. Then search by those fields.

    python custom_question.py path/to/video.mp4
"""

import sys

import vidra
from vidra import Folder, LocalEmbedder, Models, OpenAI
from vidra.video_rag import describe, search, video_rag

vidra.configure(env_file=".env")
source = sys.argv[1]
models = Models(vlm=OpenAI("gpt-5.4-mini"), embedder=LocalEmbedder())
database = Folder("data/out")

# Saved in data/prompts.json; adding it again replaces it.
describe.add_question(
    "checkout",
    "These {n} frames span {span}. Describe what happens at the checkout.",
    fields={
        "payment": {"type": "text", "about": "how the customer pays, if it can be seen",
                    "one_of": ["cash", "card", "phone", "none visible"]},
        "items": {"type": "list", "about": "each item scanned, bagged or handed over"},
        "people": {"type": "list", "about": "each person at the checkout",
                   "of": {"role": "cashier, customer or bystander",
                          "doing": "what their hands are doing"}},
    },
    about="the checkout: payment, items, who does what",
)
print(describe.question("checkout")["fields"])

# clip's frames, asked clip's own question and ours, in one decode.
run = video_rag(source, "data/out", policy="scene", sampler="clip,clip:checkout",
                models=models, database=database)

# Only answers to our question, and only those that saw cash.
moments, notes = search("a customer paying", run.video_id, question="checkout",
                        structured={"payment": "cash"},
                        models=models, database=database)
for m in moments:
    answer = m.hits[0]["structured"]
    print(f"{m.start_ts:6.1f}-{m.end_ts:6.1f}s  payment={answer['payment']}  "
          f"items={answer['items'][:3]}")
for note in notes:
    print("note:", note)
