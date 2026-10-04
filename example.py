

import re
import sys

import falconvar
from falconvar import Folder, Models, aggregates
from falconvar.video_rag import describe, search, video_rag

source = sys.argv[1] if len(sys.argv) > 1 else "samples/test.mp4"


# Read keys from this .env; Models(keys={"openai": ...}) is the other way.
falconvar.configure(env_file=".env")
models = Models(describer="openai", embedder="local")


# A question of our own, saved in data/prompts.json.
describe.add_question(
    "checkout",
    "These {n} frames span {span}. Describe what is happening at the checkout.",
    fields={
        "payment": {"type": "text", "about": "how the customer pays, if it can be seen",
                    "one_of": ["cash", "card", "phone", "none visible"]},
        "items": {"type": "list", "about": "each item being scanned, bagged or handed over"},
        "people": {"type": "list", "about": "each person at the checkout",
                   "of": {"role": "cashier, customer or bystander",
                          "doing": "what they are doing with their hands"}},
    },
    about="the checkout: payment, items, who does what")

# `clip,clip:checkout` is one pass: clip's frames are asked both questions.
run = video_rag(source, "data/out", policy="scene", sampler="clip,yolo,clip:checkout",
                models=models)
print(f"{run.video_id}: {len(run.steps)} steps, output in {run.home}\n")


queries = [
    "someone talking on a phone",
    "a cashier scanning groceries",
    "a customer handing money to the cashier",
    "a person leaning over a shopping cart",
    "someone walking away down the aisle",
    "a man looking down at something in his hands",
]


def closest(text: str, query: str) -> str:
    """The sentence of `text` sharing the most words with `query`, to show why a
    chunk came back (the ranking itself is by vectors).
    """
    words = set(re.findall(r"[a-z]{3,}", query.lower()))
    sentences = re.split(r"(?<=[.;])\s+|\n", text)
    return max(sentences, key=lambda s: len(words & set(re.findall(r"[a-z]{3,}", s.lower()))))



index = Folder(run.home.parent)
for query in queries:
    moments, notes = search(query, run.video_id, moments=3,
                            models=models, database=index)
    print(f"{query!r}")
    for note in notes:
        print(f"    note: {note}")
    for moment in moments:
        line = closest(moment.hits[0]["content"], query)
        print(f"    chunk {moment.chunk_id:>2}  {moment.start_ts:5.0f}-{moment.end_ts:<4.0f}s"
              f"  {'+'.join(moment.samplers):<9}  {line[:110]}")
    print()


# Only our question's answers, and its fields as they were answered.
print("question=checkout: 'a customer paying for groceries'")
moments, _ = search("a customer paying for groceries", run.video_id, moments=3,
                    question="checkout", models=models, database=index)
for moment in moments:
    answer = moment.hits[0]["structured"]
    print(f"    chunk {moment.chunk_id:>2}  {moment.start_ts:5.0f}-{moment.end_ts:<4.0f}s"
          f"  payment={answer.get('payment')}  items={answer.get('items', [])[:4]}")
    for person in answer.get("people", [])[:3]:
        print(f"        {person.get('role')}: {person.get('doing')}")

# `one_of` makes a field a fixed vocabulary, so it can be filtered on exactly.
for paid in ("cash", "card"):
    moments, _ = search("paying at the checkout", run.video_id, moments=5,
                        question="checkout", structured={"payment": paid},
                        models=models, database=index)
    print(f"\npayment={paid}: chunks {sorted(m.chunk_id for m in moments) or 'none'}")


# An aggregate prompt of our own, saved in data/aggregates.json, reading only
# the checkout answers: one report over the whole video.
aggregates.add_prompt(
    "checkout_report",
    "Report on the checkout across this whole video.",
    fields={
        "report": {"type": "text", "about": "what happened at the checkout, in order"},
        "payments": {"type": "list", "about": "each payment seen",
                     "of": {"method": "cash, card or phone", "when": "roughly when"}},
        "busiest": {"type": "text", "about": "when the checkout was busiest, and why"},
    },
    inputs="clip:checkout",
    about="the checkout over the whole video")
made = aggregates.aggregate(run.home, run.home / "aggregates", checkout_report=True,
                            previous=run.home / "aggregates", models=models)
report = aggregates.load(made.artifacts["checkout_report"]).payload
print(f"\ncheckout_report: {report['report'][:300]}")
for payment in report.get("payments", []):
    print(f"    {payment.get('method')}: {payment.get('when')}")
print(f"    busiest: {report.get('busiest')}")
