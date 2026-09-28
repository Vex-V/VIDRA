"""Rules first, then a model merges whole groups: does it recover what text cannot?

    python -m eval.llm_merge                     # test and test1, 2 runs each
    python -m eval.llm_merge --runs 3 --llm openai

`eval.linkers` tops out where the VLM describes one person differently by
view: test's right-hand cashier is "grey T-shirt with printed text on the back"
from behind and "white shirt, dark apron" from the front, and no similarity
over those two strings says they are one person. What does say so is context:
the same register, the same role, never on screen twice at once.

So the rules build high-precision groups (chunk-first + attach, P >= 0.99 on
both labelled videos), and one call per video is shown the groups -- not the
128 raw mentions, which is where `linking.py`'s docstring records a model
linking too eagerly -- and asked which groups are one person. Code enforces the
one hard rule afterwards: two groups holding entries of one answer are
different people, whatever the model says.
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Optional, Sequence

from .attributes import DATA, use_data
from .linkers import (LABELS, Corpus, attach, calibrated, chunk_first, errors, score,
                      similarities)

INSTRUCTION = """Below are groups of observations from a fixed overhead camera in one video, cut into 20-second chunks. Each observation is one describer's account of one person in one chunk. Every group is already believed to be a single person.

Decide which groups are the SAME person as each other.

- Two groups that share a chunk were seen at the same moment and are never the same person. Such pairs are listed at the end.
- The same person is often described differently depending on which way they face: a shirt's printed back from behind, an apron from the front; "grey" and "white" for one pale shirt; "vest", "jacket" or "cardigan" for one outer layer.
- Where they stand and what they do (which register, a cashier's role, pushing the same cart) is strong evidence for someone who stays in one place.
- A partial view at the frame edge usually belongs to someone seen whole in a neighbouring chunk.
- Do not merge people whose clothing clearly differs in a way no view explains: different colours of trousers or shorts, a cap against none, a patterned top against a plain one.

Return only the merges you are confident of. A group that is nobody else's is simply left out."""


def render_groups(c: Corpus, groups) -> tuple[str, list[set]]:
    lines, answers = [], []
    for n, g in enumerate(groups):
        chunks = sorted({c.mentions[i].chunk_id for i in g})
        lines.append(f"\nG{n} -- chunks {', '.join(map(str, chunks))}")
        for i in sorted(g, key=lambda i: (c.mentions[i].chunk_id, c.mentions[i].sampler_id))[:8]:
            e = c.mentions[i].entry
            attrs = "/".join(str(e.get(k, "?")) for k in ("gender", "age_group", "hair_color"))
            lines.append(f"  [chunk {c.mentions[i].chunk_id}] {e.get('role', '')}; {attrs}; "
                         f"{e.get('appearance', '')} | {e.get('clothing', '')} | "
                         f"{e.get('action', '')}")
        answers.append({c.answer[i] for i in g})
    clash = [f"G{a}-G{b}" for a in range(len(groups)) for b in range(a + 1, len(groups))
             if {x[0] for x in answers[a]} & {x[0] for x in answers[b]}]
    lines.append("\nSeen at the same moment, never the same person: " + ", ".join(clash))
    return "\n".join(lines), answers


async def ask(llm_spec: Optional[str], prompt: str) -> list[list[int]]:
    from falconvar.aggregates.base import listing, schema
    from falconvar.shared.models.llm import Model

    model = Model(llm_spec)
    props = listing("merges", {"groups": {"type": "array", "items": {"type": "integer"}},
                               "reason": {"type": "string"}})
    answer = await model.complete(prompt, schema("merges", props))
    return [m["groups"] for m in answer.get("merges") or []]


def apply(c: Corpus, groups, answers, merges) -> tuple[list[list[int]], int, int]:
    """Union the model's merges, refusing any that would put two entries of one
    answer in one person. Returns the groups, merges made, merges refused."""
    parent = list(range(len(groups)))
    held = [set(a) for a in answers]

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    made = refused = 0
    for ids in merges:
        ids = [i for i in ids if 0 <= i < len(groups)]
        for a, b in zip(ids, ids[1:]):
            ra, rb = root(a), root(b)
            if ra == rb:
                continue
            if held[ra] & held[rb]:
                refused += 1
                continue
            parent[rb] = ra
            held[ra] |= held[rb]
            made += 1
    out: dict[int, list[int]] = {}
    for n, g in enumerate(groups):
        out.setdefault(root(n), []).extend(g)
    return list(out.values()), made, refused


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Rules, then a model merging whole groups.")
    ap.add_argument("--video", default="test,test1")
    ap.add_argument("--similarity", default="blend[local] fieldwise+attr+places")
    ap.add_argument("--rule", default="q95")
    ap.add_argument("--llm", default=None, help="provider or provider/model")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--errors", action="store_true")
    args = ap.parse_args(argv)
    use_data(DATA)
    from falconvar.shared import env
    env.load()

    embedder = args.similarity.split("[")[1].split("]")[0]
    for video in args.video.split(","):
        c = Corpus(video, LABELS[video])
        z = similarities(c, [embedder])[args.similarity]
        base = attach(c, z, chunk_first(c, z, calibrated(c, z, args.rule)),
                      calibrated(c, z, "q80"))
        print(f"\n######## {video}  {args.similarity} chunk-first + attach {args.rule}")
        print(f"  rules only       {len(base):>3} groups  {score(c, base).row()}")
        prompt, answers = render_groups(c, base)
        for run in range(args.runs):
            merges = asyncio.run(ask(args.llm, INSTRUCTION + "\n" + prompt))
            groups, made, refused = apply(c, base, answers, merges)
            print(f"  + model, run {run + 1}  {len(groups):>3} groups  {score(c, groups).row()}"
                  f"  merges {made} refused {refused}")
            if args.errors:
                errors(c, groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
