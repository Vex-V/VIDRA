"""Grade a retrieval configuration against a query set.

A case is a query and the chunks a person says answer it:

    {"query": "...", "video_id": "Chernobyl", "relevant": [5, 6],
     "kind": "literal" | "paraphrase", "why": "..."}

Results are grouped by measured overlap (the fraction of a query's content
words the corpus contains), with `kind` reported beside it. Metrics are over
chunks, which is what retrieval returns.
"""

from __future__ import annotations

import argparse
import json
import statistics
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falconvar.video_rag.retrieve import search                 # noqa: E402

#: Word splitting for the overlap bands.
_WORD = re.compile(r"[a-z0-9£$€%.:'-]+")

#: Words dropped before measuring overlap.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to from by
for with without into onto over under again further is are was were be been
being am do does did doing have has had having i you he she it we they them
his her its our their as so such no nor not only own same too very can will
just should now there here when where why how all any both each few more most
other some what which who whom
""".split())


def tokenize(text: str) -> list[str]:
    """Words, lowercased, stopwords removed; `1:23` and `9p` stay whole."""
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


# ----------------------------------------------------------------- metrics

def reciprocal_rank(ranked: list[int], relevant: set[int]) -> float:
    """1/rank of the first relevant chunk, or 0."""
    for position, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def recall_at(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def overlap(query: str, corpus_terms: set[str]) -> float:
    """What fraction of a query's content words appear in the corpus."""
    terms = tokenize(query)
    if not terms:
        return 0.0
    return sum(1 for t in terms if t in corpus_terms) / len(terms)


# -------------------------------------------------------------------- run

#: Overlap bands the results are grouped by.
BANDS = ((0.34, "low overlap"), (0.67, "mixed"), (1.01, "high overlap"))


def band_of(share: float) -> str:
    return next(name for edge, name in BANDS if share < edge)


def run_case(case: dict[str, Any], config: dict[str, Any], moments: int,
             terms: Optional[set[str]] = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        found, notes = search(case["query"], case["video_id"],
                              moments=moments, **config)
    except Exception as exc:                              # noqa: BLE001
        return {"query": case["query"], "error": f"{type(exc).__name__}: {exc}",
                "mrr": 0.0, "top1": 0.0, "recall": 0.0, "elapsed_s": 0.0,
                "kind": case.get("kind", "literal"), "band": "unknown"}

    ranked = [m.chunk_id for m in found]
    relevant = set(case.get("relevant", []))
    lexical_fired = sum(
        1 for m in found for h in m.hits if h.get("text_rank") is not None)
    share = overlap(case["query"], terms) if terms else None
    return {
        "query": case["query"],
        "kind": case.get("kind", "literal"),
        "overlap": None if share is None else round(share, 3),
        "band": band_of(share) if share is not None else "unknown",
        "ranked": ranked,
        "relevant": sorted(relevant),
        "mrr": reciprocal_rank(ranked, relevant),
        "top1": 1.0 if ranked and ranked[0] in relevant else 0.0,
        "recall": recall_at(ranked, relevant, moments),
        # How often the lexical half returned anything.
        "lexical_hits": lexical_fired,
        "units": sum(len(m.hits) for m in found),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "notes": [n for n in notes if "gives up" not in n],
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: Iterable[float]) -> float:
        values = list(values)
        return round(statistics.fmean(values), 4) if values else 0.0

    by_kind: dict[str, Any] = {}
    # By measured overlap, in band order.
    order = [name for _, name in BANDS] + ["unknown"]
    for band in [b for b in order if any(r.get("band") == b for r in rows)]:
        part = [r for r in rows if r.get("band") == band]
        by_kind[band] = {"n": len(part), "mrr": mean(r["mrr"] for r in part),
                         "top1": mean(r["top1"] for r in part),
                         "recall": mean(r["recall"] for r in part),
                         "lexical_fired_on":
                             sum(1 for r in part if r.get("lexical_hits"))}
    return {
        "cases": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "mrr": mean(r["mrr"] for r in rows),
        "top1": mean(r["top1"] for r in rows),
        "recall": mean(r["recall"] for r in rows),
        "lexical_fired_on": sum(1 for r in rows if r.get("lexical_hits")),
        "median_s": round(statistics.median([r["elapsed_s"] for r in rows]), 3)
                    if rows else 0.0,
        "by_kind": by_kind,
    }


def corpus_terms(video_ids: Iterable[str]) -> set[str]:
    """Every content word the index holds, read from `embedded.json`."""
    from falconvar.video_rag.embed import readable

    terms: set[str] = set()
    for video_id in video_ids:
        try:
            for unit in readable.load(video_id).units:
                terms.update(tokenize(unit.get("content", "")))
        except FileNotFoundError:
            continue
    return terms


# ------------------------------------------------------------------- main

def main(argv: Optional[list[str]] = None) -> int:
    from falconvar.shared.config import env
    env.load()        # an entry point reads .env; the library never does
    ap = argparse.ArgumentParser(
        description="Grade a retrieval configuration against a query set.")
    ap.add_argument("cases", type=Path, nargs="?",
                    default=Path(__file__).with_name("cases.json"))
    ap.add_argument("--embedder", default=None,
                    help="a provider or provider/model; default as embed resolves it")
    ap.add_argument("--moments", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--question", default=None, help="run every case filtered")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--check", action="store_true",
                    help="verify each case's `kind` against corpus overlap, "
                         "and run nothing")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]

    if args.check:
        terms = corpus_terms({c["video_id"] for c in cases})
        print(f"{len(terms)} distinct content words in the corpus\n")
        print(f"  {'kind':<11} {'overlap':>8}  query")
        bad = 0
        for case in cases:
            share = overlap(case["query"], terms)
            kind = case.get("kind", "literal")
            # Flag a case whose label disagrees with its measured overlap.
            wrong = (kind == "paraphrase" and share > 0.67) or \
                    (kind == "literal" and share < 0.34)
            bad += wrong
            print(f"  {kind:<11} {share:>7.0%}  {band_of(share):<13} "
                  f"{case['query'][:46]}"
                  f"{'   <- wording disagrees' if wrong else ''}")
        print(f"\n{len(cases) - bad}/{len(cases)} labels match their wording.")
        print("Bands are what the summary groups on; the label is only intent.")
        return 0

    def configured() -> dict[str, Any]:
        config = {"embedder": args.embedder, "candidates": args.candidates}
        if args.question:
            config["question"] = args.question
        if args.strategy:
            config["strategy"] = args.strategy
        return config

    terms = corpus_terms({c["video_id"] for c in cases})
    rows = [run_case(c, configured(), args.moments, terms) for c in cases]
    report = {"postgres": summarise(rows)}

    if args.json:
        print(json.dumps({"summary": report, "rows": rows}, indent=2))
        return 0

    for name, found in report.items():
        print(f"\n{name}   {found['cases']} cases"
              f"{f'  ({found['errors']} errored)' if found['errors'] else ''}")
        print(f"  MRR        {found['mrr']:.4f}")
        print(f"  top-1      {found['top1']:.4f}")
        print(f"  recall@{args.moments}   {found['recall']:.4f}")
        print(f"  lexical    fired on {found['lexical_fired_on']}/{found['cases']}")
        print(f"  median     {found['median_s']:.3f}s")
        for band, part in found["by_kind"].items():
            print(f"    {band:<13} n={part['n']:<3} mrr {part['mrr']:.4f}  "
                  f"top1 {part['top1']:.4f}  "
                  f"lexical {part['lexical_fired_on']}/{part['n']}")

    print("\nFailures worth reading:")
    for row in sorted(rows, key=lambda r: r["mrr"])[:5]:
        if row["mrr"] >= 1.0:
            continue
        print(f"  mrr {row['mrr']:.2f}  want {row.get('relevant')} "
              f"got {row.get('ranked')}  {row['query'][:46]}")
        if row.get("error"):
            print(f"          {row['error'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
