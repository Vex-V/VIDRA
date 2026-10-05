"""A bench of entity linkers over the describer's people entries, scored
against hand labels.

    python -m eval.linkers                       # test (primary) and test1
    python -m eval.linkers --video test --top 25
    python -m eval.linkers --embedders openai,local

A linker is a similarity (cosine over a rendering of the entry, attribute
distance, or a blend) and a grouping (the production rules, agglomeration,
tracking, chunk-first). Every similarity is a z-score against the video's
provably different pairs. The calibrated row reads its threshold off the
video; the oracle row sweeps it against the labels (a ceiling). Scored on
pairwise P/R/F1, wrong links and B-cubed F1.

Labels: `people_labels_test.json`, `people_labels_test2.json` and
`people_labels.json` (test1), for videos under `data/eval/people`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Optional, Sequence

import numpy as np

from .attributes import DATA, HERE, Corpus, colour_garment, use_data

LABELS = {"test": HERE / "people_labels_test.json", "test1": HERE / "people_labels.json",
          "test2": HERE / "people_labels_test2.json",
          # test3 is a copy of test2, described again.
          "test3": HERE / "people_labels_test3.json"}
CACHE = DATA / "embedding_cache.json"


# --------------------------------------------------------------------- scoring

def bcubed(c: Corpus, groups) -> float:
    pred = {c.mentions[i].key: n for n, g in enumerate(groups) for i in g}
    keys = [k for k in pred if k in c.truth]
    p = r = 0.0
    for k in keys:
        mine = {x for x in keys if pred[x] == pred[k]}
        true = {x for x in keys if c.truth[x] == c.truth[k]}
        p += len(mine & true) / len(mine)
        r += len(mine & true) / len(true)
    p, r = p / len(keys), r / len(keys)
    return 2 * p * r / (p + r)


@dataclass
class Score:
    p: float
    r: float
    f1: float
    wrong: int
    b3: float
    people: int          # predicted groups holding a labelled mention

    def row(self) -> str:
        return (f"P {self.p:.2f} R {self.r:.2f} F1 {self.f1:.2f} wrong {self.wrong:>3} "
                f"B3 {self.b3:.2f} people {self.people:>3}")


def score(c: Corpus, groups) -> Score:
    p, r, f, fp = c.score(groups)
    people = len({n for n, g in enumerate(groups) for i in g if c.mentions[i].key in c.truth})
    return Score(p, r, f, fp, bcubed(c, groups), people)


# ---------------------------------------------------------------- similarities

def different_pairs(c: Corpus) -> np.ndarray:
    """Mask of provably different pairs: two entries of one answer."""
    a = c.answer
    return np.array([[i != j and a[i] == a[j] for j in range(c.n)] for i in range(c.n)])


def standardise(c: Corpus, sim: np.ndarray) -> np.ndarray:
    """z-score against this video's provably different pairs."""
    d = sim[different_pairs(c)]
    return (sim - d.mean()) / (d.std() or 1.0)


def _cache() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def embed(texts: Sequence[str], name: str) -> np.ndarray:
    """Unit vectors, cached on disk by (embedder, text)."""
    from vidra.shared.models import embedders

    store = _cache()
    key = lambda t: hashlib.sha1(f"{name}\0{t}".encode()).hexdigest()
    todo = sorted({t for t in texts if key(t) not in store})
    if todo:
        for t, v in zip(todo, embedders.build(name).embed(todo)):
            store[key(t)] = v
        CACHE.write_text(json.dumps(store), encoding="utf-8")
    v = np.asarray([store[key(t)] for t in texts], dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


ATTRS = ("gender", "age_group", "hair_color", "hair_style", "top_color", "top_type",
         "bottom_color", "bottom_type")

#: What gets embedded; `production` is what the `people` profile uses.
RENDER: dict[str, Callable[[dict], str]] = {
    "production": lambda e: "; ".join(str(e[k]).strip() for k in ("appearance", "clothing")
                                      if str(e.get(k) or "").strip()),
    "clothing": lambda e: e.get("clothing") or "",
    "rich": lambda e: " ".join(filter(None, (e.get("appearance"), e.get("clothing"),
                                             ", ".join(e.get("accessories") or []),
                                             e.get("distinguishing")))),
    # The closed-vocabulary fields as words.
    "attributes": lambda e: " ".join(
        f"{k.replace('_', ' ')} {str(e.get(k, '')).replace('_', ' ')}"
        for k in ATTRS if e.get(k) not in (None, "", "unclear", "covered")),
}


def cosine(c: Corpus, render: str, embedder: str) -> np.ndarray:
    v = embed([RENDER[render](m.entry) or "unknown" for m in c.mentions], embedder)
    return v @ v.T


def fieldwise(c: Corpus, embedder: str) -> np.ndarray:
    """Clothing and appearance embedded apart and averaged."""
    a = embed([m.entry.get("appearance") or "unknown" for m in c.mentions], embedder)
    b = embed([m.entry.get("clothing") or "unknown" for m in c.mentions], embedder)
    return 0.35 * (a @ a.T) + 0.65 * (b @ b.T)


def garments(c: Corpus) -> np.ndarray:
    s = np.eye(c.n)
    sets = [colour_garment(m.entry) for m in c.mentions]
    for i, j in combinations(range(c.n), 2):
        A, B = sets[i], sets[j]
        s[i, j] = s[j, i] = 2 * len(A & B) / (len(A) + len(B)) if A or B else 0.0
    return s


PLACES = {"left", "center", "centre", "middle", "central", "right", "far", "back", "upper",
          "lower", "foreground", "edge", "aisle", "register", "checkout", "lane", "cart",
          "counter", "shelves", "entrance"}


def places(c: Corpus) -> np.ndarray:
    """Place words from the action, as an identity cue."""
    import re
    sets = [set(re.findall(r"[a-z]+", (m.entry.get("action") or "").lower())) & PLACES
            for m in c.mentions]
    s = np.eye(c.n)
    for i, j in combinations(range(c.n), 2):
        A, B = sets[i], sets[j]
        s[i, j] = s[j, i] = len(A & B) / len(A | B) if A | B else 0.0
    return s


def attr_generic(c: Corpus) -> np.ndarray:
    """Agreement over the closed-vocabulary fields, every field weighted equally."""
    unknown = {"unclear", "covered", ""}
    vals = [[str(m.entry.get(k, "")) for k in ATTRS] for m in c.mentions]
    s = np.zeros((c.n, c.n))
    for i, j in combinations(range(c.n), 2):
        both = [(a, b) for a, b in zip(vals[i], vals[j]) if a not in unknown and b not in unknown]
        s[i, j] = s[j, i] = (sum(a == b for a, b in both) / len(both)) if both else 0.5
    return s


def keywise(c: Corpus, embedder: str, keys=("appearance", "clothing")) -> np.ndarray:
    """Each identity key embedded apart, averaged with equal weight."""
    out = np.zeros((c.n, c.n))
    for k in keys:
        v = embed([m.entry.get(k) or "unknown" for m in c.mentions], embedder)
        out += v @ v.T
    return out / len(keys)


def similarities(c: Corpus, embedders: Sequence[str]) -> dict[str, np.ndarray]:
    """Every similarity, already standardised."""
    raw: dict[str, np.ndarray] = {"attr distance": -c.distances(), "garment dice": garments(c),
                                  "places": places(c), "attr generic": attr_generic(c)}
    for e in embedders:
        for r in RENDER:
            raw[f"cos[{e}] {r}"] = cosine(c, r, e)
        raw[f"cos[{e}] fieldwise"] = fieldwise(c, e)
        raw[f"cos[{e}] keywise"] = keywise(c, e)
    z = {k: standardise(c, v) for k, v in raw.items()}
    for e in embedders:
        # Blends of text similarity and attribute agreement.
        z[f"blend[{e}] production+attr"] = (z[f"cos[{e}] production"] + z["attr distance"]) / 2
        z[f"blend[{e}] fieldwise+attr"] = (z[f"cos[{e}] fieldwise"] + z["attr distance"]) / 2
        z[f"blend[{e}] rich+attr"] = (z[f"cos[{e}] rich"] + z["attr distance"]) / 2
        z[f"blend[{e}] rich+attr+places"] = (z[f"cos[{e}] rich"] + z["attr distance"]
                                             + 0.5 * z["places"]) / 2.5
        z[f"blend[{e}] fieldwise+attr+places"] = (z[f"cos[{e}] fieldwise"] + z["attr distance"]
                                                  + 0.5 * z["places"]) / 2.5
        # Built only from what a profile declares.
        z[f"blend[{e}] keywise+generic"] = (z[f"cos[{e}] keywise"] + z["attr generic"]) / 2
    return z


# -------------------------------------------------------------------- grouping

def calibrated(c: Corpus, z: np.ndarray, rule: str = "max") -> float:
    d = z[different_pairs(c)]
    return float(d.max() if rule == "max" else np.quantile(d, int(rule[1:]) / 100))


def rules(c: Corpus, z: np.ndarray, thr: float, mutual: bool = True):
    """The production rules over any similarity: mutual best matches above the
    bar, greedy merge, refusing an answer clash.
    """
    n, ans = c.n, c.answer
    by: dict = {}
    for i, a in enumerate(ans):
        by.setdefault(a, []).append(i)
    best = lambda i, a: max(by[a], key=lambda k: z[i, k])
    pairs = sorted(((z[i, j], i, j) for i in range(n) for j in range(i + 1, n)
                    if ans[i] != ans[j] and z[i, j] > thr
                    and (not mutual or (best(i, ans[j]) == j and best(j, ans[i]) == i))),
                   key=lambda p: (-p[0], p[1], p[2]))
    group, members = list(range(n)), {i: [i] for i in range(n)}
    for _, i, j in pairs:
        a, b = group[i], group[j]
        if a == b or {ans[k] for k in members[a]} & {ans[k] for k in members[b]}:
            continue
        members[a] += members[b]
        for k in members[b]:
            group[k] = a
        del members[b]
    return list(members.values())


def agglomerate(c: Corpus, z: np.ndarray, thr: float, linkage: str = "average",
                start: Optional[list[list[int]]] = None):
    """Merge the two most alike groups until none clears the bar, never joining
    two entries of one answer.
    """
    groups = [list(g) for g in (start or [[i] for i in range(c.n)])]
    answers = [{c.answer[i] for i in g} for g in groups]
    agg = {"average": np.mean, "single": np.max, "complete": np.min}[linkage]
    link = {}
    for a, b in combinations(range(len(groups)), 2):
        if not answers[a] & answers[b]:
            link[(a, b)] = float(agg(z[np.ix_(groups[a], groups[b])]))
    alive = set(range(len(groups)))
    while link:
        (a, b), value = max(link.items(), key=lambda kv: kv[1])
        if value <= thr:
            break
        groups[a] += groups[b]
        answers[a] |= answers[b]
        alive.discard(b)
        link = {k: v for k, v in link.items() if a not in k and b not in k}
        for o in alive - {a}:
            if not answers[a] & answers[o]:
                link[(min(a, o), max(a, o))] = float(agg(z[np.ix_(groups[a], groups[o])]))
    return [groups[i] for i in sorted(alive)]


def chunk_first(c: Corpus, z: np.ndarray, thr: float, pair_thr: Optional[float] = None):
    """Pair each chunk's two answers first (Hungarian), then agglomerate the pairs."""
    from scipy.optimize import linear_sum_assignment

    pair_thr = thr if pair_thr is None else pair_thr
    start: list[list[int]] = []
    chunks = sorted({a[0] for a in c.answer})
    for ch in chunks:
        answers = sorted({a for a in c.answer if a[0] == ch}, key=lambda a: a[1])
        members = [[i for i in range(c.n) if c.answer[i] == a] for a in answers]
        if len(members) != 2:
            start += [[i] for m in members for i in m]
            continue
        A, B = members
        rows, cols = linear_sum_assignment(-z[np.ix_(A, B)])
        used_a, used_b = set(), set()
        for r, k in zip(rows, cols):
            if z[A[r], B[k]] > pair_thr:
                start.append([A[r], B[k]])
                used_a.add(r)
                used_b.add(k)
        start += [[A[r]] for r in range(len(A)) if r not in used_a]
        start += [[B[k]] for k in range(len(B)) if k not in used_b]
    return agglomerate(c, z, thr, "average", start)


def track(c: Corpus, z: np.ndarray, thr: float, gap: int = 2, merge: Optional[str] = "single"):
    """Hungarian tracking over answers in time order, then join tracks that never
    share an answer.
    """
    from scipy.optimize import linear_sum_assignment

    tracks: list[dict] = []
    for step, answer in enumerate(c.answers):
        dets = [i for i in range(c.n) if c.answer[i] == answer]
        live = [t for t in tracks if step - t["step"] <= gap]
        if live and dets:
            cost = np.array([[-max(z[m, d] for m in t["m"][-2:]) for d in dets] for t in live])
            taken = set()
            for r, k in zip(*linear_sum_assignment(cost)):
                if -cost[r, k] > thr:
                    live[r]["m"].append(dets[k])
                    live[r]["step"] = step
                    taken.add(dets[k])
            dets = [d for d in dets if d not in taken]
        tracks += [{"m": [d], "step": step} for d in dets]
    groups = [t["m"] for t in tracks]
    return agglomerate(c, z, thr, merge, groups) if merge else groups


def attach(c: Corpus, z: np.ndarray, groups, thr: float, largest: int = 1):
    """Attach each lone mention to the group it averages closest to, if that clears
    the lower bar `thr` and holds no entry of its answer.
    """
    groups = [list(g) for g in groups]
    stragglers = [g for g in groups if len(g) <= largest]
    anchors = [g for g in groups if len(g) > largest]
    answers = [{c.answer[i] for i in g} for g in anchors]
    moves: dict[int, list[int]] = {}
    for s in stragglers:
        mine = {c.answer[i] for i in s}
        options = [(float(z[np.ix_(s, g)].mean()), n) for n, g in enumerate(anchors)
                   if not mine & answers[n]]
        if options and max(options)[0] > thr:
            moves.setdefault(max(options)[1], []).append(s)
        else:
            anchors.append(s)
            answers.append(mine)
    out = []
    for n, g in enumerate(anchors):
        extra = moves.get(n, [])
        # Two stragglers from one answer claiming one group: keep the closer.
        taken, seen = [], set()
        for s in sorted(extra, key=lambda s: -float(z[np.ix_(s, g)].mean())):
            a = {c.answer[i] for i in s}
            if a & seen:
                out.append(s)
                continue
            seen |= a
            taken += s
        out.append(g + taken)
    return out


GROUPINGS: dict[str, Callable] = {
    "rules (today)": lambda c, z, t: rules(c, z, t),
    "rules, no mutual": lambda c, z, t: rules(c, z, t, mutual=False),
    "average link": lambda c, z, t: agglomerate(c, z, t, "average"),
    "complete link": lambda c, z, t: agglomerate(c, z, t, "complete"),
    "chunk-first + average": lambda c, z, t: chunk_first(c, z, t),
    "track": lambda c, z, t: track(c, z, t, merge=None),
    "track + merge": lambda c, z, t: track(c, z, t),
    "track + avg merge": lambda c, z, t: track(c, z, t, merge="average"),
    # Stragglers attach at the q80 of different pairs.
    "chunk-first + attach": lambda c, z, t: attach(c, z, chunk_first(c, z, t),
                                                   calibrated(c, z, "q80")),
    "average link + attach": lambda c, z, t: attach(c, z, agglomerate(c, z, t, "average"),
                                                    calibrated(c, z, "q80")),
    "track + merge + attach": lambda c, z, t: attach(c, z, track(c, z, t),
                                                     calibrated(c, z, "q80")),
}
#: Where the bar is read off the different pairs.
RULES = ("max", "q99", "q95")


def sweep(c: Corpus, z: np.ndarray, grouping: Callable) -> tuple[float, Score]:
    best = None
    for thr in np.linspace(-1.0, 6.0, 57):
        s = score(c, grouping(c, z, float(thr)))
        if best is None or (s.b3, s.f1) > (best[1].b3, best[1].f1):
            best = (float(thr), s)
    return best


class _Cached:
    """An embedder that answers from the bench's disk cache."""

    def __init__(self, name: str) -> None:
        self.name = name

    def embed(self, texts):
        return embed(list(texts), self.name).tolist()


def shipped(c: Corpus, embedder: str, profile: str = "people") -> Score:
    """The production linker, run as `entities` runs it."""
    from vidra.aggregates import definitions
    from vidra.aggregates.entities.linking import link_similar, similarity

    entry = definitions.get("profiles", profile)
    sim, _ = similarity(c.mentions, _Cached(embedder), entry)
    linked = link_similar(c.mentions, sim, entry["rule"], entry["mutual"], entry.get("threshold"))
    return score(c, linked.groups)


# ------------------------------------------------------------------------ main

def bench(video: str, embedders: Sequence[str], top: int, only: Optional[str]):
    c = Corpus(video, LABELS[video])
    true_people = len(c.labels)
    print(f"\n######## {video}: {c.n} mentions in {len(c.answers)} answers, "
          f"{len(c.truth)} labelled, {true_people} people")
    sims = similarities(c, embedders)
    if only:
        sims = {k: v for k, v in sims.items() if only in k}

    print("\n== separation, AUC (same-person pair beats a provably different one)")
    for name, z in sorted(sims.items(), key=lambda kv: -c.auc(kv[1])):
        print(f"  {name:<34} {c.auc(z):.3f}")

    rows = []
    for sname, z in sims.items():
        for gname, g in GROUPINGS.items():
            for rule in RULES:
                thr = calibrated(c, z, rule)
                s = score(c, g(c, z, thr))
                rows.append((s, sname, gname, rule, thr))
    print(f"\n== calibrated, no labels seen: best {top} by B-cubed")
    for s, sname, gname, rule, thr in sorted(rows, key=lambda r: (-r[0].b3, -r[0].f1))[:top]:
        print(f"  {sname:<34} {gname:<22} {rule:<4} z>{thr:5.2f}  {s.row()}")
    print("\n== the linker before this bench: one cosine over `appearance; clothing`, max")
    for s, sname, gname, rule, thr in rows:
        if gname == "rules (today)" and rule == "max" and sname.endswith("production") \
                and sname.startswith("cos"):
            print(f"  {sname:<34} {gname:<22} {rule:<4} z>{thr:5.2f}  {s.row()}")
    print("\n== the shipped `people` profile, through `linking`")
    for e in embedders:
        print(f"  {e:<57} {shipped(c, e).row()}")
    return c, sims, rows


def errors(c: Corpus, groups) -> None:
    """How each person was split, and every group that mixes people."""
    pred = {c.mentions[i].key: n for n, g in enumerate(groups) for i in g}
    for person, keys in c.labels.items():
        parts: dict[int, int] = {}
        for k in keys:
            parts[pred[k]] = parts.get(pred[k], 0) + 1
        print(f"  {person:<20} {len(keys):>2} mentions -> {len(parts)} groups "
              f"{sorted(parts.values(), reverse=True)}")
    for n, g in enumerate(groups):
        people: dict[str, int] = {}
        for i in g:
            who = c.truth.get(c.mentions[i].key, "?unlabelled")
            people[who] = people.get(who, 0) + 1
        if len([p for p in people if not p.startswith("?")]) > 1:
            print(f"  MIXED group {n}: {people}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Entity linker bench over people descriptions.")
    ap.add_argument("--video", default="test,test1,test2", help="comma-separated")
    ap.add_argument("--embedders", default="openai,local")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--only", help="substring of the similarity names to keep")
    ap.add_argument("--oracle", action="store_true",
                    help="also sweep the threshold against the labels (a ceiling)")
    ap.add_argument("--errors", metavar="SIM|GROUPING|RULE",
                    help="print splits and mixed groups for one calibrated config")
    args = ap.parse_args(argv)
    use_data(DATA)
    from vidra.shared.config import env
    env.load()

    embedders = [e for e in args.embedders.split(",") if e]
    videos = args.video.split(",")
    table: dict[tuple, list[Score]] = {}
    for video in videos:
        c, sims, rows = bench(video, embedders, args.top, args.only)
        for s, sname, gname, rule, _ in rows:
            # Keyed by the similarity's family, embedder lifted out.
            family = sname
            for e in embedders:
                family = family.replace(f"[{e}]", "")
            table.setdefault((family, gname, rule), []).append(s)
        if args.errors:
            sname, gname, rule = args.errors.split("|")
            z = sims[sname]
            groups = GROUPINGS[gname](c, z, calibrated(c, z, rule))
            print(f"\n== errors: {args.errors}  {score(c, groups).row()}")
            errors(c, groups)
        if args.oracle:
            print("\n== oracle: threshold swept against the labels (ceiling only)")
            out = []
            for sname, z in sims.items():
                for gname, g in GROUPINGS.items():
                    thr, s = sweep(c, z, g)
                    out.append((s, sname, gname, thr))
            for s, sname, gname, thr in sorted(out, key=lambda r: (-r[0].b3, -r[0].f1))[:args.top]:
                print(f"  {sname:<34} {gname:<22} z>{thr:5.2f}  {s.row()}")

    # Rank by the worst cell over every video and embedder.
    print(f"\n######## worst case over {' x '.join(videos)} x {' x '.join(embedders)}: "
          f"calibrated, cells are B3/P")
    ranked = sorted(table.items(), key=lambda kv: (-min(s.b3 for s in kv[1]),
                                                   -min(s.p for s in kv[1])))
    for (family, gname, rule), ss in ranked[:args.top]:
        cells = " ".join(f"{s.b3:.2f}/{s.p:.2f}" for s in ss)
        print(f"  {family:<28} {gname:<23} {rule:<4} min B3 {min(s.b3 for s in ss):.2f} "
              f"P {min(s.p for s in ss):.2f} | {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
