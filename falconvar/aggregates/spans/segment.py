"""Where the subject changes: contiguous segments from one vector per chunk.

No model call. Three passes over vectors the caller made:

    1. boundaries   TextTiling-style: compare the chunks either side of every
                    gap; a gap in a valley of similarity is a candidate, and
                    its depth is how far it sits below the peaks either side.
                    Candidates deeper than mean - std/2 of all depths are
                    boundaries.
    2. floor        while a segment is shorter than `shortest` seconds, merge
                    the shortest into whichever neighbour is more alike.
    3. reduce       while there are more segments than `most`, merge the two
                    neighbouring segments whose centroids are most alike.

A segment is a list of row positions, in order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

#: Chunks compared on each side of a gap.
BLOCK = 2


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def _mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    return _unit([sum(column) / len(vectors) for column in zip(*vectors)])


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Segments:
    #: Row positions per segment, in order, covering every row once.
    groups: list[list[int]]
    #: The first pass's boundaries: the row a segment starts at, its gap
    #: similarity and its depth.
    boundaries: list[dict[str, float]] = field(default_factory=list)
    #: How many segments the first pass made, before any merge.
    first_pass: int = 0
    #: How many merges the floor forced, then how many `most` did.
    floored: int = 0
    merged: int = 0


def gap_similarity(vectors: Sequence[Sequence[float]], block: int = BLOCK) -> list[float]:
    """Similarity across each gap: the `block` chunks before it against the
    `block` after it, each side averaged. `len(vectors) - 1` values."""
    unit = [_unit(v) for v in vectors]
    out = []
    for gap in range(len(unit) - 1):
        left = unit[max(0, gap - block + 1):gap + 1]
        right = unit[gap + 1:gap + 1 + block]
        out.append(_cosine(_mean(left), _mean(right)))
    return out


def depths(similarity: Sequence[float]) -> list[float]:
    """How far each gap sits below the peaks either side of it (climbing stops where
    similarity stops rising).
    """
    out = []
    for gap, here in enumerate(similarity):
        left = here
        for value in reversed(similarity[:gap]):
            if value < left:
                break
            left = value
        right = here
        for value in similarity[gap + 1:]:
            if value < right:
                break
            right = value
        out.append((left - here) + (right - here))
    return out


def segment(vectors: Sequence[Sequence[float]], most: Optional[int] = None,
            durations: Optional[Sequence[float]] = None,
            shortest: float = 0.0, block: int = BLOCK) -> Segments:
    """Contiguous segments over `vectors` (one per row, in time order), at
    most `most` of them and none shorter than `shortest` seconds of
    `durations` (one per row). `most=None` and `shortest=0` keep whatever
    the first pass finds."""
    if most is not None and most < 1:
        raise ValueError("most must be 1 or more")
    if shortest and (durations is None or len(durations) != len(vectors)):
        raise ValueError("a floor in seconds needs one duration per vector")
    n = len(vectors)
    if n == 0:
        return Segments([])
    if n == 1:
        return Segments([[0]], first_pass=1)

    similarity = gap_similarity(vectors, block)
    depth = depths(similarity)
    # A valley: no deeper than its neighbours, and below at least one peak.
    candidates = [g for g in range(len(similarity))
                  if depth[g] > 0
                  and (g == 0 or similarity[g] <= similarity[g - 1])
                  and (g == len(similarity) - 1 or similarity[g] <= similarity[g + 1])]
    chosen: list[int] = []
    if candidates:
        values = [depth[g] for g in candidates]
        mean = sum(values) / len(values)
        spread = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        chosen = [g for g in candidates if depth[g] >= mean - spread / 2]

    starts = [0] + [g + 1 for g in chosen]
    groups = [list(range(a, b)) for a, b in zip(starts, starts[1:] + [n])]
    found = Segments(
        groups,
        boundaries=[{"row": g + 1, "similarity": round(similarity[g], 4),
                     "depth": round(depth[g], 4)} for g in chosen],
        first_pass=len(groups))

    unit = [_unit(v) for v in vectors]
    centroids = [_mean([unit[i] for i in group]) for group in groups]

    def length(group: list[int]) -> float:
        return sum(durations[i] for i in group) if durations else float(len(group))

    def merge(left: int) -> None:
        groups[left:left + 2] = [groups[left] + groups[left + 1]]
        centroids[left:left + 2] = [_mean([unit[i] for i in groups[left]])]

    while shortest and len(groups) > 1:
        short = [i for i, g in enumerate(groups) if length(g) < shortest]
        if not short:
            break
        i = min(short, key=lambda k: length(groups[k]))
        if i == 0:
            left = 0
        elif i == len(groups) - 1:
            left = i - 1
        else:
            left = (i - 1 if _cosine(centroids[i], centroids[i - 1])
                    >= _cosine(centroids[i], centroids[i + 1]) else i)
        merge(left)
        found.floored += 1

    if most is not None:
        while len(groups) > most:
            # Merge the two most alike neighbours.
            pair = max(range(len(groups) - 1),
                       key=lambda i: _cosine(centroids[i], centroids[i + 1]))
            merge(pair)
            found.merged += 1
    found.groups = groups
    return found


__all__ = ["BLOCK", "Segments", "depths", "gap_similarity", "segment"]
