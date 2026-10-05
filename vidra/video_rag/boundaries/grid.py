"""Evidence plus a duration, into spans. The only place boundaries are decided.

`uniform` is arithmetic over the container duration. A content-derived policy
gets two guards: `min_s` merges short chunks, `max_s` splits long ones evenly.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from vidra.shared.contracts.documents import Timeline
from vidra.shared.reporting.errors import Refused, UnknownOption

#: Every policy, and which evidence it needs first (None, "video" or "audio").
POLICIES: dict[str, Optional[str]] = {
    "uniform": None,           # arithmetic over media.json. Nothing runs first.
    "scene": "video",          # boundaries.scenes -> cuts.json
    "vad": "audio",            # boundaries.speech reading transcript.raw.json
    "speaker": "audio",
}

#: A final chunk shorter than this fraction of the nominal length is merged into
#: the one before it, under every policy.
MIN_TAIL_FRACTION = 0.25


def merge_tail(spans: list[tuple[float, float]],
               min_tail_s: float) -> list[tuple[float, float]]:
    """Absorb a too-short final chunk into the one before it."""
    if len(spans) < 2:
        return spans
    start, end = spans[-1]
    if end - start >= min_tail_s:
        return spans
    return spans[:-2] + [(spans[-2][0], end)]


def enforce(spans: list[tuple[float, float]], min_s: float = 0.0,
            max_s: Optional[float] = None) -> list[tuple[float, float]]:
    """Apply `min_s` (merge) and `max_s` (split evenly)."""
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and (end - start) < min_s:
            merged[-1] = (merged[-1][0], end)      # absorb into the previous
        else:
            merged.append((start, end))
    # A short first chunk absorbs forwards.
    if min_s and len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_s:
        merged[:2] = [(merged[0][0], merged[1][1])]

    if not max_s:
        return merged

    out: list[tuple[float, float]] = []
    for start, end in merged:
        span = end - start
        if span <= max_s:
            out.append((start, end))
            continue
        pieces = int(-(-span // max_s))             # ceil, without importing math
        step = span / pieces
        edges = [start + step * i for i in range(pieces)] + [end]
        out.extend(zip(edges, edges[1:]))
    return out


def uniform(video_id: str, duration_s: float, chunk_s: float = 20.0) -> Timeline:
    """A fixed grid, truncated to where the media ends."""
    if chunk_s <= 0:
        raise Refused("chunk_s must be positive")
    if duration_s <= 0:
        raise Refused("duration_s must be positive")
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s:
        spans.append((start, min(start + chunk_s, duration_s)))
        start += chunk_s
    spans = merge_tail(spans or [(0.0, duration_s)], chunk_s * MIN_TAIL_FRACTION)
    return Timeline(video_id, spans, "uniform", {"chunk_s": chunk_s}, "grid")


def from_cuts(video_id: str, cuts: Sequence[float], duration_s: float,
              policy: str, params: Optional[dict[str, Any]] = None,
              derived_from: str = "grid",
              min_s: float = 5.0,
              max_s: Optional[float] = 30.0) -> Timeline:
    """Interior cut times plus a total length, guarded, as a Timeline. No cuts gives
    one chunk covering the file, which `max_s` then divides. The last span
    reaches `duration_s`, the container's duration.
    """
    edges = [0.0] + sorted(t for t in cuts if 0.0 < t < duration_s) + [duration_s]
    spans = [(a, b) for a, b in zip(edges, edges[1:]) if b > a]
    spans = enforce(spans or [(0.0, duration_s)], min_s, max_s)
    spans = merge_tail(spans, min_s)
    settings = {"min_s": min_s, "max_s": max_s, **(params or {})}
    return Timeline(video_id, spans, policy, settings, derived_from)


def build(video_id: str, policy: str, duration_s: float,
          cuts: Optional[Sequence[float]] = None,
          derived_from: str = "grid",
          chunk_s: float = 20.0, min_s: float = 5.0,
          max_s: Optional[float] = None,
          params: Optional[dict[str, Any]] = None) -> Timeline:
    """One grid, whatever the policy. `max_s` defaults to `chunk_s` for a
    content-derived policy.
    """
    if policy not in POLICIES:
        raise UnknownOption(f"unknown policy {policy!r}; "
                       f"known: {', '.join(POLICIES)}")
    if policy == "uniform":
        return uniform(video_id, duration_s, chunk_s)
    if cuts is None:
        raise Refused(
            f"policy {policy!r} needs cuts from "
            f"boundaries.{'scenes' if POLICIES[policy] == 'video' else 'speech'}")
    return from_cuts(video_id, cuts, duration_s, policy, params,
                     derived_from, min_s, max_s if max_s is not None else chunk_s)


__all__ = ["POLICIES", "MIN_TAIL_FRACTION", "merge_tail", "enforce", "uniform",
           "from_cuts", "build"]
