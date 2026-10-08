"""Which mentions across chunks are the same person or thing.

Linking embeds what identifies each mention and merges under rules; no
model decides identity (one writes each entity's account afterwards).

    cannot-link   two entries in one answer are different
    calibrated    the threshold is read off those provably different pairs,
                  in this video, with this embedder (`rule`: max or a
                  quantile)
    mutual        a pair links only if each is the other's best match
    merge         greedy, most similar first; a merge that would join two
                  entries of one answer is refused

A profile may measure more than one thing: `weights` embeds each identity
key apart, `attributes` (with `near` and `shared`) scores agreement over
closed-vocabulary fields. Each measure is a z-score against the provably
different pairs and the blend is their mean. A profile over a whole value
(prose, a text field) has nothing provably different and must give a
`threshold`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

from vidra.shared.contracts.documents import fingerprint_of

if TYPE_CHECKING:
    from ...core.base import Context

#: How the threshold is read off the provably different pairs.
RULES = ("max", "q95", "q90")


@dataclass
class Mention:
    """One entry in one answer's field -- or, for a whole value, the answer."""

    chunk_id: int
    sampler_id: str
    field: str
    index: int
    signature: str
    entry: dict[str, Any]

    @property
    def key(self) -> str:
        return f"c{self.chunk_id}/{self.sampler_id}/{self.field}/{self.index}"

    @property
    def answer(self) -> tuple[int, str, str]:
        """The list this entry came from. Two entries of one list differ."""
        return (self.chunk_id, self.sampler_id, self.field)


@dataclass
class Linked:
    groups: list[list[int]]
    threshold: Optional[float]
    calibration_pairs: int
    candidate_pairs: int


@dataclass
class Mentions:
    """One profile's selection, read against one video."""

    selection: Any
    items: list[Mention]

    @property
    def empty(self) -> bool:
        return not self.items

    @property
    def why_empty(self) -> str:
        what = (f"`{self.selection.field}` entries carrying "
                f"{', '.join(self.selection.keys)}" if self.selection.keys
                else f"a value for `{self.selection.field}`")
        heads = "+".join(s.head for s in self.selection.sources)
        return f"no answer read by `{heads}` has {what}"

    @property
    def chars(self) -> int:
        return sum(len(m.signature) for m in self.items)

    def fingerprint(self) -> str:
        """A hash of everything every mention carries."""
        return fingerprint_of({"mentions": [[m.key, m.signature, m.entry]
                                            for m in self.items]})


def mentions_of(context: "Context", selection: Any) -> list[Mention]:
    """Every mention a profile's selection finds, in chunk order: each object in a
    matching answer's list field, or one per answer for a whole value.
    """
    from vidra.shared.contracts.units import render
    from ...core.inputs import PROSE

    field, keys = selection.field, tuple(selection.keys)
    out: list[Mention] = []
    if field == "transcript":
        for chunk_id in context.chunk_ids():
            said = (context.transcript.text_of(chunk_id) or "").strip() \
                if context.transcript is not None else ""
            if said:
                out.append(Mention(chunk_id, "transcript", field, 0, said,
                                   {"transcript": said}))
        return out

    for chunk in (context.descriptions.chunks if context.descriptions else []):
        for sampler_id, block in (chunk.get("samplers") or {}).items():
            if not any(s.matches(sampler_id, block) for s in selection.sources):
                continue
            structured = block.get("structured") or {}
            if not keys:
                value = (block.get("description") if field == PROSE
                         else structured.get(field))
                said = (value.strip() if isinstance(value, str) else
                        render("", {field: value}).removeprefix(f"{field}: "))
                if said:
                    out.append(Mention(chunk["chunk_id"], sampler_id, field, 0,
                                       said, {field: said}))
                continue
            for index, item in enumerate(structured.get(field) or []):
                if not isinstance(item, dict):
                    continue
                signature = "; ".join(str(item[k]).strip() for k in keys
                                      if str(item.get(k) or "").strip())
                if signature:
                    out.append(Mention(chunk["chunk_id"], sampler_id, field,
                                       index, signature, item))
    return out


#: Attribute values that say nothing either way.
UNKNOWN = frozenset({"", "unclear", "covered"})
#: Words too common to count as shared detail in a free-text field.
STOP = frozenset({"a", "an", "the", "and", "with", "of", "on", "in", "none", "nothing", "no"})
#: Shared words in a free-text field count once their overlap reaches this.
OVERLAP = 0.3


def _unit(vectors: Any) -> Any:
    import numpy as np

    matrix = np.asarray(vectors, dtype=float)
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


def _different(mentions: Sequence[Mention]) -> Any:
    """Mask of provably different pairs: two entries of one answer."""
    import numpy as np

    answers = [m.answer for m in mentions]
    return np.array([[i != j and answers[i] == answers[j] for j in range(len(answers))]
                     for i in range(len(answers))])


def _standardised(sim: Any, different: Any) -> Any:
    """A z-score against the provably different pairs."""
    pool = sim[different]
    return (sim - pool.mean()) / (pool.std() or 1.0)


def _words(value: Any) -> set[str]:
    import re
    return set(re.findall(r"[a-z]+", str(value or "").lower())) - STOP


def attribute_similarity(mentions: Sequence[Mention], profile: dict[str, Any]) -> Any:
    """How far apart two entries' closed-vocabulary answers are, negated.

    Each field in `attributes` costs its weight when both answered and disagree,
    half for a pair listed in `near`, nothing when either could not see. Each
    field in `shared` takes its weight off per value two lists share, or once
    for text fields whose words overlap enough.
    """
    import numpy as np

    weights = profile.get("attributes") or {}
    shared = profile.get("shared") or {}
    near = {frozenset(pair) for pair in profile.get("near") or []}
    entries = [m.entry for m in mentions]
    words = {k: [_words(e.get(k)) for e in entries] for k, _ in shared.items()}
    count = len(entries)
    sim = np.zeros((count, count))
    for i in range(count):
        for j in range(i + 1, count):
            x, y, cost = entries[i], entries[j], 0.0
            for key, weight in weights.items():
                a, b = str(x.get(key) or ""), str(y.get(key) or "")
                if a in UNKNOWN or b in UNKNOWN or a == b:
                    continue
                cost += weight * (0.5 if frozenset((a, b)) in near else 1.0)
            for key, weight in shared.items():
                a, b = x.get(key), y.get(key)
                if isinstance(a, list) and isinstance(b, list):
                    cost -= weight * len({str(v) for v in a} & {str(v) for v in b})
                elif (wa := words[key][i]) and (wb := words[key][j]) \
                        and len(wa & wb) / len(wa | wb) >= OVERLAP:
                    cost -= weight
            sim[i, j] = sim[j, i] = -cost
    return sim


def similarity(mentions: Sequence[Mention], embedder: Any,
               profile: dict[str, Any]) -> tuple[Any, bool]:
    """The profile's measure of how alike two mentions are, and whether it is
    standardised. Without `weights` or `attributes`, the cosine over the
    signature.
    """
    import numpy as np

    weights, attributes = profile.get("weights") or {}, profile.get("attributes")
    if weights:
        # One embedding call per key.
        text = np.zeros((len(mentions), len(mentions)))
        for key, weight in weights.items():
            unit = _unit(embedder.embed([str(m.entry.get(key) or "").strip() or "unknown"
                                         for m in mentions]))
            text += weight * (unit @ unit.T)
        text /= sum(weights.values())
    else:
        unit = _unit(embedder.embed([m.signature for m in mentions]))
        text = unit @ unit.T
    if not weights and not attributes:
        return text, False
    different = _different(mentions)
    if not different.any():
        # Nothing to standardise against; `link` will not link either.
        return text, False
    measures = [_standardised(text, different)]
    if attributes:
        measures.append(_standardised(attribute_similarity(mentions, profile), different))
    return sum(measures) / len(measures), True


def link(mentions: Sequence[Mention], vectors: Sequence[Sequence[float]],
         rule: str = "max", mutual: bool = True,
         threshold: Optional[float] = None) -> Linked:
    """Group mentions of the same subject by the cosine of their vectors.
    Indices into `mentions`. See `link_similar` for the rules."""
    if len(mentions) == 0:
        return link_similar(mentions, [], rule, mutual, threshold)
    unit = _unit(vectors)
    return link_similar(mentions, unit @ unit.T, rule, mutual, threshold)


def link_similar(mentions: Sequence[Mention], sim: Any, rule: str = "max",
                 mutual: bool = True, threshold: Optional[float] = None) -> Linked:
    """Group mentions of the same subject, given how alike each pair is.
    `threshold` fixes the bar instead of reading it off the video.
    """
    import numpy as np

    if rule not in RULES:
        from vidra.shared.reporting.errors import UnknownOption
        raise UnknownOption(f"rule must be one of {', '.join(RULES)}")
    count = len(mentions)
    if count == 0:
        return Linked([], threshold, 0, 0)

    sim = np.asarray(sim, dtype=float)
    answers = [m.answer for m in mentions]

    # Calibrated over the whole video and every field.
    different = [sim[i, j] for i in range(count) for j in range(i + 1, count)
                 if answers[i] == answers[j]]
    if threshold is None:
        if not different:
            # No provably different pair to calibrate on: no links.
            return Linked([[i] for i in range(count)], None, 0, 0)
        threshold = float(max(different) if rule == "max"
                          else np.quantile(different, {"q95": 0.95, "q90": 0.90}[rule]))

    by_field: dict[str, list[int]] = {}
    for i, mention in enumerate(mentions):
        by_field.setdefault(mention.field, []).append(i)
    in_answer: dict[tuple[int, str, str], list[int]] = {}
    for i, answer in enumerate(answers):
        in_answer.setdefault(answer, []).append(i)

    def best(i: int, answer: tuple[int, str, str]) -> int:
        return max(in_answer[answer], key=lambda k: sim[i, k])

    pairs = []
    for indices in by_field.values():
        for position, i in enumerate(indices):
            for j in indices[position + 1:]:
                if answers[i] == answers[j] or sim[i, j] <= threshold:
                    continue
                if mutual and (best(i, answers[j]) != j or best(j, answers[i]) != i):
                    continue
                pairs.append((float(sim[i, j]), i, j))

    group_of = list(range(count))
    members = {i: [i] for i in range(count)}
    for _, i, j in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        a, b = group_of[i], group_of[j]
        if a == b:
            continue
        if {answers[k] for k in members[a]} & {answers[k] for k in members[b]}:
            continue
        members[a].extend(members[b])
        for k in members[b]:
            group_of[k] = a
        del members[b]

    groups = sorted((sorted(g) for g in members.values()), key=lambda g: g[0])
    return Linked(groups, float(threshold), len(different), len(pairs))


__all__ = ["Linked", "Mention", "Mentions", "RULES", "attribute_similarity", "link",
           "link_similar", "mentions_of", "similarity"]
