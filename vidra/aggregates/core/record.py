"""A record: the documents an aggregate is taken from, each named, and what
each aggregator reads from them.

    video = record(timeline="d/timeline.json", transcript="d/transcript.json",
                   descriptions="d/descriptions.json", manifest="d/manifest.json")
    said = video.excerpt(transcript=True)
    seen = video.excerpt(answers={"clip:activity": ["summary", "actors"]})
    people = video.sightings(profile="people", answers=["yolo"])

What `video_rag` wrote for one video, or `combination` for several:

    timeline      required: the grid every answer is placed on
    descriptions  what the picture was said to hold, per chunk and answer
    transcript    what was said, per chunk
    manifest      which frames were kept (only `stats` reads it)

Nothing is read by default. An excerpt names the transcript, the answers, or
both, and every answer names its fields; `summary` is the answer's prose. A
key of `answers` is an answer id as `descriptions.json` stores it:
`clip:activity`, or `yolo` for a sampler asked its own question. A name, a
field or an answer this record does not have is refused here, before
anything is paid for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from ...shared.contracts.documents import (Descriptions, Excerpt, Manifest,
                                           Sightings, Timeline, Transcript,
                                           same_video)
from ...shared.reporting.errors import VidraError
from ...shared.storage import files
from .base import Context
from .inputs import Input, Source, check, read

#: The documents a record may hold, and the type each is read as.
KINDS: dict[str, type] = {"timeline": Timeline, "descriptions": Descriptions,
                          "transcript": Transcript, "manifest": Manifest}

#: A path as a caller hands one over.
PathLike = Union[str, Path]


class RecordError(VidraError, ValueError):
    """A record that cannot be opened, or an excerpt or sightings asking for
    something the record does not have."""


def context(timeline: PathLike,
            descriptions: Optional[PathLike] = None,
            transcript: Optional[PathLike] = None,
            manifest: Optional[PathLike] = None) -> Context:
    """The documents at these paths, joined by `chunk_id`. The grid is required;
    the rest are optional. Documents from another video, or cut on another
    grid, are refused.
    """
    grid = files.read(timeline, Timeline)
    read_ = {"descriptions": files.maybe(descriptions, Descriptions),
             "transcript": files.maybe(transcript, Transcript),
             "manifest": files.maybe(manifest, Manifest)}
    video_id = same_video(timeline=grid, **read_)
    expected = grid.fingerprint()
    for kind, document in read_.items():
        cut_on = getattr(document, "timeline_fingerprint", "") if document else ""
        if cut_on and cut_on != expected:
            raise RecordError(
                f"{kind} was cut on a different grid from {timeline}: chunk ids "
                f"would mean different spans. Re-run what wrote it on this grid.")
    return Context(video_id, grid, read_["manifest"], read_["descriptions"],
                   read_["transcript"])


class Record:
    """One video's documents (or a combination's), and the inputs built from
    them. Made by `record()` or `combine()`."""

    def __init__(self, context_: Context, paths: Mapping[str, Path]) -> None:
        self.context = context_
        #: kind -> the file it was read from.
        self.paths = dict(paths)

    @property
    def video_id(self) -> str:
        return self.context.video_id

    @property
    def timeline(self) -> Timeline:
        return self.context.timeline

    def answer_ids(self) -> list[str]:
        """Every answer this record's descriptions hold, as `answers=` keys."""
        found: dict[str, None] = {}
        if self.context.descriptions is not None:
            for chunk in self.context.descriptions.chunks:
                found.update(dict.fromkeys(chunk.get("samplers") or {}))
        return list(found)

    def __repr__(self) -> str:
        held = ", ".join(f"{k}={str(v)!r}" for k, v in self.paths.items())
        return f"record({held})"

    # ------------------------------------------------------------- excerpts

    def excerpt(self, *, transcript: bool = False,
                answers: Optional[Mapping[str, Sequence[str]]] = None,
                out: Optional[PathLike] = None) -> Excerpt:
        """What a text aggregator reads: per chunk, the transcript if
        `transcript`, and each answer in `answers` reduced to its listed fields.
        `out` also writes it to a file.
        """
        if not transcript and not answers:
            raise RecordError("an excerpt reads something: pass transcript=True, "
                              "answers={...}, or both")
        sources: list[Source] = []
        if transcript:
            if self.context.transcript is None:
                raise RecordError("transcript=True, but this record was given no "
                                  "transcript")
            sources.append(Source("transcript"))
        for answer, fields in (answers or {}).items():
            sources.append(Source(self._head(answer), self._fields(answer, fields)))
        one = Input(tuple(sources))
        self._check(one)
        taken = read(self.context, one)
        made = Excerpt(
            video_id=self.video_id, selection=str(one),
            timeline=self.timeline.as_dict(),
            rows=[{"chunk_id": r.chunk_id, "start_ts": r.start, "end_ts": r.end,
                   "parts": [list(part) for part in r.parts]} for r in taken.rows],
            answers=list(taken.answers))
        if out is not None:
            files.write(Path(out), made)
        return made

    def sightings(self, *, profile: str,
                  answers: Optional[Sequence[str]] = None,
                  keys: Optional[Sequence[str]] = None,
                  out: Optional[PathLike] = None) -> Sightings:
        """What a link profile reads: every entry of the profile's list field in
        the answers named (`people` in `yolo`'s answers, for the `people`
        profile). `keys` narrows the identity keys the profile links on; it
        cannot widen them. `out` also writes it to a file.
        """
        from .. import definitions
        from ..aggregators.entities.linking import mentions_of

        name = profile.removeprefix(definitions.PROFILE_PREFIX)
        known = definitions.load()["profiles"]
        if name not in known:
            raise RecordError(f"no link profile {name!r}; known: "
                              f"{', '.join(sorted(known))}")
        entry = definitions.get("profiles", name)
        field = entry["field"]
        if field == "transcript":
            if answers:
                raise RecordError(f"profile {name!r} links the transcript; it reads "
                                  f"no answers")
            sources = [Source("transcript")]
        else:
            if not answers:
                raise RecordError(
                    f"profile {name!r} links `{field}` entries: name the answers "
                    f"to read them from, e.g. answers=[\"yolo\"]; this record has "
                    f"{', '.join(self.answer_ids()) or 'none'}")
            narrowed = tuple(f"{field}.{k}" for k in keys or ())
            sources = []
            for answer in answers:
                self._known(answer)
                sources.append(Source(self._head(answer), narrowed))
        one = Input(tuple(sources))
        chosen = definitions.selection(name, one)
        mentions = mentions_of(self.context, chosen)
        heard: dict[str, str] = {}
        # The transcript, only when the profile's account reads it and only for
        # the chunks something was seen in.
        if entry.get("transcript") and self.context.transcript is not None:
            for chunk_id in sorted({m.chunk_id for m in mentions}):
                said = (self.context.transcript.text_of(chunk_id) or "").strip()
                if said:
                    heard[str(chunk_id)] = said
        made = Sightings(
            video_id=self.video_id, profile=name, selection=str(one),
            field_name=chosen.field, keys=list(chosen.keys),
            timeline=self.timeline.as_dict(),
            mentions=[{"chunk_id": m.chunk_id, "sampler_id": m.sampler_id,
                       "field": m.field, "index": m.index, "signature": m.signature,
                       "entry": m.entry} for m in mentions],
            transcript=heard)
        if out is not None:
            files.write(Path(out), made)
        return made

    # ------------------------------------------------------------- checking

    def _known(self, answer: str) -> None:
        if self.context.descriptions is None:
            raise RecordError(f"answer {answer!r} asked for, but this record was "
                              f"given no descriptions")
        if answer not in self.answer_ids():
            raise RecordError(f"this record has no answer {answer!r}; it has "
                              f"{', '.join(self.answer_ids()) or 'none'}")

    def _head(self, answer: str) -> str:
        """An answer id as a source that matches it and nothing else: `yolo` is
        sampler `yolo` asked question `yolo`, not every answer to `yolo`."""
        self._known(answer)
        return answer if ":" in answer else f"{answer}:{answer}"

    def _fields(self, answer: str, fields: Sequence[str]) -> tuple[str, ...]:
        if isinstance(fields, str) or not fields:
            raise RecordError(f"answers[{answer!r}] is a list of fields, e.g. "
                              f"[\"summary\", \"actors\"]; `summary` is the prose")
        if "*" in fields:
            raise RecordError(f"answers[{answer!r}]: name the fields rather than `*`")
        return tuple(fields)

    def _check(self, one: Input) -> None:
        """Every field named exists in its question's shape."""
        import re

        from ...video_rag.core import vocabulary
        problems = check([one], vocabulary())
        if problems:
            # `yolo:yolo` is how the answer `yolo` is matched; say `yolo`.
            raise RecordError("; ".join(re.sub(r"\b([a-z][a-z0-9_-]*):\1\b", r"\1", p)
                                        for p in problems))


def record(*, timeline: PathLike, transcript: Optional[PathLike] = None,
           descriptions: Optional[PathLike] = None,
           manifest: Optional[PathLike] = None) -> Record:
    """The documents at these paths, as one record. `timeline` is required; the
    others are optional, and an aggregator that needs one it was not given says
    so. Documents from another video or cut on another grid are refused.
    """
    given = {"timeline": timeline, "transcript": transcript,
             "descriptions": descriptions, "manifest": manifest}
    paths_ = {kind: Path(where) for kind, where in given.items() if where is not None}
    for kind, where in paths_.items():
        if where.is_dir():
            raise RecordError(f"{kind}={str(where)!r} is a folder; name the file, "
                              f"e.g. {where / (kind + '.json')}")
    return Record(context(timeline, descriptions, transcript, manifest), paths_)


__all__ = ["KINDS", "Record", "RecordError", "context", "record"]
