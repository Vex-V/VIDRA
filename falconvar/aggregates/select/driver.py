"""The select component: a record + a selection -> what one aggregate reads.

    select("data/out/test", "in/ner.json", "ner", "transcript")
    ner.ner("in/ner.json", "answers/ner.json")

**Choosing what to read is a step of its own**, so every aggregator after it is
a component with one file in. The selection is the `inputs` grammar; what comes
out depends on the aggregator it is for:

    a text aggregator     an `Excerpt`: per chunk, what each source said,
                          with the times already resolved from the grid
    a link profile        `Sightings`: every entry the profile identifies,
                          with the answer it came from, so the cannot-link
                          rule survives the file

A count over the whole record (`stats`, `coverage`, `speakers`) reads no
selection, and asking for one is refused rather than ignored.

**One input per file.** `a,b` is two inputs and two answers -- the pipeline
selects each on its own -- so a selection naming more than one is refused here
rather than silently cut to its first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ...shared import logs
from ...shared.contracts.documents import Excerpt, Produced, Sightings
from ...shared.storage import files
from .. import definitions, kind_of, takes_inputs
from ..base import Context
from ..inputs import Input, parse, read
from ..record import Source, open_record
#: Refusals are the tier's own `FalconvarError`, never a bare `ValueError`:
#: "anything the library refused" has to stay one `except`.
from ..driver import AggregateError


def one_input(aggregator: str, selection: Union[str, Input, None]) -> Input:
    """The single input a selection names, or the aggregator's default."""
    from ..driver import default_selection
    if isinstance(selection, Input):
        return selection
    text = default_selection(aggregator) if selection is None else selection
    parsed = parse(text)
    if len(parsed) != 1:
        raise AggregateError(f"{text!r} is {len(parsed)} inputs; select one at a time "
                         f"-- each makes its own answer")
    return parsed[0]


def pick(record: Context, aggregator: str,
         selection: Union[str, Input, None] = None) -> Union[Excerpt, Sightings]:
    """What `aggregator` reads from `record`, under `selection` (its default
    when None). Reads and writes nothing."""
    if not takes_inputs(aggregator):
        raise AggregateError(f"{aggregator} counts the whole record; it reads no "
                         f"selection -- hand it the record itself")
    one = one_input(aggregator, selection)
    grid = record.timeline.as_dict()

    if kind_of(aggregator) != "link":
        taken = read(record, one)
        return Excerpt(
            video_id=record.video_id, selection=str(one), timeline=grid,
            rows=[{"chunk_id": r.chunk_id, "start_ts": r.start, "end_ts": r.end,
                   "parts": [list(part) for part in r.parts]} for r in taken.rows],
            answers=list(taken.answers))

    from ..entities.linking import mentions_of
    profile = definitions.locate(aggregator)[1]
    chosen = definitions.selection(profile, one)
    mentions = mentions_of(record, chosen)
    heard: dict[str, str] = {}
    # Only when the profile's account reads what was said, and only for the
    # chunks someone was seen in: the file carries what the run needs.
    if definitions.get("profiles", profile).get("transcript") and record.transcript:
        for chunk_id in sorted({m.chunk_id for m in mentions}):
            said = (record.transcript.text_of(chunk_id) or "").strip()
            if said:
                heard[str(chunk_id)] = said
    return Sightings(
        video_id=record.video_id, profile=profile, selection=str(one),
        field_name=chosen.field, keys=list(chosen.keys), timeline=grid,
        mentions=[{"chunk_id": m.chunk_id, "sampler_id": m.sampler_id,
                   "field": m.field, "index": m.index, "signature": m.signature,
                   "entry": m.entry} for m in mentions],
        transcript=heard)


def select(source: Source, out: str | Path, aggregator: str,
           selection: Optional[str] = None) -> Produced:
    """`pick` with a read at each end: the record at `source` (a folder or a
    mapping of documents), the excerpt or sightings to `out`."""
    record = open_record(source)
    with logs.timed("select", record.video_id) as done:
        picked = pick(record, aggregator, selection)
        where = files.write(Path(out), picked)
        count = len(picked.rows) if isinstance(picked, Excerpt) else len(picked.mentions)
        done(aggregator=aggregator, count=count)
    kind = "excerpt" if isinstance(picked, Excerpt) else "sightings"
    return Produced(video_id=record.video_id, component="select",
                    artifacts={kind: where},
                    stats={"for": aggregator, "selection": picked.selection,
                           kind: count, "empty": count == 0},
                    skipped=["nothing selected"] if count == 0 else [])


def load(path: str | Path) -> Union[Excerpt, Sightings]:
    """An excerpt or sightings file, typed as whichever it is."""
    from ..driver import load_input
    return load_input(path)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    from ..driver import report

    ap = argparse.ArgumentParser(
        description="Take what one aggregator reads out of a record, as a file.")
    ap.add_argument("source", help="a video's folder (or a combination's)")
    ap.add_argument("out", help="where to write the excerpt or sightings")
    ap.add_argument("aggregator", help="who it is for, e.g. ner, summary, entities:people")
    ap.add_argument("--selection", default=None,
                    help="one input in the inputs grammar, e.g. transcript+clip:activity; "
                         "default the aggregator's own")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        produced = select(args.source, args.out, args.aggregator, args.selection)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1
    return report(produced, args.json)


__all__ = ["load", "one_input", "pick", "select"]
