"""The embed component: descriptions + transcript -> `embedded.json`.

Embeds **only what changed**, keyed by a hash of the unit's own text. A re-run
against unchanged documents costs nothing and says so.

The vectors land in `embedded.json` and nowhere else. Putting them into a
database is the pipeline's job -- so embedding does not need a database to be
reachable, and whoever wants the vectors somewhere this project has never
heard of gets a file to read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from falconvar.shared import logs, progress
from falconvar.shared.contracts.documents import (Descriptions, Embedded,
                                                  Produced, Timeline,
                                                  Transcript)
from falconvar.shared.models import embedders as embedders_mod
from falconvar.shared.storage.files import maybe, read, write
from falconvar.shared.contracts.documents import same_video
from . import readable
from . import units as units_mod
# The published type, under the name a caller would annotate: an
# annotation is the contract, and `list[units_mod.Unit]` names an
# alias local to this file.
from .units import Unit
from falconvar.shared.errors import Refused


def encode(descriptions: Optional[Descriptions] = None,
           transcript: Optional[Transcript] = None,
           previous: Optional[Embedded] = None,
           embedder: Optional[str] = None,
           batch: int = 64,
           on_progress: Optional[progress.Reporter] = None) -> list[Unit]:
    """Documents in hand -> units carrying their vectors. Writes nothing.

    Name at least one of the two documents; both is the ordinary case.

    **`previous` is what resume used to need a data root for.** It was an
    `embedded.json` this function could not see: the diff lived one level up,
    in the function that knew the video id, so a caller reaching this one
    re-embedded everything and paid for it. Handed the previous document,
    the diff belongs here, and both ways in get it.

    A vector is comparable only inside the space that made it, so a
    `previous` written by a different embedder carries nothing forward and
    every unit is new work.

    `embedder` is a provider or `provider/model`; None resolves through
    `shared.models.providers` -- FALCONVAR_EMBEDDER, then openai -- exactly as
    `retrieve` resolves it, so the two cannot disagree about the space.
    """
    if batch < 1:
        raise Refused(f"batch must be at least 1, not {batch}")
    from falconvar.shared.models import providers
    providers.require("embed", embedder)
    if descriptions is None and transcript is None:
        raise Refused(
            "nothing to embed -- pass descriptions=, transcript=, or both")

    wanted: list[Unit] = []
    if descriptions is not None:
        wanted += units_mod.from_descriptions(descriptions)
    if transcript is not None:
        wanted += units_mod.from_transcript(transcript)

    built = embedders_mod.build(embedder)

    # What a previous run paid for, and only under the same embedder key.
    hashes: dict[str, str] = {}
    vectors: dict[str, list[float]] = {}
    if previous is not None and previous.embedder == built.key:
        hashes = previous.stored()
        vectors = {f"{previous.video_id}:{u['chunk_id']}:{u['sampler_id']}":
                   u["vector"] for u in previous.units if u.get("vector")}

    changed = [u for u in wanted if hashes.get(u.key) != u.text_hash]

    # The unchanged ones keep the vectors already paid for. Without this the
    # file is rewritten whole every run, so "unchanged" would mean "dropped"
    # -- the document is the store now, and a store that forgets what it was
    # not asked about is worse than one that re-embeds everything.
    for unit in wanted:
        if unit.key in vectors and unit not in changed:
            unit.vector = vectors[unit.key]

    progress.report(on_progress, "embed", len(changed), 0,
                    len(wanted) - len(changed))
    for start in range(0, len(changed), batch):
        window = changed[start:start + batch]
        for unit, vector in zip(window,
                                built.embed([u.content for u in window])):
            unit.vector = vector
        # Per batch, not per unit: one request carries the whole window, so a
        # unit has no completion of its own to report.
        progress.report(on_progress, "embed", len(changed),
                        min(start + batch, len(changed)),
                        len(wanted) - len(changed),
                        f"{start}-{min(start + batch, len(changed))}", window)
    return wanted


def embed(out: str | Path,
          descriptions: Optional[str | Path] = None,
          transcript: Optional[str | Path] = None,
          previous: Optional[str | Path] = None,
          timeline: Optional[str | Path] = None,
          embedder: Optional[str] = None,
          batch: int = 64,
          on_progress: Optional[progress.Reporter] = None) -> Produced:
    """Embed what changed, into `out`. Name at least one input document.

    `encode` plus a read at each end.

    **There is no destination parameter, and there have been two.** It took a
    `sink` it never read, and then an `index_name` naming a vector store to
    upsert into. Both made the paid half of this component depend on
    something being reachable: a database down, or an embedded Qdrant lock
    still held, failed a run whose API calls had already been made. Now it
    writes one file and the pipeline decides where a copy goes.

    `previous` is usually `out` itself -- the file this run is about to
    replace. `timeline` is read for its fingerprint alone, which is what
    tells a later reader whether these units still describe this video.
    """
    if batch < 1:
        raise Refused(f"batch must be at least 1, not {batch}")
    from falconvar.shared.models import providers
    providers.require("embed", embedder)

    described = maybe(descriptions, Descriptions)
    spoken = maybe(transcript, Transcript)
    if described is None and spoken is None:
        raise Refused(
            f"nothing to embed -- no descriptions and no transcript "
            f"(descriptions={descriptions}, transcript={transcript})")

    stored = maybe(previous, Embedded)
    grid = maybe(timeline, Timeline)
    video_id = same_video(descriptions=described, transcript=spoken,
                          previous=stored, timeline=grid)

    with logs.timed("embed", video_id) as done:
        units = encode(described, spoken, stored, embedder, batch, on_progress)
        key = embedders_mod.build(embedder).key
        carried = 0 if stored is None or stored.embedder != key else sum(
            1 for u in units if stored.stored().get(u.key) == u.text_hash)

        # Rewritten whole, so there is no prune: a unit the grid no longer has
        # is simply not in `units`. A table keeps what it was never told to
        # remove, and that is the pipeline's problem, not this one's.
        document = readable.build(video_id, units, key,
                                  "" if grid is None else grid.fingerprint())
        where = write(out, document)
        done(units=len(units), embedded=len(units) - carried,
             unchanged=carried, embedder=key)

    return Produced(
        video_id=video_id, component="embed",
        artifacts={"embedded": where},
        stats={"units": len(units), "embedded": len(units) - carried,
               "unchanged": carried, "embedder": key,
               "samplers": sorted({u.sampler_id for u in units})},
    )


def load(path: str | Path) -> Embedded:
    """Read an `embedded.json` back, typed -- text and vectors together."""
    return read(path, Embedded)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Embed what changed.")
    ap.add_argument("out", help="where to write embedded.json")
    ap.add_argument("--descriptions", default=None,
                    help="path to descriptions.json")
    ap.add_argument("--transcript", default=None, help="path to transcript.json")
    ap.add_argument("--previous", default=None,
                    help="an earlier embedded.json; unchanged vectors are "
                         "carried forward. Usually the same path as `out`")
    ap.add_argument("--timeline", default=None,
                    help="path to timeline.json, for the fingerprint")
    ap.add_argument("--embedder", default=None,
                    help="a provider or provider/model; default FALCONVAR_EMBEDDER, "
                         f"then openai. Known: {', '.join(embedders_mod.available())}")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = embed(args.out, args.descriptions, args.transcript,
                         args.previous, args.timeline, args.embedder,
                         args.batch)
    except (KeyError, ValueError, FileNotFoundError,
            embedders_mod.EmbedderUnavailable) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    print(f"{produced.video_id}   {s['embedder']}")
    print(f"  units        {s['units']}   from {', '.join(s['samplers'])}")
    print(f"  embedded     {s['embedded']}   ({s['unchanged']} unchanged)")
    print()
    for name, where in produced.artifacts.items():
        print(f"  {name:<12} -> {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
