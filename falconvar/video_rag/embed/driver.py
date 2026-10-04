"""The embed component: descriptions + transcript -> `embedded.json`.

Embeds only what changed, keyed by a hash of each unit's text. The vectors
are written to `embedded.json`; copying them to a database is the pipeline's
job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from falconvar.shared.reporting import logs, progress
from falconvar.shared.contracts.documents import (Descriptions, Embedded,
                                                  Produced, Timeline,
                                                  Transcript)
from falconvar.shared.models import embedders as embedders_mod
from falconvar.shared.storage.files import maybe, read, write
from falconvar.shared.contracts.documents import same_video
from . import readable
from . import units as units_mod
# The published type.
from .units import Unit
from falconvar.shared.reporting.errors import Refused


def encode(descriptions: Optional[Descriptions] = None,
           transcript: Optional[Transcript] = None,
           previous: Optional[Embedded] = None,
           embedder: Optional[str] = None,
           batch: int = 64,
           on_progress: Optional[progress.Reporter] = None) -> list[Unit]:
    """Documents in hand -> units carrying their vectors. Writes nothing.

    Name at least one of the two documents. `previous` is an earlier
    `Embedded`: units whose text is unchanged under the same embedder keep their
    vectors. `embedder` is a provider or `provider/model`; None resolves to
    FALCONVAR_EMBEDDER, then openai, as `retrieve` does.
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

    # Hashes from a previous run, only under the same embedder.
    hashes: dict[str, str] = {}
    vectors: dict[str, list[float]] = {}
    if previous is not None and previous.embedder == built.key:
        hashes = previous.stored()
        vectors = {f"{previous.video_id}:{u['chunk_id']}:{u['sampler_id']}":
                   u["vector"] for u in previous.units if u.get("vector")}

    changed = [u for u in wanted if hashes.get(u.key) != u.text_hash]

    # Unchanged units keep their stored vectors.
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
        # Reported per batch.
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
    """Embed what changed, into `out`. Name at least one input document. `encode`
    plus a read at each end. `previous` is usually `out` itself; `timeline` is
    read for its fingerprint.
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

        # Rewritten whole: units the grid no longer has are simply absent.
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
    """Read an `embedded.json` back, typed: text and vectors."""
    return read(path, Embedded)


def main(argv: Optional[list[str]] = None) -> int:
    from falconvar.shared.config import env
    env.load()        # an entry point reads .env; the library never does
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
