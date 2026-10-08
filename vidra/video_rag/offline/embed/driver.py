"""The embed component: descriptions + transcript -> `embedded.json`.

Embeds only what changed, keyed by a hash of each unit's text. The vectors
are written to `embedded.json`; copying them to a database is the pipeline's
job.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from vidra.shared.reporting import logs, progress
from vidra.shared.contracts.documents import (Descriptions, Embedded,
                                                  Produced, Timeline,
                                                  Transcript)
from vidra.shared.models.base import require
from vidra.shared.models.roles import resolve
from vidra.shared.storage.files import maybe, read, write
from vidra.shared.contracts.documents import same_video
from ...core.embed import readable
from ...core.embed import units as units_mod
# The published type.
from ...core.embed.units import Unit
from vidra.shared.reporting.errors import Refused

if TYPE_CHECKING:
    from vidra.shared.models.base import Embedder


def encode(descriptions: Optional[Descriptions] = None,
           transcript: Optional[Transcript] = None,
           previous: Optional[Embedded] = None,
           embedder: Optional["Embedder"] = None,
           batch: int = 64,
           on_progress: Optional[progress.Reporter] = None) -> list[Unit]:
    """Documents in hand -> units carrying their vectors. Writes nothing.

    Name at least one of the two documents. `previous` is an earlier
    `Embedded`: units whose text is unchanged under the same embedder keep their
    vectors. `embedder` is the model (None is OpenAI's default); a search must
    use the same one.
    """
    if batch < 1:
        raise Refused(f"batch must be at least 1, not {batch}")
    built = resolve("embedder", embedder)
    require("embedder", built)
    if descriptions is None and transcript is None:
        raise Refused(
            "nothing to embed -- pass descriptions=, transcript=, or both")

    wanted: list[Unit] = []
    if descriptions is not None:
        wanted += units_mod.from_descriptions(descriptions)
    if transcript is not None:
        wanted += units_mod.from_transcript(transcript)

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

    def reported(done: int, window: list[Unit]) -> None:
        # Reported per batch.
        progress.report(on_progress, "embed", len(changed), done,
                        len(wanted) - len(changed),
                        f"{done - len(window)}-{done}", window)
    units_mod.embed_all(changed, built, batch, reported)
    return wanted


def embed(out: str | Path,
          descriptions: Optional[str | Path] = None,
          transcript: Optional[str | Path] = None,
          previous: Optional[str | Path] = None,
          timeline: Optional[str | Path] = None,
          embedder: Optional["Embedder"] = None,
          batch: int = 64,
          on_progress: Optional[progress.Reporter] = None) -> Produced:
    """Embed what changed, into `out`. Name at least one input document. `encode`
    plus a read at each end. `previous` is usually `out` itself; `timeline` is
    read for its fingerprint.
    """
    if batch < 1:
        raise Refused(f"batch must be at least 1, not {batch}")
    built = resolve("embedder", embedder)
    require("embedder", built)

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
        units = encode(described, spoken, stored, built, batch, on_progress)
        key = built.key
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
