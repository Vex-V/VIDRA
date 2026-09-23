"""The embed component: descriptions + transcript -> vectors.

Embeds **only what changed**, keyed by a hash of the unit's own text. A re-run
against unchanged documents costs nothing and says so.

The vectors land in `embedded.json` and nowhere else. Putting them into a
database is the pipeline's job, through `shared/storage/supabase.py` -- so
embedding does not need a database to be reachable, and whoever wants the
vectors somewhere this project has never heard of gets a file to read.
"""

from __future__ import annotations

from typing import Optional

from ...shared import logs, paths, progress
from ...shared.contracts.documents import (Descriptions, Produced,
                                           Transcript)
from ...shared.models import embedders as embedders_mod
from . import readable
from . import units as units_mod
# The published type, under the name a caller would annotate: an
# annotation is the contract, and `list[units_mod.Unit]` names an
# alias local to this file.
from .units import Unit


def collect(video_id: str) -> list[units_mod.Unit]:
    """Every embeddable unit this video has, from both modalities."""
    out: list[units_mod.Unit] = []
    if paths.exists(video_id, "descriptions"):
        from ..describe import load as load_descriptions
        out += units_mod.from_descriptions(load_descriptions(video_id))
    if paths.exists(video_id, "transcript"):
        from ..cut import load as load_transcript
        out += units_mod.from_transcript(load_transcript(video_id))
    return out


def encode(descriptions: Optional[Descriptions] = None,
           transcript: Optional[Transcript] = None,
           embedder: Optional[str] = None,
           batch: int = 64,
           on_progress: Optional[progress.Reporter] = None) -> list[Unit]:
    """Documents in hand -> units carrying their vectors. Writes nothing.

    Name at least one of the two documents; both is the ordinary case, and is
    what `run` passes when the video has both modalities.

    **This does not resume, and `run` does.** Resume needs to know what was
    already paid for, which is a fact about a stored `embedded.json` -- and
    there is none here, only the documents handed in. Every unit is embedded,
    and a caller re-embedding unchanged text pays for it. `run` reads the
    previous document and carries its vectors forward instead; collapsing the
    two would mean either this one reading a data root or that one losing its
    resume.

    `embedder` is a provider or `provider/model`; None resolves through
    `shared.models.providers` -- FALCONVAR_EMBEDDER, then openai -- exactly as
    `retrieve` resolves it, so the two cannot disagree about the space.
    """
    if batch < 1:
        raise ValueError(f"batch must be at least 1, not {batch}")
    from ...shared.models import providers
    providers.require("embed", embedder)
    if descriptions is None and transcript is None:
        raise ValueError(
            "nothing to embed -- pass descriptions=, transcript=, or both")

    wanted: list[Unit] = []
    if descriptions is not None:
        wanted += units_mod.from_descriptions(descriptions)
    if transcript is not None:
        wanted += units_mod.from_transcript(transcript)

    built = embedders_mod.build(embedder)
    progress.report(on_progress, "embed", len(wanted), 0)
    for start in range(0, len(wanted), batch):
        window = wanted[start:start + batch]
        for unit, vector in zip(window, built.embed([u.content for u in window])):
            unit.vector = vector
        # Per batch, not per unit: one request carries the whole window, so a
        # unit has no completion of its own to report.
        progress.report(on_progress, "embed", len(wanted),
                        min(start + batch, len(wanted)), 0,
                        f"{start}-{min(start + batch, len(wanted))}", window)
    return wanted


def embed(video_id: str, embedder: Optional[str] = None,
          batch: int = 64,
          on_progress: Optional[progress.Reporter] = None) -> Produced:
    """Embed what changed, into `embedded.json`.

    `embedder` is a provider or `provider/model`; None resolves through
    `shared.models.providers` -- FALCONVAR_EMBEDDER, then openai -- exactly as
    `retrieve` resolves it, so the two cannot disagree about the space.

    **There is no destination parameter, and there have been two.** It took a
    `sink` it never read, and then an `index_name` naming a vector store to
    upsert into. Both made the paid half of this component depend on something
    being reachable: a database down, or an embedded Qdrant lock still held,
    failed a run whose API calls had already been made. Now it writes one file
    and the pipeline decides where a copy goes -- which is also what lets
    anyone put these vectors in a store this project has never heard of.

    Resume survived that move. It used to ask each index for `stored_hashes()`;
    it now reads the previous `embedded.json` and carries unchanged vectors
    forward, which is the same diff against a document rather than a server.
    """
    # Arguments first: neither of these should cost an embedder build. `batch`
    # was left to `range()`, which answers "arg 3 must not be zero" -- the
    # interpreter talking about a loop the caller cannot see.
    if batch < 1:
        raise ValueError(f"batch must be at least 1, not {batch}")
    from ...shared.models import providers
    providers.require("embed", embedder)

    built = embedders_mod.build(embedder)
    with logs.timed("embed", video_id) as done:
        produced = _embed(built, video_id, batch, on_progress)
        done(units=produced.stats["units"],
             embedded=produced.stats["embedded"],
             unchanged=produced.stats["unchanged"],
             embedder=produced.stats["embedder"])
    return produced


#: The uniform name every component also answers to: what a dispatch
#: table calls and what a form introspects. The same function object.
#: See `media/driver.py`.
run = embed


def _embed(built, video_id: str, batch: int,
           on_progress: Optional[progress.Reporter] = None) -> Produced:

    wanted = collect(video_id)
    if not wanted:
        raise FileNotFoundError(
            f"{video_id}: nothing to embed -- no descriptions and no transcript")

    # What the previous run paid for, and only if it was the same embedder: a
    # vector is comparable only inside the space that made it, so a switched
    # model has nothing to carry forward and every unit is new work. An absent
    # or unreadable document is simply an empty diff.
    previous: dict[str, str] = {}
    vectors: dict[str, list[float]] = {}
    try:
        stored = readable.load(video_id)
        if stored.embedder == built.key:
            previous = stored.stored()
            vectors = {f"{stored.video_id}:{u['chunk_id']}:{u['sampler_id']}":
                       u["vector"] for u in stored.units if u.get("vector")}
    except (FileNotFoundError, KeyError, ValueError):
        pass

    changed = [u for u in wanted if previous.get(u.key) != u.text_hash]

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
        for unit, vector in zip(window, built.embed([u.content for u in window])):
            unit.vector = vector
        progress.report(on_progress, "embed", len(changed),
                        min(start + batch, len(changed)),
                        len(wanted) - len(changed),
                        f"{start}-{min(start + batch, len(changed))}", window)

    # Free -- the units are in hand, and the grid they were built on is what
    # tells a reader whether they still describe this video.
    from ..boundaries import load as load_timeline
    try:
        fingerprint = load_timeline(video_id).fingerprint()
    except Exception:                                    # noqa: BLE001
        fingerprint = ""

    # Rewritten whole, so there is no prune: a unit the grid no longer has is
    # simply not in `wanted`. Postgres needs one and gets it in
    # `supabase.write_embeddings`, because a table keeps what it was never
    # told to remove.
    where = readable.write(video_id, wanted, built.key, fingerprint)

    # No whole-video vector here: that is made from the summary, and
    # `aggregates.index_summary` writes it, through `shared`. Reading it back
    # from `aggregates/summary.json` made this tier depend on the one above --
    # and on a first run the file did not exist yet, because embed runs first.
    return Produced(
        video_id=video_id, component="embed",
        artifacts={"embedded": where},
        stats={"units": len(wanted), "embedded": len(changed),
               "unchanged": len(wanted) - len(changed),
               "embedder": built.key,
               "samplers": sorted({u.sampler_id for u in wanted})},
    )


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Embed what changed.")
    ap.add_argument("video_id")
    ap.add_argument("--embedder", default=None,
                    help="a provider or provider/model; default FALCONVAR_EMBEDDER, "
                         f"then openai. Known: {', '.join(embedders_mod.available())}")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = run(args.video_id, args.embedder, args.batch)
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
