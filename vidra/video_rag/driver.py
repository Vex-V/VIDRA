"""The video_rag pipeline: every component in order, over one folder of files.

    video_rag("x.mp4", "data/out", policy="scene", sampler="clip:[text,scene]")

`media` makes `<into>/<id>/` and every later path is composed inside it by
`layout()`. The run order follows from the policy (`boundaries.POLICIES`):

    uniform    nothing runs first
    scene      boundaries.evidence decodes the picture first
    vad        audio runs first
    speaker    audio runs first

Per-stage tuning (a scene threshold, a sampler's `confidence`) lives on the
components; call them directly to set it. `vocabulary()` at the bottom is
what `aggregates` asks of this tier.
"""

from __future__ import annotations

import inspect

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from vidra.shared.reporting import logs
from vidra.shared.config import paths
from vidra.shared.contracts.documents import Produced
from . import audio, boundaries, cut, describe, embed, media, video
#: The conflict rules `media` accepts.
from .media.driver import ON_CONFLICT
from vidra.shared.reporting.errors import Refused
from vidra.shared.models.roles import Models, keys_of, unpack
from vidra.shared.storage.database import DATABASES as _DATABASES
from vidra.shared.storage.database import Database, as_database

#: The databases a run may name by string.
DATABASES = tuple(_DATABASES)


def layout(home: str | Path) -> dict[str, Path]:
    """Every artifact's path inside one video's folder, from the library's own
    filename table. Includes `store`, the frame directory.
    """
    root = Path(home)
    return {**{name: root / filename for name, filename in paths.ARTIFACTS.items()},
            **{name: root / folder for name, folder in paths.DIRECTORIES.items()}}


@dataclass
class Options:
    """The settings of one extraction."""

    source: Path
    #: The directory that holds one folder per video.
    into: Path
    video_id: Optional[str] = None
    #: When a different file wants a taken id: `new` mints `clip-2`, `replace`
    #: deletes the folder, `refuse` raises.
    on_conflict: str = "new"
    #: The video's name and recording time; None means the filename and the
    #: container's tags (or what an earlier run of this file was given).
    name: Optional[str] = None
    recorded_at: Optional[str | datetime] = None
    policy: str = "uniform"                  # decides who runs first
    use_video: bool = True
    use_audio: bool = True
    sampler: str = "uniform"                 # what to look at
    # A provider or `provider/model`; None resolves when the stage runs.
    describer: Optional[str] = None          # frames -> answers
    embedder: Optional[str] = None           # text -> vectors
    #: Reuse what an earlier run described and embedded when it is still current.
    resume: bool = True
    # Also copy every artifact to this database: a name or a built `Database`.
    database: Optional[str | Database] = None


@dataclass
class Run:
    #: The id the run settled on (`media` may mint a new one).
    video_id: str
    #: The folder every artifact went into.
    home: Path
    #: Every component's receipt, in the order they ran.
    steps: list[Produced] = field(default_factory=list)
    #: `{component: why}` for each component that did not run.
    skipped: dict[str, str] = field(default_factory=dict)
    #: Database writes that failed while the files were written.
    problems: list[str] = field(default_factory=list)

    def artifacts(self) -> dict[str, str]:
        """What is actually in the folder, by artifact name."""
        return {name: str(where) for name, where in sorted(layout(self.home).items())
                if where.exists()}

    def as_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id,
                "home": str(self.home),
                "steps": [s.as_dict() for s in self.steps],
                "skipped": self.skipped,
                "problems": self.problems,
                "artifacts": self.artifacts()}


def validate(options: Options) -> list[str]:
    """Problems no single component can see, as messages; empty means valid."""
    problems: list[str] = []
    if options.policy not in boundaries.POLICIES:
        problems.append(f"policy must be one of {', '.join(boundaries.POLICIES)}")
    else:
        needs = boundaries.POLICIES[options.policy]
        if needs == "video" and not options.use_video:
            problems.append(f"policy {options.policy!r} is found in the picture, "
                            "which this run is not reading")
        if needs == "audio" and not options.use_audio:
            problems.append(f"policy {options.policy!r} is derived from the "
                            "soundtrack, which this run is not reading")
    if not options.use_video and not options.use_audio:
        problems.append("nothing to do: read the picture, the soundtrack, or both")
    if not Path(options.source).exists():
        problems.append(f"{options.source} does not exist")
    if options.on_conflict not in ON_CONFLICT:
        problems.append(f"on_conflict must be one of "
                        f"{', '.join(ON_CONFLICT)}")
    if not (options.database is None or isinstance(options.database, Database)
            or options.database in DATABASES):
        problems.append(f"unknown database {options.database!r}; pass a "
                        f"Database, or one of: {', '.join(DATABASES)}")

    # A file where `into` should be makes every write fail.
    into = Path(options.into)
    if into.exists() and not into.is_dir():
        problems.append(f"{into} is a file; `into` is the directory that holds "
                        f"one folder per video")

    # Both halves of every `name:question` pair, against the sampler registry and
    # the question vocabulary.
    if options.use_video:
        from .describe import prompts
        from .video import samplers as _samplers
        from .video.driver import parse_spec, split_specs
        known_samplers, known_questions = _samplers.available(), prompts.questions()
        for spec in split_specs(options.sampler):
            name, asked = parse_spec(spec)
            if name not in known_samplers:
                problems.append(f"unknown sampler {name!r} in {spec!r}; "
                                f"known: {', '.join(known_samplers)}")
            for question in asked:
                if question not in known_questions:
                    problems.append(f"unknown question {question!r} in {spec!r}; "
                                    f"known: {', '.join(known_questions)}")

    # Every provider the run will call: known, able to serve the role, with a key.
    from vidra.shared.models import providers
    wanted = [("embed", options.embedder)]
    if options.use_video:
        wanted.append(("describe", options.describer))
    for role, spec in wanted:
        problems += providers.problems(role, spec)
    return problems


def export(produced: Produced, database: str | Database) -> list[str]:
    """Write a component's artifacts to a database. Returns what failed.

    Reads each artifact file back from the receipt and hands it to
    `database.write`. Directories are skipped, and a failure is reported rather
    than raised.
    """
    target = as_database(database)
    from vidra.shared.storage import files

    problems: list[str] = []
    for artifact, where in sorted(produced.artifacts.items()):
        if not where or Path(where).is_dir():
            continue
        try:
            target.write(produced.video_id, artifact,
                         files.read_json(Path(where)))
        except Exception as exc:                          # noqa: BLE001
            message = f"{artifact} -> {target.name}: {exc}"
            problems.append(message)
            logs.logger(produced.component).warning(
                "%s", message,
                extra={"component": produced.component, "event": "export",
                       "video_id": produced.video_id, "artifact": artifact,
                       "reason": str(exc)[:300]})

    # The prompt text behind each question hash this run used.
    if produced.component == "describe":
        try:
            from .describe.driver import prompt_rows
            document = files.read_json(Path(produced.artifacts["descriptions"]))
            target.write_prompts(
                prompt_rows((document.get("model") or {}).get("prompts") or {}))
        except Exception as exc:                          # noqa: BLE001
            problems.append(f"prompts -> {target.name}: {exc}")
    return problems


def process(options: Options,
            on_step: Optional[Callable[..., None]] = None) -> Run:
    """One extraction, top to bottom: one component call per step.

    `on_step(component, produced)` is called before each component (with None)
    and after it (with its `Produced`). A callback taking a third argument also
    receives a `Progress` for every unit inside the long stages.
    """
    with logs.timed("video_rag") as whole:
        return _run(options, whole, on_step)


def _run(options: Options, whole: Any,
         on_step: Optional[Callable[..., None]] = None) -> Run:
    problems = validate(options)
    if problems:
        raise Refused("; ".join(problems))
    # Built once, so a missing setting fails before the first step.
    database = as_database(options.database)

    say = on_step or (lambda *arguments: None)

    # Does the callback take a third argument (progress)?
    wants_progress = False
    if on_step is not None:
        try:
            parameters = inspect.signature(on_step).parameters
            wants_progress = (
                len(parameters) >= 3
                or any(p.kind is inspect.Parameter.VAR_POSITIONAL
                       for p in parameters.values()))
        except (TypeError, ValueError):          # a builtin, or a C callable
            wants_progress = False

    def forward(event: Any) -> None:
        say(event.component, None, event)

    def announce(component: str, produced: Optional[Produced]) -> None:
        """Two arguments always; the third is only ever a progress event."""
        say(component, produced)

    ticking = {"on_progress": forward} if wants_progress else {}

    def starting(name: str) -> None:
        """Announce a component by name before it runs."""
        announce(name, None)

    # 1 · the file, and the folder everything else lands in
    starting("media")
    first = media.media(options.source, options.into, options.video_id,
                        options.on_conflict, options.name, options.recorded_at)
    home = Path(first.stats["home"])
    at = layout(home)
    run = Run(video_id=first.video_id, home=home)
    whole(video_id=run.video_id)

    def step(produced: Produced) -> Produced:
        run.steps.append(produced)
        if database is not None:
            run.problems += export(produced, database)
        announce(produced.component, produced)
        return produced

    step(first)

    def earlier(name: str) -> Optional[Path]:
        """The document an earlier run left, for `previous=`; None when resume is off
        or there is none.
        """
        if not options.resume:
            return None
        return at[name] if at[name].exists() else None

    # Streams the file does not carry, from the media receipt.
    absent = set(first.skipped)
    use_audio = options.use_audio and "audio" not in absent
    use_video = options.use_video and "video" not in absent
    for name in ("audio", "video"):
        if getattr(options, f"use_{name}") and name in absent:
            run.skipped[name] = f"the file carries no {name} stream"
            if boundaries.POLICIES[options.policy] == name:
                raise Refused(f"policy {options.policy!r} needs the {name} "
                                 f"stream, and {options.source} has none")

    # 2 · the soundtrack
    if use_audio:
        starting("audio")
        step(audio.audio(at["media"], at["raw_transcript"]))
    else:
        run.skipped.setdefault("audio", "this run is not reading the soundtrack")

    # 3 · boundary evidence, if this policy needs any
    starting("boundaries.evidence")
    found = boundaries.evidence(at["cuts"], options.policy,
                                media=at["media"],
                                raw_transcript=(at["raw_transcript"]
                                                if use_audio else None))
    if "evidence" in found.skipped:
        run.skipped["boundaries.evidence"] = (
            f"policy {options.policy!r} needs none -- it is arithmetic over "
            "the container duration")
    else:
        step(found)

    # 4 · the grid. `cuts=` only when evidence was written.
    starting("boundaries")
    step(boundaries.boundaries(at["media"], at["timeline"], options.policy,
                               cuts=at["cuts"] if at["cuts"].exists() else None,
                               ))

    # 5 · the picture, onto the grid. `store` is where frames go; describe reads it.
    if use_video:
        starting("video")
        step(video.video(at["media"], at["timeline"], at["manifest"],
                         store=at["store"], sampler=options.sampler, **ticking))
    else:
        run.skipped.setdefault("video", "this run is not reading the picture")

    # 6 · the transcript, onto the grid
    if use_audio:
        starting("cut")
        step(cut.cut(at["timeline"], at["raw_transcript"], at["transcript"]))

    # 7 · one answer per (chunk, sampler)
    if use_video:
        starting("describe")
        step(describe.describe(at["manifest"], at["timeline"], at["store"],
                               at["descriptions"],
                               previous=earlier("descriptions"),
                               describer=options.describer, **ticking))

    # 8 · vectors, from whichever documents this run wrote
    starting("embed")
    step(embed.embed(at["embedded"],
                     descriptions=at["descriptions"] if use_video else None,
                     transcript=at["transcript"] if use_audio else None,
                     previous=earlier("embedded"),
                     timeline=at["timeline"],
                     embedder=options.embedder, **ticking))

    whole(policy=options.policy, sampler=options.sampler,
          steps=len(run.steps), skipped=len(run.skipped))
    return run


def video_rag(source: str | Path,
              into: str | Path,
              video_id: Optional[str] = None,
              on_conflict: str = "new",
              name: Optional[str] = None,
              recorded_at: Optional[str | datetime] = None,
              policy: str = "uniform",
              use_video: bool = True,
              use_audio: bool = True,
              sampler: str = "uniform",
              describer: Optional[str] = None,
              embedder: Optional[str] = None,
              resume: bool = True,
              database: Optional[str | Database] = None,
              on_step: Optional[Callable[..., None]] = None,
              models: Optional[Models] = None) -> Run:
    """The whole extraction, as keyword arguments: `process` with an `Options`.

    Everything lands under `<into>/<video_id>/`. `models` carries the describer,
    the embedder and any keys; `database` is a name or a built `Database`.
    `name` and `recorded_at` override the video's filename and recording time.
    """
    roles = unpack(models, describer=describer, embedder=embedder)
    with keys_of(models):
        return process(Options(
            source=Path(source), into=Path(into), video_id=video_id,
            on_conflict=on_conflict, name=name, recorded_at=recorded_at,
            policy=policy, use_video=use_video,
            use_audio=use_audio, sampler=sampler, describer=roles["describer"],
            embedder=roles["embedder"], resume=resume, database=database,
        ), on_step)


# ------------------------------------------- the one thing aggregates asks
def vocabulary() -> dict[str, Any]:
    """What an aggregate's input may name: every sampler, and every question with
    its fields -- `{field: [entry keys]}` for a list of objects, None otherwise.
    """
    from .describe import library
    from .video import samplers as _samplers

    def fields(question: str) -> dict[str, Optional[list[str]]]:
        out: dict[str, Optional[list[str]]] = {}
        for name, spec in (library.shape_of(question).get("fields") or {}).items():
            items = spec.get("items") if isinstance(spec, dict) else None
            nested = items.get("properties") if isinstance(items, dict) else None
            out[name] = list(nested) if nested else None
        return out

    return {"samplers": _samplers.available(),
            "questions": {q: fields(q) for q in library.questions()}}


def main(argv: Optional[list[str]] = None) -> int:
    from vidra.shared.config import env
    env.load()        # an entry point reads .env; the library never does
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract once: media to embed, into a folder you name. "
                    "Per-stage tuning lives on each component's own CLI, e.g. "
                    "`python -m proto.video`.")
    ap.add_argument("source", type=Path)
    ap.add_argument("into", type=Path,
                    help="the directory that holds one folder per video; the "
                         "run writes everything under <into>/<video-id>/")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--on-conflict", default="new", choices=sorted(ON_CONFLICT),
                    help="what to do when a different file wants a taken id")
    ap.add_argument("--name", default=None,
                    help="what to call the video; default the filename")
    ap.add_argument("--recorded-at", default=None,
                    help="when it was recorded, ISO 8601; default the "
                         "container's creation time, if it has one")
    ap.add_argument("--policy", default="uniform", choices=sorted(boundaries.POLICIES))
    ap.add_argument("--sampler", default="uniform",
                    help="comma-separated; any may carry a question after a "
                         "colon, e.g. `yolo:overview`")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--describer", default=None,
                    help="a provider or provider/model, e.g. ollama/gemma3:4b; "
                         "default VIDRA_DESCRIBER, then openai")
    ap.add_argument("--embedder", default=None,
                    help="a provider or provider/model, e.g. local; "
                         "default VIDRA_EMBEDDER, then openai")
    ap.add_argument("--no-resume", action="store_true",
                    help="describe and embed everything again, ignoring what "
                         "an earlier run left in the folder")
    ap.add_argument("--database", default=None,
                    help=f"also write a copy to a database; known: "
                         f"{', '.join(DATABASES)}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    options = Options(
        source=args.source, into=args.into, video_id=args.video_id,
        on_conflict=args.on_conflict, name=args.name,
        recorded_at=args.recorded_at, policy=args.policy,
        use_video=not args.no_video, use_audio=not args.no_audio,
        sampler=args.sampler, describer=args.describer, embedder=args.embedder,
        resume=not args.no_resume, database=args.database)
    return report(options, lambda on_step: process(options, on_step), args.json)


def report(options: Any, execute: Callable[[Callable], Run],
           as_json: bool) -> int:
    """Validate, run and print, for this CLI and `workflow`'s. `options` may be
    `workflow.Options`, which `workflow.validate` has already checked.
    """
    import json

    problems = validate(options) if isinstance(options, Options) else []
    if problems:
        for problem in problems:
            print(f"error: {problem}")
        return 2

    def on_step(component: str, produced: Optional[Produced]) -> None:
        if as_json:
            return
        if produced is None:                       # about to run
            print(f"  {component:<22} ...", flush=True)
            return
        print(f"  {component:<22} -> {', '.join(produced.artifacts) or '-'}")

    if not as_json:
        where = getattr(options, "into", None)
        print(f"{options.source}   policy={options.policy}"
              f"{f'   into={where}' if where else ''}")
    try:
        run = execute(on_step)
    except Exception as exc:                             # noqa: BLE001
        print(f"error: {exc}")
        return 1

    if as_json:
        print(json.dumps(run.as_dict(), indent=2))
        return 0
    for name, why in run.skipped.items():
        print(f"  {name:<22} -- skipped: {why}")
    print(f"\n{run.video_id} -> {run.home}")
    print(f"  artifacts: {', '.join(run.artifacts())}")
    return 0


__all__ = ["DATABASES", "Options", "Run", "export", "layout",
           "main", "process", "report", "validate", "video_rag", "vocabulary"]
