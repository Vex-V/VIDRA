"""The video_rag pipeline: every component in order, over one folder of files.

    video_rag("x.mp4", "data/out", policy="scene", sampler="clip:[text,scene]")

`media` makes `<into>/<id>/` and every later path is composed inside it by
`layout()`. The run order follows from the policy (`boundaries.POLICIES`):

    uniform    nothing runs first
    scene      boundaries.evidence decodes the picture first
    vad        audio runs first
    speaker    audio runs first

Per-stage tuning (a scene threshold, a sampler's `confidence`) lives on the
components; call them directly to set it. What `aggregates` asks of this tier,
`vocabulary()`, is in `core`: it is the same for every pipeline.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from vidra.shared.config.lookup import bound, signature_without
from vidra.shared.config.paths import layout
from vidra.shared.contracts.documents import Produced
from vidra.shared.models.roles import Models, given
from vidra.shared.reporting import logs
from vidra.shared.reporting.errors import Refused
from vidra.shared.storage.database import Database, as_database
from ..core.export import export
from . import audio, boundaries, cut, describe, embed, glance, media, video
#: The conflict rules `media` accepts.
from .media.driver import ON_CONFLICT


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
    #: What to look at: a spec string, or a list of strings and Sampler objects.
    sampler: str | Sequence[Any] = "uniform"
    #: Who answers: the vlm, the embedder and the visual_embedder (frames ->
    #: vectors, `glance`). None, or a role left None, is that role's default.
    models: Optional[Models] = None
    #: Ask the vlm about the kept frames. False with a visual_embedder: no VLM.
    describe: bool = True
    #: Reuse what an earlier run described and embedded when it is still current.
    resume: bool = True
    # Also copy every artifact to this database, built: `Supabase()`, `Folder(...)`.
    database: Optional[Database] = None


@dataclass
class Run:
    #: The id the run settled on (`media` may mint a new one).
    video_id: str
    #: The folder every artifact went into.
    folder: Path
    #: Every component's receipt, in the order they ran.
    steps: list[Produced] = field(default_factory=list)
    #: `{component: why}` for each component that did not run.
    skipped: dict[str, str] = field(default_factory=dict)
    #: Database writes that failed while the files were written.
    problems: list[str] = field(default_factory=list)

    def artifacts(self) -> dict[str, str]:
        """What is actually in the folder, by artifact name."""
        return {name: str(where) for name, where in sorted(layout(self.folder).items())
                if where.exists()}

    def as_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id,
                "folder": str(self.folder),
                "steps": [s.as_dict() for s in self.steps],
                "skipped": self.skipped,
                "problems": self.problems,
                "artifacts": self.artifacts()}


def problems_of(options: Options) -> list[str]:
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
    try:
        models = given(options.models)
    except Refused as exc:
        return problems + [str(exc)]
    if not options.use_video and models.visual_embedder is not None:
        problems.append("a visual_embedder embeds frames, which this run is not "
                        "reading")
    if not options.use_video and not options.describe:
        problems.append("describe=False changes nothing when the picture is not read")
    if options.use_video and not options.describe and models.visual_embedder is None:
        problems.append("describe=False and no visual_embedder: the frames would be "
                        "kept and nothing made of them; pass a visual_embedder")
    from ..core.checks import run_problems
    roles: dict[str, Any] = {}
    # The text embedder only when there may be text: answers or a transcript.
    if options.use_audio or (options.use_video and options.describe):
        roles["embedder"] = models.embedder
    if options.use_video and options.describe:
        roles["vlm"] = models.vlm
    if options.use_video and models.visual_embedder is not None:
        roles["visual_embedder"] = models.visual_embedder
    problems += run_problems(options.into, options.database,
                             options.sampler if options.use_video else None, roles)
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
    problems = problems_of(options)
    if problems:
        raise Refused("; ".join(problems))
    # Built once, before the first step.
    database = as_database(options.database)
    models = given(options.models)

    say = on_step or (lambda *_: None)

    # A callback taking a third argument also gets every unit's progress.
    ticking = ({"on_progress": lambda event: say(event.component, None, event)}
               if _takes_progress(on_step) else {})

    # 1 · the file, and the folder everything else lands in
    say("media", None)
    first = media.media(options.source, options.into, options.video_id,
                        options.on_conflict, options.name, options.recorded_at)
    folder = Path(first.stats["folder"])
    at = layout(folder)
    run = Run(video_id=first.video_id, folder=folder)
    whole(video_id=run.video_id)

    def step(produced: Produced) -> Produced:
        run.steps.append(produced)
        if database is not None:
            run.problems += export(produced, database)
        say(produced.component, produced)
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
        say("audio", None)
        step(audio.audio(at["media"], at["raw_transcript"]))
    else:
        run.skipped.setdefault("audio", "this run is not reading the soundtrack")

    # 3 · boundary evidence, if this policy needs any
    say("boundaries.evidence", None)
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
    say("boundaries", None)
    step(boundaries.boundaries(at["media"], at["timeline"], options.policy,
                               cuts=at["cuts"] if at["cuts"].exists() else None))

    # 5 · the picture, onto the grid. `store` is where frames go; describe reads it.
    if use_video:
        say("video", None)
        step(video.video(at["media"], at["timeline"], at["manifest"],
                         store=at["store"], sampler=options.sampler, **ticking))
    else:
        run.skipped.setdefault("video", "this run is not reading the picture")

    # 6 · the transcript, onto the grid
    if use_audio:
        say("cut", None)
        step(cut.cut(at["timeline"], at["raw_transcript"], at["transcript"]))

    # 7 · one answer per (chunk, sampler)
    described = use_video and options.describe
    if described:
        say("describe", None)
        step(describe.describe(at["manifest"], at["timeline"], at["store"],
                               at["descriptions"],
                               previous=earlier("descriptions"),
                               vlm=models.vlm, **ticking))
    elif use_video:
        run.skipped["describe"] = "describe=False: the frames are embedded, not described"

    # 7b · one vector per (chunk, sampler run), from the frames alone
    if use_video and models.visual_embedder is not None:
        say("glance", None)
        step(glance.glance(at["manifest"], at["timeline"], at["store"],
                           at["glances"], previous=earlier("glances"),
                           visual_embedder=models.visual_embedder, **ticking))

    # 8 · vectors, from whichever documents this run wrote
    if described or use_audio:
        say("embed", None)
        step(embed.embed(at["embedded"],
                         descriptions=at["descriptions"] if described else None,
                         transcript=at["transcript"] if use_audio else None,
                         previous=earlier("embedded"),
                         timeline=at["timeline"],
                         embedder=models.embedder, **ticking))
    else:
        run.skipped["embed"] = "no text to embed: nothing described, no soundtrack"

    from ..core.sampling.specs import spec_text
    whole(policy=options.policy, sampler=spec_text(options.sampler),
          steps=len(run.steps), skipped=len(run.skipped))
    return run


def _takes_progress(on_step: Optional[Callable[..., None]]) -> bool:
    """Whether a step callback takes a third argument, a `Progress`, read off
    its signature."""
    if on_step is None:
        return False
    try:
        parameters = inspect.signature(on_step).parameters
    except (TypeError, ValueError):              # a builtin, or a C callable
        return False
    return (len(parameters) >= 3
            or any(p.kind is inspect.Parameter.VAR_POSITIONAL
                   for p in parameters.values()))


def video_rag(source: str | Path,
              into: str | Path,
              video_id: Optional[str] = None,
              on_conflict: str = "new",
              name: Optional[str] = None,
              recorded_at: Optional[str | datetime] = None,
              policy: str = "uniform",
              use_video: bool = True,
              use_audio: bool = True,
              sampler: str | Sequence[Any] = "uniform",
              models: Optional[Models] = None,
              describe: bool = True,
              resume: bool = True,
              database: Optional[Database] = None,
              on_step: Optional[Callable[..., None]] = None) -> Run:
    """The whole extraction: every component in order, under one folder.

    Everything lands under `<into>/<video_id>/`. `models` carries the vlm, the
    embedder and the visual_embedder (None is every default); `database` is a
    built `Database`. `name` and `recorded_at` override the video's
    filename and recording time. A `visual_embedder` also embeds the kept
    frames (`glance`); `describe=False` with one asks no vlm at all.
    """
    given = {k: v for k, v in locals().items() if k != "on_step"}
    return process(_options(given), on_step)


def _options(given: dict[str, Any]) -> Options:
    return Options(**{**given, "source": Path(given["source"]),
                      "into": Path(given["into"])})


def validate(*args: Any, **kwargs: Any) -> list[str]:
    """Every problem `video_rag(...)` would refuse with these same arguments, as
    messages; empty means it would start. Nothing runs."""
    return problems_of(_options(bound(video_rag, args, kwargs, drop=("on_step",))))


validate.__signature__ = signature_without(video_rag, ("on_step",), "list[str]")


__all__ = ["Run", "export", "layout", "validate", "video_rag"]
