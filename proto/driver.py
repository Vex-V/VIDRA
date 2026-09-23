"""The pipeline: every component in order, over one folder of files.

The components take filepaths and resolve nothing. Something still has to
decide where the files go, in what order they are written, and which of them
this run needs at all -- and that is this module. It is the same driver as
before with the addressing moved into it: where `falconvar/video_rag/driver.py`
passed a video id to eight components that each resolved their own paths, this
asks `media` for a folder and composes every later path inside it.

    video_rag("x.mp4", "data/out", policy="scene", sampler="clip:[text,scene]")

`into` is the whole of the output decision. `media` makes `<into>/<id>/`, the
receipt says which folder it settled on -- a mint under `on_conflict="new"`
means the id is not the one that was asked for -- and `layout()` names every
file inside it from the library's own table.

Order is not decided here either. It falls out of what the chosen policy
depends on, and `boundaries.POLICIES` is that table:

    uniform    nothing;  arithmetic over a duration `media.json` already has
    scene      the picture;  boundaries.evidence decodes and scores it
    vad        a transcript; audio must finish first
    speaker    a transcript; audio must finish first

**What this driver has that the id one did not is the answer to "which file".**
Three things were implicit in addressing by id and are decisions here:

    the folder      `media(source, into)` mints or replaces, and every later
                    path is composed from the receipt's `home` rather than
                    from the id that was asked for
    resume          `describe` read its own last output whenever
                    `resume=True`, and `embed` diffed against an index. Both
                    are `previous=` now, so this passes the earlier document
                    when there is one -- which is the same behaviour, spelled
                    where it can be turned off
    consistency     two documents from different videos in one call is
                    possible under paths and was not under ids. The components
                    refuse it themselves (`_io.same_video`); keeping every
                    file in one folder is what makes it not arise

Per-stage tuning is deliberately absent, as before: a scene threshold, a
sampler's `confidence`, an audio `compute_type`. Those live on the component
that owns them, and a caller who wants them drives the components directly --
every one takes the paths this function composes, in the order it uses them.
"""

from __future__ import annotations

import inspect

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from falconvar.shared import logs, paths
from falconvar.shared.contracts.documents import Produced
from falconvar.shared.errors import UnknownOption
from . import audio, boundaries, cut, describe, embed, media, video
#: Unadvertised machinery, reached through the module it lives in: the
#: component publishes its entry points, and a caller's `on_conflict` is
#: checked against the same table `media` checks it against.
from .media.driver import ON_CONFLICT

#: Every component, in the order it can run.
COMPONENTS = ("media", "audio", "boundaries.evidence", "boundaries",
              "video", "cut", "describe", "embed")

#: The databases a run may also write to. A component never writes to one --
#: it produces a file and stops -- so this is the pipeline's question alone,
#: and `None` is the honest default: files are not a choice.
DATABASES = ("supabase",)


def layout(home: str | Path) -> dict[str, Path]:
    """Every artifact's path inside one video's folder.

    **Read from the library's tables rather than spelled here.** A pipeline
    typing its own filenames would write `raw_transcript.json` where every
    component writes `transcript.raw.json`, and nothing would say so until a
    read failed -- which is the one thing naming-in-the-library buys now that
    resolving has left.

    Directories are in it too: `store` is not a document, but it is a path
    this run decides and `video` and `describe` both need the same one.
    """
    root = Path(home)
    return {**{name: root / filename for name, filename in paths.ARTIFACTS.items()},
            **{name: root / folder for name, folder in paths.DIRECTORIES.items()}}


@dataclass
class Options:
    """The shape of one extraction. Tuning lives on the component CLIs."""

    source: Path
    #: The directory that holds one folder per video. The output decision, and
    #: the only one: everything else is composed inside whatever `media`
    #: settles on under it.
    into: Path
    video_id: Optional[str] = None
    #: What to do when a *different* file wants a taken id. `media`'s, and so
    #: the pipeline's, because `into` is the pipeline's argument: `new` mints
    #: `clip-2`, `replace` deletes the folder, `refuse` raises.
    on_conflict: str = "new"
    policy: str = "uniform"                  # decides who runs first
    use_video: bool = True
    use_audio: bool = True
    sampler: str = "uniform"                 # what to look at
    # Who answers: a provider or `provider/model`. None resolves through
    # `shared.models.providers` when the stage runs -- FALCONVAR_* from .env,
    # then openai -- rather than being captured at import, before .env is read.
    describer: Optional[str] = None          # frames -> answers
    embedder: Optional[str] = None           # text -> vectors
    #: Hand `describe` and `embed` the documents an earlier run wrote, so a
    #: pair or a vector still current is not paid for twice. True is what
    #: addressing by id did unconditionally; False is "describe it all again",
    #: which was previously only reachable by deleting the file.
    resume: bool = True
    # Where a *copy* goes. The files are written either way; this is the
    # pipeline reading each one back and handing it to `storage/supabase.py`.
    database: Optional[str] = None


@dataclass
class Run:
    #: The id the run settled on, read off the first step rather than off the
    #: request -- `media` may mint a new one when a different file wants a
    #: taken id, and everything after it is addressed by the answer.
    video_id: str
    #: The folder every artifact went into, likewise off the receipt. Under
    #: id addressing a caller could re-derive this from the id and the data
    #: root; here it is the run's own answer and there is nowhere else to get
    #: it, so it is reported.
    home: Path
    #: Every component's receipt, in the order they ran.
    steps: list[Produced] = field(default_factory=list)
    #: `{component: why}`. A reason, not a flag: a component that did not run
    #: is only useful beside why it did not -- a file with no soundtrack and
    #: a policy that needs no precursor are different absences.
    skipped: dict[str, str] = field(default_factory=dict)
    #: What was reported and continued past -- a database write that failed
    #: while the files landed. Empty on a run that named no database. Reported
    #: rather than raised, and *listed* rather than counted: "3 writes failed"
    #: cannot say whether the same table failed three times.
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
    """Contradictions only, as messages. Empty means valid.

    Just the ones no single component can see -- a policy derived from a stream
    this run is not reading. Everything else is checked by the component that
    owns it, where the message can be specific. Returned rather than raised so
    a CLI prints them all at once and an API can answer 422 with the list.
    """
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

    # `into` is a directory that will be created, so the check is that nothing
    # is already *in the way* -- a file at that path makes every later write
    # fail, one component at a time, after the first has run.
    into = Path(options.into)
    if into.exists() and not into.is_dir():
        problems.append(f"{into} is a file; `into` is the directory that holds "
                        f"one folder per video")

    # Both halves of every `name:question` pair, against the two registries.
    #
    # Checked here because the alternative is where it used to be caught: in
    # `describe`, after ingest has decoded the whole video. `yolo:overvew` was
    # a 202 that ran media, audio, boundaries and a full video pass before
    # failing on the typo -- which is the late failure this function exists to
    # prevent. `question_for` falls back to the scene question, so a spelling
    # nobody checks is a run that completes and answers something nobody asked.
    # `video` still does not import `describe`: this driver is the composition
    # root, and a sampler records its question as an opaque string.
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

    # Every model extraction will call: a known provider, able to do the job,
    # with a model and a key. A missing key discovered by `describe` arrives
    # after the whole video has been decoded. Whether a local server is up is
    # not asked: that would make validation a network call.
    from falconvar.shared.models import providers
    wanted = [("embed", options.embedder)]
    if options.use_video:
        wanted.append(("describe", options.describer))
    for role, spec in wanted:
        problems += providers.problems(role, spec)
    return problems


def export(produced: Produced, database: str) -> list[str]:
    """A component's artifacts into a database. Returns what went wrong.

    **The pipeline's job, never a component's.** A component writes its file
    and stops, so this reads that file back -- from the path in its own
    receipt, which is the whole of what a component now says about where its
    output went -- and hands it to the function named for it in
    `shared/storage/supabase.py`. The re-read is cheap: the document is on
    disk because the next component is about to read it anyway.

    Best-effort, and an *informed* choice rather than a default nobody picked.
    The file has landed and the next component reads the file, so a database
    that is down is a report.

    A directory artifact -- `store` -- is skipped: there is no document to
    read, and `WRITERS` has no entry for it either.
    """
    if database not in DATABASES:
        raise UnknownOption(f"unknown database {database!r}; "
                            f"known: {', '.join(DATABASES)}")
    from falconvar.shared.storage import files, supabase

    problems: list[str] = []
    for artifact, where in sorted(produced.artifacts.items()):
        if supabase.writer_for(artifact) is None or not where:
            continue
        try:
            supabase.write(produced.video_id, artifact,
                           files.read_json(Path(where)))
        except Exception as exc:                          # noqa: BLE001
            message = f"{artifact} -> {database}: {exc}"
            problems.append(message)
            logs.logger(produced.component).warning(
                "%s", message,
                extra={"component": produced.component, "event": "export",
                       "video_id": produced.video_id, "artifact": artifact,
                       "reason": str(exc)[:300]})

    # Provenance for the questions this run asked, at the version it asked
    # them under. `descriptions.model` records the hashes; only these rows can
    # say what a hash *meant*, because editing an instruction loses the old
    # text. Assembled only when a database was named.
    if produced.component == "describe":
        try:
            from .describe.driver import prompt_rows
            document = files.read_json(Path(produced.artifacts["descriptions"]))
            supabase.write_prompts(
                prompt_rows((document.get("model") or {}).get("prompts") or {}))
        except Exception as exc:                          # noqa: BLE001
            problems.append(f"prompts -> {database}: {exc}")
    return problems


def process(options: Options,
            on_step: Optional[Callable[..., None]] = None) -> Run:
    """One extraction, top to bottom. Every step is one component call.

    `on_step` is called twice per component -- once by name before it runs,
    once with its `Produced` after. It may take a **third** argument, and if
    it does it also receives a `Progress` for every unit inside the long
    stages:

        def on_step(component, produced, progress=None):
            if progress:
                print(f"  {progress.completed}/{progress.total}")

    Whether it takes one is read off the callback rather than announced,
    because every two-argument callback already written must keep working
    untouched.

    A wrapper around `_run`, so the pipeline is one INFO in and one INFO out,
    and a failure part way through is an ERROR carrying how far it got.
    """
    with logs.timed("video_rag") as whole:
        return _run(options, whole, on_step)


def _run(options: Options, whole: Any,
         on_step: Optional[Callable[..., None]] = None) -> Run:
    problems = validate(options)
    if problems:
        raise ValueError("; ".join(problems))

    say = on_step or (lambda *arguments: None)

    # Arity read once, not per call. A callback taking *args counts as wanting
    # progress: it asked for whatever it is given.
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
        """Announce a component before it runs.

        The name is the caller's, because before a component runs it is the
        only word for it there is -- `boundaries.evidence` answers as
        `boundaries.scenes` or `boundaries.audio`, naming which modality
        supplied it, and that is not knowable in advance. Without this, the
        latest thing a poller hears is the last component to *finish*, so the
        longest stage in the run reports as the one before it.
        """
        announce(name, None)

    # 1 · what the file is, and -- the part addressing by id did not have --
    #     which folder everything else lands in.
    starting("media")
    first = media.media(options.source, options.into, options.video_id,
                        options.on_conflict)
    home = Path(first.stats["home"])
    at = layout(home)
    run = Run(video_id=first.video_id, home=home)
    whole(video_id=run.video_id)

    def step(produced: Produced) -> Produced:
        run.steps.append(produced)
        if options.database:
            run.problems += export(produced, options.database)
        announce(produced.component, produced)
        return produced

    step(first)

    def earlier(name: str) -> Optional[Path]:
        """The document an earlier run left, for `previous=`.

        `None` when resume is off or there is nothing there, which is what
        both components read as "describe/embed it all". A path that does not
        exist would do as well -- `_io.maybe` answers None for either -- but
        saying it here keeps the decision in the one place that made it.
        """
        if not options.resume:
            return None
        return at[name] if at[name].exists() else None

    # Whether a file carries a soundtrack is a property of the file, not of the
    # request. The receipt says so -- `skipped` lists the streams it lacks --
    # so this reads the answer rather than opening the container again.
    absent = set(first.skipped)
    use_audio = options.use_audio and "audio" not in absent
    use_video = options.use_video and "video" not in absent
    for name in ("audio", "video"):
        if getattr(options, f"use_{name}") and name in absent:
            run.skipped[name] = f"the file carries no {name} stream"
            if boundaries.POLICIES[options.policy] == name:
                raise ValueError(f"policy {options.policy!r} needs the {name} "
                                 f"stream, and {options.source} has none")

    # 2 · the soundtrack. Before the grid when the policy needs a transcript to
    #     derive one; the order is the dependency, not a rule about modalities.
    if use_audio:
        starting("audio")
        step(audio.audio(at["media"], at["raw_transcript"]))
    else:
        run.skipped.setdefault("audio", "this run is not reading the soundtrack")

    # 3 · boundary evidence, if this policy needs any. Both inputs are handed
    #     over and `POLICIES` decides which is opened -- a scene pass never
    #     reads the transcript, and under `uniform` neither is read for
    #     anything but the receipt's id.
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

    # 4 · THE GRID. `cuts=` only when something wrote one: under `uniform`
    #     nothing did, and pointing the component at a file that is not there
    #     would be a missing input rather than a policy that needs none.
    starting("boundaries")
    step(boundaries.boundaries(at["media"], at["timeline"], options.policy,
                               cuts=at["cuts"] if at["cuts"].exists() else None,
                               ))

    # 5 · the picture, onto that grid. The grid is an input here and is never
    #     edited, which is why nothing needs a Chunker. `store=` is where the
    #     pixels go, and it is the same directory `describe` is pointed at --
    #     the one thing two components must agree on that is not a document.
    if use_video:
        starting("video")
        step(video.video(at["media"], at["timeline"], at["manifest"],
                         store=at["store"], sampler=options.sampler, **ticking))
    else:
        run.skipped.setdefault("video", "this run is not reading the picture")

    # 6 · the transcript, onto the same grid. Cheap: Whisper timestamped every
    #     word, so this can be redone against a different grid for nothing.
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

    # 8 · vectors, from both modalities. Whichever documents this run wrote --
    #     `embed` needs at least one and takes either.
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
              policy: str = "uniform",
              use_video: bool = True,
              use_audio: bool = True,
              sampler: str = "uniform",
              describer: Optional[str] = None,
              embedder: Optional[str] = None,
              resume: bool = True,
              database: Optional[str] = None,
              on_step: Optional[Callable[..., None]] = None) -> Run:
    """The whole extraction, as keyword arguments. `process` with an `Options`.

    The library front door. `Options` is the validated record a form or an API
    posts and `process` is what runs it; this is the spelling a caller writes
    by hand, so the arguments are named and checked at the call rather than
    assembled into a dataclass first.

    `into` is the second positional argument because it is the one decision
    the components no longer make: they are handed paths, and these are the
    paths. Everything a run writes lands under `<into>/<video_id>/`.
    """
    return process(Options(
        source=Path(source), into=Path(into), video_id=video_id,
        on_conflict=on_conflict, policy=policy, use_video=use_video,
        use_audio=use_audio, sampler=sampler, describer=describer,
        embedder=embedder, resume=resume, database=database,
    ), on_step)


def main(argv: Optional[list[str]] = None) -> int:
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
    ap.add_argument("--policy", default="uniform", choices=sorted(boundaries.POLICIES))
    ap.add_argument("--sampler", default="uniform",
                    help="comma-separated; any may carry a question after a "
                         "colon, e.g. `yolo:overview`")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--describer", default=None,
                    help="a provider or provider/model, e.g. ollama/gemma3:4b; "
                         "default FALCONVAR_DESCRIBER, then openai")
    ap.add_argument("--embedder", default=None,
                    help="a provider or provider/model, e.g. local; "
                         "default FALCONVAR_EMBEDDER, then openai")
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
        on_conflict=args.on_conflict, policy=args.policy,
        use_video=not args.no_video, use_audio=not args.no_audio,
        sampler=args.sampler, describer=args.describer, embedder=args.embedder,
        resume=not args.no_resume, database=args.database)
    return report(options, lambda on_step: process(options, on_step), args.json)


def report(options: Options, execute: Callable[[Callable], Run],
           as_json: bool) -> int:
    """Validate, run and print. The CLI's half, kept out of `main`."""
    import json

    problems = validate(options)
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
        print(f"{options.source}   policy={options.policy}   into={options.into}")
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


__all__ = ["COMPONENTS", "DATABASES", "Options", "Run", "export", "layout",
           "main", "process", "report", "validate", "video_rag"]
