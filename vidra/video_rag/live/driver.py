"""The live pipeline: a stream in, an answer per kept frame as it happens, and
the same documents as `offline` when the stream ends.

    video_rag_live("tcp://0.0.0.0:9000?listen=1", "data/out", video_id="dock-3",
                   sampler="clip:safety", on_unit=check)

Two threads. The sampling thread decodes, decimates, finds the chunk and runs
the samplers (`observe`). The event loop describes each kept frame as soon as
its context is complete, embeds the answer, appends it to
`observations.jsonl` and hands it to `on_unit`. Between them is a bounded
queue: when the model falls behind, the oldest waiting frame is dropped and
counted.

When the stream ends (or `stop_after_s`, `stop`, or `StopStream` from a
callback), the run is written out as `media.json`, `timeline.json`,
`manifest.json`, `descriptions.json` and `embedded.json`, in `offline`'s
shapes. A chunk's description there is its frames' answers joined in time
order; each answer is kept whole under `observations`.

There is no audio; `record=` keeps the stream for an `offline` run afterwards.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from vidra.shared.config import paths
from vidra.shared.contracts.documents import (Descriptions, Manifest, Media,
                                              Produced, Timeline,
                                              fingerprint_of)
from vidra.shared.contracts.units import render
from vidra.shared.models.roles import Models, given, resolve
from vidra.shared.reporting import logs
from vidra.shared.reporting.errors import Refused
from vidra.shared.storage import files
from vidra.shared.storage.database import Database, as_database
from ..core.describe import prompts
from ..core.frames import FrameStore
from ..core.sampling.decimate import Decimator
from .observe import Ask, Tally, observe
from .source import Source, now

#: The per-frame answers, one JSON object per line, appended as they arrive.
OBSERVATIONS = paths.OBSERVATIONS
#: Where `record=True` keeps the stream, inside the run's folder.
RECORDING = "recording.ts"


class StopStream(Exception):
    """Raise from `on_unit` to end the run cleanly: reading stops, frames already
    kept are still answered, and the documents are written."""


@dataclass
class Options:
    """The settings of one live run."""

    #: Anything PyAV opens: `tcp://0.0.0.0:9000?listen=1`, `rtsp://...`, a file.
    source: str | Path
    #: The directory that holds one folder per video.
    into: Path
    #: None is the file's stem, or `live-<UTC time>` for a stream.
    video_id: Optional[str] = None
    #: What to call the video; None is the id.
    name: Optional[str] = None
    #: A spec string, or a list of strings and Sampler objects.
    sampler: str | Sequence[Any] = "uniform"
    #: The vlm and the embedder; None, or a role left None, is its default.
    models: Optional[Models] = None
    database: Optional[str | Database] = None
    #: The uniform grid: a chunk is `int(media_ts // chunk_s)`.
    chunk_s: float = 20.0
    #: Decimated frames per second of media.
    per_second: float = 1.0
    #: Decimated frames sent with each kept frame: `(before, after)`.
    context: tuple[int, int] = (0, 0)
    #: Kept frames that may wait for the model before the oldest is dropped.
    queue: int = 64
    #: Keep the stream: True for `recording.ts` in the run's folder, or a path.
    record: bool | str | Path = False
    #: Stop once the media clock reaches this many seconds.
    stop_after_s: Optional[float] = None
    #: Set from another thread to end the run as `StopStream` would.
    stop: Optional[threading.Event] = None
    #: How long to wait for a sender, and for the next packet once one is there.
    open_timeout_s: Optional[float] = 60.0
    read_timeout_s: Optional[float] = 10.0


@dataclass
class Observation:
    """One answer about one kept frame, as `on_unit` receives it."""

    video_id: str
    chunk_id: int
    #: `name:question`, or the bare sampler name for its own question.
    sampler_id: str
    sampler: str
    question: str
    #: The kept frame.
    frame_index: int
    media_ts: float
    #: Every frame the model was shown, in time order, and their times.
    frames: list[int]
    frame_ts: list[float]
    description: str
    structured: dict[str, Any]
    #: What was embedded: the description and the structured fields.
    content: str
    vector: Optional[list[float]]
    #: When the kept frame arrived, and when this answer was ready (UTC).
    seen_at: str
    answered_at: str
    #: Seconds from the kept frame's arrival to this answer.
    lag_s: float
    #: True when the stream's clock jumped just before the kept frame.
    gap_before: bool = False
    #: The embedder's key: the space `vector` is in.
    embedder: str = ""

    @property
    def text_hash(self) -> str:
        return fingerprint_of({"content": self.content})

    def as_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id, "chunk_id": self.chunk_id,
                "sampler_id": self.sampler_id, "sampler": self.sampler,
                "question": self.question, "frame_index": self.frame_index,
                "media_ts": round(self.media_ts, 3), "frames": self.frames,
                "frame_ts": [round(t, 3) for t in self.frame_ts],
                "description": self.description, "structured": self.structured,
                "content": self.content, "text_hash": self.text_hash,
                "seen_at": self.seen_at, "answered_at": self.answered_at,
                "lag_s": round(self.lag_s, 3), "gap_before": self.gap_before,
                "embedder": self.embedder,
                "vector": self.vector}


@dataclass
class LiveRun:
    video_id: str
    folder: Path
    source: str
    #: Why reading stopped: `ended`, `stopped`, `stop_after_s`, or `error`.
    ended: str = "ended"
    duration_s: float = 0.0
    chunks: int = 0
    frames_decimated: int = 0
    frames_sampled: int = 0
    #: Answers written, calls that failed, kept frames dropped from the queue.
    observations: int = 0
    #: Answers copied to `database` as they arrived, and those that did not get
    #: there (still in `observations.jsonl`, so they can be sent later).
    written: int = 0
    unwritten: int = 0
    failed: int = 0
    dropped: int = 0
    #: Seconds from a kept frame's arrival to its answer: p50, p95, max.
    lag_s: dict[str, float] = field(default_factory=dict)
    #: `(last_ts, next_ts)` for each jump in the stream's clock.
    gaps: list[tuple[float, float]] = field(default_factory=list)
    recording: Optional[str] = None
    #: The documents written when the stream ended.
    steps: list[Produced] = field(default_factory=list)
    #: Failed model calls and database writes, as messages.
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {**{k: v for k, v in self.__dict__.items() if k not in ("steps", "folder")},
                "folder": str(self.folder),
                "steps": [s.as_dict() for s in self.steps]}


# ------------------------------------------------------------------ checks
def validate(options: Options) -> list[str]:
    """Problems found before the stream is opened, as messages; empty is valid."""
    problems: list[str] = []
    source = (os.fspath(options.source) if isinstance(options.source, os.PathLike)
              else options.source)
    if not isinstance(source, str) or not source.strip():
        problems.append("source is a URL or a path")
    elif "://" not in source and not Path(source).exists():
        problems.append(f"{source} does not exist, and is not a URL")
    if options.video_id is not None:
        try:
            paths.check_id(options.video_id)
        except Exception as exc:                             # noqa: BLE001
            problems.append(str(exc))
    if options.chunk_s <= 0:
        problems.append(f"chunk_s must be positive, not {options.chunk_s}")
    if options.per_second <= 0:
        problems.append(f"per_second must be positive, not {options.per_second}")
    context = options.context
    if (not isinstance(context, (tuple, list)) or len(context) != 2
            or not all(isinstance(n, int) and n >= 0 for n in context)):
        problems.append(f"context is (before, after), two counts of frames >= 0, "
                        f"not {context!r}")
    if options.queue < 1:
        problems.append(f"queue must be at least 1, not {options.queue}")
    if options.stop_after_s is not None and options.stop_after_s <= 0:
        problems.append(f"stop_after_s must be positive, not {options.stop_after_s}")
    try:
        models = given(options.models)
    except Refused as exc:
        return problems + [str(exc)]
    from ..core.checks import run_problems
    problems += run_problems(options.into, options.database, options.sampler,
                             {"vlm": models.vlm, "embedder": models.embedder})
    return problems


# --------------------------------------------------------------------- run
def process(options: Options,
            on_unit: Optional[Callable[[Observation], None]] = None,
            on_step: Optional[Callable[..., None]] = None) -> LiveRun:
    """One live run, until the stream ends or is stopped. `on_unit` receives every
    answer as it is ready; `on_step(component, produced)` is called once the
    stream is open (with None) and for each document written at the end.
    """
    problems = validate(options)
    if problems:
        raise Refused("; ".join(problems))
    if isinstance(options.source, os.PathLike):
        options = replace(options, source=os.fspath(options.source))
    database = as_database(options.database)
    say = on_step or (lambda *_: None)

    from vidra.shared.models.base import require
    from ..core.describe import base as describe_base
    from ..core.sampling.specs import build_samplers, split_specs

    # Everything that can fail is built before the stream is opened.
    built = build_samplers(split_specs(options.sampler),
                           questions=prompts.questions())
    models = given(options.models)
    describer = describe_base.build(models.vlm)
    embedder = resolve("embedder", models.embedder)
    require("embedder", embedder)
    from vidra.shared.models.embedders.local import LocalEmbedder
    if isinstance(embedder, LocalEmbedder):
        # Loaded now, or the first answers wait seconds for the weights. A
        # remote embedder is not called: nothing is sent before the stream.
        embedder.embed(["warm up"])

    video_id = _claim(options)
    folder = Path(options.into) / video_id
    folder.mkdir(parents=True, exist_ok=True)
    store = FrameStore(folder / paths.DIRECTORIES["store"])
    recording = (None if not options.record
                 else folder / RECORDING if options.record is True
                 else Path(options.record))

    with logs.timed("live", video_id) as whole:
        source = Source(options.source, options.open_timeout_s,
                        options.read_timeout_s, record=recording).open()
        run = LiveRun(video_id=video_id, folder=folder, source=options.source,
                      recording=str(recording) if recording else None)
        # The video exists while it streams: no duration yet, the real one at the end.
        run.problems += _media(run, options, source, None, now(), database)
        say("live", None)
        tally = Tally()
        state = _State(folder / OBSERVATIONS)
        stop = _Stop(options.stop)
        try:
            asyncio.run(_serve(options, video_id, source, built, describer,
                               embedder, store, tally, state, stop, on_unit,
                               database))
        finally:
            stop.set()
            source.stop()
            source.close()
            state.close()
            run.ended = ("error" if tally.error is not None or state.error is not None
                         else "stopped" if state.stopped else tally.ended)
            _finish(run, options, source, built, describer, embedder, store,
                    tally, state, database, say)
            whole(observations=run.observations, failed=run.failed,
                  dropped=run.dropped, chunks=run.chunks, ended=run.ended)
    # A callback's own error, or the stream's, after the documents are written.
    if state.error is not None:
        raise state.error
    if tally.error is not None:
        raise tally.error
    return run


class _Stop:
    """Set by the run itself, or by the caller through their own `stop` event,
    which the run reads but never sets."""

    def __init__(self, theirs: Optional[threading.Event]) -> None:
        self.theirs = theirs
        self.ours = threading.Event()

    def set(self) -> None:
        self.ours.set()

    def is_set(self) -> bool:
        return self.ours.is_set() or (self.theirs is not None and self.theirs.is_set())


class _State:
    """What the event loop accumulates: answers, failures, lag."""

    def __init__(self, path: Path) -> None:
        self.file = path.open("a", encoding="utf-8")
        #: Answers without vectors, for the documents written at the end.
        self.answers: list[dict[str, Any]] = []
        self.failed: list[str] = []
        self.dropped: list[dict[str, Any]] = []
        self.lags: list[float] = []
        self.questions: set[str] = set()
        #: Rows waiting for the database, how many reached it, how many did not,
        #: and the first 20 reasons.
        self.unsent: list[dict[str, Any]] = []
        self.written = 0
        self.unwritten = 0
        self.write_errors: list[str] = []
        #: True once a failure was counted past the reasons kept.
        self.errors_cut = False
        self.stopped = False
        self.error: Optional[BaseException] = None

    def write(self, observation: Observation) -> None:
        self.file.write(json.dumps(observation.as_dict()) + "\n")
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


class _Inbox:
    """The bounded queue between the threads. Full, it drops its oldest entry."""

    def __init__(self, size: int, dropped: list[dict[str, Any]]) -> None:
        self.size = size
        self.items: deque[Ask] = deque()
        self.dropped = dropped
        self.closed = False
        self.ready = asyncio.Event()

    def put(self, ask: Ask) -> None:
        if len(self.items) >= self.size:
            old = self.items.popleft()
            self.dropped.append({"chunk_id": old.chunk_id, "sampler": old.run_id,
                                 "frame_index": old.frame.index,
                                 "media_ts": round(old.frame.media_ts, 3)})
        self.items.append(ask)
        self.ready.set()

    def close(self) -> None:
        self.closed = True
        self.ready.set()

    async def get(self) -> Optional[Ask]:
        while True:
            if self.items:
                return self.items.popleft()
            if self.closed:
                return None
            self.ready.clear()
            await self.ready.wait()


async def _serve(options: Options, video_id: str, source: Source, built: list,
                 describer: Any, embedder: Any, store: FrameStore, tally: Tally,
                 state: _State, stop: "_Stop",
                 on_unit: Optional[Callable[[Observation], None]],
                 database: Optional[Database] = None) -> None:
    loop = asyncio.get_running_loop()
    inbox = _Inbox(options.queue, state.dropped)
    # One writer: an answer goes out as soon as it is ready, and when the
    # database is slower than the answers, what queued meanwhile goes in one call.
    copying = database is not None and database.implements("write_observations")
    pending = asyncio.Event()
    finished = False

    async def writer() -> None:
        while True:
            await pending.wait()
            pending.clear()
            while state.unsent:
                batch = list(state.unsent)
                state.unsent.clear()
                try:
                    await asyncio.to_thread(database.write_observations,
                                            video_id, batch)
                    state.written += len(batch)
                except Exception as exc:                     # noqa: BLE001
                    state.unwritten += len(batch)
                    if len(state.write_errors) >= 20:
                        state.errors_cut = True
                    else:
                        message = (f"{len(batch)} observations -> {database.name}: "
                                   f"{exc}")
                        state.write_errors.append(message)
                        logs.logger("live").warning("%s", message, extra={
                            "component": "live", "event": "export",
                            "video_id": video_id, "reason": str(exc)[:300]})
            if finished:
                return

    def post(ask: Ask) -> None:
        loop.call_soon_threadsafe(inbox.put, ask)

    def sample() -> None:
        try:
            observe(source, built, Decimator(options.per_second), options.chunk_s,
                    tuple(options.context), store, post, tally, stop,
                    options.stop_after_s)
        except BaseException as exc:                         # noqa: BLE001
            tally.error = exc
        finally:
            loop.call_soon_threadsafe(inbox.close)

    async def answer(ask: Ask, answer_id: str, question: str) -> None:
        images = ask.images()
        context = {"video_id": video_id, "chunk_id": ask.chunk_id,
                   "start_ts": images[0].media_ts, "end_ts": images[-1].media_ts,
                   "sampler": answer_id, "question": question,
                   "sampler_config": ask.config}
        state.questions.add(question)
        try:
            said = await describer.describe(images, context)
            content = render(said.summary, said.fields)
            vector = (await asyncio.to_thread(embedder.embed, [content]))[0]
        except Exception as exc:                             # noqa: BLE001
            message = f"chunk {ask.chunk_id} frame {ask.frame.index} {answer_id}: {exc}"
            state.failed.append(message)
            logs.logger("live").warning("%s", message, extra={
                "component": "live", "event": "fail", "video_id": video_id,
                "reason": str(exc)[:300]})
            return
        lag = time.perf_counter() - ask.arrival.seen
        # Each frame once, however many views were made from it.
        shown = {f.index: f.media_ts for f in images}
        observation = Observation(
            video_id=video_id, chunk_id=ask.chunk_id, sampler_id=answer_id,
            sampler=ask.name, question=question, frame_index=ask.frame.index,
            media_ts=ask.frame.media_ts, frames=list(shown),
            frame_ts=list(shown.values()), description=said.summary,
            structured=said.fields, content=content, vector=list(vector),
            seen_at=ask.arrival.seen_at, answered_at=now(), lag_s=lag,
            gap_before=ask.arrival.gap_before, embedder=embedder.key)
        state.write(observation)
        if copying:
            state.unsent.append(observation.as_dict())
            pending.set()
        state.lags.append(lag)
        state.answers.append({k: v for k, v in observation.as_dict().items()
                              if k != "vector"})
        if on_unit is not None and state.error is None:
            try:
                on_unit(observation)
            except StopStream:
                state.stopped = True
                stop.set()
            except Exception as exc:                         # noqa: BLE001
                state.error = exc
                stop.set()

    async def worker() -> None:
        while (ask := await inbox.get()) is not None:
            if state.error is not None:
                continue                  # stopping on an error: drain, ask nothing
            await asyncio.gather(*(answer(ask, a, q) for a, q in ask.asked))

    thread = threading.Thread(target=sample, name=f"live-{video_id}", daemon=True)
    thread.start()
    copier = asyncio.create_task(writer()) if copying else None
    try:
        await asyncio.gather(*(worker() for _ in range(max(1, describer.concurrency))))
    finally:
        stop.set()
        if copier is not None:
            finished = True
            pending.set()                 # the last answers, then the writer ends
            await copier
        await asyncio.to_thread(thread.join, 30)


# ----------------------------------------------------------- the documents
def _finish(run: LiveRun, options: Options, source: Source, built: list,
            describer: Any, embedder: Any, store: FrameStore, tally: Tally,
            state: _State, database: Optional[Database],
            say: Callable[..., None]) -> None:
    """Write the run as `offline`'s documents, and copy them to `database`."""
    from ..core.embed import readable, units
    from ..core.export import export

    at = paths.layout(run.folder)
    duration = source.duration_s
    count = (max(tally.chunks) + 1) if tally.chunks else 0
    spans = [(i * options.chunk_s, min((i + 1) * options.chunk_s, duration))
             for i in range(count)]
    run.duration_s = round(duration, 3)
    run.chunks = count
    run.frames_decimated, run.frames_sampled = tally.decimated, tally.sampled
    run.observations = len(state.answers)
    run.failed, run.dropped = len(state.failed), len(state.dropped)
    run.gaps = list(source.gaps)
    run.problems += state.failed + state.write_errors
    if state.errors_cut:
        run.problems.append(f"... {state.unwritten} observations in all did not reach "
                            f"{database.name if database else 'the database'}; "
                            "they are in observations.jsonl")
    run.written, run.unwritten = state.written, state.unwritten
    if state.lags:
        ordered = sorted(state.lags)
        run.lag_s = {"p50": round(statistics.median(ordered), 3),
                     "p95": round(ordered[min(len(ordered) - 1,
                                              int(len(ordered) * 0.95))], 3),
                     "max": round(ordered[-1], 3)}
    if not count:
        return                                     # nothing was read

    def step(component: str, artifact: str, document: Any, **stats: Any) -> None:
        where = files.write(at[artifact], document)
        produced = Produced(video_id=run.video_id, component=component,
                            artifacts={artifact: where}, stats=stats)
        run.steps.append(produced)
        if database is not None:
            run.problems += export(produced, database)
        say(component, produced)

    media = _media_document(run, options, source, duration, source.started_at)
    step("media", "media", media, live=True)

    grid = Timeline(video_id=run.video_id, spans=spans, policy="uniform",
                    params={"chunk_s": options.chunk_s, "live": True},
                    derived_from="grid")
    step("boundaries", "timeline", grid, chunks=count)

    before, after = options.context
    manifest = Manifest(
        video_id=run.video_id, timeline_fingerprint=grid.fingerprint(),
        source={"path": options.source, "container": source.container_format,
                "duration_s": duration, **source.describe(duration).as_dict(),
                **({"rotation": int(source.rotation)} if int(source.rotation) else {})},
        config={"decimator": {"per_second": options.per_second},
                "samplers": [s.config() for s in built],
                "frame_store": {**store.config(), "scope": "sampled"},
                "live": {"chunk_s": options.chunk_s,
                         "context": {"before": before, "after": after},
                         "queue": options.queue}},
        stats={"frames_decimated": tally.decimated, "frames_sampled": tally.sampled,
               **({"views_made": tally.views} if tally.views else {}),
               "chunks": count, "stored_frames": store.written,
               **({"stored_views": store.views_written} if store.views_written else {}),
               "stored_mb": round(store.bytes_written / 1024 / 1024, 2),
               "gaps": len(source.gaps), "rewinds": source.rewinds,
               "corrupt_packets": source.corrupt,
               "unrecorded_packets": source.unrecorded},
        chunks=[tally.chunks.get(i) or {"chunk_id": i, "decimated_frames": 0,
                                        "samplers": {}} for i in range(count)])
    step("video", "manifest", manifest, sampled=tally.sampled)

    descriptions = Descriptions(
        video_id=run.video_id, timeline_fingerprint=grid.fingerprint(),
        manifest_fingerprint=manifest.fingerprint(),
        model={**describer.config(), "prompts": prompts.versions(sorted(state.questions))},
        chunks=_chunks(state.answers, count),
        stats={"described": len(state.answers), "failed": len(state.failed),
               "dropped": len(state.dropped), "live": True})
    step("describe", "descriptions", descriptions, described=len(state.answers))

    if state.answers:
        unit_list = units.from_descriptions(descriptions)
        units.embed_all(unit_list, embedder)
        document = readable.build(run.video_id, unit_list, embedder.key,
                                  grid.fingerprint())
        step("embed", "embedded", document, units=len(unit_list),
             embedder=embedder.key)


def _media_document(run: LiveRun, options: Options, source: Source,
                    duration: Optional[float], recorded_at: Optional[str]) -> Media:
    return Media(video_id=run.video_id, path=str(options.source),
                 container_format=source.container_format, duration_s=duration,
                 video=source.describe(duration), audio=None,
                 name=options.name or run.video_id, recorded_at=recorded_at)


def _media(run: LiveRun, options: Options, source: Source,
           duration: Optional[float], recorded_at: Optional[str],
           database: Optional[Database]) -> list[str]:
    """`media.json`, and its copy, when the stream opens. Returns what failed."""
    from ..core.export import export
    document = _media_document(run, options, source, duration, recorded_at)
    where = files.write(run.folder / paths.ARTIFACTS["media"], document)
    if database is None:
        return []
    return export(Produced(video_id=run.video_id, component="media",
                           artifacts={"media": where}, stats={"live": True}), database)


def _chunks(answers: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """`descriptions.chunks`: per chunk and answer id, the frames' answers joined in
    time order, and each kept whole under `observations`."""
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for a in sorted(answers, key=lambda a: (a["chunk_id"], a["media_ts"])):
        grouped.setdefault(a["chunk_id"], {}).setdefault(a["sampler_id"], []).append(a)
    chunks = []
    for chunk_id in range(count):
        blocks = {}
        for sampler_id, said in sorted(grouped.get(chunk_id, {}).items()):
            frames = sorted({i for a in said for i in a["frames"]})
            blocks[sampler_id] = {
                "sampler": said[0]["sampler"], "question": said[0]["question"],
                "frame_count": len(frames), "frame_indexes": frames,
                "description": " ".join(f"At {a['media_ts']:.1f}s: {a['description']}"
                                        for a in said),
                "structured": _merge([a["structured"] for a in said]),
                "observations": [{"frame_index": a["frame_index"],
                                  "media_ts": a["media_ts"], "frames": a["frames"],
                                  "description": a["description"],
                                  "structured": a["structured"],
                                  "lag_s": a["lag_s"]} for a in said],
            }
        chunks.append({"chunk_id": chunk_id, "samplers": blocks})
    return chunks


def _merge(answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Several frames' fields as one: lists concatenated without repeats, a value
    every frame agrees on kept as it is, and values they disagree on listed."""
    merged: dict[str, Any] = {}
    for key in dict.fromkeys(k for a in answers for k in a):
        values = [a[key] for a in answers if a.get(key) not in (None, "", [])]
        if not values:
            continue
        seen: dict[str, Any] = {}
        flatten = all(isinstance(v, list) for v in values)
        for value in values:
            for item in (value if flatten else [value]):
                seen.setdefault(json.dumps(item, sort_keys=True), item)
        distinct = list(seen.values())
        merged[key] = distinct if flatten or len(distinct) > 1 else distinct[0]
    return merged


def _claim(options: Options) -> str:
    """The run's id: given, the file's stem, or `live-<UTC time>`; suffixed `-2`,
    `-3` ... rather than written into a folder another run already holds."""
    if options.video_id is not None:
        wanted = options.video_id
    elif "://" not in options.source:
        wanted = Path(options.source).stem
    else:
        wanted = "live-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    paths.check_id(wanted)
    candidate, n = wanted, 1
    while (Path(options.into) / candidate).exists() and any(
            (Path(options.into) / candidate).iterdir()):
        n += 1
        candidate = f"{wanted}-{n}"
    return candidate


def video_rag_live(source: str | Path,
                   into: str | Path,
                   video_id: Optional[str] = None,
                   name: Optional[str] = None,
                   sampler: str | Sequence[Any] = "uniform",
                   models: Optional[Models] = None,
                   database: Optional[str | Database] = None,
                   on_unit: Optional[Callable[[Observation], None]] = None,
                   on_step: Optional[Callable[..., None]] = None,
                   chunk_s: float = 20.0,
                   per_second: float = 1.0,
                   context: tuple[int, int] = (0, 0),
                   queue: int = 64,
                   record: bool | str | Path = False,
                   stop_after_s: Optional[float] = None,
                   stop: Optional[threading.Event] = None,
                   open_timeout_s: Optional[float] = 60.0,
                   read_timeout_s: Optional[float] = 10.0) -> LiveRun:
    """A live run, as keyword arguments: `process` with an `Options`.

    Blocks until the stream ends, `stop_after_s` is reached, `stop` is set or
    `on_unit` raises `StopStream`. `on_unit` is called on the event loop for
    every answer, so a slow one delays the answers behind it. Everything lands
    under `<into>/<video_id>/`. `models` carries the vlm and the embedder
    (None is every default).
    """
    return process(Options(
        source=source, into=Path(into), video_id=video_id, name=name,
        sampler=sampler, models=models, database=database, chunk_s=chunk_s, per_second=per_second,
        context=context, queue=queue, record=record, stop_after_s=stop_after_s,
        stop=stop, open_timeout_s=open_timeout_s, read_timeout_s=read_timeout_s,
    ), on_unit, on_step)


__all__ = ["LiveRun", "OBSERVATIONS", "Observation", "Options", "StopStream",
           "process", "validate", "video_rag_live"]
