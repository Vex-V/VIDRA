"""The describe component: `manifest.json` + `store/` -> `descriptions.json`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from vidra.shared.reporting import logs, progress
from vidra.shared.contracts.documents import (Descriptions, Manifest,
                                           Produced, Timeline)
from vidra.shared.storage.files import maybe, read, write
from vidra.shared.contracts.documents import same_video
from ...core.describe import base, prompts
from ...core.describe.base import Description
from ...core.describe.frames import FrameSource, LoadedFrame, store_of
from ...core.frames import FrameStore
from vidra.shared.reporting.errors import Refused, UnknownOption

if TYPE_CHECKING:
    from vidra.shared.models.base import VLM

def answer(manifest: Manifest, timeline: Timeline, frames: FrameStore,
           vlm: Optional["VLM"] = None,
           samplers: Optional[Sequence[str]] = None,
           limit: Optional[int] = None,
           existing: Optional[Descriptions] = None,
           max_output_tokens: Optional[int] = None,
           on_progress: Optional[progress.Reporter] = None,
           max_images: Optional[int] = None) -> Descriptions:
    """One call per (chunk, sampler, question), over documents and frames in hand.
    Reads and writes no artifact.

    `frames` is the store ingest wrote. `existing` is an earlier `Descriptions`:
    every pair still current is skipped. `vlm` is the model asked (None is
    OpenAI's default). `max_output_tokens` caps one answer (None is 2000) and is
    part of the resume key. `max_images` refuses, before any call, a run whose
    frames and views would put more images than that in one call.
    """
    # The model is checked (its kind, its key) before any frame is read.
    built = base.build(vlm, **({} if max_output_tokens is None
                               else {"max_output_tokens": max_output_tokens}))

    # `limit=0` would describe nothing; refused.
    if limit is not None and limit < 1:
        raise Refused(
            f"limit must be at least 1, not {limit}; "
            f"leave it None for no limit")

    if manifest.timeline_fingerprint != timeline.fingerprint():
        raise Refused(
            f"{manifest.video_id}: the manifest was built on a different grid "
            f"({manifest.timeline_fingerprint} vs {timeline.fingerprint()}). "
            "Re-run ingest against the current timeline.")

    if max_images is not None:
        if max_images < 1:
            raise Refused(f"max_images must be at least 1, not {max_images}; "
                          f"leave it None for no cap")
        over = [(chunk["chunk_id"], run_id, count)
                for chunk in manifest.chunks
                for run_id, block in chunk.get("samplers", {}).items()
                if (count := sum(len(f.get("views") or [None])
                                 for f in block["frames"])) > max_images]
        if over:
            chunk_id, run_id, count = max(over, key=lambda o: o[2])
            raise Refused(
                f"{len(over)} call(s) would send more than max_images={max_images} "
                f"images, the most {count} (chunk {chunk_id}, {run_id}); keep fewer "
                f"frames (max_per_chunk) or make fewer views")

    known = prompts.questions()
    unknown = sorted({q for s in manifest.config.get("samplers", [])
                      for q in prompts.questions_of(s, s.get("id", ""))
                      if q not in known})
    if unknown:
        raise UnknownOption(
            f"manifest names unknown question(s) {', '.join(unknown)}; "
            f"known: {', '.join(prompts.questions())}")

    from .reader import answer as _pass

    with FrameSource(frames, manifest) as source:
        return _pass(manifest, timeline, built, source, samplers, existing,
                     limit, on_progress)


def describe(manifest: str | Path, timeline: str | Path,
             store: str | Path, out: str | Path,
             previous: Optional[str | Path] = None,
             vlm: Optional["VLM"] = None,
             samplers: Optional[Sequence[str]] = None,
             limit: Optional[int] = None,
             max_output_tokens: Optional[int] = None,
             on_progress: Optional[progress.Reporter] = None,
             max_images: Optional[int] = None) -> Produced:
    """One model call per (chunk, sampler, question). `answer` plus a read at each
    end. `previous` is an earlier `descriptions.json`: every pair still current
    is skipped.
    """
    plan = read(manifest, Manifest)
    grid = read(timeline, Timeline)
    existing = maybe(previous, Descriptions)
    video_id = same_video(manifest=plan, timeline=grid, previous=existing)

    with logs.timed("describe", video_id) as done:
        document = answer(plan, grid, store_of(store), vlm, samplers,
                          limit, existing, max_output_tokens, on_progress,
                          max_images)

        where = write(out, document)
        done(described=document.stats.get("described"),
             skipped_pairs=document.stats.get("skipped"),
             vlm=_named(document.model))
    return Produced(
        video_id=video_id, component="describe",
        artifacts={"descriptions": where},
        # The VLM's name from the stored model block.
        stats={**document.stats, "vlm": _named(document.model),
               "model": ((document.model.get("params") or {})
                         .get("model", _named(document.model))),
               },
    )


def ask(images: Sequence[Any], question: str, *,
        timestamps: Optional[Sequence[float]] = None,
        vlm: Optional["VLM"] = None,
        max_output_tokens: Optional[int] = None) -> Description:
    """One model call about images you hold: no manifest, grid or store.

    `images` are JPEG or PNG bytes, image file paths, or BGR arrays (what
    `video.frames` yields). `question` is any question `questions()` lists; its
    shape decides the fields. `timestamps` are each image's time in seconds,
    shown to the model beside it (by default 0, 1, 2, ...). Nothing is stored
    and nothing is resumed. Inside a running event loop, await `ask_async`.
    """
    import asyncio
    return asyncio.run(ask_async(images, question, timestamps=timestamps, vlm=vlm,
                                 max_output_tokens=max_output_tokens))


async def ask_async(images: Sequence[Any], question: str, *,
                    timestamps: Optional[Sequence[float]] = None,
                    vlm: Optional["VLM"] = None,
                    max_output_tokens: Optional[int] = None) -> Description:
    """`ask`, as a coroutine: gather several at once, and the VLM's `concurrency`
    caps how many are in flight."""
    if question not in prompts.questions():
        raise UnknownOption(f"unknown question {question!r}; "
                            f"known: {', '.join(prompts.questions())}")
    if not images:
        raise Refused("images is empty; ask about at least one")
    times = list(range(len(images))) if timestamps is None else list(timestamps)
    if len(times) != len(images):
        raise Refused(f"{len(times)} timestamps for {len(images)} images")
    built = base.build(vlm, **({} if max_output_tokens is None
                               else {"max_output_tokens": max_output_tokens}))
    frames = [LoadedFrame(index=i, media_ts=float(t), jpeg=_jpeg(image))
              for i, (image, t) in enumerate(zip(images, times))]
    return await built.describe(frames, {
        "chunk_id": "-", "sampler": question, "question": question,
        "start_ts": float(min(times)), "end_ts": float(max(times))})


def _jpeg(image: Any) -> bytes:
    """An image as JPEG bytes: JPEG bytes as they are, anything else encoded."""
    from ...core.frames import encode

    if isinstance(image, (str, Path)):
        image = Path(image).read_bytes()
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
        if data[:2] == b"\xff\xd8":
            return data
        import cv2
        import numpy as np
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise Refused("an image is neither JPEG nor any format OpenCV reads")
        return encode(decoded)
    if hasattr(image, "shape"):
        return encode(image)
    raise Refused(f"an image must be bytes, a path or an array, not "
                  f"{type(image).__name__}")


def _named(model: dict[str, object]) -> str:
    """The VLM's name, whichever key the stored block records it under."""
    return str(model.get("name") or model.get("describer") or "")


def load(path: str | Path) -> Descriptions:
    """Read a `descriptions.json` back, typed."""
    return read(path, Descriptions)
