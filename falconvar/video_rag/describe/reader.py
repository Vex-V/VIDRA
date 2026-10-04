"""`answer()` -- one model call per (chunk, sampler, question).

Every call is planned first, in manifest order, with its slot in the document
reserved; then one task per (chunk, sampler run) reads that run's frames once
and asks its questions, `describer.concurrency` at a time. A stored answer is
reused when its manifest, describer and question hash still match.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional, Sequence

from falconvar.shared.contracts.documents import Descriptions, Manifest, Timeline
from . import prompts
from .base import Describer
from falconvar.shared.reporting import progress
from .frames import FrameSource


def _model_block(describer: Describer, questions: Sequence[str]) -> dict[str, Any]:
    """What a stored answer must match to count as done: the describer's config and
    `{question: hash}`.
    """
    return {**describer.config(), "prompts": prompts.versions(questions)}


def _resumable(existing: Optional[Descriptions], manifest: Manifest,
               model: dict[str, Any]) -> set[tuple[int, str]]:
    """The (chunk, sampler) pairs whose stored answer is still current: same
    manifest, same describer, and the pair's own question hashing the same. A
    stored `prompts` that is a single string (an older format) matches nothing.
    """
    if existing is None:
        return set()
    if existing.manifest_fingerprint != manifest.fingerprint():
        return set()

    def describer_half(block: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in (block or {}).items() if k != "prompts"}

    if describer_half(existing.model) != describer_half(model):
        return set()

    stored = (existing.model or {}).get("prompts")
    if not isinstance(stored, dict):
        return set()
    current = model.get("prompts") or {}

    return {(chunk["chunk_id"], sampler_id)
            for chunk in existing.chunks
            for sampler_id, block in (chunk.get("samplers") or {}).items()
            if (q := block.get("question")) is not None
            and stored.get(q) is not None and stored.get(q) == current.get(q)}


def answer(manifest: Manifest, timeline: Timeline, describer: Describer,
             source: FrameSource,
             samplers: Optional[Sequence[str]] = None,
             existing: Optional[Descriptions] = None,
             limit: Optional[int] = None,
             on_progress: Optional[progress.Reporter] = None) -> Descriptions:
    """Describe every (chunk, sampler) the manifest names. `on_progress` fires once
    per answer, in completion order.
    """
    # Every question the manifest asks, resolved before the first call.
    model = _model_block(describer, prompts.questions_in(manifest))
    started = time.perf_counter()

    done = _resumable(existing, manifest, model)
    kept: dict[int, dict[str, Any]] = (
        {c["chunk_id"]: c for c in existing.chunks} if existing is not None else {})
    by_id = {s["id"]: s for s in manifest.config.get("samplers", [])}

    # -- plan: every call, in manifest order -----------------------------------
    chunks: list[dict[str, Any]] = []
    runs: list[tuple[int, str, str, dict[str, Any], list[tuple[str, str]],
                     dict[str, Any]]] = []
    planned = skipped = 0
    for chunk in manifest.chunks:
        chunk_id = chunk["chunk_id"]
        out: dict[str, Any] = {"chunk_id": chunk_id, "samplers": {}}
        previous = kept.get(chunk_id, {}).get("samplers", {})

        # One sampler run: one set of frames, one call per question.
        for run_id in chunk.get("samplers", {}):
            config = by_id.get(run_id, {})
            name = config.get("name") or run_id
            asked: list[tuple[str, str]] = []
            for question in prompts.questions_of(config, run_id):
                sampler_id = prompts.answer_id(name, question)
                if samplers is not None and sampler_id not in samplers:
                    continue
                if (chunk_id, sampler_id) in done:
                    out["samplers"][sampler_id] = previous[sampler_id]
                    skipped += 1
                    continue
                if limit is not None and planned >= limit:
                    continue
                # Reserved now, so the answer lands in manifest order.
                out["samplers"][sampler_id] = None
                asked.append((sampler_id, question))
                planned += 1
            if asked:
                runs.append((chunk_id, run_id, name, config, asked, out))

        # No chunk-level rollup: every sampler's answer stays in its own block.
        chunks.append(out)

    # -- run: concurrently, each run's frames read once ------------------------
    finished = 0
    progress.report(on_progress, "describe", planned, 0, skipped)

    async def run(chunk_id: int, run_id: str, name: str, config: dict[str, Any],
                  asked: list[tuple[str, str]], out: dict[str, Any],
                  gate: asyncio.Semaphore) -> None:
        async with gate:
            images = source.images_for(chunk_id, run_id)
            if not images:
                for sampler_id, _ in asked:
                    del out["samplers"][sampler_id]
                return
            start_ts, end_ts = timeline.bounds_of(chunk_id)

            async def ask(sampler_id: str, question: str) -> None:
                context = {
                    "video_id": manifest.video_id,
                    "chunk_id": chunk_id,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "sampler": sampler_id,
                    "question": question,
                    "sampler_config": config,
                }
                call_started = time.perf_counter()
                answer = await describer.describe(images, context)
                out["samplers"][sampler_id] = {
                    # Both halves of the sampler id, written out.
                    "sampler": name,
                    "question": question,
                    "frame_count": len(images),
                    "frame_indexes": [f.index for f in images],
                    "description": answer.summary,
                    "structured": answer.fields,
                    "elapsed_s": round(time.perf_counter() - call_started, 3),
                }
                # Counted after the answer is stored; every task runs on one event loop.
                nonlocal finished
                finished += 1
                progress.report(on_progress, "describe", planned, finished,
                                skipped, f"{chunk_id}:{sampler_id}",
                                out["samplers"][sampler_id])

            await asyncio.gather(*(ask(s, q) for s, q in asked))

    async def run_all() -> None:
        gate = asyncio.Semaphore(max(1, describer.concurrency))
        await asyncio.gather(*(run(*r, gate) for r in runs))

    if runs:
        asyncio.run(run_all())

    described = sum(len(c["samplers"]) for c in chunks) - skipped
    return Descriptions(
        video_id=manifest.video_id,
        timeline_fingerprint=manifest.timeline_fingerprint,
        manifest_fingerprint=manifest.fingerprint(),
        model=model,
        chunks=chunks,
        stats={"described": described, "skipped": skipped,
               "chunks": len(chunks),
               "elapsed_s": round(time.perf_counter() - started, 3)},
    )


__all__ = ["answer"]
