"""What every artifact is: one dataclass per document a component writes.

Every component imports its documents from here and none imports another.
`schemas.py` generates `db/json/*.schema.json` from these classes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from ..reporting.errors import Refused

#: Decimal places kept for times in identity and serialisation.
PRECISION = 3


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, PRECISION)


def _hash(payload: dict[str, Any]) -> str:
    """A short, stable hash of a document's identifying content (keys sorted)."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# 1 · media  --  what the file is
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoStream:
    """How to open and address the picture. `time_base` is a string so the
    exact rational survives JSON.
    """

    #: Its stream index in the container.
    index: int
    #: The codec name as the container reports it, e.g. `h264`.
    codec: str
    #: Frames per second. None when the container does not say.
    rate: Optional[float]
    #: The stream's clock, as `num/den`.
    time_base: Optional[str]
    #: Coded width in pixels, before any rotation metadata.
    width: int
    #: Coded height in pixels, before any rotation metadata.
    height: int
    #: How many frames the container claims; None when it does not say.
    frames: Optional[int]
    #: This stream's own duration.
    duration_s: Optional[float]

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "codec": self.codec, "rate": self.rate,
                "time_base": self.time_base, "width": self.width,
                "height": self.height, "frames": self.frames,
                "duration_s": _round(self.duration_s)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VideoStream":
        return cls(index=d["index"], codec=d["codec"], rate=d.get("rate"),
                   time_base=d.get("time_base"), width=d["width"],
                   height=d["height"], frames=d.get("frames"),
                   duration_s=d.get("duration_s"))


@dataclass(frozen=True)
class AudioStream:
    """How to open the soundtrack."""

    #: Its stream index in the container.
    index: int
    #: The codec name as the container reports it, e.g. `aac`.
    codec: str
    #: Samples per second. None when the container will not say.
    rate: Optional[int]
    #: How many channels. None when the container will not say.
    channels: Optional[int]
    #: This stream's own duration; see `VideoStream.duration_s`.
    duration_s: Optional[float]

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "codec": self.codec, "rate": self.rate,
                "channels": self.channels, "duration_s": _round(self.duration_s)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AudioStream":
        return cls(index=d["index"], codec=d["codec"], rate=d.get("rate"),
                   channels=d.get("channels"), duration_s=d.get("duration_s"))


@dataclass(frozen=True)
class Media:
    """One file, seen as the two streams it carries. `duration_s` is the
    container's, which the grid uses.
    """

    #: The id every later component is addressed by, and the name of its folder.
    video_id: str
    #: Where the file was when it was described.
    path: str
    #: The container as the demuxer names it, e.g. `mov,mp4,m4a,3gp,3g2,mj2`.
    container_format: str
    #: The container's duration; None when it reports none.
    duration_s: Optional[float]
    #: The picture stream, or None if the file carries none.
    video: Optional[VideoStream]
    #: The soundtrack stream, or None if the file carries none.
    audio: Optional[AudioStream]
    #: A fingerprint of the file's content, telling the same file at another path
    #: from a different file with the same name. None in an older `media.json`.
    source: Optional[str] = None
    #: What to call the video: the filename unless the caller named it.
    name: Optional[str] = None
    #: When the video was recorded, as ISO 8601: from the container's tags unless
    #: given, otherwise None.
    recorded_at: Optional[str] = None

    @property
    def has_video(self) -> bool:
        return self.video is not None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "document": "media", "version": 1,
            "video_id": self.video_id,
            "path": self.path,
            "container_format": self.container_format,
            "duration_s": _round(self.duration_s),
            "video": self.video.as_dict() if self.video else None,
            "audio": self.audio.as_dict() if self.audio else None,
            "source": self.source,
            "name": self.name,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Media":
        return cls(
            video_id=d["video_id"], path=d["path"],
            container_format=d["container_format"],
            duration_s=d.get("duration_s"),
            video=VideoStream.from_dict(d["video"]) if d.get("video") else None,
            audio=AudioStream.from_dict(d["audio"]) if d.get("audio") else None,
            source=d.get("source"),
            name=d.get("name"),
            recorded_at=d.get("recorded_at"),
        )


# --------------------------------------------------------------------------
# 2 · raw transcript  --  what was said, before any grid exists
# --------------------------------------------------------------------------

@dataclass
class RawTranscript:
    """Words, segments and speaker turns, before any grid is applied.

    `turns` is empty when no diarizer ran, and every segment keeps
    `speaker: None`.
    """

    #: Which video this is the soundtrack of.
    video_id: str
    #: Which transcriber and diarizer produced this, and their settings.
    model: dict[str, Any] = field(default_factory=dict)
    #: The decoded audio: sample rate, channels, duration, RMS and peak.
    track: dict[str, Any] = field(default_factory=dict)
    #: Whisper's utterances: `start`, `end`, `text` and `speaker`. No chunk ids.
    segments: list[dict[str, Any]] = field(default_factory=list)
    #: One entry per word with `start`/`end`, and the `speaker` whose turn its
    #: midpoint falls in.
    words: list[dict[str, Any]] = field(default_factory=list)
    #: Diarization's turns: who spoke from when to when.
    turns: list[dict[str, Any]] = field(default_factory=list)
    #: Segments, words, speakers, turns, speech seconds and timings.
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def speakers(self) -> list[str]:
        seen = {t.get("speaker") for t in self.turns}
        return sorted(s for s in seen if s)

    @property
    def silent(self) -> bool:
        return bool(self.track.get("silent", False))

    def as_dict(self) -> dict[str, Any]:
        return {
            "document": "raw_transcript", "version": 1,
            "video_id": self.video_id,
            "model": self.model,
            "track": self.track,
            "speakers": self.speakers,
            "segments": self.segments,
            "words": self.words,
            "turns": self.turns,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RawTranscript":
        return cls(video_id=d["video_id"], model=d.get("model", {}),
                   track=d.get("track", {}), segments=d.get("segments", []),
                   words=d.get("words", []), turns=d.get("turns", []),
                   stats=d.get("stats", {}))


# --------------------------------------------------------------------------
# 3 · cuts  --  raw boundary evidence, before any guard
# --------------------------------------------------------------------------

@dataclass
class Cuts:
    """Interior cut times, plus the per-frame score series they were thresholded
    from. `retune` re-thresholds the series without decoding.
    """

    #: Which video these boundaries were found in.
    video_id: str
    #: Which modality they came from: `video` or `audio`.
    source: str                      # "video" | "audio"
    #: What found them: `content` for the scene detector, or the policy name.
    detector: str                    # "content" | "vad" | "speaker"
    #: The settings the pass ran under.
    params: dict[str, Any] = field(default_factory=dict)
    #: The boundary timestamps themselves, in media seconds.
    cuts: list[float] = field(default_factory=list)
    #: The per-frame score series, for re-thresholding. None for a detector with no
    #: continuous score.
    scores: Optional[dict[str, Any]] = None
    #: How many cuts, over how many scored frames.
    stats: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "document": "cuts", "version": 1,
            "video_id": self.video_id,
            "source": self.source,
            "detector": self.detector,
            "params": self.params,
            "cuts": [_round(c) for c in self.cuts],
            "scores": self.scores,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Cuts":
        return cls(video_id=d["video_id"], source=d["source"],
                   detector=d["detector"], params=d.get("params", {}),
                   cuts=list(d.get("cuts", [])), scores=d.get("scores"),
                   stats=d.get("stats", {}))


# --------------------------------------------------------------------------
# 4 · timeline  --  THE GRID
# --------------------------------------------------------------------------

@dataclass
class Timeline:
    """Where every chunk starts and ends, and the policy that decided it. A
    `chunk_id` is the index into `spans`.
    """

    #: Which video this grid divides.
    video_id: str
    #: `(start_s, end_s)` per chunk, in order and contiguous.
    spans: list[tuple[float, float]]
    #: How the boundaries were decided: `uniform`, `scene`, `vad` or `speaker`.
    policy: str = "uniform"                       # uniform|scene|vad|speaker
    #: The settings the grid was built under.
    params: dict[str, Any] = field(default_factory=dict)
    #: "video", "audio", or "grid" when the policy is arithmetic.
    derived_from: str = "grid"

    def __len__(self) -> int:
        return len(self.spans)

    @property
    def duration_s(self) -> float:
        return self.spans[-1][1] if self.spans else 0.0

    def bounds_of(self, chunk_id: int) -> tuple[float, float]:
        """Start and end of one chunk."""
        return self.spans[chunk_id]

    def index_at(self, ts: float) -> Optional[int]:
        """Which chunk contains ``ts``. The last chunk owns its own end."""
        for index, (start, end) in enumerate(self.spans):
            if start <= ts < end:
                return index
        if self.spans and ts == self.spans[-1][1]:
            return len(self.spans) - 1
        return None

    def nearest(self, ts: float) -> int:
        """`index_at`, but never None: a time past either end belongs to the nearest
        edge chunk.
        """
        index = self.index_at(ts)
        if index is not None:
            return index
        if not self.spans:
            raise Refused("an empty timeline contains no chunk")
        return 0 if ts < self.spans[0][0] else len(self.spans) - 1

    def fingerprint(self) -> str:
        """A hash of the grid and its policy. Documents cut on it record this."""
        return _hash({
            "spans": [[_round(s), _round(e)] for s, e in self.spans],
            "policy": self.policy,
            "params": self.params,
        })

    def as_dict(self) -> dict[str, Any]:
        return {
            "document": "timeline", "version": 1,
            "video_id": self.video_id,
            "policy": self.policy,
            "derived_from": self.derived_from,
            "params": self.params,
            "fingerprint": self.fingerprint(),
            "duration_s": _round(self.duration_s),
            "chunk_count": len(self.spans),
            "chunks": [{"chunk_id": i, "start_ts": _round(s), "end_ts": _round(e)}
                       for i, (s, e) in enumerate(self.spans)],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Timeline":
        return cls(
            video_id=d["video_id"],
            spans=[(float(c["start_ts"]), float(c["end_ts"]))
                   for c in sorted(d["chunks"], key=lambda c: c["chunk_id"])],
            policy=d.get("policy", "uniform"),
            params=d.get("params", {}),
            derived_from=d.get("derived_from", "grid"),
        )


# --------------------------------------------------------------------------
# 5 · manifest  --  which frames were kept, why, and how to fetch them again
# --------------------------------------------------------------------------

@dataclass
class Manifest:
    """What each sampler kept from each chunk. Times come from the timeline.

    `manifest_fingerprint` hashes the source (minus its path), the config
    (minus the frame store), and any views a sampler made of its frames.
    """

    #: Which video was ingested.
    video_id: str
    #: The grid this was cut on.
    timeline_fingerprint: str
    #: What was decoded: path, container, duration and the video stream.
    #: `video.recreate` rebuilds a store from this.
    source: dict[str, Any] = field(default_factory=dict)
    #: The decimator, every sampler's configuration, and where frames went.
    config: dict[str, Any] = field(default_factory=dict)
    #: Frames decimated and sampled, chunks, and stored files and megabytes.
    stats: dict[str, Any] = field(default_factory=dict)
    #: [{chunk_id, decimated_frames, samplers: {id: {frame_count, frames[]}}}];
    #: a frame record carries `views` when its sampler showed more than the frame
    chunks: list[dict[str, Any]] = field(default_factory=list)

    def fingerprint(self) -> str:
        views = [[c["chunk_id"], run_id, f["index"], f["views"]]
                 for c in self.chunks for run_id, b in c.get("samplers", {}).items()
                 for f in b.get("frames", []) if f.get("views")]
        return _hash({
            "source": {k: v for k, v in self.source.items() if k != "path"},
            "config": {k: v for k, v in self.config.items()
                       if k != "frame_store"},
            # Only when a sampler made images: what the model is shown.
            **({"views": views} if views else {}),
        })

    def sampler_ids(self) -> list[str]:
        return [s["id"] for s in self.config.get("samplers", [])]

    def frames_of(self, chunk_id: int, sampler_id: str) -> list[dict[str, Any]]:
        for chunk in self.chunks:
            if chunk["chunk_id"] == chunk_id:
                block = chunk.get("samplers", {}).get(sampler_id)
                return list(block["frames"]) if block else []
        return []

    def as_dict(self) -> dict[str, Any]:
        return {
            "document": "manifest", "version": 1,
            "video_id": self.video_id,
            "timeline_fingerprint": self.timeline_fingerprint,
            "manifest_fingerprint": self.fingerprint(),
            "source": self.source,
            "config": self.config,
            "stats": self.stats,
            "chunks": self.chunks,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        return cls(video_id=d["video_id"],
                   timeline_fingerprint=d.get("timeline_fingerprint", ""),
                   source=d.get("source", {}), config=d.get("config", {}),
                   stats=d.get("stats", {}), chunks=d.get("chunks", []))


# --------------------------------------------------------------------------
# 6 · transcript  --  what was said, on the grid
# --------------------------------------------------------------------------

@dataclass
class Transcript:
    """The raw transcript cut to chunks. Chunks with no speech are kept, with
    empty text.
    """

    #: Which video this is the soundtrack of.
    video_id: str
    #: The grid it was cut on; see `Manifest.timeline_fingerprint`.
    timeline_fingerprint: str = ""
    #: The transcriber and diarizer, carried through from the `RawTranscript`.
    model: dict[str, Any] = field(default_factory=dict)
    #: One entry per chunk: its text, words, turns and speakers.
    chunks: list[dict[str, Any]] = field(default_factory=list)
    #: Chunks, chunks with speech, words placed, words outside the grid, speakers.
    stats: dict[str, Any] = field(default_factory=dict)

    def text_of(self, chunk_id: int) -> str:
        for chunk in self.chunks:
            if chunk["chunk_id"] == chunk_id:
                return chunk.get("text", "")
        return ""

    def as_dict(self) -> dict[str, Any]:
        return {"document": "transcript", "version": 1,
                "video_id": self.video_id,
                "timeline_fingerprint": self.timeline_fingerprint,
                "model": self.model, "stats": self.stats,
                "chunks": self.chunks}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Transcript":
        return cls(video_id=d["video_id"],
                   timeline_fingerprint=d.get("timeline_fingerprint", ""),
                   model=d.get("model", {}), chunks=d.get("chunks", []),
                   stats=d.get("stats", {}))


# --------------------------------------------------------------------------
# 7 · descriptions  --  one answer per (chunk, sampler)
# --------------------------------------------------------------------------

@dataclass
class Descriptions:
    """What a model said about each chunk, per sampler id (`name` or
    `name:question`).
    """

    #: Which video was described.
    video_id: str
    #: The grid it was described against; see `Manifest.timeline_fingerprint`.
    timeline_fingerprint: str = ""
    #: Which ingest produced the frames.
    manifest_fingerprint: str = ""
    #: Which VLM, its settings, and `{question: hash}` for every prompt.
    model: dict[str, Any] = field(default_factory=dict)
    #: One entry per chunk, with a block per sampler id: its summary and
    #: structured answer.
    chunks: list[dict[str, Any]] = field(default_factory=list)
    #: Described, skipped, failed, and the elapsed time.
    stats: dict[str, Any] = field(default_factory=dict)

    def done(self) -> set[tuple[int, str]]:
        """Which (chunk, sampler) pairs already have an answer."""
        return {(c["chunk_id"], sid)
                for c in self.chunks for sid in c.get("samplers", {})}

    def as_dict(self) -> dict[str, Any]:
        return {"document": "descriptions", "version": 1,
                "video_id": self.video_id,
                "timeline_fingerprint": self.timeline_fingerprint,
                "manifest_fingerprint": self.manifest_fingerprint,
                "model": self.model, "stats": self.stats,
                "chunks": self.chunks}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Descriptions":
        return cls(video_id=d["video_id"],
                   timeline_fingerprint=d.get("timeline_fingerprint", ""),
                   manifest_fingerprint=d.get("manifest_fingerprint", ""),
                   model=d.get("model", {}), chunks=d.get("chunks", []),
                   stats=d.get("stats", {}))


# --------------------------------------------------------------------------
# 8 · embedded  --  what text was embedded. For reading, not for searching.
# --------------------------------------------------------------------------

@dataclass
class Embedded:
    """What `embed` produced: every unit's text, hash and vector, under one embedder."""

    #: Which video these units belong to.
    video_id: str
    #: The grid they were built on; see `Manifest.timeline_fingerprint`.
    timeline_fingerprint: str = ""
    #: Who made the vectors: the embedder's `key`.
    embedder: str = ""
    #: One entry per unit: the text, its hash, and `vector` when it was embedded.
    units: list[dict[str, Any]] = field(default_factory=list)

    def text_of(self, chunk_id: int, sampler_id: str) -> str:
        for unit in self.units:
            if unit["chunk_id"] == chunk_id and unit["sampler_id"] == sampler_id:
                return unit.get("content", "")
        return ""

    def as_dict(self) -> dict[str, Any]:
        return {"document": "embedded", "version": 2,
                "video_id": self.video_id,
                "timeline_fingerprint": self.timeline_fingerprint,
                "embedder": self.embedder,
                "count": len(self.units),
                "samplers": sorted({u["sampler_id"] for u in self.units}),
                "units": self.units}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Embedded":
        return cls(video_id=d["video_id"],
                   timeline_fingerprint=d.get("timeline_fingerprint", ""),
                   embedder=d.get("embedder", ""),
                   units=d.get("units", []))

    def stored(self) -> dict[str, str]:
        """`{unit key: text_hash}` for every unit that has a vector."""
        return {f"{self.video_id}:{u['chunk_id']}:{u['sampler_id']}":
                u.get("text_hash", "")
                for u in self.units if u.get("vector")}


# --------------------------------------------------------------------------
# 9 · aggregate  --  video-level structure over what the chunks said
# --------------------------------------------------------------------------

@dataclass
class Aggregate:
    """One aggregator's answer about a whole record."""

    #: Which video was answered about.
    video_id: str
    #: Which aggregate this is, e.g. `summary` or `entities:people`. Stored with
    #: `.` for `:` in its filename.
    aggregate_id: str
    #: The cost ceiling it ran under: `free`, `local` or `llm`.
    tier: str
    #: The answer itself, shaped by the definition that produced it.
    payload: dict[str, Any] = field(default_factory=dict)
    #: A hash of what the answer read.
    inputs_fingerprint: str = ""
    #: Headline numbers, and `model` for an aggregate made by a model.
    stats: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"document": "aggregate", "version": 1,
                "video_id": self.video_id, "aggregate_id": self.aggregate_id,
                "tier": self.tier, "inputs_fingerprint": self.inputs_fingerprint,
                "payload": self.payload, "stats": self.stats}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Aggregate":
        return cls(video_id=d["video_id"], aggregate_id=d["aggregate_id"],
                   tier=d.get("tier", "free"), payload=d.get("payload", {}),
                   inputs_fingerprint=d.get("inputs_fingerprint", ""),
                   stats=d.get("stats", {}))


# --------------------------------------------------------------------------
# 10 · excerpt and sightings  --  what ONE aggregate reads, as a file
# --------------------------------------------------------------------------

@dataclass
class Excerpt:
    """The rows one selection took from a record: what a text aggregate reads.

    A row is `{chunk_id, start_ts, end_ts, parts: [[source, text], ...]}`.
    Chunks the selection found nothing in are absent.
    """

    #: Which video, or combination, the rows came from.
    video_id: str
    #: The selection as written, e.g. `transcript+clip:activity`.
    selection: str
    #: The grid, as `timeline.json` holds it.
    timeline: dict[str, Any] = field(default_factory=dict)
    #: One entry per chunk that the selection found something in.
    rows: list[dict[str, Any]] = field(default_factory=list)
    #: The answer ids that contributed, and `transcript` when it did.
    answers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"document": "excerpt", "version": 1, "video_id": self.video_id,
                "selection": self.selection, "count": len(self.rows),
                "answers": self.answers, "timeline": self.timeline,
                "rows": self.rows}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Excerpt":
        return cls(video_id=d["video_id"], selection=d.get("selection", ""),
                   timeline=d.get("timeline", {}), rows=d.get("rows", []),
                   answers=d.get("answers", []))


@dataclass
class Sightings:
    """The mentions one link profile reads: every list entry that might be an
    entity, with the answer it came from.
    """

    #: Which video, or combination, the mentions came from.
    video_id: str
    #: The link profile they were selected for, e.g. `people`.
    profile: str
    #: The input as written, e.g. `*` or `yolo[people.clothing]`.
    selection: str
    #: The list field linked, e.g. `people`.
    field_name: str = ""
    #: The identity keys each mention is signed by.
    keys: list[str] = field(default_factory=list)
    #: The grid, as `timeline.json` holds it.
    timeline: dict[str, Any] = field(default_factory=dict)
    #: `{chunk_id, sampler_id, field, index, signature, entry}` per mention.
    mentions: list[dict[str, Any]] = field(default_factory=list)
    #: `{chunk_id: text}` for the chunks mentioned, when the profile reads the
    #: transcript.
    transcript: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"document": "sightings", "version": 1, "video_id": self.video_id,
                "profile": self.profile, "selection": self.selection,
                "field": self.field_name, "keys": self.keys,
                "count": len(self.mentions), "timeline": self.timeline,
                "mentions": self.mentions, "transcript": self.transcript}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sightings":
        return cls(video_id=d["video_id"], profile=d.get("profile", ""),
                   selection=d.get("selection", ""), field_name=d.get("field", ""),
                   keys=d.get("keys", []), timeline=d.get("timeline", {}),
                   mentions=d.get("mentions", []),
                   transcript={str(k): v for k, v in (d.get("transcript") or {}).items()})


def fingerprint_of(payload: dict[str, Any]) -> str:
    """A stable hash of arbitrary content."""
    return _hash(payload)


# --------------------------------------------------------------------------
# what every component hands back
# --------------------------------------------------------------------------

@dataclass
class Produced:
    """What a component wrote: its artifacts, headline stats, and what it skipped."""

    #: Which video. Read the id from here: `media` may mint a new one.
    video_id: str
    #: Which component produced this.
    component: str
    #: `{artifact name: where it went}`, for what was actually written.
    artifacts: dict[str, str] = field(default_factory=dict)
    #: The component's headline numbers.
    stats: dict[str, Any] = field(default_factory=dict)
    #: What did not happen, e.g. `["evidence"]` for a policy that needs none.
    skipped: list[str] = field(default_factory=list)

    def path(self, name: str) -> str:
        return self.artifacts[name]

    def as_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id, "component": self.component,
                "artifacts": self.artifacts,
                "stats": self.stats, "skipped": self.skipped}


def same_video(**documents: Any) -> str:
    """The one video id these documents agree on, or a refusal naming the argument
    that differs.
    """
    seen = {name: doc.video_id for name, doc in documents.items()
            if doc is not None}
    ids = set(seen.values())
    if len(ids) > 1:
        detail = ", ".join(f"{name}={vid!r}" for name, vid in sorted(seen.items()))
        raise Refused(
            f"these documents are not the same video: {detail}. "
            f"Each carries its own id, and a component works on one video.")
    return next(iter(ids)) if ids else ""


DOCUMENTS = {"media": Media, "raw_transcript": RawTranscript,
             "cuts": Cuts, "timeline": Timeline, "manifest": Manifest,
             "transcript": Transcript, "descriptions": Descriptions,
             "embedded": Embedded,
             "aggregate": Aggregate,
             "excerpt": Excerpt, "sightings": Sightings}

__all__ = ["PRECISION", "VideoStream", "AudioStream", "Media",
           "RawTranscript", "Cuts", "Timeline", "Manifest", "Transcript",
           "Descriptions", "Embedded", "Aggregate", "Excerpt", "Sightings",
           "Produced", "fingerprint_of", "DOCUMENTS", "same_video"]
