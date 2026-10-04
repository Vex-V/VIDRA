"""Reading what streams a file carries.

Opens the container once, records what each half needs to open its own
decoder, and closes it. A missing stream is a fact (`has_video`,
`has_audio`); only a file carrying neither raises.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import av

from falconvar.shared.contracts.documents import (AudioStream, Media, VideoStream,
                                           fingerprint_of)
from falconvar.shared.reporting.errors import FalconvarError, Refused


class UnusableMedia(FalconvarError, RuntimeError):
    """The file cannot be opened, or carries neither a video nor an audio stream."""


#: The tags a recording time is read from, in order, compared without case.
TIME_TAGS = ("com.apple.quicktime.creationdate", "creation_time",
             "date_recorded", "date")


def parse_time(value: Any) -> Optional[str]:
    """A recording time as ISO 8601, or None. Takes a string or a `datetime`; a
    year alone, a date with no time, or anything at or before 1970 is None.
    """
    if isinstance(value, datetime):
        when = value
    else:
        try:
            when = datetime.fromisoformat(str(value).strip())
        except ValueError:
            return None
        if "T" not in str(value) and " " not in str(value).strip():
            return None                          # a date, not a moment
    return None if when.year <= 1970 else when.isoformat()


def _tagged_time(container: Any) -> Optional[str]:
    """The first tag, container then streams, that holds a usable time."""
    tags = [container.metadata, *(s.metadata for s in container.streams)]
    for wanted in TIME_TAGS:
        for found in tags:
            for key, value in found.items():
                if key.lower() == wanted and (when := parse_time(value)):
                    return when
    return None


def _given(name: Optional[str], when: Any) -> tuple[Optional[str], Optional[str]]:
    """The caller's name and time, checked before the file is opened."""
    if name is not None and not str(name).strip():
        # None means the default; an empty name is refused.
        raise Refused("name must not be empty; pass None for the filename")
    stamp = None
    if when is not None:
        stamp = parse_time(when)
        if stamp is None:
            raise Refused(f"recorded_at {when!r} is not a date and time after "
                          f"1970; pass a datetime or ISO 8601, e.g. "
                          f"2024-03-15T09:30:00+00:00")
    return (str(name).strip() if name is not None else None), stamp


def _seconds(value: Optional[int], time_base) -> Optional[float]:
    """A stream duration in seconds, or None when the container does not say."""
    if value is None or time_base is None:
        return None
    return float(value * time_base)


def fingerprint(path: Path, described: Media) -> str:
    """What file this is, independently of its name or where it sits: a hash of
    its size, duration, container and both streams.
    """
    return fingerprint_of({
        "bytes": path.stat().st_size,
        "duration_s": described.duration_s,
        "container": described.container_format,
        "video": described.video.as_dict() if described.video else None,
        "audio": described.audio.as_dict() if described.audio else None,
    })


def split(path: str | Path, video_id: Optional[str] = None,
          name: Optional[str] = None,
          recorded_at: Optional[str | datetime] = None) -> Media:
    """Open `path` once and describe the two streams it carries.

    `name` defaults to the filename and `recorded_at` to the container's tags (or
    None); either given here replaces its default. Neither is part of `source`.
    """
    given_name, given_time = _given(name, recorded_at)
    path = Path(path)
    if not path.exists():
        raise UnusableMedia(f"{path} does not exist")

    try:
        container = av.open(str(path))
    except Exception as exc:                                # noqa: BLE001
        raise UnusableMedia(
            f"cannot open {path} -- unsupported, missing or corrupt "
            f"({type(exc).__name__})") from None

    try:
        # The container's duration: the streams can disagree by milliseconds.
        duration = (container.duration / av.time_base
                    if container.duration is not None else None)

        v = next(iter(container.streams.video), None)
        a = next(iter(container.streams.audio), None)

        video = None if v is None else VideoStream(
            index=v.index,
            codec=v.codec_context.name,
            # `guessed_rate` is derived from the timestamps; prefer it.
            rate=float(v.guessed_rate or v.average_rate or 0) or None,
            # As a string, so the exact rational survives JSON.
            time_base=str(v.time_base) if v.time_base else None,
            width=v.codec_context.width,
            height=v.codec_context.height,
            # 0 means the container did not count.
            frames=v.frames or None,
            duration_s=_seconds(v.duration, v.time_base),
        )

        audio = None if a is None else AudioStream(
            index=a.index,
            codec=a.codec_context.name,
            rate=a.rate,
            channels=a.channels,
            duration_s=_seconds(a.duration, a.time_base),
        )
        container_format = container.format.name
        stamped = given_time or _tagged_time(container)
    finally:
        container.close()

    if video is None and audio is None:
        raise UnusableMedia(f"{path} carries neither a video nor an audio stream")

    described = Media(
        video_id=video_id or path.stem,
        path=str(path),
        container_format=container_format,
        duration_s=duration,
        video=video,
        audio=audio,
        name=given_name or path.name,
        recorded_at=stamped,
    )
    return replace(described, source=fingerprint(path, described))


__all__ = ["TIME_TAGS", "UnusableMedia", "fingerprint", "parse_time", "split"]
