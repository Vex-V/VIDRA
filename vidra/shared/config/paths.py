"""Where everything lives: the data root, the weights cache, the checkout, and
every artifact's filename.

The data and weights roots resolve on first use, highest first:
`configure()`, then `VIDRA_DATA` / `VIDRA_WEIGHTS`, then the
checkout when there is one, then `~/.vidra`. A checkout counts only when
`root/vidra/shared/config/paths.py` is this file. Imports only `errors`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
from ..reporting.errors import VidraError, Unavailable


class NotACheckout(Unavailable, RuntimeError):
    """A development tool run from an installed copy: it writes source files
    under the repository (`db/json`), and there is none."""


#: Where an installed copy writes when nothing says otherwise.
FALLBACK_HOME = Path.home() / ".vidra"


def _checkout_root() -> Optional[Path]:
    """The checkout this file belongs to, or None when installed."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if not (parent / "pyproject.toml").exists():
            continue
        candidate = parent / "vidra" / "shared" / "config" / "paths.py"
        if candidate.exists() and candidate.resolve() == here:
            return parent
    return None


#: The checkout, or None when installed.
CHECKOUT = _checkout_root()

#: What `configure()` set.
_configured: dict[str, Path] = {}

def configure(data_root: Optional[Path | str] = None,
              weights: Optional[Path | str] = None) -> None:
    """Say where this process reads and writes; outranks the environment and the
    checkout. Call it before anything runs.
    """
    for key, value in (("data", data_root), ("weights", weights)):
        if value is not None:
            _configured[key] = Path(value).expanduser().resolve()


def _resolve(key: str, variable: str, in_checkout: str) -> Path:
    # `configure()` first, then the environment.
    if key in _configured:
        return _configured[key]
    value = os.environ.get(variable)
    if value:
        return Path(value).expanduser().resolve()
    if CHECKOUT is not None:
        return CHECKOUT / in_checkout
    return FALLBACK_HOME / in_checkout


def data_root() -> Path:
    return _resolve("data", "VIDRA_DATA", "data")


def weights_root() -> Path:
    return _resolve("weights", "VIDRA_WEIGHTS", "weights")


def out_root() -> Path:
    return data_root() / "out"


def checkout_root() -> Path:
    """The checkout, for tools that write source files (`db/json`).
    Raises when installed.
    """
    if CHECKOUT is None:
        raise NotACheckout(
            "not running from a VIDRA checkout -- this is a development "
            "tool and needs the repository, not an installed copy")
    return CHECKOUT


#: Module attributes computed on access: `paths.NAME` calls its function.
_LAZY: dict[str, Any] = {
    "REPO_ROOT": checkout_root,
    "DATA_ROOT": data_root,
    "OUT_ROOT": out_root,
    "WEIGHTS": weights_root,
    #: Custom describe questions (the built-ins ship in the package).
    "PROMPTS": lambda: data_root() / "prompts.json",
    #: Custom aggregate prompts and link profiles.
    "AGGREGATE_DEFINITIONS": lambda: data_root() / "aggregates.json",
}


def __getattr__(name: str) -> Any:
    """PEP 562: `paths.OUT_ROOT` resolves on access, not at import."""
    if name in _LAZY:
        return _LAZY[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Artifact name -> filename under `data/out/<video-id>/`.
ARTIFACTS: dict[str, str] = {
    "media": "media.json",
    "raw_transcript": "transcript.raw.json",
    "cuts": "cuts.json",
    "timeline": "timeline.json",
    "manifest": "manifest.json",
    "transcript": "transcript.json",
    "descriptions": "descriptions.json",
    "embedded": "embedded.json",
}

#: Artifacts that are directories rather than documents.
DIRECTORIES: dict[str, str] = {
    "store": "store",
    "aggregates": "aggregates",
}

#: Artifact name -> the component that writes it, named in a missing-artifact
#: error.
PRODUCED_BY: dict[str, str] = {
    "media": "media",
    "raw_transcript": "audio",
    "cuts": "boundaries.evidence",
    "timeline": "boundaries",
    "manifest": "video",
    "transcript": "cut",
    "descriptions": "describe",
    "embedded": "embed",
    "store": "video",
    "aggregates": "aggregates",
    # Aggregate inputs and answers, which live wherever their caller says.
    "excerpt": "record.excerpt(out=...)",
    "sightings": "record.sightings(out=...)",
    "aggregate": "aggregates",
}


class MissingArtifact(VidraError, FileNotFoundError):
    """An artifact that has not been produced yet. Still a `FileNotFoundError`."""


class UnusableVideoId(VidraError, ValueError):
    """A video id that cannot name a directory under OUT_ROOT."""


#: A leading underscore marks a directory under OUT_ROOT that is not a video:
#: `check_id` refuses it and listings skip it.
RESERVED_PREFIX = "_"

#: The characters a video id may contain.
_ALLOWED = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"


def check_id(video_id: str) -> str:
    """The id, or raise. Every path this module hands out goes through it: an id
    may not climb out of the data root, start with `_`, or use other characters.
    """
    if not video_id:
        raise UnusableVideoId("a video id cannot be empty")
    bad = sorted({c for c in video_id if c not in _ALLOWED})
    if bad:
        raise UnusableVideoId(
            f"video id {video_id!r} cannot contain {', '.join(repr(c) for c in bad)} "
            f"-- an id names a directory under {out_root()}, and travels on into "
            f"URLs, database rows and index payloads; ASCII letters, digits, "
            f"'.', '_' and '-' only. An id defaults to the filename stem, so "
            f"name one explicitly when the file's does not qualify.")
    if video_id.startswith(RESERVED_PREFIX):
        raise UnusableVideoId(
            f"video id {video_id!r} cannot start with {RESERVED_PREFIX!r}: that "
            f"marks a directory under OUT_ROOT that is not a video, so it "
            f"would never be listed")
    if set(video_id) == {"."}:                  # '.' and '..', which pass the alphabet
        raise UnusableVideoId(f"video id {video_id!r} is not a directory name")
    return video_id


__all__ = ["ARTIFACTS", "CHECKOUT", "DIRECTORIES", "FALLBACK_HOME",
           "PRODUCED_BY", "RESERVED_PREFIX", "MissingArtifact", "NotACheckout",
           "UnusableVideoId", "check_id", "checkout_root", "configure",
           "data_root", "out_root", "weights_root"]
