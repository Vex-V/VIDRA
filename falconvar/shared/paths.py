"""Where everything lives.

Imports only `errors`, which imports nothing -- so this stays a leaf and no
cycle is possible. The only module that knows an artifact's filename: a caller
names a video and an artifact and never concatenates a path.

**A checkout is recognised, not guessed.** Searching upward for
`pyproject.toml` finds *a* marker, not necessarily this project's: installed
into a venv inside someone else's repo, the search walked out of
`site-packages` and returned *their* root, so the library imported fine and
wrote every artifact and 338 MB of checkpoints into their tree with nothing
reporting it. A marker counts only when the package under it is this one --
`root/falconvar/shared/paths.py` has to be this very file. That is also why a
parent count is wrong: it is a fact about a file's depth, which is what a
reorganisation changes, and this module has been moved once already.

**The roots resolve on first use, not at import.** They used to be module
constants, so an installed copy raised `RuntimeError` from `import falconvar`
before a caller could say where its data should go. Now `configure()` can be
called after the import that triggers it, and a library with no checkout
around it has somewhere to write.

Precedence, highest first: `configure()`, then `FALCONVAR_DATA` /
`FALCONVAR_WEIGHTS`, then the checkout when there is one, then
`~/.falconvar`. `configure()` wins because an embedding application must be
able to guarantee where it writes; the variables are for when you do not
control the calling code.

**Not for keys.** `shared/env.py` reads `.env`, which happens later than this
resolves -- so a data root set in `.env` would take effect or not depending on
which module was imported first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
from .errors import FalconvarError, Unavailable


class NotACheckout(Unavailable, RuntimeError):
    """A development tool run from an installed copy: it writes source files
    under the repository (`db/json`, `TYPES.txt`), and there is none."""


#: Where an installed copy writes when nothing says otherwise.
FALLBACK_HOME = Path.home() / ".falconvar"


def _checkout_root() -> Optional[Path]:
    """The checkout this file belongs to, or None if it is installed.

    A `pyproject.toml` above us is only ours if the package beneath it is this
    package -- otherwise it belongs to whoever we were installed into.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if not (parent / "pyproject.toml").exists():
            continue
        candidate = parent / "falconvar" / "shared" / "paths.py"
        if candidate.exists() and candidate.resolve() == here:
            return parent
    return None


#: The checkout, or None when installed. Resolved once: it cannot change.
CHECKOUT = _checkout_root()

#: What `configure()` set, if anything. One process, one default.
_configured: dict[str, Path] = {}

def configure(data_root: Optional[Path | str] = None,
              weights: Optional[Path | str] = None) -> None:
    """Say where this process reads and writes. Takes precedence over both the
    environment and the checkout.

    Call it before anything runs. Paths already handed out are not revisited --
    they are computed per call, so only work started earlier keeps the old root.
    """
    for key, value in (("data", data_root), ("weights", weights)):
        if value is not None:
            _configured[key] = Path(value).expanduser().resolve()


def _resolve(key: str, variable: str, in_checkout: str) -> Path:
    # `configure()` outranks the environment: an application embedding this
    # must be able to guarantee where it writes.
    if key in _configured:
        return _configured[key]
    value = os.environ.get(variable)
    if value:
        return Path(value).expanduser().resolve()
    if CHECKOUT is not None:
        return CHECKOUT / in_checkout
    return FALLBACK_HOME / in_checkout


def data_root() -> Path:
    return _resolve("data", "FALCONVAR_DATA", "data")


def weights_root() -> Path:
    return _resolve("weights", "FALCONVAR_WEIGHTS", "weights")


def out_root() -> Path:
    return data_root() / "out"


def checkout_root() -> Path:
    """The checkout, for the tools that write *source* rather than output.

    Raises when installed: `schemas --check` regenerates files under `db/`,
    which only exists in a checkout, and a silent wrong directory is what this
    module exists to prevent.
    """
    if CHECKOUT is None:
        raise NotACheckout(
            "not running from a FalCONvar checkout -- this is a development "
            "tool and needs the repository, not an installed copy")
    return CHECKOUT


#: Lazily-resolved module attributes, each `paths.NAME` a call to its function.
#: Kept as names so every reader stays `paths.OUT_ROOT`, while the value is
#: computed now rather than at import.
_LAZY: dict[str, Any] = {
    "REPO_ROOT": checkout_root,
    "DATA_ROOT": data_root,
    "OUT_ROOT": out_root,
    "WEIGHTS": weights_root,
    "UPLOADS": lambda: data_root() / "uploads",
    #: Custom describe prompts, added through the API. Under the data root
    #: rather than in the package because a request writes it; the built-ins
    #: stay in `falconvar/video_rag/describe/prompts.json`, read-only.
    "PROMPTS": lambda: data_root() / "prompts.json",
    #: Custom aggregate prompts and link profiles. The built-ins live in
    #: `falconvar/aggregates/definitions/definitions.json`.
    "AGGREGATE_DEFINITIONS": lambda: data_root() / "aggregates.json",
    #: Model endpoints added or overridden per deployment. Hand-edited, and
    #: never holds a key -- it names the variables keys are read from.
    "PROVIDERS": lambda: data_root() / "providers.json",
}


def __getattr__(name: str) -> Any:
    """PEP 562: `paths.OUT_ROOT` resolves on access, not at import."""
    if name in _LAZY:
        return _LAZY[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: artifact name -> filename under `data/out/<video-id>/`.
#: The whole naming convention, in one place. A component asks for "timeline";
#: nothing anywhere else spells "timeline.json".
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

#: artifact name -> the component that writes it, spelled as a caller would
#: run it. Beside the filename because it answers the other half of the same
#: question: a reader that has just failed to find `timeline.json` wants to
#: know who makes one, and only this module knew the filename in the first
#: place.
#:
#: Every component that reads another's output arrives here, so this is what
#: turns `[Errno 2] No such file or directory: 'C:\...\timeline.json'` -- which
#: asks a user to already know which component writes that file -- into the
#: name of the step they skipped. `boundaries` was the only component saying
#: this for itself, and it had to hand-write the sentence twice.
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
    # Not artifacts of a video's folder -- an aggregate's inputs and answers
    # go wherever their caller says -- but a reader that finds none still
    # wants the step that writes one.
    "excerpt": "aggregates.select",
    "sightings": "aggregates.select",
    "aggregate": "aggregates",
}


class UnknownArtifact(FalconvarError, KeyError):
    """An artifact name nothing in ARTIFACTS or DIRECTORIES answers to."""


class MissingArtifact(FalconvarError, FileNotFoundError):
    """An artifact that has not been produced yet.

    A `FileNotFoundError` still, so every `except FileNotFoundError` already
    written keeps working -- including the ones in each driver's `main`, which
    is what turns this into `error: ...` on a CLI rather than a traceback.
    """


class UnusableVideoId(FalconvarError, ValueError):
    """A video id that cannot name a directory under OUT_ROOT."""


#: A leading underscore marks a directory under OUT_ROOT that is not a video.
#: Written for the embedded Qdrant store, which lived at `_qdrant` beside the
#: videos rather than inside one and was listed as a video with no artifacts --
#: by `paths.videos()`, and so by `GET /videos`. That store is gone; the rule
#: is not, because the next directory that is not a video would repeat it.
#: Read by `videos()`, which skips such a directory, and by `check_id`, which
#: refuses to make one.
RESERVED_PREFIX = "_"

#: What an id may be made of. Deliberately `api/main.safe_id`'s alphabet, so
#: whatever that cleaning produces is guaranteed to be accepted here -- one
#: rule, stated once, rather than a cleaner and a checker that can disagree.
_ALLOWED = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"


def check_id(video_id: str) -> str:
    """The id, or raise. Every path this module hands out goes through it.

    **An id becomes a directory name, so it is checked rather than trusted.**
    A library takes its id from whoever calls it -- an HTTP path segment, a
    filename, a job record -- and `out_root() / video_id` is a join, not a
    containment. `../../escaped` wrote two levels *above* the data root while
    `Produced.video_id` still read it back unchanged, so nothing anywhere
    reported that a run had left its own tree.

    A leading underscore is refused for a different reason, and it is the
    quieter failure: `RESERVED_PREFIX` marks a directory that is not a video,
    and it was enforced on read only. `_hidden` wrote a complete, correct
    output directory that `videos()` -- and so `GET /videos`, and so the whole
    client -- would never list again.

    Cleaning instead of refusing would be worse: two ids differing only in a
    refused character would silently land in one directory and overwrite each
    other's artifacts.
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
            f"marks a directory under OUT_ROOT that is not a video, so `videos()` "
            f"would never list it again")
    if set(video_id) == {"."}:                  # '.' and '..', which pass the alphabet
        raise UnusableVideoId(f"video id {video_id!r} is not a directory name")
    return video_id


def home(video_id: str) -> Path:
    """Everything one video produced, in one directory.

    Grouped by video rather than by artifact type, so a video's whole output is
    one thing to inspect, copy or delete, and a later component adds to it
    without a new top-level directory.

    The id is checked here because this is the one place both tiers join it to
    a root -- `artifact`, `exists`, `present` and every component's `load` all
    arrive through this line.
    """
    return out_root() / check_id(video_id)


def artifact(video_id: str, name: str) -> Path:
    """The path an artifact would occupy. It need not exist yet."""
    if name in ARTIFACTS:
        return home(video_id) / ARTIFACTS[name]
    if name in DIRECTORIES:
        return home(video_id) / DIRECTORIES[name]
    known = ", ".join(sorted({*ARTIFACTS, *DIRECTORIES}))
    raise UnknownArtifact(f"unknown artifact {name!r}; known: {known}")


def exists(video_id: str, name: str) -> bool:
    return artifact(video_id, name).exists()


def require(video_id: str, name: str) -> Path:
    """The artifact's path, or raise saying which component would make it.

    **"That video does not exist" and "you skipped a step" are different
    answers**, and they used to be the same `[Errno 2]`. A caller driving the
    components itself -- which is half of what the library is for -- gets the
    order wrong regularly, and the raw path tells them nothing unless they
    already know which component writes `timeline.json`.
    """
    path = artifact(video_id, name)
    if path.exists():
        # The one line every artifact read passes through, which is why the
        # DEBUG record lives here rather than in eleven callers. `PRODUCED_BY`
        # names the component, so a record says who wrote what is being read.
        from . import logs
        logs.read(PRODUCED_BY.get(name, "shared"), video_id, name)
        return path
    if not home(video_id).exists():
        raise MissingArtifact(
            f"no video {video_id!r} under {out_root()} -- "
            f"`media(path)` is what creates one")
    producer = PRODUCED_BY.get(name)
    made_by = f"; `{producer}` writes it" if producer else ""
    raise MissingArtifact(
        f"{video_id} has no {name}{made_by}. Present: "
        f"{', '.join(present(video_id)) or 'nothing yet'}")


def present(video_id: str) -> list[str]:
    """Which artifacts this video actually has, in pipeline order.

    Read from disk rather than remembered, so a restarted process still knows
    everything it produced -- and so a listing advertises no artifact that
    would 404, which reads as breakage rather than as a stage never run.
    """
    order = [*ARTIFACTS, *DIRECTORIES]
    return [name for name in order if exists(video_id, name)]


def videos() -> list[str]:
    """Every video id with an output directory."""
    root = out_root()
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith(RESERVED_PREFIX))


__all__ = ["ARTIFACTS", "CHECKOUT", "DIRECTORIES", "FALLBACK_HOME",
           "PRODUCED_BY", "RESERVED_PREFIX", "MissingArtifact", "NotACheckout",
           "UnknownArtifact", "UnusableVideoId",
           "artifact", "check_id", "checkout_root", "configure", "data_root",
           "exists", "home", "out_root", "present", "require", "videos",
           "weights_root"]
