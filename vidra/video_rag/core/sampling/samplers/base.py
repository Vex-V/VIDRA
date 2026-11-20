"""The sampler contract.

A sampler sees the decimated stream one frame at a time and answers yes or
no; it cannot look ahead or revisit a frame. The base class enforces:

    min_interval_s  smallest gap between two kept frames
    max_per_chunk   ceiling on frames kept from one chunk
    prompts         which questions describe should ask about these frames

Rate limits apply before `propose` runs, and every chunk keeps at least one
frame. Prompts are opaque strings here.

`views` says how a kept frame is shown to the model: by default the frame
itself (`View.FRAME`); a sampler may add or substitute images made from it --
crops, enlargements, a marked copy -- each stored beside the frame.

`config()` is what the manifest records, and the manifest's fingerprint is
what decides whether stored answers are still current -- so a setting left
out of it changes the frames kept and not the fingerprint, and describe
reuses answers about frames it no longer has. A subclass's own constructor
parameters are therefore recorded without being asked: each is read off the
instance (`self.<name>`, else `self._<name>`) and added to the base settings.
A setting with a `config()` of its own (a model part) is recorded by it; one
that is not plain JSON is refused by name. A class that writes `config()`
itself records by hand from there down, as the built-ins do -- their
fingerprints are unchanged.
"""

from __future__ import annotations

import inspect
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional, Sequence

from ..reader import Frame
from vidra.shared.reporting.errors import Refused


@dataclass
class View:
    """One image a kept frame is shown to the model as. `View.FRAME` is the frame
    itself, as stored; any other view carries its own pixels (BGR), a `label`
    the model reads beside it, and `meta` recorded in the manifest."""

    image: Any = field(default=None, repr=False)
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    FRAME: ClassVar["View"]

    @property
    def is_frame(self) -> bool:
        return self.image is None


View.FRAME = View()


class Deferred:
    """A part a sampler builds on first use, so making a sampler loads no
    model: `Deferred(CLIPEmbedder)`. Every attribute but `load` is the built
    part's, so code using the part never knows. Copies of a sampler share it,
    and it is built once."""

    def __init__(self, build: Callable[[], Any]) -> None:
        self._build = build
        self._part: Any = None
        self._lock = threading.Lock()

    def load(self) -> Any:
        """The part, built now if it was not."""
        with self._lock:
            if self._part is None:
                self._part = self._build()
        return self._part

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.load(), name)


#: Keys the base settings use; a subclass setting may not take one.
_BASE_KEYS = ("id", "name", "min_interval_s", "max_per_chunk", "prompts")


def _init_parameters(cls: type) -> list[str]:
    """The named parameters of `cls.__init__`, without self, *args, **kwargs."""
    try:
        params = inspect.signature(cls.__init__).parameters.values()
    except (TypeError, ValueError):
        return []
    return [p.name for p in params if p.name != "self"
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]


class Sampler(ABC):
    """Base class for frame-selection strategies."""

    name: str = "base"

    #: Constructor parameters recorded in `config()` without being written
    #: there: a subclass's own, from the last class that wrote `config()` down.
    _captured: ClassVar[tuple[str, ...]] = ()
    #: True on a class whose settings reach `config()` some other way (a
    #: built-in records its detector's settings through the detector).
    _settings_by_hand: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "config" in cls.__dict__ or cls.__dict__.get("_settings_by_hand"):
            cls._captured = ()
            return
        if "__init__" not in cls.__dict__:
            return                                     # inherits the parent's
        known = {name for base in cls.__mro__[1:] if "__init__" in base.__dict__
                 for name in _init_parameters(base)}
        new = [name for name in _init_parameters(cls) if name not in known]
        cls._captured = tuple(dict.fromkeys([*cls._captured, *new]))

    def __init__(self, min_interval_s: float = 0.0,
                 max_per_chunk: Optional[int] = None,
                 sampler_id: Optional[str] = None,
                 prompts: Optional[Sequence[str]] = None) -> None:
        if min_interval_s < 0:
            raise Refused("min_interval_s must be >= 0")
        if max_per_chunk is not None and max_per_chunk < 1:
            raise Refused("max_per_chunk must be >= 1; every chunk keeps a frame")
        self.min_interval_s = min_interval_s
        self.max_per_chunk = max_per_chunk
        self._sampler_id = sampler_id
        self.prompts: list[str] = list(prompts or [])
        self._kept_in_chunk = 0
        self._last_kept_ts: Optional[float] = None

    @property
    def sampler_id(self) -> str:
        """The manifest key for this run of this sampler: the strategy's name."""
        return self._sampler_id or self.name

    def reset(self, chunk_id: int) -> None:
        """Called once when a chunk opens, before any frame is offered: samplers
        forget everything at a boundary.
        """
        self._kept_in_chunk = 0
        self._last_kept_ts = None
        self.on_reset(chunk_id)

    def on_reset(self, chunk_id: int) -> None:
        """Subclass hook for clearing strategy state at a boundary."""

    def accepts(self, frame: Frame, chunk_local_index: int) -> bool:
        """Final decision for one frame. Do not override -- implement propose."""
        if self.max_per_chunk is not None and self._kept_in_chunk >= self.max_per_chunk:
            # Chunk is full.
            return False

        first_of_chunk = self._last_kept_ts is None
        if (not first_of_chunk
                and frame.media_ts - self._last_kept_ts < self.min_interval_s):
            return False

        # The strategy sees every allowed frame; the first frame of a chunk is kept
        # regardless.
        keep = self.propose(frame, chunk_local_index) or first_of_chunk
        if keep:
            self._kept_in_chunk += 1
            self._last_kept_ts = frame.media_ts
        return keep

    @abstractmethod
    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        """The strategy's opinion, before rate limits and the chunk guarantee.
        `chunk_local_index` counts decimated frames in the current chunk from 0.
        """

    # The two halves of a change sampler: describe a frame, compare two.

    def describe(self, frame: Frame) -> Any:
        """The model's output for this frame. None for positional samplers."""
        return None

    def compare(self, current: Any, reference: Any) -> Optional[float]:
        """Similarity in [0, 1]. Lower means more changed."""
        return None

    def last_score(self) -> Optional[float]:
        """What the last decision was based on, recorded in the manifest."""
        return None

    def views(self, frame: Frame) -> list[View]:
        """How a frame this sampler kept is shown to the model, in order. Called
        once per kept frame, with its pixels. The frame itself by default."""
        return [View.FRAME]

    def _base_config(self) -> dict[str, Any]:
        # `prompts` is omitted when empty.
        config: dict[str, Any] = {
            "id": self.sampler_id,
            "name": self.name,
            "min_interval_s": self.min_interval_s,
            "max_per_chunk": self.max_per_chunk,
        }
        if self.prompts:
            config["prompts"] = list(self.prompts)
        config.update(self._captured_settings())
        return config

    def _captured_settings(self) -> dict[str, Any]:
        """This class's own constructor settings, as the manifest records them."""
        cls = type(self).__name__
        out: dict[str, Any] = {}
        for setting in self._captured:
            if setting in _BASE_KEYS:
                raise Refused(f"{cls}'s setting {setting!r} has the name of a base "
                              f"setting; rename it, or write config() yourself")
            missing = object()
            value = getattr(self, setting, missing)
            if value is missing:
                value = getattr(self, "_" + setting, missing)
            if value is missing:
                raise Refused(
                    f"{cls} takes {setting!r} but keeps no attribute of that name, so "
                    f"the manifest cannot record it: store it as self.{setting} "
                    f"(or self._{setting}), or write config() yourself")
            if callable(getattr(value, "config", None)):
                value = value.config()
            try:
                out[setting] = json.loads(json.dumps(value))
            except (TypeError, ValueError):
                raise Refused(
                    f"{cls}'s setting {setting!r} is of type {type(value).__name__}, which the "
                    f"manifest cannot record: give it a config() returning plain JSON, "
                    f"or write {cls}.config() yourself") from None
        return out

    def questions(self) -> list[str]:
        """The questions to ask about this run's frames; empty means one named after
        the strategy.
        """
        return list(self.prompts) or [self.name]

    def config(self) -> dict[str, Any]:
        """Serialised into the manifest: the base settings and this class's own
        constructor settings. Override only to record something else."""
        return self._base_config()


__all__ = ["Deferred", "Sampler", "View"]
