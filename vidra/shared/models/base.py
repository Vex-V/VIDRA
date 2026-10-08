"""What a model is to this library: three base classes, one per role.

    VLM        frames and text in, a structured answer out   describe
    LLM        text in, text or a structured answer out      the llm aggregates
    Embedder   text in, vectors out                          embed, search, linking

Subclass one and fill in its method to bring any model:

    class MyVLM(VLM):
        key = "mylab:vision-v2"
        async def generate(self, parts, schema=None, system=None,
                           max_output_tokens=4000):
            ...   # parts: {"type": "text", "text": ...} / {"type": "image", "data": bytes, "mime": ...}
            return {"summary": ...}          # a dict when `schema` is given

`key` is the model's identity and is required. It is recorded with every
answer and every vector, and it decides whether stored work is still current:
a different model needs a different key. For an embedder it names the vector
space -- vectors under different keys are never compared.

A method may be `async def` or a plain `def` (run in a thread). The library
calls it through `calls()`, which caps calls in flight at `concurrency`, parses
a JSON string returned for a schema, and turns any failure into `ModelFailed`.

The ready-made classes are in `llm.py` (`OpenAI`, `Chat`, `Anthropic`,
`Stub`) and `embedders/` (`OpenAIEmbedder`, `LocalEmbedder`, `HashEmbedder`).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import weakref
from typing import Any, Optional, Sequence

from ..reporting.errors import ModelUnavailable, Refused, Unavailable, VidraError


class ModelFailed(Unavailable):
    """A model call that failed: a refusal, a network error, an answer that
    was not the shape asked for."""


# ------------------------------------------------------------ request parts

def text(value: str) -> dict[str, Any]:
    """A part of a request: some text."""
    return {"type": "text", "text": value}


def image(data: bytes, mime: str = "image/jpeg") -> dict[str, Any]:
    """A part of a request: an image, as the bytes the store already holds."""
    return {"type": "image", "data": data, "mime": mime}


def parse_json(raw: str) -> Any:
    """JSON out of a model's text, tolerating code fences."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    body = raw.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = body.find("{"), body.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(body[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ModelFailed(f"response was not JSON: {exc}") from None
    raise ModelFailed("response was not JSON: no object in it")


# ------------------------------------------------------------- the classes

class _Model:
    #: The model's identity, e.g. `openai:gpt-5.4-mini`. Required.
    key: str = ""
    #: Calls in flight at once.
    concurrency: int = 8
    #: How a structured answer is enforced; recorded in describe's resume key.
    response_mode: str = "custom"

    @property
    def name(self) -> str:
        """The part of `key` before the first colon."""
        return self.key.partition(":")[0]

    @property
    def model(self) -> str:
        """The part of `key` after the first colon."""
        return self.key.partition(":")[2]

    def problems(self) -> list[str]:
        """What stops this model running, checked before any work starts: a
        missing API key, for instance. Override to add checks."""
        return []

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.key!r})"


class VLM(_Model):
    """A vision-language model: describes frames."""

    async def generate(self, parts: Sequence[dict[str, Any]],
                       schema: Optional[dict[str, Any]] = None,
                       system: Optional[str] = None,
                       max_output_tokens: int = 4000) -> Any:
        """`parts` in order: the instruction, then a label and an image per frame.
        `schema` is `{"name", "schema"}` (a JSON Schema); return a dict matching
        it, or a JSON string. Without a schema, return text."""
        raise NotImplementedError(f"{type(self).__name__} does not implement generate()")


class LLM(_Model):
    """A text model: summaries, chapters, events, entity accounts."""

    async def complete(self, prompt: str, schema: Optional[dict[str, Any]] = None,
                       system: Optional[str] = None,
                       max_output_tokens: int = 4000) -> Any:
        """`schema` is `{"name", "schema"}` (a JSON Schema); return a dict
        matching it, or a JSON string. Without a schema, return text."""
        raise NotImplementedError(f"{type(self).__name__} does not implement complete()")


class Embedder(_Model):
    """Text to vectors. Every vector it makes must have the same width."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in order."""
        raise NotImplementedError(f"{type(self).__name__} does not implement embed()")

    def embed_query(self, text: str) -> list[float]:
        """The vector for a search. Override for a model that embeds a query
        differently from a passage."""
        return self.embed([text])[0]


#: Role -> the class a model must be to fill it.
ROLES: dict[str, type] = {"vlm": VLM, "llm": LLM, "embedder": Embedder}


# --------------------------------------------------------------- checking

def _misfit(role: str, model: Any) -> Optional[str]:
    """Why `model` is not the right kind of thing for `role`, or None. A `key`
    that is a property is trusted: reading it may cost a call (an embedder
    probing its width), and this check runs before any work."""
    wanted = ROLES[role]
    if not isinstance(model, wanted):
        article = "a" if wanted is VLM else "an"
        return (f"{role} must be {article} {wanted.__name__}, not {type(model).__name__}"
                + (" -- pass a model object, e.g. OpenAI(\"gpt-5.4-mini\")"
                   if isinstance(model, str) else ""))
    declared = inspect.getattr_static(model, "key", "")
    if isinstance(declared, property):
        return None
    if not isinstance(declared, str) or not declared.strip():
        return (f"{type(model).__name__} has no key: set `key` to a name for the "
                "model, e.g. \"mylab:vision-v2\"")
    return None


def problems(role: str, model: Any) -> list[str]:
    """What stops `model` filling `role`, as messages; empty means it can."""
    misfit = _misfit(role, model)
    if misfit:
        return [misfit]
    try:
        return [f"{role}: {p}" for p in model.problems()]
    except Exception as exc:                                  # noqa: BLE001
        return [f"{role}: {type(model).__name__}.problems() failed: {exc}"]


def require(role: str, model: Any) -> None:
    """`problems`, raising the first: a wrong type or no key is `Refused`,
    anything the model reports (a missing API key) is `ModelUnavailable`."""
    misfit = _misfit(role, model)
    if misfit:
        raise Refused(misfit)
    found = problems(role, model)
    if found:
        raise ModelUnavailable(found[0])


# ---------------------------------------------------------------- calling

def _gate(model: Any) -> asyncio.Semaphore:
    """`model`'s cap on calls in flight, for this event loop (a semaphore
    belongs to one loop). Kept on the model: every caller shares it."""
    gates = model.__dict__.setdefault("_vidra_gates", weakref.WeakKeyDictionary())
    loop = asyncio.get_running_loop()
    if loop not in gates:
        gates[loop] = asyncio.Semaphore(max(1, int(getattr(model, "concurrency", 1) or 1)))
    return gates[loop]


async def _call(model: Any, method: str, first: Any, schema: Optional[dict[str, Any]],
                system: Optional[str], max_output_tokens: int) -> Any:
    bound = getattr(model, method)
    kwargs = {"schema": schema, "system": system, "max_output_tokens": max_output_tokens}
    async with _gate(model):
        try:
            if inspect.iscoroutinefunction(bound):
                answer = await bound(first, **kwargs)
            else:
                answer = await asyncio.to_thread(bound, first, **kwargs)
        except VidraError:
            raise
        except Exception as exc:                              # noqa: BLE001
            raise ModelFailed(f"{model.key}: {type(exc).__name__}: {exc}") from None
    if schema is not None:
        if isinstance(answer, str):
            answer = parse_json(answer)
        if not isinstance(answer, dict):
            raise ModelFailed(f"{model.key} answered {type(answer).__name__}, "
                              f"not an object matching {schema.get('name')}")
    elif not isinstance(answer, str):
        raise ModelFailed(f"{model.key} answered {type(answer).__name__}, not text")
    return answer


class Calls:
    """A model as the library calls it: see `calls`."""

    def __init__(self, model: Any) -> None:
        self.target = model
        self.key: str = model.key
        self.concurrency: int = max(1, int(getattr(model, "concurrency", 1) or 1))
        self.response_mode: str = getattr(model, "response_mode", "custom")

    @property
    def name(self) -> str:
        return self.target.name

    @property
    def model(self) -> str:
        return self.target.model

    async def generate(self, parts: Sequence[dict[str, Any]],
                       schema: Optional[dict[str, Any]] = None,
                       system: Optional[str] = None,
                       max_output_tokens: int = 4000) -> Any:
        return await _call(self.target, "generate", list(parts), schema, system,
                           max_output_tokens)

    async def complete(self, prompt: str, schema: Optional[dict[str, Any]] = None,
                       system: Optional[str] = None,
                       max_output_tokens: int = 4000) -> Any:
        return await _call(self.target, "complete", prompt, schema, system,
                           max_output_tokens)


def calls(role: str, model: Any) -> Calls:
    """`model`, checked for `role`, wrapped the way the library calls it."""
    require(role, model)
    return Calls(model)


__all__ = ["Calls", "Embedder", "LLM", "ModelFailed", "ROLES",
           "VLM", "calls", "image", "parse_json", "problems", "require", "text"]
