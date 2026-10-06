"""The ready-made models: one class per wire format, each both a VLM and an LLM.

    OpenAI("gpt-5.4-mini")                       the Responses API
    Chat("gemma3:4b", base_url="http://localhost:11434/v1", name="ollama")
                                                 anything serving Chat Completions:
                                                 Ollama, LM Studio, llama.cpp, vLLM,
                                                 Gemini, Mistral, Groq, OpenRouter, ...
    Anthropic("claude-haiku-4-5")                the Messages API
    Stub()                                       no model: obviously fake answers

How a shape is enforced differs by format:

    OpenAI     `text.format` json_schema with `strict`
    Chat       `response_format` json_schema by default; json_object, or the
               schema in the prompt, for a server with no schema support
               (`structured=`)
    Anthropic  the schema as the input of a forced tool

A key is read when a call is made: `api_key=`, else the environment variable
named below. A client is opened per call (an async client belongs to the loop
that opened it). A truncated answer is detected from the response itself.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Sequence

from .base import LLM, VLM, ModelFailed, parse_json, text

#: Retries on the Anthropic path for a rate limit or an overloaded server.
RETRIES = 2
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})

#: How a `Chat` server is asked for a shape.
STRUCTURED = ("json_schema", "json_object", "prompt")


def _data_uri(part: dict[str, Any]) -> str:
    return f"data:{part['mime']};base64,{base64.b64encode(part['data']).decode('ascii')}"


def _schema_prompt(schema: dict[str, Any]) -> str:
    return ("Answer with a single JSON object and nothing else -- no prose, no "
            "code fence. It must match this JSON Schema exactly:\n"
            + json.dumps(schema["schema"], separators=(",", ":")))


def _retry_after(response: Any, attempt: int) -> float:
    """The server's `retry-after` when it sends one, else 1 s then 2 s."""
    try:
        return min(60.0, max(0.0, float(response.headers.get("retry-after", ""))))
    except ValueError:
        return float(2 ** attempt)


class _Remote(VLM, LLM):
    """What every API-backed model shares: a key, a client, the error wording."""

    #: Where a key is read from when `api_key=` is not given, first set wins.
    key_vars: tuple[str, ...] = ()
    #: Whether a call can be made with no key at all (a local server).
    keyless = False

    def __init__(self, label: str, model: str, api_key: Optional[str],
                 base_url: Optional[str], concurrency: int, client: Any) -> None:
        if not model or not str(model).strip():
            raise ModelFailed(f"{type(self).__name__} needs a model name")
        self.key = f"{label}:{model}"
        self.base_url = base_url
        self.concurrency = concurrency
        self._api_key = api_key
        self._client = client

    def __repr__(self) -> str:                    # never the API key
        where = f", base_url={self.base_url!r}" if self.base_url else ""
        return f"{type(self).__name__}({self.key!r}{where})"

    def api_key(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        for variable in self.key_vars:
            if os.environ.get(variable):
                return os.environ[variable]
        return None

    def problems(self) -> list[str]:
        if self.keyless or self.api_key():
            return []
        return [f"no key for {self.key}: pass api_key= or set "
                f"{' or '.join(self.key_vars)} in the environment"]

    def _needed_key(self) -> Optional[str]:
        found = self.api_key()
        if found is None and not self.keyless:
            raise ModelFailed(self.problems()[0])
        return found

    @asynccontextmanager
    async def _openai(self) -> AsyncIterator[Any]:
        if self._client is not None:
            yield self._client
            return
        from openai import AsyncOpenAI
        # The SDK needs a key string even for a server that ignores it.
        options: dict[str, Any] = {"api_key": self._needed_key() or "not-needed"}
        if self.base_url:
            options["base_url"] = self.base_url
        async with AsyncOpenAI(**options) as client:
            yield client

    def _failed(self, exc: Exception) -> ModelFailed:
        message = str(exc)
        lowered = message.lower()
        if "connect" in lowered and self.base_url:
            return ModelFailed(f"could not reach {self.key} at {self.base_url} "
                               f"-- is it running? ({message})")
        if "model" in lowered and ("not" in lowered or "unknown" in lowered):
            return ModelFailed(f"{self.name} rejected model {self.model!r}: {message}. "
                               "Name a model this server serves.")
        return ModelFailed(f"the {self.key} call failed: {message}")

    @staticmethod
    def _cut_short(reason: Any, limit: int) -> ModelFailed:
        return ModelFailed(f"response cut short ({reason}); raise max_output_tokens "
                           f"above {limit} or reduce how much is being sent")

    def _finish(self, raw: Optional[str], schema: Optional[dict[str, Any]]) -> Any:
        body = (raw or "").strip()
        if not body:
            raise ModelFailed(f"{self.key} returned nothing")
        return body if schema is None else parse_json(body)

    async def complete(self, prompt: str, schema: Optional[dict[str, Any]] = None,
                       system: Optional[str] = None,
                       max_output_tokens: int = 4000) -> Any:
        return await self.generate([text(prompt)], schema, system, max_output_tokens)


class OpenAI(_Remote):
    """OpenAI's Responses API. The key is `api_key=`, else `OPENAI_API_KEY`."""

    key_vars = ("OPENAI_API_KEY", "OPENAI_API")
    response_mode = "json_schema"

    def __init__(self, model: str = "gpt-5.4-mini", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, concurrency: int = 8,
                 client: Any = None) -> None:
        super().__init__("openai", model, api_key, base_url, concurrency, client)

    async def generate(self, parts: Sequence[dict[str, Any]],
                       schema: Optional[dict[str, Any]] = None,
                       system: Optional[str] = None,
                       max_output_tokens: int = 4000) -> Any:
        limit = max_output_tokens
        if len(parts) == 1 and parts[0]["type"] == "text":
            # A text-only request is sent as a bare string.
            content: Any = parts[0]["text"]
        else:
            content = [{"type": "input_text", "text": p["text"]} if p["type"] == "text"
                       else {"type": "input_image", "image_url": _data_uri(p)}
                       for p in parts]
        request: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": limit,
        }
        if system:
            request["instructions"] = system
        if schema is not None:
            request["text"] = {"format": {"type": "json_schema", "strict": True,
                                          **schema}}
        async with self._openai() as client:
            try:
                response = await client.responses.create(**request)
            except Exception as exc:                    # noqa: BLE001
                raise self._failed(exc) from None
        if getattr(response, "status", None) == "incomplete":
            raise self._cut_short(getattr(getattr(response, "incomplete_details", None),
                                          "reason", "unknown"), limit)
        return self._finish(getattr(response, "output_text", ""), schema)


class Chat(_Remote):
    """Any server speaking OpenAI's Chat Completions.

    `name` is the first half of the key (`ollama:gemma3:4b`), so two servers
    serving one model id can be told apart. `api_key` may be None for a local
    server; a hosted one wants it. `structured` is how a shape is asked for:
    `json_schema` (enforced by the server), `json_object`, or `prompt` (the
    schema in the prompt, parsed). `token_field` is the name the server takes
    the output cap under; a refusal naming one is retried with the other. Set
    `concurrency=1` for a server that answers one call at a time.
    """

    keyless = True

    def __init__(self, model: str, base_url: str, name: str = "chat",
                 api_key: Optional[str] = None, structured: str = "json_schema",
                 token_field: str = "max_tokens", concurrency: int = 8,
                 client: Any = None) -> None:
        if not base_url:
            raise ModelFailed("Chat needs base_url, e.g. http://localhost:11434/v1")
        if structured not in STRUCTURED:
            raise ModelFailed(f"structured must be one of {', '.join(STRUCTURED)}")
        if token_field not in ("max_tokens", "max_completion_tokens"):
            raise ModelFailed("token_field must be max_tokens or max_completion_tokens")
        super().__init__(name, model, api_key, base_url, concurrency, client)
        self.structured = structured
        self.token_field = token_field
        self.response_mode = structured

    async def generate(self, parts: Sequence[dict[str, Any]],
                       schema: Optional[dict[str, Any]] = None,
                       system: Optional[str] = None,
                       max_output_tokens: int = 4000) -> Any:
        limit = max_output_tokens
        mode = self.structured if schema is not None else None
        parts = list(parts)
        if mode in ("json_object", "prompt"):
            parts.append(text(_schema_prompt(schema)))

        if all(p["type"] == "text" for p in parts):
            content: Any = "\n\n".join(p["text"] for p in parts)
        else:
            content = [{"type": "text", "text": p["text"]} if p["type"] == "text"
                       else {"type": "image_url", "image_url": {"url": _data_uri(p)}}
                       for p in parts]
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        request: dict[str, Any] = {"model": self.model, "messages": messages}
        if mode == "json_schema":
            request["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema["name"], "strict": True, "schema": schema["schema"]}}
        elif mode == "json_object":
            request["response_format"] = {"type": "json_object"}

        field = self.token_field
        async with self._openai() as client:
            try:
                response = await client.chat.completions.create(**request, **{field: limit})
            except Exception as exc:                    # noqa: BLE001
                other = ("max_completion_tokens" if field == "max_tokens" else "max_tokens")
                if field not in str(exc):
                    raise self._failed(exc) from None
                try:
                    response = await client.chat.completions.create(**request,
                                                                    **{other: limit})
                except Exception as again:              # noqa: BLE001
                    raise self._failed(again) from None

        if not getattr(response, "choices", None):
            raise ModelFailed(f"{self.key} returned no choices")
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise self._cut_short("length", limit)
        message = choice.message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ModelFailed(f"{self.key} refused: {refusal}")
        return self._finish(getattr(message, "content", None), schema)


class Anthropic(_Remote):
    """Claude through the Messages API. The key is `api_key=`, else
    `ANTHROPIC_API_KEY`."""

    key_vars = ("ANTHROPIC_API_KEY",)
    response_mode = "tool"

    def __init__(self, model: str = "claude-haiku-4-5", api_key: Optional[str] = None,
                 base_url: str = "https://api.anthropic.com", concurrency: int = 8,
                 client: Any = None) -> None:
        super().__init__("anthropic", model, api_key, base_url, concurrency, client)

    @asynccontextmanager
    async def _http(self) -> AsyncIterator[Any]:
        if self._client is not None:
            yield self._client
            return
        import httpx
        async with httpx.AsyncClient(timeout=600) as client:
            yield client

    def _failed(self, exc: Exception) -> ModelFailed:
        # The base's "is it running?" is for a local server, which this is not.
        message = str(exc)
        if "model" in message.lower() and ("not" in message.lower()
                                           or "unknown" in message.lower()):
            return ModelFailed(f"anthropic rejected model {self.model!r}: {message}")
        return ModelFailed(f"the {self.key} call failed: {message}")

    async def generate(self, parts: Sequence[dict[str, Any]],
                       schema: Optional[dict[str, Any]] = None,
                       system: Optional[str] = None,
                       max_output_tokens: int = 4000) -> Any:
        limit = max_output_tokens
        key = self._needed_key()
        content = [{"type": "text", "text": p["text"]} if p["type"] == "text"
                   else {"type": "image", "source": {
                       "type": "base64", "media_type": p["mime"],
                       "data": base64.b64encode(p["data"]).decode("ascii")}}
                   for p in parts]
        body: dict[str, Any] = {"model": self.model, "max_tokens": limit,
                                "messages": [{"role": "user", "content": content}]}
        if system:
            body["system"] = system
        if schema is not None:
            # A forced tool whose input is the schema; its arguments are the answer.
            body["tools"] = [{"name": schema["name"],
                              "description": "Record the answer in exactly this shape.",
                              "input_schema": schema["schema"]}]
            body["tool_choice"] = {"type": "tool", "name": schema["name"]}
        headers = {"anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        if key:
            headers["x-api-key"] = key
        url = (self.base_url or "").rstrip("/") + "/v1/messages"

        async with self._http() as http:
            for attempt in range(RETRIES + 1):
                try:
                    response = await http.post(url, json=body, headers=headers,
                                               timeout=600)
                except Exception as exc:                # noqa: BLE001
                    raise self._failed(exc) from None
                if response.status_code not in RETRY_STATUS or attempt == RETRIES:
                    break
                await asyncio.sleep(_retry_after(response, attempt))
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", {}).get("message") or response.text
            except ValueError:
                detail = response.text
            raise self._failed(RuntimeError(f"{response.status_code}: {detail}"))

        data = response.json()
        if data.get("stop_reason") == "max_tokens":
            raise self._cut_short("max_tokens", limit)
        blocks = data.get("content") or []
        if schema is not None:
            for block in blocks:
                if block.get("type") == "tool_use":
                    return block.get("input") or {}
            raise ModelFailed(f"{self.key} answered without the {schema['name']} tool")
        return self._finish("".join(b.get("text", "") for b in blocks
                                    if b.get("type") == "text"), None)


# -------------------------------------------------------------------- stub

def _empty(spec: Any) -> Any:
    """The emptiest value a JSON Schema allows."""
    if not isinstance(spec, dict):
        return None
    if spec.get("enum"):
        return spec["enum"][0]
    kind = spec.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    if kind == "object":
        return {k: _empty(v) for k, v in (spec.get("properties") or {}).items()}
    return {"string": "", "array": [], "integer": 0, "number": 0.0,
            "boolean": False}.get(kind)


class Stub(VLM, LLM):
    """No model and no network: answers are obviously stub text, in the shape
    asked for. For exercising a pipeline without a key."""

    key = "stub"
    response_mode = "stub"

    def _shaped(self, summary: str, schema: Optional[dict[str, Any]]) -> Any:
        if schema is None:
            return summary
        properties = (schema.get("schema") or {}).get("properties") or {}
        return {k: summary if k == "summary" else _empty(v)
                for k, v in properties.items()}

    async def generate(self, parts: Sequence[dict[str, Any]],
                       schema: Optional[dict[str, Any]] = None,
                       system: Optional[str] = None,
                       max_output_tokens: int = 4000) -> Any:
        # The frame labels name each frame's index and time, so every answer differs.
        labels = [p["text"] for p in list(parts)[1:] if p["type"] == "text"]
        images = sum(1 for p in parts if p["type"] == "image")
        return self._shaped(f"[stub] {images} frames. " + " ".join(labels), schema)

    async def complete(self, prompt: str, schema: Optional[dict[str, Any]] = None,
                       system: Optional[str] = None,
                       max_output_tokens: int = 4000) -> Any:
        return self._shaped("[stub] " + " ".join(prompt.split())[:300], schema)


__all__ = ["Anthropic", "Chat", "OpenAI", "STRUCTURED", "Stub"]
