"""Which model answers a call, and how to reach it.

A provider is a name and a protocol (the wire format):

    openai     OpenAI's API: Responses for answers, /embeddings for vectors
    chat       anything serving OpenAI's Chat Completions and /embeddings --
               Ollama, LM Studio, llama.cpp, vLLM, Gemini, Mistral, Groq,
               OpenRouter, Together, DeepSeek, xAI, Voyage
    anthropic  the Messages API; answers only
    local      a Hugging Face model in this process; vectors only

Built-ins are declared below; `data/providers.json` adds or overrides
endpoints and names the variables keys are read from. A choice is one string,
`provider` or `provider/model`, resolved when the call is made.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, fields, replace
from typing import Any, Iterator, Mapping, Optional

from ..config import paths
from ..reporting.errors import VidraError, Unavailable

PROTOCOLS = ("openai", "chat", "anthropic", "local")
ROLES = ("describe", "llm", "embed")

#: How a `chat` provider is asked for a shape: enforced by the server
#: (`json_schema`) or put in the prompt and parsed.
STRUCTURED = ("json_schema", "json_object", "prompt")

#: What a role uses when neither the call nor the environment names a provider.
FALLBACK = "openai"

#: role -> the variable naming its provider, or `provider/model`.
ENV: dict[str, str] = {
    "describe": "VIDRA_DESCRIBER",
    "llm": "VIDRA_LLM",
    "embed": "VIDRA_EMBEDDER",
}

#: Names that serve a role offline: no model, no key.
OFFLINE: dict[str, str] = {"describe": "stub", "embed": "hash"}

NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class ProviderError(VidraError, ValueError):
    """An unknown provider, one that cannot do what was asked, or no model."""


class ProviderUnavailable(Unavailable):
    """A provider that needs a key, and none is set."""


@dataclass(frozen=True)
class Provider:
    name: str
    protocol: str
    base_url: Optional[str] = None
    #: Variables the key is read from, first set wins. Names, never values.
    key_vars: tuple[str, ...] = ()
    chat_model: Optional[str] = None
    embed_model: Optional[str] = None
    can_chat: bool = True
    can_embed: bool = True
    structured: str = "json_schema"
    #: Accepts `dimensions` on /embeddings.
    dimensions: bool = False
    #: Wants `input_type: query | document` on /embeddings (Voyage).
    input_type: bool = False
    #: `max_tokens` or `max_completion_tokens`; a refusal naming one is retried
    #: with the other.
    token_field: str = "max_tokens"
    #: Calls in flight at once.
    concurrency: int = 8
    #: Runs on this machine: no key required.
    local: bool = False
    about: str = ""
    builtin: bool = True

    def can(self, role: str) -> bool:
        return self.can_embed if role == "embed" else self.can_chat

    def default_model(self, role: str) -> Optional[str]:
        return self.embed_model if role == "embed" else self.chat_model


_BUILTIN: tuple[Provider, ...] = (
    Provider("openai", "openai", key_vars=("OPENAI_API_KEY", "OPENAI_API"),
             chat_model="gpt-5.4-mini", embed_model="text-embedding-3-small",
             dimensions=True, about="OpenAI"),
    Provider("anthropic", "anthropic", base_url="https://api.anthropic.com",
             key_vars=("ANTHROPIC_API_KEY",), chat_model="claude-haiku-4-5",
             can_embed=False,
             about="Claude, through the Messages API. No embeddings"),
    Provider("gemini", "chat",
             base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
             key_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
             chat_model="gemini-2.5-flash", embed_model="gemini-embedding-001",
             dimensions=True,
             about="Google Gemini, through its OpenAI-compatible endpoint"),
    Provider("mistral", "chat", base_url="https://api.mistral.ai/v1",
             key_vars=("MISTRAL_API_KEY",), chat_model="mistral-small-latest",
             embed_model="mistral-embed", about="Mistral"),
    Provider("openrouter", "chat", base_url="https://openrouter.ai/api/v1",
             key_vars=("OPENROUTER_API_KEY",), can_embed=False,
             about="OpenRouter: one key, many vendors. Name the model"),
    Provider("groq", "chat", base_url="https://api.groq.com/openai/v1",
             key_vars=("GROQ_API_KEY",), can_embed=False,
             about="Groq. Name the model"),
    Provider("together", "chat", base_url="https://api.together.xyz/v1",
             key_vars=("TOGETHER_API_KEY",),
             about="Together AI. Name the model"),
    Provider("deepseek", "chat", base_url="https://api.deepseek.com/v1",
             key_vars=("DEEPSEEK_API_KEY",), chat_model="deepseek-chat",
             can_embed=False, structured="json_object",
             about="DeepSeek. Text only, so aggregates rather than describe"),
    Provider("xai", "chat", base_url="https://api.x.ai/v1",
             key_vars=("XAI_API_KEY",), can_embed=False,
             about="xAI Grok. Name the model"),
    Provider("voyage", "chat", base_url="https://api.voyageai.com/v1",
             key_vars=("VOYAGE_API_KEY",), can_chat=False,
             embed_model="voyage-3.5", input_type=True,
             about="Voyage AI embeddings"),
    Provider("ollama", "chat", base_url="http://localhost:11434/v1",
             key_vars=("OLLAMA_API_KEY",), chat_model="gemma3:4b",
             embed_model="nomic-embed-text", local=True, concurrency=1,
             about="Ollama on this machine. `ollama pull` the model first"),
    Provider("lmstudio", "chat", base_url="http://localhost:1234/v1",
             key_vars=("LMSTUDIO_API_KEY",), local=True, concurrency=1,
             about="LM Studio's server. Name the model it has loaded"),
    Provider("llamacpp", "chat", base_url="http://localhost:8080/v1",
             key_vars=("LLAMACPP_API_KEY",), local=True, concurrency=1,
             about="llama.cpp `llama-server`. Start it with --embeddings for vectors"),
    Provider("local", "local", can_chat=False,
             embed_model="BAAI/bge-small-en-v1.5", local=True,
             about="a Hugging Face embedding model loaded into this process"),
)

#: Fields `data/providers.json` may set.
_FILE_FIELDS = tuple(f.name for f in fields(Provider)
                     if f.name not in ("name", "builtin"))


# ------------------------------------------------------------------ loading

def _check(name: str, entry: Any, existing: Optional[Provider]) -> list[str]:
    """Why a `providers.json` entry cannot be used, as messages."""
    if not isinstance(entry, dict):
        return ["must be an object of fields"]
    problems = []
    if not NAME.match(name):
        # `provider/model` splits on the first slash, so a name may not contain one.
        problems.append("name must be lowercase letters, digits, `_`, `.` or `-`")
    if name in OFFLINE.values():
        problems.append(f"{name!r} is reserved: it loads no model")
    for key in entry:
        if "key" in key and key != "key_vars":
            problems.append(f"{key!r}: keys never live in this file -- name the "
                            "environment variable in key_vars and set it in .env")
        elif key not in _FILE_FIELDS:
            problems.append(f"unknown field {key!r}; known: {', '.join(_FILE_FIELDS)}")
    protocol = entry.get("protocol", existing.protocol if existing else None)
    if protocol not in PROTOCOLS:
        problems.append(f"protocol must be one of {', '.join(PROTOCOLS)}")
    if existing is None and protocol == "chat" and not entry.get("base_url"):
        problems.append("a new chat provider needs base_url")
    if entry.get("structured", "json_schema") not in STRUCTURED:
        problems.append(f"structured must be one of {', '.join(STRUCTURED)}")
    if entry.get("token_field", "max_tokens") not in ("max_tokens",
                                                     "max_completion_tokens"):
        problems.append("token_field must be max_tokens or max_completion_tokens")
    if "concurrency" in entry and not (type(entry["concurrency"]) is int
                                       and entry["concurrency"] >= 1):
        problems.append("concurrency must be a whole number, 1 or more")
    if "key_vars" in entry and not (isinstance(entry["key_vars"], list)
                                    and all(isinstance(v, str) for v in entry["key_vars"])):
        problems.append("key_vars must be a list of variable names")
    return problems


def load() -> tuple[dict[str, Provider], list[str]]:
    """Every provider, and why any `providers.json` entry was left out (a bad
    entry is dropped and reported, not raised).
    """
    found = {p.name: p for p in _BUILTIN}
    problems: list[str] = []
    path = paths.PROVIDERS
    if not path.exists():
        return found, problems
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return found, [f"{path}: {exc}"]
    entries = data.get("providers", {}) if isinstance(data, dict) else {}
    for name, entry in entries.items():
        existing = found.get(name)
        wrong = _check(name, entry, existing)
        if wrong:
            problems += [f"{path}: {name}: {w}" for w in wrong]
            continue
        values = {k: (tuple(v) if k == "key_vars" else v) for k, v in entry.items()}
        found[name] = (replace(existing, **values) if existing
                       else Provider(name=name, builtin=False, **values))
    return found, problems


def providers() -> dict[str, Provider]:
    return load()[0]


def get(name: str) -> Provider:
    found = providers()
    if name not in found:
        raise ProviderError(f"unknown provider {name!r}; known: {', '.join(sorted(found))}")
    return found[name]


def names(role: str) -> list[str]:
    """Every name that can serve a role, providers and offline ones alike."""
    out = [p.name for p in providers().values() if p.can(role)]
    if role in OFFLINE:
        out.append(OFFLINE[role])
    return sorted(out)


# --------------------------------------------------------------- resolving

def split(spec: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """`ollama/gemma3:4b` -> (`ollama`, `gemma3:4b`). Splits on the first slash only
    when the head is a provider; a bare name has no model.
    """
    if not spec or not spec.strip():
        return None, None
    spec = spec.strip()
    head, slash, rest = spec.partition("/")
    if slash and (head in providers() or head in OFFLINE.values()):
        return head, rest or None
    return spec, None


def choose(role: str, spec: Optional[str] = None) -> tuple[str, Optional[str]]:
    """The provider and model a call should use: `spec`, then the role's variable,
    then `FALLBACK`. The model is None only for an offline name.
    """
    if role not in ROLES:
        raise ProviderError(f"unknown role {role!r}; known: {', '.join(ROLES)}")
    chosen, model = split(spec)
    if chosen is None:
        chosen, model = split(os.environ.get(ENV[role]))
    chosen = chosen or FALLBACK

    if chosen == OFFLINE.get(role):
        return chosen, None
    if chosen in OFFLINE.values():
        raise ProviderError(f"{chosen!r} cannot serve {role}; "
                            f"known: {', '.join(names(role))}")

    provider = get(chosen)
    if not provider.can(role):
        what = "vectors" if role == "embed" else "answers"
        raise ProviderError(f"{chosen} serves no {what}; for {role} use one of "
                            f"{', '.join(names(role))}")
    model = model or provider.default_model(role)
    if not model:
        raise ProviderError(
            f"{chosen} has no default {'embedding' if role == 'embed' else 'chat'} "
            f"model: name one, as `{chosen}/<model>`")
    return chosen, model


def base_url(provider: Provider) -> Optional[str]:
    """`<NAME>_BASE_URL` if set, else the declared one."""
    variable = re.sub(r"[^A-Z0-9]", "_", provider.name.upper()) + "_BASE_URL"
    return os.environ.get(variable) or provider.base_url


# ------------------------------------------------------------------- keys
#
# Keys given in code (`Models(keys={"openai": ...})`) outrank the environment.
# A pipeline sets them in a context variable for the length of its call;
# `api_key` reads them from there.

_KEYS: ContextVar[Mapping[str, str]] = ContextVar("vidra_keys", default={})


def use_keys(given: Optional[Mapping[str, str]]) -> Token:
    """Make `given` the keys every call below sees, until `release`."""
    return _KEYS.set({**_KEYS.get(), **(given or {})})


def release(token: Token) -> None:
    _KEYS.reset(token)


@contextmanager
def keys(given: Optional[Mapping[str, str]]) -> Iterator[None]:
    """`use_keys` for the length of a `with` block."""
    token = use_keys(given)
    try:
        yield
    finally:
        release(token)


def api_key(provider: Provider) -> Optional[str]:
    """The key: one given in code, else the environment; None where none is
    needed. Raises naming both ways to give one.
    """
    given = _KEYS.get().get(provider.name)
    if given:
        return given
    for variable in provider.key_vars:
        if os.environ.get(variable):
            return os.environ[variable]
    if provider.local or not provider.key_vars:
        return None
    raise ProviderUnavailable(
        f"no key for {provider.name}: pass Models(keys={{{provider.name!r}: ...}}) "
        f"or set {' or '.join(provider.key_vars)} in the environment")


def problems(role: str, spec: Optional[str] = None) -> list[str]:
    """What stops a role running, as messages: unknown names and missing keys."""
    try:
        chosen, _ = choose(role, spec)
    except ProviderError as exc:
        return [str(exc)]
    if chosen in OFFLINE.values():
        return []
    try:
        api_key(get(chosen))
    except ProviderUnavailable as exc:
        return [f"{role}: {exc}"]
    return []


def require(role: str, spec: Optional[str] = None) -> None:
    """`problems`, raising the first: for a component called on its own."""
    found = problems(role, spec)
    if found:
        raise ProviderUnavailable(found[0])


__all__ = ["ENV", "FALLBACK", "OFFLINE", "PROTOCOLS", "Provider", "ProviderError",
           "ProviderUnavailable", "ROLES", "api_key", "base_url",
           "choose", "get", "keys", "load", "names", "problems",
           "providers", "release", "require", "split", "use_keys"]
