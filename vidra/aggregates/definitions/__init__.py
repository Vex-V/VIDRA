"""Aggregate definitions, as data: prompts and link profiles.

    vidra/aggregates/definitions/definitions.json   built in, read-only
    data/aggregates.json                                custom

A custom entry may not shadow a built-in. A prompt's kind is how it is asked:

    fold    batch the chunks, summarise each batch, summarise the summaries
    spans   contiguous ranges, placed by embeddings and named by the model
    items   discrete things, each citing the chunk it happened in

A link profile names a field, the keys that identify an entry of it, how
entries are compared, and what to write about each entity once linked
(`check: flag` marks observations that contradict the rest). An answer's
`fields` use describe's builder; the keys a kind writes itself cannot be
declared. `version_of` hashes everything a model is told.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ...shared.config import paths
from ..core import inputs as inputs_mod
from ...shared.reporting.errors import VidraError

BUILTIN_PATH = Path(__file__).with_name("definitions.json")
SECTIONS = ("prompts", "profiles")
#: A profile's aggregate id is this plus its name: `entities:people`.
PROFILE_PREFIX = "entities:"

NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
KINDS = ("fold", "spans", "items")
#: Names a definition cannot take.
RESERVED = frozenset({"stats", "speakers", "coverage", "ner", "sentiment", "entities"})
#: Keys a kind writes into an answer itself.
OWNED: dict[str, frozenset[str]] = {
    "fold": frozenset(), "spans": frozenset({"chunk_ids", "start_ts", "end_ts"}),
    "items": frozenset({"chunk_id", "start_ts", "end_ts"}), "link": frozenset({"doubts"}),
}
CHECKS = ("flag", "off")
MAX_INSTRUCTION = 4000
MAX_NARRATIVES = 50

PROMPT_KEYS = frozenset({"kind", "about", "instruction", "fields",
                         "key", "fold_instruction"})
PROFILE_DEFAULTS: dict[str, Any] = {
    "identity": [], "story": [], "transcript": False,
    "threshold": None, "rule": "max", "mutual": True, "check": "flag",
    "min_appearances": 2, "max_narratives": 12,
}
#: Optional profile keys for comparing entries beyond one cosine;
#: `linking.similarity` says what each does.
PROFILE_MEASURES = ("weights", "attributes", "near", "shared")
PROFILE_KEYS = frozenset({"about", "field", "instruction", "fields", *PROFILE_DEFAULTS,
                          *PROFILE_MEASURES})

_lock = threading.Lock()
#: Keyed by data root.
_cache: dict[str, dict[str, Any]] = {}


class DefinitionError(VidraError, ValueError):
    """A definition that will not be accepted, or does not exist. Carries the
    problems as a list.
    """

    def __init__(self, message: str, problems: Optional[list[str]] = None) -> None:
        super().__init__(message)
        self.problems = problems or [message]


class ProtectedDefinition(DefinitionError):
    """Built in, so not yours to change."""


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DefinitionError(f"{path} is not valid JSON: {exc}") from None


def _with_defaults(section: str, entry: dict[str, Any]) -> dict[str, Any]:
    return {**PROFILE_DEFAULTS, **entry} if section == "profiles" else dict(entry)


def load(refresh: bool = False) -> dict[str, Any]:
    """Built-ins, then custom on top. Cached; `refresh` re-reads. A custom entry
    that shadows a built-in or fails its check is dropped and listed under
    `problems`.
    """
    key = str(paths.AGGREGATE_DEFINITIONS)
    with _lock:
        if key in _cache and not refresh:
            return _cache[key]
        builtin = _read(BUILTIN_PATH)
        if not builtin:
            raise DefinitionError(f"{BUILTIN_PATH} is missing; the package is incomplete")
        merged: dict[str, Any] = {
            "system": builtin["system"],
            "kinds": builtin["kinds"],
            "prompts": {n: {**e, "builtin": True} for n, e in builtin["prompts"].items()},
            "profiles": {n: {**_with_defaults("profiles", e), "builtin": True}
                         for n, e in builtin["profiles"].items()},
            "problems": {},
        }
        custom = _read(paths.AGGREGATE_DEFINITIONS)
        for section in SECTIONS:
            for name, entry in (custom.get(section) or {}).items():
                where = f"{section}/{name}"
                if name in merged[section]:
                    merged["problems"][where] = ["shadows a built-in; ignored"]
                    continue
                if not isinstance(entry, dict):
                    merged["problems"][where] = ["not an object"]
                    continue
                entry = _with_defaults(section, entry)
                found = CHECKERS[section](name, entry)
                if found:
                    merged["problems"][where] = found
                    continue
                merged[section][name] = {**entry, "builtin": False}
        _cache[key] = merged
        return merged


# --------------------------------------------------------------------- reading

def system() -> str:
    return load()["system"]


def kind_text(kind: str) -> dict[str, str]:
    """What a kind appends to every instruction, from the data."""
    return dict(load()["kinds"].get(kind) or {})


def prompts() -> dict[str, dict[str, Any]]:
    return load()["prompts"]


def profiles() -> dict[str, dict[str, Any]]:
    return load()["profiles"]


def get(section: str, name: str) -> dict[str, Any]:
    if section not in SECTIONS:
        raise DefinitionError(f"section must be one of {', '.join(SECTIONS)}")
    entry = load()[section].get(name)
    if entry is None:
        raise DefinitionError(f"unknown {section[:-1]} {name!r}; known: "
                              f"{', '.join(sorted(load()[section])) or 'none'}")
    return entry


def ids() -> list[str]:
    """Every definition's aggregate id: prompts by name, profiles prefixed."""
    return [*sorted(prompts()), *(PROFILE_PREFIX + p for p in sorted(profiles()))]


def locate(definition_id: str) -> tuple[str, str]:
    """`entities:people` -> ("profiles", "people"); `summary` -> ("prompts", "summary")."""
    if definition_id.startswith(PROFILE_PREFIX):
        return "profiles", definition_id[len(PROFILE_PREFIX):]
    return "prompts", definition_id


def version_of(section: str, name: str) -> str:
    """A hash of everything the model is told for this definition: the entry,
    its kind's own wording and the system preamble."""
    entry = {k: v for k, v in get(section, name).items() if k != "builtin"}
    kind = entry.get("kind", "link")
    payload = json.dumps({"system": system(), "kind": kind_text(kind), "entry": entry},
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Selection:
    """What one link runs over: which answers, which field, which keys."""

    sources: tuple[inputs_mod.Source, ...]
    field: str
    #: The identity keys. Empty links the field's whole value.
    keys: tuple[str, ...]


def selection(name: str, chosen: inputs_mod.Input) -> Selection:
    """A profile's selection over the answers `chosen` names, narrowed by its
    fields if it has any: `people.clothing` links on clothing alone. An input
    may narrow the identity keys, never widen them.
    """
    entry = get("profiles", name)
    field = entry["field"]
    identity = tuple(entry.get("identity") or ())
    keys = identity
    asked = [f for s in chosen.sources for f in s.fields]
    if asked:
        if not identity:
            raise inputs_mod.InputError(
                f"profile {name!r} links whole `{field}` values; its input takes no fields")
        allowed = {f"{field}.{k}" for k in identity}
        wrong = [f for f in asked if f not in allowed]
        if wrong:
            raise inputs_mod.InputError(
                f"{', '.join(wrong)}: profile {name!r} identifies by "
                f"{', '.join(sorted(allowed))}; an input may narrow that, not widen it")
        keys = tuple(dict.fromkeys(f.split(".", 1)[1] for f in asked))
    return Selection(tuple(inputs_mod.Source(s.head) for s in chosen.sources),
                     field, keys)


# ------------------------------------------------------------------ validating

def _check_name(name: str) -> list[str]:
    problems = []
    if not NAME.match(name or ""):
        problems.append(f"name {name!r} must match {NAME.pattern}")
    if name in RESERVED:
        problems.append(f"{name!r} is reserved: {', '.join(sorted(RESERVED))}")
    return problems


def _check_text(value: Any, what: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"`{what}` is required"]
    if len(value) > MAX_INSTRUCTION:
        return [f"`{what}` is {len(value)} characters; {MAX_INSTRUCTION} max"]
    return []


def _check_fields(fields: Any, owned: frozenset[str]) -> list[str]:
    from ...shared.contracts.fields import problems as field_problems
    problems = field_problems(fields)
    if isinstance(fields, dict):
        taken = sorted(set(fields) & owned)
        if taken:
            problems.append(f"{', '.join(taken)}: written by the kind itself, "
                            "so a definition cannot declare it")
    return problems


def check_prompt(name: str, entry: dict[str, Any],
                 vocabulary: Optional[dict[str, Any]] = None) -> list[str]:
    """Everything wrong with a proposed prompt."""
    problems = _check_name(name)
    unknown = set(entry) - PROMPT_KEYS - {"builtin"}
    if unknown:
        problems.append(f"unknown key(s) {', '.join(sorted(unknown))}; a prompt "
                        f"takes {', '.join(sorted(PROMPT_KEYS))}")
    kind = entry.get("kind")
    if kind not in KINDS:
        problems.append(f"kind must be one of {', '.join(KINDS)}")
    problems += _check_text(entry.get("instruction"), "instruction")
    if entry.get("fold_instruction") is not None:
        problems += (["`fold_instruction` is for kind fold"] if kind != "fold"
                     else _check_text(entry["fold_instruction"], "fold_instruction"))
    if entry.get("key") is not None:
        if kind not in ("spans", "items"):
            problems.append("`key` names the list a spans or items answer is under")
        elif not FIELD_NAME.match(str(entry["key"])):
            problems.append(f"`key` must match {FIELD_NAME.pattern}")
    problems += _check_fields(entry.get("fields"), OWNED.get(kind, frozenset()))
    return problems


def _weights(value: Any, what: str) -> list[str]:
    if not isinstance(value, dict) or not value or not all(
            isinstance(k, str) and FIELD_NAME.match(k) and isinstance(w, (int, float))
            and not isinstance(w, bool) and w > 0 for k, w in value.items()):
        return [f"`{what}` maps entry keys to positive weights"]
    return []


def _check_measures(entry: dict[str, Any], identity: list[str]) -> list[str]:
    """`weights`, `attributes`, `near`, `shared`: all optional, all about entries."""
    present = [k for k in PROFILE_MEASURES if k in entry]
    if not present:
        return []
    if not identity:
        return [f"{', '.join(f'`{k}`' for k in present)} measure entries, so they need "
                f"`identity`"]
    problems = []
    if entry.get("threshold") is not None:
        problems.append("`threshold` is a cosine, and with "
                        f"{', '.join(f'`{k}`' for k in present)} the measure is a "
                        "z-score read off the video -- drop `threshold`")
    if "weights" in entry:
        problems += _weights(entry["weights"], "weights")
        if isinstance(entry["weights"], dict) and set(entry["weights"]) - set(identity):
            problems.append("`weights` weighs identity keys only")
    for key in ("attributes", "shared"):
        if key in entry:
            problems += _weights(entry[key], key)
    if "near" in entry:
        near = entry["near"]
        if not isinstance(near, list) or not all(
                isinstance(p, list) and len(p) == 2 and all(isinstance(v, str) for v in p)
                for p in near):
            problems.append("`near` is a list of value pairs, e.g. [[\"dark_blue\", \"black\"]]")
        if "attributes" not in entry:
            problems.append("`near` softens `attributes`, so it needs them")
    return problems


def check_profile(name: str, entry: dict[str, Any],
                  vocabulary: Optional[dict[str, Any]] = None) -> list[str]:
    """Everything wrong with a proposed link profile (defaults applied)."""
    problems = _check_name(name)
    unknown = set(entry) - PROFILE_KEYS - {"builtin"}
    if unknown:
        problems.append(f"unknown key(s) {', '.join(sorted(unknown))}; a profile "
                        f"takes {', '.join(sorted(PROFILE_KEYS))}")
    field = entry.get("field")
    identity, story = entry.get("identity"), entry.get("story")
    if not isinstance(field, str) or not FIELD_NAME.match(field):
        problems.append("`field` names what is linked: a list field such as "
                        "`people`, a text field, `summary` or `transcript`")
    for what, keys in (("identity", identity), ("story", story)):
        if not isinstance(keys, list) or not all(
                isinstance(k, str) and FIELD_NAME.match(k) for k in keys):
            problems.append(f"`{what}` must be a list of entry keys")
    if problems:
        return problems

    if identity and field in (inputs_mod.PROSE, "transcript"):
        problems.append(f"`{field}` has no entries, so no `identity` keys")
    if story and not identity:
        problems.append("`story` picks entry keys, so it needs `identity`")
    threshold = entry.get("threshold")
    if threshold is not None and not (isinstance(threshold, (int, float))
                                      and 0 < threshold < 1):
        problems.append("`threshold` is a cosine similarity between 0 and 1")
    if not identity and threshold is None:
        problems.append(
            "a profile without `identity` links whole values, and nothing in one "
            "answer is provably different from anything else, so no threshold "
            "can be read off the video -- give `threshold`")
    problems += _check_measures(entry, identity)
    from ..aggregators.entities.linking import RULES
    if entry.get("rule") not in RULES:
        problems.append(f"`rule` must be one of {', '.join(RULES)}")
    if entry.get("check") not in CHECKS:
        problems.append(f"`check` must be one of {', '.join(CHECKS)}")
    for key in ("mutual", "transcript"):
        if not isinstance(entry.get(key), bool):
            problems.append(f"`{key}` is true or false")
    appearances, narratives = entry.get("min_appearances"), entry.get("max_narratives")
    if not isinstance(appearances, int) or appearances < 1:
        problems.append("`min_appearances` is a whole number, at least 1")
    if not isinstance(narratives, int) or not 0 <= narratives <= MAX_NARRATIVES:
        problems.append(f"`max_narratives` is a whole number, 0 to {MAX_NARRATIVES}")
    problems += _check_text(entry.get("instruction"), "instruction")
    problems += _check_fields(entry.get("fields"), OWNED["link"])

    if vocabulary is not None and field not in (inputs_mod.PROSE, "transcript"):
        shapes = list(vocabulary["questions"].values())
        if identity:
            wanted = (set(identity) | set(story) | set(entry.get("attributes") or {})
                      | set(entry.get("shared") or {}))
            if not any(isinstance(s.get(field), list) and wanted <= set(s[field])
                       for s in shapes):
                problems.append(f"no question's shape has a `{field}` list whose "
                                f"entries carry {', '.join(sorted(wanted))}")
        elif not any(field in s and s[field] is None for s in shapes):
            problems.append(f"no question's shape has a `{field}` field holding a value")
    return problems


CHECKERS = {"prompts": check_prompt, "profiles": check_profile}


# -------------------------------------------------------------------- writing

def add(section: str, name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Add or replace a custom definition. Built-ins are refused."""
    if section not in SECTIONS:
        raise DefinitionError(f"section must be one of {', '.join(SECTIONS)}")
    if load()[section].get(name, {}).get("builtin"):
        raise ProtectedDefinition(f"{name!r} is a built-in {section[:-1]} and cannot be "
                        "replaced; pick another name")
    clean = {k: v for k, v in entry.items() if v is not None and k != "builtin"}
    from ...video_rag import driver as video_rag
    problems = CHECKERS[section](name, _with_defaults(section, clean),
                                 video_rag.vocabulary())
    if problems:
        raise DefinitionError("; ".join(problems), problems)
    with _lock:
        doc = _read(paths.AGGREGATE_DEFINITIONS) or {
            "document": "aggregate_definitions", "version": 1}
        doc.setdefault(section, {})[name] = clean
        _write(doc)
    load(refresh=True)
    return get(section, name)


def remove(section: str, name: str) -> None:
    """Delete a custom definition. Answers it already wrote are untouched."""
    if load()[section].get(name, {}).get("builtin"):
        raise ProtectedDefinition(f"{name!r} is built in and cannot be deleted")
    with _lock:
        doc = _read(paths.AGGREGATE_DEFINITIONS)
        if not (doc.get(section) or {}).pop(name, None):
            raise DefinitionError(f"no custom {section[:-1]} {name!r}")
        _write(doc)
    load(refresh=True)


def add_prompt(name: str, instruction: str, fields: dict[str, Any], *,
               kind: str = "fold",
               key: Optional[str] = None, fold_instruction: Optional[str] = None,
               about: str = "") -> dict[str, Any]:
    """Add an aggregate prompt of your own, or replace one you added. Returns
    `definition(name)`.

        aggregates.add_prompt(
            "incident_report",
            "Write an incident report for this video.",
            fields={
                "report":   {"type": "text", "about": "what happened, in order"},
                "severity": {"type": "text", "about": "how serious",
                             "one_of": ["none", "minor", "major"]},
            },
        )
        aggregates.custom(name="incident_report", input=excerpt, out=...)

    `kind` is how it is asked:

        fold    one answer over the whole video (as `summary`)
        spans   contiguous chapters, each answering `fields` (as `chapters`)
        items   discrete things, each citing its chunk (as `events`)

    `fields` uses describe's builder: `type` is `text` or `list`, `about` is what
    the model is told to put there, `one_of` fixes the values, `of` makes a
    list's entries objects. `key` names the list a spans or items answer is
    under (the kind when None); `fold_instruction` is how a fold merges partial summaries.
    Saved in `aggregates.json` under the data root. A built-in's name raises
    `ProtectedDefinition`; anything else wrong raises `DefinitionError`.
    """
    add("prompts", name, {"kind": kind, "about": about, "instruction": instruction,
                          "fields": fields, "key": key,
                          "fold_instruction": fold_instruction})
    return definition(name)


def add_profile(name: str, field: str, instruction: str, fields: dict[str, Any], *,
                identity: Optional[list[str]] = None, story: Optional[list[str]] = None,
                transcript: Optional[bool] = None,
                threshold: Optional[float] = None, rule: Optional[str] = None,
                mutual: Optional[bool] = None, check: Optional[str] = None,
                min_appearances: Optional[int] = None,
                max_narratives: Optional[int] = None,
                weights: Optional[dict[str, float]] = None,
                attributes: Optional[dict[str, float]] = None,
                near: Optional[list[list[str]]] = None,
                shared: Optional[dict[str, float]] = None,
                about: str = "") -> dict[str, Any]:
    """Add a link profile of your own, or replace one you added. Runs as
    `entities:<name>`. Returns `definition("entities:<name>")`.

        aggregates.add_profile(
            "tools", "objects",
            "Below are observations of what may be the same object, in time "
            "order. Say what it is and who used it for what.",
            fields={"use": {"type": "text", "about": "who used it, for what"}},
            identity=["object", "appearance"], story=["context"])
        aggregates.entities(profile="tools", input=video.sightings(
            profile="tools", answers=["objects"]), out=...)

    `field` is the answer field whose entries are linked; `identity` the entry
    keys that say who is who, `story` the keys the account reads besides.
    `instruction` and
    `fields` describe the account written for each linked entity. The rest
    tune linking and default as the built-in profiles do: `threshold` (needed
    without `identity`), `rule` (`max`, `q95`), `mutual`, `check` (`flag`,
    `off`), `min_appearances`, `max_narratives`, `transcript`, and the
    measures `weights`, `attributes`, `near`, `shared`.
    """
    add("profiles", name, {
        "about": about, "field": field, "instruction": instruction, "fields": fields,
        "identity": identity, "story": story, "transcript": transcript,
        "threshold": threshold, "rule": rule, "mutual": mutual, "check": check,
        "min_appearances": min_appearances, "max_narratives": max_narratives,
        "weights": weights, "attributes": attributes, "near": near, "shared": shared})
    return definition(PROFILE_PREFIX + name)


def remove_prompt(name: str) -> None:
    """Remove a prompt you added. Answers it already wrote are kept."""
    remove("prompts", name)


def remove_profile(name: str) -> None:
    """Remove a link profile you added. Answers it already wrote are kept."""
    remove("profiles", name)


def definition(name: str) -> dict[str, Any]:
    """One prompt or profile by aggregate id (`summary`, `entities:people`): its
    entry, with `name`, `kind` (`link` for a profile) and `builtin`.
    """
    section, short = locate(name)
    entry = {k: v for k, v in get(section, short).items() if k != "builtin"}
    return {"name": name, "kind": entry.pop("kind", "link"),
            "builtin": bool(get(section, short).get("builtin")), **entry}


def _write(doc: dict[str, Any]) -> None:
    target = paths.AGGREGATE_DEFINITIONS
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(target)                  # atomic: a torn file is a broken run


__all__ = ["CHECKS", "DefinitionError", "KINDS", "PROFILE_PREFIX", "ProtectedDefinition",
           "Selection", "add", "add_profile", "add_prompt", "check_profile",
           "check_prompt", "definition", "get", "ids",
           "kind_text", "load", "locate", "profiles", "prompts", "remove",
           "remove_profile", "remove_prompt", "selection", "system", "version_of"]
