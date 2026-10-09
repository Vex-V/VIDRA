"""The question vocabulary: built-in questions, plus any a user has added.

    vidra/video_rag/describe/prompts.json   built in, read-only
    data/prompts.json                           custom, written by `add_question`

A question is an instruction and a shape; the shape holds the response
schema. A custom question names a shipped shape or brings its own, built from
`fields` (`text` or `list`, `of` for a list of objects, `one_of` for a fixed
vocabulary). A custom entry may not shadow a built-in question or shape, and
an unknown question resolves to the shape marked `fallback`.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Optional

from vidra.shared.config import paths
from vidra.shared.reporting.errors import VidraError
from vidra.shared.contracts.fields import check_fields, compile_fields
from vidra.shared.contracts.units import IMAGE_QUESTION

BUILTIN_PATH = Path(__file__).with_name("prompts.json")

#: Allowed question names: lowercase, no colon (`sampler:question` splits on one).
NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

#: Placeholders an instruction may use.
PLACEHOLDERS = {"n", "span", "vocabulary"}


_lock = threading.Lock()
#: The merged vocabulary, cached per data root.
_cache: dict[str, dict[str, Any]] = {}


class PromptError(VidraError, ValueError):
    """A prompt the vocabulary will not accept, with the reason."""


class ProtectedPrompt(PromptError):
    """The question is built in, so it cannot be changed."""


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromptError(f"{path} is not valid JSON: {exc}") from None


# ------------------------------------------------------- building a shape


def compile_shape(spec: dict[str, Any]) -> dict[str, Any]:
    """A field spec -> a shape, in the form the built-ins are written. Never the
    fallback shape.
    """
    fields = spec.get("fields") or {}
    return {
        "summary": spec.get("summary") or "standard",
        "fallback": False,
        "fields": compile_fields(fields),
    }


def check_shape(spec: Any) -> list[str]:
    """Everything wrong with a proposed shape, as messages."""
    problems: list[str] = []
    if not isinstance(spec, dict):
        return ["shape must be an object with a `fields` map"]

    summary = spec.get("summary") or "standard"
    known_summaries = load()["summaries"]
    if summary not in known_summaries:
        problems.append(f"unknown summary {summary!r}; "
                        f"known: {', '.join(sorted(known_summaries))}")

    fields = spec.get("fields")
    if not isinstance(fields, dict) or not fields:
        problems.append("a shape needs at least one field; use the `prose` "
                        "shape for a summary-only question")
        return problems
    return problems + check_fields(fields)


def load(refresh: bool = False) -> dict[str, Any]:
    """The merged vocabulary: built-ins, then custom on top. Cached; `refresh`
    re-reads the files.
    """
    key = str(paths.PROMPTS)
    with _lock:
        if key in _cache and not refresh:
            return _cache[key]

        builtin = _read(BUILTIN_PATH)
        if not builtin:
            raise PromptError(f"{BUILTIN_PATH} is missing; the package is "
                              "incomplete and no question can be resolved")

        merged = {
            "system": builtin["system"],
            "summaries": dict(builtin["summaries"]),
            "shapes": dict(builtin["shapes"]),
            "questions": {name: {**q, "builtin": True}
                          for name, q in builtin["questions"].items()},
        }

        custom = _read(paths.PROMPTS)

        # Shapes first: a question resolves its shape by name.
        for name, spec in (custom.get("shapes") or {}).items():
            # A custom shape may not redefine a shipped one.
            if name in merged["shapes"]:
                continue
            try:
                merged["shapes"][name] = compile_shape(spec)
            except Exception as exc:                       # noqa: BLE001
                # A custom shape that will not compile is dropped, with a warning; its
                # question falls back to the general shape.
                from vidra.shared.reporting import logs
                logs.logger("describe").warning(
                    "custom shape %r dropped: %s", name, exc,
                    extra={"component": "describe", "event": "dropped",
                           "reason": f"{type(exc).__name__}: {exc}"[:300]})
                continue

        for name, entry in (custom.get("questions") or {}).items():
            # A custom question may not shadow a built-in.
            if name in merged["questions"]:
                continue
            merged["questions"][name] = {**entry, "builtin": False}

        _cache[key] = merged
        return merged


# --------------------------------------------------------------------- reading

def questions() -> list[str]:
    """Every question name, built in or added."""
    return sorted(load()["questions"])


def question(name: str) -> dict[str, Any]:
    """One question: its instruction, about, shape, summary length and
    `{field: what the model is told to put there}`.
    """
    entry = load()["questions"].get(name)
    if entry is None:
        raise PromptError(f"unknown question {name!r}; "
                          f"known: {', '.join(questions())}")
    shape = shape_of(name)
    return {
        "name": name,
        "instruction": entry.get("instruction", ""),
        "about": entry.get("about", ""),
        "builtin": bool(entry.get("builtin")),
        # Its own shape is stored under the question's name.
        "shape": None if entry.get("shape") == name and not entry.get("builtin")
                 else entry.get("shape"),
        "summary": shape.get("summary", "standard"),
        "fields": {field: spec.get("description", "")
                   for field, spec in (shape.get("fields") or {}).items()},
    }


def shapes() -> dict[str, Any]:
    return load()["shapes"]


def builtin_shapes() -> set[str]:
    """Which shapes ship in the package."""
    return set(_read(BUILTIN_PATH).get("shapes") or {})


def shape_of(name: str) -> dict[str, Any]:
    """The shape a question answers in; an unknown question gets the fallback shape."""
    entry = load()["questions"].get(name)
    shape = entry.get("shape") if entry else None
    return load()["shapes"].get(shape) or _fallback_shape()


def _fallback_shape() -> dict[str, Any]:
    for shape in load()["shapes"].values():
        if shape.get("fallback"):
            return shape
    raise PromptError("no shape is marked `fallback`; the general question "
                      "has nothing to resolve to")


def instruction_of(name: str) -> str:
    entry = load()["questions"].get(name)
    if entry and entry.get("instruction"):
        return entry["instruction"]
    fallback = next((q for q in load()["questions"].values()
                     if (load()["shapes"].get(q.get("shape")) or {}).get("fallback")),
                    None)
    return (fallback or {}).get("instruction", "")


# ------------------------------------------------------------------ validating

def check(name: str, entry: dict[str, Any],
          shape_spec: Optional[dict[str, Any]] = None) -> list[str]:
    """Everything wrong with a proposed question, as messages. `shape_spec` is
    checked instead of `entry["shape"]` when the question brings its own shape.
    """
    problems: list[str] = []
    if not NAME.match(name or ""):
        problems.append(f"name {name!r} must match {NAME.pattern} -- lowercase, "
                        "no colon, since `sampler:question` splits on one")
    if name == IMAGE_QUESTION:
        problems.append(f"{name!r} is taken: it is the question half of every "
                        "unit `glance` makes from frames")

    instruction = (entry.get("instruction") or "").strip()
    if not instruction:
        problems.append("instruction is required")
    elif len(instruction) > 4000:
        problems.append(f"instruction is {len(instruction)} characters; 4000 max")
    else:
        # Only the known placeholders may appear in braces.
        try:
            unknown = {f for _, f, _, _ in __import__("string").Formatter()
                       .parse(instruction) if f} - PLACEHOLDERS
        except ValueError as exc:
            problems.append(f"instruction has malformed braces: {exc}")
        else:
            if unknown:
                problems.append(
                    f"instruction uses unknown placeholder(s) "
                    f"{', '.join(sorted(unknown))}; known: "
                    f"{', '.join(sorted(PLACEHOLDERS))}")

    if shape_spec is not None:
        # A question's own shape may not take a built-in shape's name.
        if name in builtin_shapes():
            problems.append(f"{name!r} is a built-in shape; a question that "
                            "defines its own shape cannot take that name")
        problems.extend(check_shape(shape_spec))
        return problems

    shape = entry.get("shape")
    if shape not in load()["shapes"]:
        problems.append(f"unknown shape {shape!r}; "
                        f"known: {', '.join(sorted(load()['shapes']))}")
    return problems


# -------------------------------------------------------------------- writing

def _write_custom(doc: dict[str, Any]) -> None:
    paths.PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.PROMPTS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(paths.PROMPTS)          # atomic: a torn file is a broken run


def add(name: str, instruction: str, shape: str = "scene",
        about: str = "", fields: Optional[dict[str, Any]] = None,
        summary: str = "standard") -> dict[str, Any]:
    """Add or replace a custom question. Built-ins are refused. `fields` gives
    the question its own shape, stored under its name.
    """
    shape_spec = (None if fields is None
                  else {"summary": summary, "fields": fields})
    entry = {"shape": name if shape_spec is not None else shape,
             "instruction": instruction.strip(),
             **({"about": about.strip()} if about.strip() else {})}

    if load()["questions"].get(name, {}).get("builtin"):
        raise ProtectedPrompt(
            f"{name!r} is a built-in question and cannot be replaced. Built-ins "
            "live in the package so that every deployment's `yolo` means the "
            "same thing; pick another name.")

    problems = check(name, entry, shape_spec)
    if problems:
        raise PromptError("; ".join(problems))

    with _lock:
        doc = _read(paths.PROMPTS) or {"document": "prompts", "version": 1,
                                       "questions": {}}
        doc.setdefault("questions", {})[name] = entry
        shapes = doc.setdefault("shapes", {})
        if shape_spec is not None:
            shapes[name] = shape_spec
        else:
            # A question that switched to a shipped shape drops its own.
            shapes.pop(name, None)
        if not shapes:
            doc.pop("shapes")
        _write_custom(doc)
    load(refresh=True)
    return {**entry, "builtin": False}


def add_question(name: str, instruction: str, *,
                 fields: Optional[dict[str, Any]] = None,
                 shape: Optional[str] = None,
                 summary: Optional[str] = None,
                 about: str = "") -> dict[str, Any]:
    """Add a question of your own, or replace one you added. Returns `question(name)`.

        describe.add_question(
            "hazards",
            "These {n} frames span {span}. List every hazard you can see.",
            fields={
                "hazards":  {"type": "list", "about": "each hazard, briefly"},
                "severity": {"type": "text", "about": "the worst one",
                             "one_of": ["none", "low", "high"]},
                "exposed":  {"type": "list", "about": "who is at risk",
                             "of": {"who": "by appearance", "how": "how exposed"}},
            })
        video_rag("x.mp4", "data/out", sampler="clip:hazards")

    Pair it with any sampler as `sampler:name`. What it answers is one of:

        fields=   its own fields: `type` is `text` or `list`; `about` (required)
                  is what the model is told to put there; `one_of` fixes the
                  values; `of` makes a list's entries objects
        shape=    a shipped shape: `scene`, `people`, `objects`, `text`, `prose`
        neither   prose only (the `prose` shape)

    `summary` sets the prose length for a shape of your own: `standard` (at least
    150 words) or `brief` (4 to 5 sentences). `instruction` may use `{n}`,
    `{span}` and `{vocabulary}`. Saved in `prompts.json` under the data root. A
    built-in's name raises `ProtectedPrompt`.
    """
    if fields is not None and shape is not None:
        raise PromptError("pass fields= for a shape of your own or shape= for "
                          "a shipped one, not both")
    if shape is not None and shape not in builtin_shapes():
        # Only shipped shapes: another question's own shape is removed with it.
        raise PromptError(f"unknown shape {shape!r}; shipped: "
                          f"{', '.join(sorted(builtin_shapes()))} -- or give "
                          f"fields= for one of your own")
    if summary is not None and fields is None:
        raise PromptError("summary= sets the length for a shape of your own, "
                          "so it needs fields=; a shipped shape has its own")
    if summary is not None and summary not in load()["summaries"]:
        raise PromptError(f"unknown summary {summary!r}; "
                          f"known: {', '.join(load()['summaries'])}")
    add(name, instruction, shape=shape or "prose", about=about, fields=fields,
        summary=summary or "standard")
    return question(name)


def remove_question(name: str) -> None:
    """Remove a question you added, with any shape it brought. A later describe
    naming it is refused as unknown.
    """
    remove(name)


def remove(name: str) -> None:
    """Delete a custom question and its own shape."""
    if load()["questions"].get(name, {}).get("builtin"):
        raise ProtectedPrompt(f"{name!r} is built in and cannot be deleted")
    with _lock:
        doc = _read(paths.PROMPTS)
        if not (doc.get("questions") or {}).pop(name, None):
            raise PromptError(f"no custom question {name!r}")
        (doc.get("shapes") or {}).pop(name, None)
        if "shapes" in doc and not doc["shapes"]:
            doc.pop("shapes")
        _write_custom(doc)
    load(refresh=True)
