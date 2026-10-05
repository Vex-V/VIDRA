"""The field builder: a compact field spec in, strict JSON Schema out.

`{"severity": {"type": "text", "one_of": [...]}}` becomes the schema a
structured model call is sent. Describe shapes and aggregate prompts both
declare their fields this way. Field counts are capped.
"""

from __future__ import annotations

import re
from typing import Any


#: Allowed field names.
FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


#: What a field may be: `text` or `list`; `of` makes a list's entries objects.
FIELD_TYPES = ("text", "list")


#: Caps on fields, nested keys and choices.
MAX_FIELDS = 12


MAX_NESTED_KEYS = 8


MAX_ENUM = 24


def _compile_field(spec: dict[str, Any]) -> dict[str, Any]:
    """One field spec -> its JSON Schema fragment, in the strict subset (every
    property required, no additional properties).
    """
    about = str(spec.get("about") or "").strip()
    one_of = list(spec.get("one_of") or [])

    if spec.get("type") == "text":
        leaf: dict[str, Any] = {"type": "string", "description": about}
        if one_of:
            leaf["enum"] = one_of
        return leaf

    nested = spec.get("of")
    if nested:
        keys = list(nested)
        # A list of objects, one per entity, every key required.
        return {
            "type": "array", "description": about,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": keys,
                "properties": {k: {"type": "string",
                                   "description": str(nested[k]).strip()}
                               for k in keys},
            },
        }

    items: dict[str, Any] = {"type": "string"}
    if one_of:
        items["enum"] = one_of
    return {"type": "array", "description": about, "items": items}


def compile_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """`{name: JSON Schema fragment}`, in the order the fields were written."""
    return {name: _compile_field(f) for name, f in fields.items()}


def check_fields(fields: dict[str, Any]) -> list[str]:
    """Everything wrong with a field spec, as messages."""
    problems: list[str] = []
    if len(fields) > MAX_FIELDS:
        problems.append(f"{len(fields)} fields; {MAX_FIELDS} max -- a longer "
                        "answer truncates rather than failing")

    for name, field in fields.items():
        where = f"field {name!r}"
        if not FIELD_NAME.match(str(name)):
            problems.append(f"{where} must match {FIELD_NAME.pattern}")
        if not isinstance(field, dict):
            problems.append(f"{where} must be an object with a `type`")
            continue

        kind = field.get("type")
        if kind not in FIELD_TYPES:
            problems.append(f"{where}: unknown type {kind!r}; "
                            f"known: {', '.join(FIELD_TYPES)}")
        if not str(field.get("about") or "").strip():
            # `about` is what the model is told to put in the field.
            problems.append(f"{where} needs an `about` describing what to put "
                            "in it -- it is what the model is steered by")

        one_of = field.get("one_of")
        if one_of is not None:
            if not isinstance(one_of, list) or not one_of:
                problems.append(f"{where}: `one_of` must be a non-empty list")
            elif len(one_of) > MAX_ENUM:
                problems.append(f"{where}: {len(one_of)} choices; {MAX_ENUM} max")
            elif not all(isinstance(v, str) and v.strip() for v in one_of):
                problems.append(f"{where}: `one_of` values must be strings")

        nested = field.get("of")
        if nested is None:
            continue
        if kind == "text":
            problems.append(f"{where}: `of` needs type 'list' -- it makes each "
                            "entry an object, so there must be entries")
        if one_of is not None and nested:
            problems.append(f"{where}: `one_of` and `of` are exclusive; a "
                            "vocabulary constrains a value, `of` replaces it")
        if not isinstance(nested, dict) or not nested:
            problems.append(f"{where}: `of` must be a non-empty "
                            "{key: description} map")
            continue
        if len(nested) > MAX_NESTED_KEYS:
            problems.append(f"{where}: {len(nested)} keys; {MAX_NESTED_KEYS} max")
        for key, description in nested.items():
            if not FIELD_NAME.match(str(key)):
                problems.append(f"{where}: key {key!r} must match "
                                f"{FIELD_NAME.pattern}")
            if not str(description or "").strip():
                problems.append(f"{where}: key {key!r} needs a description")
    return problems


def problems(fields: Any) -> list[str]:
    """`check_fields`, plus the check that it is a non-empty field map."""
    if not isinstance(fields, dict) or not fields:
        return ["`fields` must be a non-empty {name: {type, about}} map"]
    return check_fields(fields)


__all__ = ["FIELD_NAME", "FIELD_TYPES", "MAX_ENUM", "MAX_FIELDS",
           "MAX_NESTED_KEYS", "check_fields", "compile_fields", "problems"]
