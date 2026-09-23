"""The type reference, generated from the dataclasses.

Every type the library hands back is a dataclass, so `TYPES.txt` is read off
them rather than written beside them -- the argument `schemas.py` already
makes for the JSON Schemas, and for the same reason: a restated field list
drifts, and a drifted one documents a field that no longer exists or hides one
that does. `--check` fails on a stale file, so the two cannot separate.

**Field documentation is read from the source, not from the objects.** Python
keeps no per-attribute docstring at runtime -- `#:` above a field and a bare
string below it are both conventions the interpreter discards -- so this parses
the module with `ast`. Both spellings are understood, because both are already
used in this tree.

`--undocumented` lists fields with nothing said about them, which is what makes
this a check rather than a dump.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
import typing
from pathlib import Path
from typing import Any, Optional, get_args, get_origin

from .. import paths


def out_path() -> Path:
    """Where the reference is written. A checkout only: this is source."""
    return paths.checkout_root() / "TYPES.txt"


# --------------------------------------------------------------- field docs

def _field_docs(cls: type) -> dict[str, str]:
    """`{field: what the source says about it}` for one dataclass.

    Two spellings, both already in this tree:

        #: a comment above the field
        name: str

        name: str
        \"\"\"a string below the field\"\"\"
    """
    try:
        source = inspect.getsource(inspect.getmodule(cls))
    except (OSError, TypeError):
        return {}
    lines = source.splitlines()

    node = next((n for n in ast.walk(ast.parse(source))
                 if isinstance(n, ast.ClassDef) and n.name == cls.__name__),
                None)
    if node is None:
        return {}

    docs: dict[str, str] = {}
    body = node.body
    for i, stmt in enumerate(body):
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target,
                                                                 ast.Name):
            continue
        name = stmt.target.id

        # `#:` comments immediately above, read upward until they stop.
        above: list[str] = []
        row = stmt.lineno - 2                      # 0-based, one line up
        while row >= 0 and lines[row].strip().startswith("#:"):
            above.insert(0, lines[row].strip()[2:].strip())
            row -= 1

        # A bare string literal on the next statement.
        below = ""
        if i + 1 < len(body):
            nxt = body[i + 1]
            if (isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant)
                    and isinstance(nxt.value.value, str)):
                below = " ".join(nxt.value.value.split())

        text = " ".join(above) or below
        if text:
            docs[name] = text
    return docs


# ------------------------------------------------------------------- typing

def _render(annotation: Any) -> str:
    """One annotation, as a reader would write it.

    `typing.get_type_hints` is deliberately not used: it resolves
    `Optional[X]` into `Union[X, None]` and loses the spelling the signature
    published, and `/capabilities` already learned that the annotation as
    written is the contract.
    """
    if isinstance(annotation, str):
        return annotation
    if annotation is type(None):
        return "None"
    if annotation is Any:
        return "Any"

    origin, args = get_origin(annotation), get_args(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))
    if origin is typing.Union:
        rendered = [_render(a) for a in args]
        if "None" in rendered and len(rendered) == 2:
            return f"Optional[{next(r for r in rendered if r != 'None')}]"
        return " | ".join(rendered)
    name = getattr(origin, "__name__", str(origin))
    return f"{name}[{', '.join(_render(a) for a in args)}]" if args else name


# ------------------------------------------------------------------ writing

def _wrap(text: str, indent: int, width: int = 78) -> list[str]:
    return textwrap.wrap(text, width=width,
                         initial_indent=" " * indent,
                         subsequent_indent=" " * indent) or []


def describe_type(cls: type) -> list[str]:
    """One dataclass, as the lines it contributes to the reference."""
    out = [cls.__name__, "-" * len(cls.__name__)]

    doc = inspect.getdoc(cls) or ""
    if doc and not doc.startswith(f"{cls.__name__}("):
        for para in doc.split("\n\n"):
            out += _wrap(" ".join(para.split()), 4)
            out.append("")
    if out[-1] != "":
        out.append("")

    docs = _field_docs(cls)
    for field in dataclasses.fields(cls):
        default = ""
        if field.default is not dataclasses.MISSING:
            default = f" = {field.default!r}"
        elif field.default_factory is not dataclasses.MISSING:   # type: ignore[misc]
            default = " = <factory>"
        out.append(f"    {field.name}: {_render(field.type)}{default}")
        if field.name in docs:
            out += _wrap(docs[field.name], 8)

    # Properties are part of what a caller reads, and several carry the only
    # answer to "how do I ask whether this has audio".
    props = [(n, v) for n, v in vars(cls).items()
             if isinstance(v, property) and not n.startswith("_")]
    if props:
        out.append("")
        out.append("    properties")
        for name, prop in sorted(props):
            summary = " ".join((inspect.getdoc(prop.fget) or "").split())
            out.append(f"        .{name}")
            if summary:
                out += _wrap(summary, 12)

    methods = [n for n, v in vars(cls).items()
               if callable(v) and not n.startswith("_")
               and n not in ("as_dict", "from_dict")]
    if methods:
        out.append("")
        out.append(f"    methods   {' '.join(sorted(methods))}")

    out.append("")
    return out


def _types() -> list[tuple[str, list[type]]]:
    """Every public type, grouped by what it is for.

    Imported here rather than at module scope: `units` and `search` pull in
    their components, and this is a development tool.
    """
    from . import documents, units
    # The class by its full module path. `from ..retrieve import search`
    # binds the *function*, because the package's `__init__` exports it under
    # that name -- which is the surface trim working as intended.
    from ...video_rag.retrieve.search import Moment
    from ... import workflow

    return [
        ("WHAT A COMPONENT WRITES", [
            documents.Media, documents.VideoStream, documents.AudioStream,
            documents.RawTranscript, documents.Cuts, documents.Timeline,
            documents.Manifest, documents.Transcript, documents.Descriptions,
            documents.Embedded, documents.Aggregate]),
        ("WHAT A RUN REPORTS", [
            documents.Produced, workflow.Run]),
        ("WHAT A SEARCH RETURNS", [
            Moment]),
        ("WHAT EMBEDDING WORKS ON", [units.Unit]),
    ]


HEADER = """FALCONVAR -- TYPE REFERENCE
===========================
Generated from the dataclasses by `python -m falconvar.shared.contracts.reference`.
Do not edit: the dataclasses are the source, and `--check` fails on a stale copy.

Every type here is a plain dataclass. The documents carry `as_dict()` and
`from_dict()`, which is how a component writes one and how `load` reads it
back; nothing else here needs a constructor call from outside the library.

A component's `run(video_id, ...)` returns `Produced`, a receipt for a write.
A component's pure verb -- `split`, `listen`, `detect`, `timeline`, `ingest`,
`apply`, `answer`, `encode` -- returns the document itself, which is one of
the types under WHAT A COMPONENT WRITES.
"""


def generate() -> str:
    lines = HEADER.splitlines()
    for heading, group in _types():
        lines += ["", "", heading, "=" * len(heading), ""]
        for cls in group:
            lines += describe_type(cls)
    return "\n".join(lines).rstrip() + "\n"


def undocumented() -> list[str]:
    """Fields the source says nothing about. The reason this is a check."""
    missing = []
    for _, group in _types():
        docs_by_class = {cls: _field_docs(cls) for cls in group}
        for cls in group:
            for field in dataclasses.fields(cls):
                if field.name not in docs_by_class[cls]:
                    missing.append(f"{cls.__name__}.{field.name}")
    return missing


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Generate the type reference.")
    ap.add_argument("--check", action="store_true",
                    help="fail if TYPES.txt is stale rather than rewriting it")
    ap.add_argument("--undocumented", action="store_true",
                    help="list fields with nothing said about them")
    args = ap.parse_args(argv)

    if args.undocumented:
        missing = undocumented()
        for name in missing:
            print(f"  {name}")
        print(f"{len(missing)} undocumented field(s)")
        return 0

    text = generate()
    path = out_path()
    if args.check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != text:
            print(f"{path} is stale -- re-run without --check")
            return 1
        print(f"{path.name} current")
        return 0

    path.write_text(text, encoding="utf-8")
    print(f"{path}   {len(text.splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
