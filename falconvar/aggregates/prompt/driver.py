"""The prompt component: any aggregate prompt, by name, over one excerpt.

    prompt("summary", "in/summary.json", "answers/summary.json")
    prompt("incident_report", excerpt, out, llm="anthropic")

`summary`, `chapters` and `events` are not classes -- they are entries in
`definitions.json`, each of a kind (`fold`, `spans`, `items`) whose runner does
the work, and a custom prompt is one more entry. So there is one component for
all of them, named for the section of `definitions` they live in, and the
definition's name says which. A link profile is not a prompt: it reads
sightings, and runs through `entities`.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import available, definitions, kind_of


def prompts() -> list[str]:
    """Every prompt this deployment has: built-in and custom."""
    return [name for name in available()
            if name in definitions.ids() and kind_of(name) != "link"]


def prompt(name: str, excerpt: Any, out: Any, previous: Any = None,
           llm: Optional[str] = None) -> Any:
    """Answer the prompt `name` over the excerpt file at `excerpt`, into the
    answer file `out`. `llm` answers it; `previous` is an earlier answer's
    file, reused when it read the same text under the same words and model."""
    if name not in prompts():
        hint = (f" -- {name} is a link profile; run it through `entities`"
                if name in available() and kind_of(name) == "link" else "")
        from ..driver import AggregateError
        raise AggregateError(f"no prompt {name!r}{hint}; known: {', '.join(prompts())}")
    from ..driver import run_one
    return run_one(name, excerpt, out, previous, llm)


def main(argv: Any = None) -> int:
    from ..driver import component_main
    return component_main(argv, "Answer one aggregate prompt -- summary, chapters, "
                                "events or a custom one -- over an excerpt.",
                          named="the prompt, e.g. summary")


__all__ = ["main", "prompt", "prompts"]
