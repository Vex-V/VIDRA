"""The prompt component: any aggregate prompt, by name, over one excerpt.

    prompt("summary", "in/summary.json", "answers/summary.json")
    prompt("incident_report", excerpt, out, llm="anthropic")

`summary`, `chapters`, `events` and custom prompts are entries in the
definitions, each run by its kind's runner. Link profiles run through
`entities`.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import available, definitions, kind_of


def prompts() -> list[str]:
    """Every prompt this deployment has: built-in and custom."""
    return [name for name in available()
            if name in definitions.ids() and kind_of(name) != "link"]


def prompt(name: str, excerpt: Any, out: Any, previous: Any = None,
           llm: Optional[str] = None, embedder: Optional[str] = None,
           models: Optional[Any] = None, **settings: Any) -> Any:
    """Answer the prompt `name` over the excerpt file at `excerpt`, into `out`.
    `previous` is an earlier answer's file, reused while current. `models` carries
    the llm, the embedder and their keys. A `spans` prompt (`chapters`) also
    takes `embedder`, `max_spans` and `min_span_s`.
    """
    if name not in prompts():
        hint = (f" -- {name} is a link profile; run it through `entities`"
                if name in available() and kind_of(name) == "link" else "")
        from ..driver import AggregateError
        raise AggregateError(f"no prompt {name!r}{hint}; known: {', '.join(prompts())}")
    from ..driver import run_one
    return run_one(name, excerpt, out, previous, llm, embedder, settings, models)


def main(argv: Any = None) -> int:
    from vidra.shared.config import env
    env.load()        # an entry point reads .env; the library never does
    from ..driver import component_main
    return component_main(argv, "Answer one aggregate prompt -- summary, chapters, "
                                "events or a custom one -- over an excerpt.",
                          named="the prompt, e.g. summary")


__all__ = ["main", "prompt", "prompts"]
