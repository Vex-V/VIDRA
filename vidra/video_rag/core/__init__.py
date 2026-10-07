"""What every video_rag pipeline uses, whatever it reads frames from.

    frames      the frame store: one JPEG per kept frame
    sampling    the decimator, the frame reader, the samplers and their spec
    describe    the question vocabulary and the model call that answers one
    retrieve    a query to ranked moments, over what any pipeline wrote

`vocabulary()` is the one thing `aggregates` asks of this tier. It lives here
because a sampler or question name means the same thing in every pipeline.
Nothing here imports a pipeline.
"""

from __future__ import annotations

from typing import Any, Optional


# ------------------------------------------- the one thing aggregates asks
def vocabulary() -> dict[str, Any]:
    """What an aggregate's input may name: every sampler, and every question with
    its fields -- `{field: [entry keys]}` for a list of objects, None otherwise.
    """
    from .describe import library
    from .sampling import samplers as _samplers

    def fields(question: str) -> dict[str, Optional[list[str]]]:
        out: dict[str, Optional[list[str]]] = {}
        for name, spec in (library.shape_of(question).get("fields") or {}).items():
            items = spec.get("items") if isinstance(spec, dict) else None
            nested = items.get("properties") if isinstance(items, dict) else None
            out[name] = list(nested) if nested else None
        return out

    return {"samplers": _samplers.available(),
            "questions": {q: fields(q) for q in library.questions()}}


__all__ = ["vocabulary"]
