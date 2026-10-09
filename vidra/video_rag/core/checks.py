"""What both pipelines check before any work, beside their own settings.

    problems += run_problems(into, database, sampler, {"vlm": vlm, "embedder": e})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from vidra.shared.storage.database import Database, problem


def run_problems(into: str | Path, database: Optional[Database],
                 sampler: Optional[str | Sequence[Any]],
                 models: Mapping[str, Any]) -> list[str]:
    """Problems with where a run writes, what it looks at and who answers, as
    messages; empty means valid. `sampler` None skips the spec (a run not
    reading the picture); `models` is role -> model, None for the default.
    """
    problems: list[str] = []
    if problem(database):
        problems.append(problem(database))
    # `into` must be a directory, when it exists.
    where = Path(into)
    if where.exists() and not where.is_dir():
        problems.append(f"{where} is a file; `into` is the directory that holds "
                        f"one folder per video")
    if sampler is not None:
        # Every sampler and question the spec names, against the registry and
        # the question vocabulary, including a bare name's own question.
        from .describe import prompts
        from .sampling import specs
        problems += specs.problems(sampler, prompts.questions())
    # Every model the run will call: the right kind, with a key and an API key.
    from vidra.shared.models import base
    from vidra.shared.models.roles import resolve
    for role, model in models.items():
        problems += base.problems(role, resolve(role, model))
    return problems


__all__ = ["run_problems"]
