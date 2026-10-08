"""What both pipelines check before any work, beside their own settings.

    problems += run_problems(into, database, sampler, {"vlm": vlm, "embedder": e})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from vidra.shared.storage.database import DATABASES, Database


def run_problems(into: str | Path, database: Optional[str | Database],
                 sampler: Optional[str | Sequence[Any]],
                 models: Mapping[str, Any]) -> list[str]:
    """Problems with where a run writes, what it looks at and who answers, as
    messages; empty means valid. `sampler` None skips the spec (a run not
    reading the picture); `models` is role -> model, None for the default.
    """
    problems: list[str] = []
    if not (database is None or isinstance(database, Database)
            or database in DATABASES):
        problems.append(f"unknown database {database!r}; pass a "
                        f"Database, or one of: {', '.join(DATABASES)}")
    # A file where `into` should be makes every write fail.
    where = Path(into)
    if where.exists() and not where.is_dir():
        problems.append(f"{where} is a file; `into` is the directory that holds "
                        f"one folder per video")
    if sampler is not None:
        # Every sampler and question the spec names, against the registry and
        # the question vocabulary -- including a bare name's own question, which
        # would otherwise fall back to the general one in silence.
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
