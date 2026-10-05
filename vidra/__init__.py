"""VIDRA (Video Description, Retrieval & Aggregation): video in, searchable
moments and higher-level answers out.

    video_rag/   extraction and search: picture and soundtrack read onto one
                 chunk grid, described, embedded and searchable
    aggregates/  answers over what video_rag extracted: counts, speakers,
                 summaries, chapters, events, entities
    shared/      what both tiers use: config, reporting, document contracts,
                 storage, model providers

`workflow.py` runs video_rag's driver, then aggregates'.
"""

from __future__ import annotations

import logging

from .shared.reporting.errors import VidraError, ModelUnavailable, Unavailable
from .shared.config.settings import configure

#: A library configures no logging; see `shared/reporting/logs.py`.
logging.getLogger("vidra").addHandler(logging.NullHandler())

#: Resolved on first access: `__version__` from the installed metadata, and the
#: values a caller builds once (`Models`, `Database`, `Supabase`, `Folder`).
_LAZY = {"Models": "vidra.shared.models.roles",
         "Database": "vidra.shared.storage.database",
         "Supabase": "vidra.shared.storage.supabase",
         "Folder": "vidra.shared.storage.folder"}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version
        try:
            value = version("vidra")
        except PackageNotFoundError:          # a checkout that was never installed
            value = "0+unknown"
        globals()["__version__"] = value      # resolve once
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Database", "VidraError", "Folder", "ModelUnavailable", "Models", "Supabase",
           "Unavailable", "__version__", "configure"]

