"""VIDRA (Video Description, Retrieval & Aggregation): video in, searchable
moments and higher-level answers out.

    video_rag/   extraction and search: picture and soundtrack read onto one
                 chunk grid, described, embedded and searchable
    aggregates/  answers over what video_rag extracted: counts, speakers,
                 summaries, chapters, events, entities
    shared/      what both tiers use: config, reporting, document contracts,
                 storage, models

`workflow.py` runs video_rag's driver, then aggregates'.
"""

from __future__ import annotations

import logging

from vidra.shared.reporting.errors import VidraError, ModelUnavailable, Unavailable
from vidra.shared.config.settings import configure

#: A library configures no logging; see `shared/reporting/logs.py`.
logging.getLogger("vidra").addHandler(logging.NullHandler())

#: Resolved on first access: `__version__` from the installed metadata, and the
#: values a caller builds once: the models, `Models`, and the databases.
_LAZY = {"Models": "vidra.shared.models.roles",
         "VLM": "vidra.shared.models.base",
         "LLM": "vidra.shared.models.base",
         "Embedder": "vidra.shared.models.base",
         "OpenAI": "vidra.shared.models.llm",
         "Chat": "vidra.shared.models.llm",
         "Anthropic": "vidra.shared.models.llm",
         "Stub": "vidra.shared.models.llm",
         "OpenAIEmbedder": "vidra.shared.models.embedders.remote",
         "LocalEmbedder": "vidra.shared.models.embedders.local",
         "HashEmbedder": "vidra.shared.models.embedders",
         "VisualEmbedder": "vidra.shared.models.base",
         "LocalVisualEmbedder": "vidra.shared.models.embedders.visual",
         "LocalMultimodalEmbedder": "vidra.shared.models.embedders.multimodal",
         "Database": "vidra.shared.storage.database",
         "Supabase": "vidra.shared.storage.supabase",
         "Folder": "vidra.shared.storage.folder",
         # The base of a sampler of your own; `vidra.video_rag.samplers` builds
         # and registers them.
         "Sampler": "vidra.video_rag.core.sampling.samplers.base",
         "View": "vidra.video_rag.core.sampling.samplers.base"}


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


__all__ = ["Anthropic", "Chat", "Database", "Embedder", "Folder", "HashEmbedder", "LLM",
           "LocalEmbedder", "LocalMultimodalEmbedder", "LocalVisualEmbedder", "ModelUnavailable", "Models", "OpenAI",
           "OpenAIEmbedder", "Sampler", "Stub", "Supabase", "Unavailable", "VLM", "VidraError",
           "View", "VisualEmbedder", "__version__", "configure"]

