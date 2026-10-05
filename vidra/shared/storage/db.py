"""The Supabase client, batched writes, and the key variable names.

The secret key writes; the publishable key reads.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..reporting.errors import Unavailable

#: The Postgres schema every table lives in.
SCHEMA = "vidra"

URL_VARS = ("SUPABASE_URL",)
SECRET_VARS = ("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY",
               "SUPABASE_SERVICE_KEY")
PUBLISHABLE_VARS = ("SUPABASE_PUBLISHABLE_KEY", "SUPABASE_ANON_KEY")


class DatabaseUnavailable(Unavailable):
    """No URL, no key, no client library, or a server that will not answer."""


class SchemaOutOfDate(DatabaseUnavailable, RuntimeError):
    """A database whose schema predates this code: re-run the schema files."""


def _first(names: tuple[str, ...]) -> Optional[str]:
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def client(url: Optional[str] = None, key: Optional[str] = None,
           write: bool = True) -> Any:
    """A Supabase client. ``write=False`` uses the read-only key."""
    url = url or _first(URL_VARS)
    names = SECRET_VARS if write else PUBLISHABLE_VARS
    key = key or _first(names)
    if not url or not key:
        raise DatabaseUnavailable(
            "set SUPABASE_URL and " + " or ".join(names) +
            " in .env (it is gitignored) or in the environment")
    from supabase import ClientOptions, create_client
    try:
        return create_client(url, key,
                             options=ClientOptions(schema=SCHEMA))
    except Exception as exc:                             # noqa: BLE001
        raise DatabaseUnavailable(f"could not connect: {exc}") from None


def upsert_vectors(table: str, rows: list[dict[str, Any]], api: Any = None) -> int:
    """`upsert`, turning a fixed-width vector column's refusal into "re-run the
    schema file".
    """
    try:
        return upsert(table, rows, api)
    except Exception as exc:                             # noqa: BLE001
        if "dimension" not in str(exc).lower():
            raise
        raise SchemaOutOfDate(
            f"{table} refused these vectors ({exc}). The column still has a "
            "fixed width: re-run db/supabase/video_rag.sql, which lets it hold "
            "any embedder's") from None



def upsert(table: str, rows: list[dict[str, Any]], api: Any = None,
           chunk: int = 200) -> int:
    """Upsert rows in batches. Returns how many were sent."""
    if not rows:
        return 0
    api = api or client()
    for start in range(0, len(rows), chunk):
        api.table(table).upsert(rows[start:start + chunk]).execute()
    return len(rows)


def delete_stale_chunks(table: str, video_id: str, keep_below: int,
                        api: Any = None) -> None:
    """Delete rows for chunks past `keep_below` (a grid that shrank). Call after the
    upserts.
    """
    api = api or client()
    (api.table(table).delete()
        .eq("video_id", video_id).gte("chunk_id", keep_below).execute())


def delete_except(table: str, match: dict[str, Any], column: str,
                  keep: list[Any], api: Any = None) -> None:
    """Delete the rows matching `match` whose `column` is not in `keep`. Call after
    the upserts.
    """
    api = api or client()
    query = api.table(table).delete()
    for name, value in match.items():
        query = query.eq(name, value)
    if keep:
        query = query.not_.in_(column, list(keep))
    query.execute()


__all__ = ["upsert_vectors", "DatabaseUnavailable", "PUBLISHABLE_VARS", "SECRET_VARS",
           "URL_VARS", "client", "delete_except",
           "delete_stale_chunks", "upsert"]
