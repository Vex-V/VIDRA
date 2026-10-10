"""Delete every video, locally and in Supabase.

    python db/wipe.py                  # shows what it would delete, then asks
    python db/wipe.py --local          # only data/out
    python db/wipe.py --supabase       # only the rows
    python db/wipe.py --yes            # no question

Irreversible. Keeps the schema, custom prompts and definitions
(`data/*.json`), `data/eval/` and `weights/`. Uses whichever data root the
pipeline would.
"""
from __future__ import annotations

import argparse
import shutil
import sys

from vidra.shared.config import env, paths
from vidra.shared.storage import db

#: Children before parents. A table a database does not have is skipped.
TABLES = ("ag_mentions", "ag_embeddings", "ag_items", "ag_answers", "ag_sources",
          "vr_observations", "vr_embeddings", "vr_descriptions", "vr_chunk_samplers",
          "vr_manifests", "vr_transcript_chunks", "vr_transcripts", "vr_chunks", "vr_timelines",
          "vr_videos", "vr_prompts", "ag_definitions")


def columns(table: str) -> tuple[str, ...]:
    """The key column to filter on: `source_id` for an aggregate table, `video_id`
    otherwise.
    """
    if table in ("vr_prompts", "ag_definitions"):
        return ("name",)
    return ("source_id", "video_id")


def count(api, table: str):
    """(rows, the column that keys them), or None for a table this database
    does not have."""
    for c in columns(table):
        try:
            n = (api.table(table).select(c, count="exact")
                 .not_.is_(c, "null").limit(1).execute().count)
            return n, c
        except Exception as exc:                            # noqa: BLE001
            text = str(exc)
            if "PGRST205" in text or "42P01" in text:       # no such table
                return None
            if "42703" in text:                             # no such column
                continue
            raise
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Delete every video, locally and in Supabase.")
    ap.add_argument("--local", action="store_true", help="only the local files")
    ap.add_argument("--supabase", action="store_true", help="only the Supabase rows")
    ap.add_argument("--yes", action="store_true", help="do not ask")
    args = ap.parse_args()
    local = args.local or not args.supabase
    remote = args.supabase or not args.local
    env.load()

    folders = [p for p in (paths.OUT_ROOT,) if p.exists()]
    api = counts = None
    if remote:
        api = db.client()                       # the secret key: deletes need it
        counts = {t: found for t in TABLES if (found := count(api, t)) is not None}

    print("Will delete:")
    if local:
        videos = sorted(p.name for p in paths.OUT_ROOT.iterdir()) if paths.OUT_ROOT.exists() else []
        print(f"  local     {', '.join(map(str, folders)) or 'nothing, already empty'}")
        if videos:
            print(f"            {', '.join(videos)}")
    if remote:
        print(f"  supabase  {sum(n for n, _ in counts.values())} rows: "
              + ", ".join(f"{t} {n}" for t, (n, _) in counts.items() if n))
    if not args.yes and input("\nType 'delete' to continue: ").strip() != "delete":
        print("Nothing deleted.")
        return 1

    failed = False
    if local:
        for folder in folders:
            try:
                shutil.rmtree(folder)
                print(f"removed {folder}")
            except OSError as exc:
                failed = True
                print(f"could not remove {folder}: {exc}\n"
                      "  (is another run still using it?)", file=sys.stderr)
    if remote:
        for table in TABLES:
            n, key = counts.get(table) or (0, "")
            if n:
                # PostgREST refuses a delete with no filter.
                api.table(table).delete().not_.is_(key, "null").execute()
                print(f"emptied {table} ({n} rows)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
