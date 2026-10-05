"""Where a document goes.

`files` writes one document to one path, atomically. `database` is the base
every backend implements; `supabase` and `folder` are backends, and `db` is
the Supabase client. Components only write files; a pipeline copies them to
a database.
"""
