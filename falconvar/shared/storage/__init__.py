"""Where a document goes.

`files` is what every component writes through -- one document, one path,
atomically. `db` is the Supabase client, and `supabase` the only module that
knows a table name.

**Nothing below a component reaches `supabase`.** A component produces a
document and stops; a pipeline reads it back and decides whether a copy also
belongs in a database. That is what makes a second one -- `sql.py`, offering
the same `write_<artifact>` names -- a sibling file rather than a new branch
inside a fan-out.
"""
