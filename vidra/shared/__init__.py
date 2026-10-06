"""What more than one component needs.

    config/         where things are, and what a process was told
      paths.py      where things live; the only module that knows a filename
      env.py        reading a `.env` -- by an entry point, or when asked
      settings.py   `vidra.configure()`
    reporting/      how a run tells its caller something
      errors.py     every deliberate failure, as a `VidraError`
      logs.py       records at the choke points
      progress.py   events from inside the long stages
    contracts/      what components hand each other
      documents.py  what every artifact is
      schemas.py    JSON Schema generated from the dataclasses
    storage/        where a document goes
      files.py      writing a document to its path, atomically
      db.py         the Supabase client, and the one list of its key names
      supabase.py   documents -> rows; the only module that knows table names
    models/         who answers a model call
      base.py       VLM, LLM, Embedder: what a model is, and how it is called
      llm.py        the ready-made VLMs and LLMs: OpenAI, Chat, Anthropic, Stub
      embedders/    the ready-made embedders: OpenAI, local, hash
      roles.py      `Models`: one model per role, and the defaults

`reporting/errors` and `contracts/documents` import nothing of ours but each
other's leaf; `config/paths` only errors. A module only one component needs
does not belong here.
"""
