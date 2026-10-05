"""The implementations, one package per aggregator and per prompt kind.

Not imported by name from outside: `aggregates.stats(...)` and the rest are
the public calls. They live a level down because a package attribute and a
submodule of one name are one slot -- `aggregates.stats` could not be both.
"""
