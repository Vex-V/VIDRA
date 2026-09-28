"""What this library raises, and one base under all of it.

Imports nothing -- like `paths` and `documents`, and for the same reason. A
module of exception classes with no dependencies is a leaf, so anything may
import it and no cycle is possible. It is the one import `paths` allows itself.

**Every deliberate failure is a `FalconvarError`.** Twenty classes across ten
modules had no common ancestor, so a caller embedding this had no way to say
"anything the library refused" short of listing them -- and a list drifts the
moment one is added. `except FalconvarError` is the whole point of the base.

**The builtin base each one already had is kept.** `UnknownBackend` was a
`ValueError` and several `except ValueError` clauses depend on that, inside
this tree and in anyone's code already written against it. So these are mixed
in rather than substituted: `class UnknownBackend(FalconvarError, ValueError)`
catches under both, and nothing that worked before stops working.

**`Unavailable` is the one worth catching separately.** A missing package,
missing weights, an absent key, a server that is not there: all of them mean
*this deployment cannot do that*, and none of them is fixed by retrying. That
is a different response from a bad argument, which is why it is a branch and
not a flat list.
"""

from __future__ import annotations


class FalconvarError(Exception):
    """Anything this library refuses or cannot do, raised on purpose."""


class Unavailable(FalconvarError):
    """This deployment cannot do that: no package, no weights, no key, no
    server. Retrying will not fix it; installing or configuring something
    might."""


class ModelUnavailable(Unavailable):
    """A model that will not load here.

    One class, not two. `audio` and the local aggregates each had their own
    with this name and different bases, so a caller importing both had to
    alias one, and `except ModelUnavailable` silently covered only half of
    what it looked like it covered.
    """


class UnknownOption(FalconvarError, ValueError, KeyError):
    """A value outside a fixed vocabulary -- a policy, a sampler, a
    transcriber, a conflict rule.

    `sinks.UnknownBackend` covered exactly this while documents fanned out to
    several destinations, and removing it left the two remaining vocabularies
    -- `media.on_conflict` and the pipeline's `database` -- raising a bare
    `ValueError`, which "anything the library refused" could not catch. Not
    exported: nobody catches a bad literal by name, and `except ValueError`
    already fires. It is here so the hierarchy has no hole in it.

    **Both builtin bases.** The registries -- policies, samplers, audio
    backends -- raised `KeyError` for an unknown name and the rest `ValueError`,
    for the same mistake. Each is kept, so every `except` already written
    against either still fires. `KeyError` renders its message quoted, as a
    missing key's repr; this is a sentence, so it renders as one.
    """

    __str__ = Exception.__str__


class Refused(FalconvarError, ValueError):
    """A request the library will not carry out as given.

    A setting out of range (`limit=0`, `batch=0`, a stride of 0), settings that
    contradict each other (a floor above the ceiling), a setting nothing it
    would run reads (`silence_s` under `speaker`), documents that do not belong
    together (two videos, two grids), or nothing to do at all. Every one used
    to be a bare `ValueError` -- measured by `example.py`, seven of the
    refusals a caller meets first could not be caught as "anything the library
    refused". The message says what to do instead.
    """


__all__ = ["FalconvarError", "ModelUnavailable", "Refused", "Unavailable",
           "UnknownOption"]
