"""What this library raises, under one base.

Every deliberate failure is a `FalconvarError` and keeps its builtin base, so
`except FalconvarError` and `except ValueError` both catch it. `Unavailable`
is the branch for "this deployment cannot do that" (no package, weights, key
or server). Imports nothing.
"""

from __future__ import annotations


class FalconvarError(Exception):
    """Anything this library refuses or cannot do, raised on purpose."""


class Unavailable(FalconvarError):
    """This deployment cannot do that: no package, no weights, no key, no
    server. Retrying will not fix it; installing or configuring something
    might."""


class ModelUnavailable(Unavailable):
    """A model that will not load here."""


class UnknownOption(FalconvarError, ValueError, KeyError):
    """A value outside a fixed vocabulary: a policy, a sampler, a transcriber, a
    conflict rule. Both `ValueError` and `KeyError`.
    """

    __str__ = Exception.__str__


class Refused(FalconvarError, ValueError):
    """A request the library will not carry out as given: a setting out of range,
    settings that contradict each other, or a setting nothing reads. The message
    says what to do instead.
    """


__all__ = ["FalconvarError", "ModelUnavailable", "Refused", "Unavailable",
           "UnknownOption"]
