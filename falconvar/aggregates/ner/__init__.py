"""`ner` -- named entities, and which chunks each appears in.

    ner.ner(excerpt, out, labels=("person", "product"))   an excerpt in, one answer out

Imported only when asked for by name: a free-tier run must not pull in torch.
The registry holds `\"ner:NERAggregator\"` as a string for that reason, and
GLiNER itself is imported inside the model call.
"""

from .driver import NERAggregator, ner

__all__ = ["NERAggregator", "ner"]
