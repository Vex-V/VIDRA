"""The `link` kind: the same person or thing across chunks, and an account of each.

    entities.entities("people", sightings, out)    sightings in, one answer out

The folder that earned the split. `linking` decides who is who -- embeddings
under rules, no model -- and `driver` asks for the account afterwards. Every
link profile in `definitions` runs through here.
"""

from .driver import EntitiesAggregator, entities

__all__ = ["EntitiesAggregator", "entities"]
