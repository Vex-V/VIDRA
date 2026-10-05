"""Named entities (GLiNER, zero-shot), and which chunks each appears in."""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from ...shared.contracts.documents import fingerprint_of
from ..core.base import Context, ModelUnavailable
from ..core.inputs import Input, Read, read
from ..core.rendering import pieces, plain

#: What to look for: the label set is the configuration.
DEFAULT_LABELS = ("person", "organisation", "location", "product",
                  "event", "date")

DEFAULT_NER_MODEL = "urchade/gliner_small-v2.1"

#: The longest piece GLiNER is handed; it reads 384 tokens.
NER_CHARS = 1200


class NERAggregator:
    """Named entities across the video, with the chunks they appear in."""

    name = "ner"
    tier = "local"
    about = "named entities, and which chunks each appears in"
    depends_on: tuple[str, ...] = ()
    takes_inputs = True

    def __init__(self, model: Optional[str] = None,
                 labels: tuple[str, ...] = DEFAULT_LABELS,
                 threshold: float = 0.5) -> None:
        self.model_name = model or DEFAULT_NER_MODEL
        self.labels = list(labels)
        self.threshold = threshold

    @property
    def version(self) -> str:
        return fingerprint_of({"model": self.model_name, "labels": self.labels,
                               "threshold": self.threshold, "chars": NER_CHARS})

    def _model(self) -> Any:
        from gliner import GLiNER
        try:
            return GLiNER.from_pretrained(self.model_name)
        except Exception as exc:                         # noqa: BLE001
            raise ModelUnavailable(
                f"could not load {self.model_name!r}: {exc}") from None

    def read(self, context: Context, one: Input) -> Read:
        return read(context, one)

    def run(self, context: Context, read: Read) -> dict[str, Any]:
        model = self._model()

        # (text, label) -> chunks.
        found: dict[tuple[str, str], set[int]] = {}
        cut = 0
        for row in read.rows:
            parts = pieces(plain(row), NER_CHARS)
            cut += len(parts)
            for part in parts:
                for entity in model.predict_entities(part, self.labels,
                                                     threshold=self.threshold):
                    key = (entity["text"].strip(), entity["label"])
                    if key[0]:
                        found.setdefault(key, set()).add(row.chunk_id)

        entities = [{"text": text, "label": label,
                     "chunk_ids": sorted(chunks), "mentions": len(chunks)}
                    for (text, label), chunks in found.items()]
        entities.sort(key=lambda e: (-e["mentions"], e["text"].lower()))
        return {
            "entities": entities,
            "count": len(entities),
            "by_label": dict(Counter(e["label"] for e in entities)),
            "chunks_read": len(read.rows),
            "pieces": cut,
            "model": self.model_name,
            "labels": self.labels,
        }


def ner(excerpt: Any, out: Any, previous: Any = None, model: Optional[str] = None,
        labels: tuple[str, ...] = DEFAULT_LABELS, threshold: float = 0.5) -> Any:
    """Named entities in the excerpt file at `excerpt`, into `out`. `labels` is part
    of the answer's version, so changing it recomputes.
    """
    from ..driver import run_one
    return run_one("ner", excerpt, out, previous,
                   settings={"model": model, "labels": tuple(labels),
                             "threshold": threshold})


def main(argv: Any = None) -> int:
    from vidra.shared.config import env
    env.load()        # an entry point reads .env; the library never does
    from ..driver import component_main
    return component_main(argv, "Named entities in an excerpt, and which chunks "
                                "each appears in.", aggregator="ner")


__all__ = ["NERAggregator", "main", "ner"]
