"""A describer that loads nothing: its output is obviously stub text, filling the
same keys a real describer would for the question.
"""

from __future__ import annotations

from typing import Any, Sequence

from .. import library, prompts
from ..base import Description, register
from ..frames import LoadedFrame


@register
class StubDescriber:
    name = "stub"
    concurrency = 8

    async def describe(self, images: Sequence[LoadedFrame],
                       context: dict[str, Any]) -> Description:
        span = f"{context['start_ts']:.1f}-{context['end_ts']:.1f}s"
        indexes = ", ".join(str(f.index) for f in images)
        question = prompts.question_for(context)
        fields = library.fields_of(question)
        return Description(
            summary=(f"[stub] {context['sampler']} chunk {context['chunk_id']} "
                     f"({span}): {len(images)} frames [{indexes}]"),
            fields={key: "" if key == "setting" else [] for key in fields},
        )

    def config(self) -> dict[str, Any]:
        return {"describer": self.name}
