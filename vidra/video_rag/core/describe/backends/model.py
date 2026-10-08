"""The describer: frames in, one structured answer out, from any VLM.

The stored JPEG bytes are sent as they are, each labelled with its timestamp.
How the model is asked for a shape is the VLM's business; this checks what
comes back.
"""

from __future__ import annotations

from typing import Any, Sequence

from vidra.shared.models.base import Calls, ModelFailed, image, text
from .. import prompts
from ..base import DescriberUnavailable, Description
from ..frames import LoadedFrame

#: The default ceiling on one answer's output tokens.
DEFAULT_MAX_TOKENS = 2000


class ModelDescriber:
    """Describes one (chunk, sampler) pair with a single call."""

    def __init__(self, vlm: Calls,
                 max_output_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self.vlm = vlm
        self.name = vlm.name
        self.model = vlm.model
        self.max_output_tokens = max_output_tokens

    @property
    def concurrency(self) -> int:
        """Runs in flight at once: the VLM's cap on calls."""
        return self.vlm.concurrency

    # -- request assembly -----------------------------------------------------
    def parts_for(self, images: Sequence[LoadedFrame],
                  context: dict[str, Any]) -> list[dict[str, Any]]:
        """The instruction, then every frame labelled with its own timestamp."""
        # The question this call is for, which may differ from the sampler's name.
        parts = [text(prompts.for_sampler(prompts.question_for(context),
                                          context, len(images)))]
        for position, frame in enumerate(images, start=1):
            parts.append(text(prompts.frame_label(
                frame.index, frame.media_ts, position, len(images))))
            parts.append(image(frame.jpeg))
        return parts

    async def describe(self, images: Sequence[LoadedFrame],
                       context: dict[str, Any]) -> Description:
        where = f"chunk {context['chunk_id']} / {context['sampler']}"
        if not images:
            # Every chunk keeps a frame, so this means the manifest and the store disagree.
            raise DescriberUnavailable(f"no frames for {where}")
        question = prompts.question_for(context)
        try:
            # The question's own schema.
            payload = await self.vlm.generate(
                self.parts_for(images, context),
                {"name": f"description_{question}",
                 "schema": prompts.schema_for(question)},
                prompts.SYSTEM, self.max_output_tokens)
        except ModelFailed as exc:
            raise DescriberUnavailable(f"{self.vlm.key}, {where}: {exc}") from None

        if not isinstance(payload, dict):
            raise DescriberUnavailable(f"{self.vlm.key}, {where}: answer was not an object")
        summary = (payload.get("summary") or "").strip()
        if not summary:
            # An empty answer is not recorded as a description.
            raise DescriberUnavailable(f"{self.vlm.key}, {where}: no summary in the answer")
        return Description(summary=summary,
                           fields={k: v for k, v in payload.items() if k != "summary"})

    def config(self) -> dict[str, Any]:
        """Half of the resume key: the VLM's key, split at its first colon, and
        the settings."""
        return {
            "name": self.name,
            "params": {
                "model": self.model,
                "max_output_tokens": self.max_output_tokens,
                # Which shape produced the answer and how it was enforced.
                "response": f"{self.vlm.response_mode}/description_<sampler>",
                # Prompt hashes are recorded per question by `reader`.
            },
        }


__all__ = ["DEFAULT_MAX_TOKENS", "ModelDescriber"]
