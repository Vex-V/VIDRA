"""7 · describe -- one model answer per (chunk, sampler, question).

Reads the frame store. A stored answer is reused while its manifest,
describer and question hash still match. The question vocabulary
(`add_question`, `question`, `questions`) is published here too.
"""

from __future__ import annotations

#: The public surface: entry points, errors and return types.
from ...core.describe.base import DescriberUnavailable
from ...core.describe.frames import StoreUnavailable
from ...core.describe.library import (PromptError, ProtectedPrompt, add_question,
                                     question, questions, remove_question)
from .driver import answer, describe, load

__all__ = ["DescriberUnavailable", "PromptError", "ProtectedPrompt",
           "StoreUnavailable", "add_question", "answer", "describe", "load", "question", "questions", "remove_question"]
