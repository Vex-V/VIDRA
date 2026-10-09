"""7 · describe -- one model answer per (chunk, sampler, question).

Reads the frame store. A stored answer is reused while its manifest,
describer and question hash still match. The question vocabulary
(`add_question`, `question`, `questions`) is published here too, and `ask`:
one question about images you hold, with no documents.
"""

from __future__ import annotations

#: The public surface: entry points, errors and return types.
from ...core.describe.base import DescriberUnavailable, Description
from ...core.describe.frames import StoreUnavailable
from ...core.describe.library import (PromptError, ProtectedPrompt, add_question,
                                     question, questions, remove_question)
from .driver import answer, ask, ask_async, describe, load

__all__ = ["DescriberUnavailable", "Description", "PromptError", "ProtectedPrompt",
           "StoreUnavailable", "add_question", "answer", "ask", "ask_async", "describe",
           "load", "question", "questions", "remove_question"]
