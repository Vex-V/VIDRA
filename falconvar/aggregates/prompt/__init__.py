"""`prompt` -- any aggregate prompt, by name: summary, chapters, events, custom.

    prompt.prompt("summary", excerpt, out)      an excerpt in, one answer out
"""

from .driver import prompt, prompts

__all__ = ["prompt", "prompts"]
