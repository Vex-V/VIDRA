"""Who spoke, for how long, and how often the voice changed."""

from __future__ import annotations

from typing import Any

from ..core.base import Context

class SpeakersAggregator:
    name = "speakers"
    tier = "free"
    about = "who spoke, for how long, and how often the voice changed"
    depends_on = ("transcript",)

    def run(self, context: Context) -> dict[str, Any]:
        held: dict[str, float] = {}
        handovers = 0
        previous = None
        turn_count = 0
        for chunk in context.transcript.chunks:
            for turn in chunk.get("turns", []):
                speaker = turn.get("speaker")
                if speaker is None:
                    continue
                turn_count += 1
                held[speaker] = held.get(speaker, 0.0) + (
                    float(turn["end"]) - float(turn["start"]))
                if previous is not None and speaker != previous:
                    handovers += 1
                previous = speaker

        duration = context.timeline.duration_s
        speech = sum(held.values())
        ranked = sorted(held.items(), key=lambda kv: -kv[1])
        return {
            "speakers": len(held),
            "turns": turn_count,
            "handovers": handovers,
            # One voice and no handovers.
            "monologue": len(held) <= 1 and handovers == 0,
            "speech_s": round(speech, 3),
            "speech_ratio": round(speech / duration, 4) if duration else 0.0,
            "by_speaker": [{"speaker": s, "seconds": round(v, 3),
                            "share": round(v / speech, 4) if speech else 0.0}
                           for s, v in ranked],
            "dominant": ranked[0][0] if ranked else None,
        }


def speakers(source: Any, out: Any, previous: Any = None) -> Any:
    """Who spoke in the record at `source`, into the answer file `out`. Needs
    a transcript; a record without one raises `Inapplicable`, saying so."""
    from ..driver import run_one
    return run_one("speakers", source, out, previous)


def main(argv: Any = None) -> int:
    from falconvar.shared.config import env
    env.load()        # an entry point reads .env; the library never does
    from ..driver import component_main
    return component_main(argv, "Who spoke in a record, for how long, and how "
                                "often the voice changed.", aggregator="speakers")


__all__ = ["SpeakersAggregator", "main", "speakers"]
