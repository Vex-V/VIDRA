"""Speech evidence: cuts from a finished transcript.

    vad       cut in the middle of a silence longer than `silence_s`
    speaker   cut where the voice changes

Cuts fall in the middle of a gap. Consecutive turns by one speaker are not a
boundary. No guards here: `grid.enforce` applies `min_s`.
"""

from __future__ import annotations

from typing import Any

from vidra.shared.contracts.documents import Cuts, RawTranscript
from vidra.shared.reporting.errors import Refused, UnknownOption

#: Below this, a gap between speech is not a boundary.
DEFAULT_SILENCE_S = 0.65


def speech_spans(transcript: RawTranscript) -> list[tuple[float, float]]:
    """Where there is speech: diarization turns when present, else Whisper's
    segments.
    """
    if transcript.turns:
        return sorted((t["start"], t["end"]) for t in transcript.turns)
    return sorted((s["start"], s["end"]) for s in transcript.segments)


def vad_cuts(transcript: RawTranscript,
             silence_s: float = DEFAULT_SILENCE_S) -> list[tuple[float, float]]:
    """Cut in the middle of every silence longer than `silence_s`. Returns
    `(cut_ts, gap_length)`, so the threshold can be retuned over the cached gaps.
    """
    spans = speech_spans(transcript)
    return [((end + start) / 2.0, start - end)
            for (_, end), (start, _) in zip(spans, spans[1:])
            if start - end >= silence_s]


def speaker_cuts(transcript: RawTranscript) -> list[tuple[float, float]]:
    """Cut between turns wherever the speaker changes; the score is the gap."""
    turns = sorted(transcript.turns, key=lambda t: t["start"])
    cuts: list[tuple[float, float]] = []
    for previous, current in zip(turns, turns[1:]):
        if current.get("speaker") == previous.get("speaker"):
            continue
        gap = current["start"] - previous["end"]
        at = ((previous["end"] + current["start"]) / 2.0 if gap > 0
              else current["start"])
        cuts.append((at, max(gap, 0.0)))
    return cuts


def detect(transcript: RawTranscript, policy: str,
           silence_s: float = DEFAULT_SILENCE_S) -> Cuts:
    """Cuts from a finished transcript, as a `Cuts` document."""
    if policy == "vad":
        scored = vad_cuts(transcript, silence_s)
        params: dict[str, Any] = {"silence_s": silence_s}
    elif policy == "speaker":
        if not transcript.turns:
            # One voice or none: no speaker cuts.
            scored = []
            params = {"speakers": 0}
        else:
            scored = speaker_cuts(transcript)
            params = {"speakers": len(transcript.speakers)}
    else:
        raise UnknownOption(f"unknown speech policy {policy!r}; known: vad, speaker")

    return Cuts(
        video_id=transcript.video_id,
        source="audio",
        detector=policy,
        params=params,
        cuts=[at for at, _ in scored],
        scores={
            "metric": "gap_s",
            "stride": 1,
            "at": [round(at, 3) for at, _ in scored],
            "values": [round(gap, 3) for _, gap in scored],
        },
        stats={"speech_spans": len(speech_spans(transcript)),
               "cuts_found": len(scored),
               "source": "turns" if transcript.turns else "segments"},
    )


def rethreshold(cuts: Cuts, silence_s: float) -> Cuts:
    """A different silence threshold over the cached gaps; `vad` only."""
    if cuts.detector != "vad":
        raise Refused(
            f"{cuts.detector!r} cuts are not thresholded, so there is nothing "
            "to retune; re-run the pass to change them")
    if not cuts.scores:
        raise Refused("this cuts document carries no gap series")
    scored = list(zip(cuts.scores["at"], cuts.scores["values"]))
    kept = [(at, gap) for at, gap in scored if gap >= silence_s]
    return Cuts(
        video_id=cuts.video_id, source=cuts.source, detector=cuts.detector,
        params={**cuts.params, "silence_s": silence_s},
        cuts=[at for at, _ in kept], scores=cuts.scores,
        stats={**cuts.stats, "cuts_found": len(kept), "rethresholded": True},
    )


__all__ = ["DEFAULT_SILENCE_S", "speech_spans", "vad_cuts", "speaker_cuts",
           "detect", "rethreshold"]
