"""The listen component: `media.json` -> `transcript.raw.json`.

Defaults are named here explicitly rather than taken from the head of a
registry. `falconvar` built a form from `transcribe.available()` in order, so
the first option was `stub`; an audio-only run through it completed in 10.6 s,
reported 42 segments and 205 words, and wrote a transcript of
`[stub0.0][stub0.1]`. Nothing was wrong enough to report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from falconvar.shared import env, logs
from falconvar.shared.contracts.documents import Media, Produced, RawTranscript
from falconvar.shared.storage.files import read, write
from . import models
#: Aliased, not renamed. The reader's takes *built* models and this
#: module's takes setting names, so they are two functions -- and two
#: public functions of one name in one package is the infinite recursion
#: `describe` had to be rescued from. See CLAUDE.md.
from .reader import listen as _pass
from .source import NoAudio
from falconvar.shared.errors import Refused

#: Named, not positional. See the module docstring.
DEFAULT_TRANSCRIBER = "whisper"
DEFAULT_DIARIZER = "pyannote"

#: This component's settings, and the constructor argument each one sets on
#: each half. Two maps rather than one because the two models are configured
#: independently and both happen to call their checkpoint `model`: `model` is
#: the transcriber's and `diarizer_model` the diarizer's, which is also how
#: `PyannoteDiarizer.config()` already reports it. `device` is in both, because
#: it is a fact about the machine rather than about either model.
SPEECH_SETTINGS = {"model": "model", "language": "language",
                   "vad_filter": "vad_filter", "compute_type": "compute_type",
                   "device": "device"}
VOICE_SETTINGS = {"diarizer_model": "model", "exclusive": "exclusive",
                  "device": "device"}


def _for(mapping: dict[str, str], takes: set[str],
         named: dict[str, object]) -> dict[str, object]:
    """The subset of ``named`` this backend can actually be told."""
    return {argument: named[setting] for setting, argument in mapping.items()
            if setting in named and argument in takes}


def listen(media: Media,
           transcriber: str = DEFAULT_TRANSCRIBER,
           diarizer: str = DEFAULT_DIARIZER,
           model: Optional[str] = None,
           language: Optional[str] = None,
           vad_filter: Optional[bool] = None,
           compute_type: Optional[str] = None,
           device: Optional[str] = None,
           diarizer_model: Optional[str] = None,
           exclusive: Optional[bool] = None) -> RawTranscript:
    """Transcribe and diarize a file described by a `Media` in hand.

    Reads and writes no artifact -- but it does open the file `media.path`
    names, because a soundtrack is not a document and there is nothing else to
    transcribe. That is the one thing this component cannot be handed.

    Every setting defaults to None, meaning *leave the backend's own default*,
    so a call that names none of them constructs exactly what it always did.

    `exclusive` is the one worth knowing about. pyannote answers with
    overlapping speech either resolved or not, and the whole grid rests on the
    resolved form: a word cannot belong to two speakers, and chunk boundaries
    derived from turns must not overlap. It was previously unreachable -- the
    diarizer was built with no arguments at all -- so the safe reading was the
    only reading. Turning it off is a deliberate choice about overlapping
    speech, not a tuning knob.

    A setting no chosen backend takes is refused rather than dropped. `stub`
    has no `language` and `none` has no `exclusive`, and quietly ignoring one
    would mean a run that reports success having transcribed under settings
    nobody asked for -- the same silence the named defaults at the top of this
    file exist to prevent. The check is here rather than in `run` so that both
    ways into this component get it.
    """
    # Before a model is constructed, not after: pyannote is gated and reads a
    # token from the environment at load time.
    env.load()
    if not media.has_audio:
        raise NoAudio(f"{media.video_id} has no audio stream")

    named: dict[str, object] = {
        setting: value for setting, value in
        (("model", model), ("language", language), ("vad_filter", vad_filter),
         ("compute_type", compute_type), ("device", device),
         ("diarizer_model", diarizer_model), ("exclusive", exclusive))
        if value is not None}

    speech = _for(SPEECH_SETTINGS, models.settings("transcriber", transcriber),
                  named)
    voices = _for(VOICE_SETTINGS, models.settings("diarizer", diarizer), named)
    reached = ({s for s in SPEECH_SETTINGS if SPEECH_SETTINGS[s] in speech}
               | {s for s in VOICE_SETTINGS if VOICE_SETTINGS[s] in voices})
    unreachable = sorted(set(named) - reached)
    if unreachable:
        raise Refused(
            f"transcriber {transcriber!r} and diarizer {diarizer!r} take no "
            f"{', '.join(repr(u) for u in unreachable)}")

    return _pass(media.path,
                 models.transcriber(transcriber, **speech),
                 models.diarizer(diarizer, **voices),
                 video_id=media.video_id)


def audio(media: str | Path, out: str | Path,
          transcriber: str = DEFAULT_TRANSCRIBER,
          diarizer: str = DEFAULT_DIARIZER,
          model: Optional[str] = None,
          language: Optional[str] = None,
          vad_filter: Optional[bool] = None,
          compute_type: Optional[str] = None,
          device: Optional[str] = None,
          diarizer_model: Optional[str] = None,
          exclusive: Optional[bool] = None) -> Produced:
    """Transcribe and diarize the whole file. Writes no chunk ids.

    `listen` plus a read at each end; every check lives down there.

    The file it decodes is `Media.path`, recorded by `media` -- so this takes
    one path and opens a second it was never handed. That is true of every
    addressing, and it is the one place the filepath story does not reach.
    """
    described = read(media, Media)
    with logs.timed("audio", described.video_id) as done:
        raw = listen(described, transcriber, diarizer, model,
                     language, vad_filter, compute_type, device,
                     diarizer_model, exclusive)
        where = write(out, raw)
        done(segments=raw.stats.get("segments"), words=raw.stats.get("words"),
             speakers=raw.stats.get("speakers"), silent=raw.silent)
    return Produced(
        video_id=described.video_id, component="audio",
        artifacts={"raw_transcript": where},
        stats={**raw.stats, "silent": raw.silent},
        skipped=["transcribe", "diarize"] if raw.silent else [],
    )


def load(path: str | Path) -> RawTranscript:
    """Read a `transcript.raw.json` back, typed."""
    return read(path, RawTranscript)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Transcribe and diarize a whole file. No chunking.")
    ap.add_argument("media", help="path to media.json")
    ap.add_argument("out", help="where to write transcript.raw.json")
    ap.add_argument("--transcriber", default=DEFAULT_TRANSCRIBER,
                    choices=sorted(models.TRANSCRIBERS))
    ap.add_argument("--diarizer", default=DEFAULT_DIARIZER,
                    choices=sorted(models.DIARIZERS))
    ap.add_argument("--model", default=None, help="whisper: tiny|base|small|...")
    ap.add_argument("--language", default=None, help="skip detection")
    ap.add_argument("--no-vad-filter", dest="vad_filter", action="store_false",
                    default=None,
                    help="whisper: transcribe the whole track, including what "
                         "its voice-activity filter would drop")
    ap.add_argument("--compute-type", default=None,
                    help="whisper: float16 | int8 | ... (default float16 on "
                         "cuda, int8 on cpu)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda "
                                                   "where torch finds one)")
    ap.add_argument("--diarizer-model", default=None,
                    help="pyannote: a checkpoint id (default "
                         "pyannote/speaker-diarization-3.1). Named here rather "
                         "than read off the backend, which is imported lazily")
    ap.add_argument("--overlaps", dest="exclusive", action="store_false",
                    default=None,
                    help="pyannote: keep overlapping speech rather than "
                         "resolving it. A word then belongs to two speakers")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = audio(args.media, args.out, args.transcriber,
                         args.diarizer, args.model, args.language,
                         args.vad_filter, args.compute_type, args.device,
                         args.diarizer_model, args.exclusive)
    except (NoAudio, KeyError, ValueError, FileNotFoundError,
            models.ModelUnavailable) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    print(f"{produced.video_id}")
    if s.get("silent"):
        print("  silent -- no model was loaded, and no speech is the finding")
    else:
        print(f"  segments     {s['segments']}   words {s['words']}")
        print(f"  speakers     {s['speakers']}   turns {s['turns']}   "
              f"speech {s['speech_s']:g}s")
        print(f"  attributed   {s['attributed']}/{s['words']} words")
        print(f"  decode       {s['decode_s']:.2f}s   "
              f"transcribe {s['transcribe_s']:.2f}s   "
              f"diarize {s['diarize_s']:.2f}s")
    print()
    print(f"raw transcript -> {produced.artifacts['raw_transcript']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
