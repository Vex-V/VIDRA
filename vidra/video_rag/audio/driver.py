"""The audio component: `media.json` -> `transcript.raw.json`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vidra.shared.config import env
from vidra.shared.reporting import logs
from vidra.shared.contracts.documents import Media, Produced, RawTranscript
from vidra.shared.storage.files import read, write
from . import models
#: The reader's `listen` takes built models; this module's takes setting names.
from .reader import listen as _pass
from .source import NoAudio
from vidra.shared.reporting.errors import Refused

#: The default transcriber, by name.
DEFAULT_TRANSCRIBER = "whisper"
DEFAULT_DIARIZER = "pyannote"

#: This component's settings -> the constructor argument each sets on the
#: transcriber and on the diarizer. `device` goes to both.
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
    """Transcribe and diarize the file a `Media` names. Reads and writes no
    artifact.

    Every setting defaults to None: the backend's own default. `exclusive`
    (pyannote) resolves overlapping speech so each word has one speaker. A
    setting no chosen backend takes is refused.
    """
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
    """Transcribe and diarize the whole file; writes no chunk ids. `listen` plus a
    read at each end. The audio decoded is the file at `Media.path`.
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
    env.load()        # an entry point reads .env; the library never does
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
    ap.add_argument("--device", default=None,
                    help="cuda | cpu (default: cuda where torch finds one; "
                         "pyannote then tries mps, which whisper cannot use)")
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
