"""The boundaries component: evidence -> `cuts.json`, then -> `timeline.json`.

    evidence(out, policy, ...)        run the pass this policy needs, if any
    boundaries(media, out, policy)    decide the grid and write it
    retune(cuts, out, threshold)      re-threshold cached scores
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from falconvar.shared.reporting import logs
from falconvar.shared.config.paths import MissingArtifact
from falconvar.shared.contracts.documents import (Cuts, Media, Produced,
                                           RawTranscript, Timeline)
from falconvar.shared.storage.files import maybe, read, write
from falconvar.shared.contracts.documents import same_video
from . import scenes, speech
from .grid import POLICIES, build
from falconvar.shared.reporting.errors import Refused, UnknownOption


def load_cuts(path: str | Path) -> Cuts:
    """Read a `cuts.json` back, typed."""
    return read(path, Cuts)


def load(path: str | Path) -> Timeline:
    """Read a `timeline.json` back, typed."""
    return read(path, Timeline)


#: Policies whose evidence is a scene pass.
_PICTURE = tuple(p for p, needs in POLICIES.items() if needs == "video")

#: Setting -> (the policies that read it, its default).
EVIDENCE_SETTINGS: dict[str, tuple[tuple[str, ...], object]] = {
    "stride": (_PICTURE, scenes.DEFAULT_STRIDE),
    "threshold": (_PICTURE, scenes.DEFAULT_THRESHOLD),
    "detect_width": (_PICTURE, scenes.DETECT_WIDTH),
    "silence_s": (("vad",), speech.DEFAULT_SILENCE_S),
}


def _check_settings(policy: str, given: dict[str, object]) -> None:
    """A setting this policy's evidence pass does not read is refused (a default
    value passes).
    """
    unreachable = sorted(
        name for name, (reads, default) in EVIDENCE_SETTINGS.items()
        if given[name] != default and policy not in reads)
    if unreachable:
        detail = "; ".join(
            f"{n} is read by {', '.join(EVIDENCE_SETTINGS[n][0]) or 'no policy'}"
            for n in unreachable)
        raise Refused(
            f"policy {policy!r} does not read {', '.join(unreachable)} -- "
            f"{detail}")


def detect(policy: str,
           media: Optional[Media] = None,
           transcript: Optional[RawTranscript] = None,
           stride: int = scenes.DEFAULT_STRIDE,
           threshold: float = scenes.DEFAULT_THRESHOLD,
           detect_width: int = scenes.DETECT_WIDTH,
           silence_s: float = speech.DEFAULT_SILENCE_S) -> Optional[Cuts]:
    """Run this policy's evidence pass over documents in hand. Writes nothing.

        detect("scene", media=media, stride=5)      -> Cuts
        detect("vad", transcript=raw)               -> Cuts
        detect("uniform")                           -> None
    """
    if policy not in POLICIES:
        raise UnknownOption(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")
    _check_settings(policy, {"stride": stride, "threshold": threshold,
                             "detect_width": detect_width,
                             "silence_s": silence_s})
    needs = POLICIES[policy]
    if needs is None:
        return None

    if needs == "video":
        if media is None:
            raise Refused(
                f"policy {policy!r} is found in the picture, so `detect` needs "
                f"the media=... it should decode")
        return scenes.detect(media, stride, threshold, detect_width)

    if transcript is None:
        raise Refused(
            f"policy {policy!r} is derived from the soundtrack, so `detect` "
            f"needs the transcript=... to read it from")
    return speech.detect(transcript, policy, silence_s)


def evidence(out: str | Path, policy: str,
             media: Optional[str | Path] = None,
             raw_transcript: Optional[str | Path] = None,
             stride: int = scenes.DEFAULT_STRIDE,
             threshold: float = scenes.DEFAULT_THRESHOLD,
             detect_width: int = scenes.DETECT_WIDTH,
             silence_s: float = speech.DEFAULT_SILENCE_S) -> Produced:
    """Run whichever evidence pass this policy needs and write the cuts to `out`.
    `POLICIES` decides which input is read. Under `uniform` nothing runs and the
    receipt says `skipped: ["evidence"]`.
    """
    if policy not in POLICIES:
        raise UnknownOption(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")
    _check_settings(policy, {"stride": stride, "threshold": threshold,
                             "detect_width": detect_width,
                             "silence_s": silence_s})
    needs = POLICIES[policy]

    if needs is None:
        described = maybe(media, Media)
        video_id = "" if described is None else described.video_id
        logs.skipped("boundaries.evidence", video_id, "evidence",
                     f"policy {policy!r} is arithmetic over the duration")
        return Produced(
            video_id=video_id, component="boundaries.evidence",
            artifacts={}, stats={"policy": policy}, skipped=["evidence"])

    if needs == "video" and media is None:
        raise Refused(
            f"policy {policy!r} is found in the picture, so `evidence` needs "
            f"media=... pointing at a media.json")
    if needs == "audio" and raw_transcript is None:
        raise Refused(
            f"policy {policy!r} is derived from the soundtrack, so `evidence` "
            f"needs raw_transcript=... pointing at a transcript.raw.json")

    with logs.timed(f"boundaries.{needs}") as done:
        cuts = detect(
            policy,
            media=read(media, Media) if needs == "video" else None,
            transcript=(None if needs == "video"
                        else read(raw_transcript, RawTranscript)),
            stride=stride, threshold=threshold, detect_width=detect_width,
            silence_s=silence_s)
        assert cuts is not None               # `needs is None` returned above

        where = write(out, cuts)
        done(policy=policy, detector=cuts.detector, cuts=len(cuts.cuts))
    return Produced(
        video_id=cuts.video_id, component=f"boundaries.{needs}",
        artifacts={"cuts": where},
        stats={**cuts.stats, **cuts.params},
    )


def retune(cuts: str | Path, out: str | Path, threshold: float) -> Produced:
    """A different threshold over the cached scores. Runs no model."""
    stored = load_cuts(cuts)
    module = scenes if stored.source == "video" else speech
    retuned = module.rethreshold(stored, threshold)
    where = write(out, retuned)
    return Produced(
        video_id=stored.video_id, component="boundaries.retune",
        artifacts={"cuts": where},
        stats={**retuned.stats, "threshold": threshold,
               "was": stored.params.get("threshold",
                                        stored.params.get("silence_s"))},
    )


def _cuts_for(cuts: Optional[str | Path], policy: str) -> Optional[Cuts]:
    """The cuts this policy needs, read from a file."""
    needs = POLICIES[policy]
    if needs is None:
        return None
    # A missing cuts file names the policy and the pass that writes it.
    producer = "scenes" if needs == "video" else "speech"
    if cuts is None:
        raise MissingArtifact(
            f"policy {policy!r} needs cuts from boundaries.{producer}; "
            f"pass cuts=... pointing at a cuts.json -- run `evidence` first")
    if not Path(cuts).exists():
        raise MissingArtifact(
            f"policy {policy!r} needs cuts from boundaries.{producer}; "
            f"{cuts} does not exist -- run `evidence` first")
    return load_cuts(cuts)


def timeline(media: Media, policy: str = "uniform",
             cuts: Optional[Cuts] = None,
             chunk_s: float = 20.0, min_s: float = 5.0,
             max_s: Optional[float] = None) -> Timeline:
    """Decide the grid from documents in hand. Reads and writes nothing.

        timeline(media)                              # uniform
        timeline(media, "scene", cuts=detect("scene", media=media))
        timeline(media, "vad", cuts=detect("vad", transcript=raw))
    """
    if policy not in POLICIES:
        raise UnknownOption(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")

    # A floor above the ceiling is refused.
    ceiling = chunk_s if max_s is None else max_s
    if ceiling and min_s > ceiling:
        raise Refused(
            f"--min-chunk {min_s:g} is larger than the ceiling {ceiling:g} "
            f"({'--max-chunk' if max_s is not None else '--chunk-duration, '
               'which --max-chunk defaults to'}). Raise the ceiling or lower "
            "the floor.")

    if media.duration_s is None:
        raise Refused(f"{media.video_id}: the container reports no duration, "
                         "so no grid can be derived from it")
    needs = POLICIES[policy]
    if needs == "video" and not media.has_video:
        raise Refused(f"policy {policy!r} is found in the picture, and "
                         f"{media.video_id} has no video stream")
    if needs == "audio" and not media.has_audio:
        raise Refused(f"policy {policy!r} is derived from the soundtrack, and "
                         f"{media.video_id} has no audio stream")

    if needs is None:
        offered, derived_from, params = None, "grid", {}
    else:
        if cuts is None:
            producer = "scenes" if needs == "video" else "speech"
            raise Refused(
                f"policy {policy!r} needs cuts from boundaries.{producer}; "
                f"pass cuts=detect({policy!r}, ...)")
        expected = "content" if policy == "scene" else policy
        if cuts.detector != expected:
            raise Refused(
                f"those are {cuts.detector!r} cuts, but the policy is "
                f"{policy!r}. Re-run the evidence pass for this policy.")
        offered, derived_from, params = list(cuts.cuts), needs, cuts.params

    return build(media.video_id, policy, media.duration_s, cuts=offered,
                 derived_from=derived_from, chunk_s=chunk_s,
                 min_s=min_s, max_s=max_s, params=params)


def boundaries(media: str | Path, out: str | Path, policy: str = "uniform",
               cuts: Optional[str | Path] = None,
               chunk_s: float = 20.0, min_s: float = 5.0,
               max_s: Optional[float] = None) -> Produced:
    """Decide the grid and write it. `timeline` plus a read at each end."""
    if policy not in POLICIES:
        raise UnknownOption(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")

    described = read(media, Media)
    evidence_doc = _cuts_for(cuts, policy)
    video_id = same_video(media=described, cuts=evidence_doc)

    with logs.timed("boundaries", video_id) as done:
        grid = timeline(described, policy, cuts=evidence_doc, chunk_s=chunk_s,
                        min_s=min_s, max_s=max_s)
        where = write(out, grid)
        done(policy=policy, chunks=len(grid), fingerprint=grid.fingerprint())
    spans = [e - s for s, e in grid.spans]
    return Produced(
        video_id=video_id, component="boundaries",
        artifacts={"timeline": where},
        stats={"policy": policy, "derived_from": grid.derived_from,
               "chunks": len(grid), "fingerprint": grid.fingerprint(),
               "duration_s": round(grid.duration_s, 3),
               "shortest_s": round(min(spans), 3) if spans else 0.0,
               "longest_s": round(max(spans), 3) if spans else 0.0,
               "cuts_offered": 0 if evidence_doc is None
                               else len(evidence_doc.cuts)},
    )


def main(argv: Optional[list[str]] = None) -> int:
    from falconvar.shared.config import env
    env.load()        # an entry point reads .env; the library never does
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Find boundary evidence, and decide the chunk grid.")
    ap.add_argument("media", help="path to media.json")
    ap.add_argument("out", help="where to write timeline.json (or cuts.json "
                                "with --evidence / --retune)")
    ap.add_argument("--cuts", default=None,
                    help="path to cuts.json, for a policy that needs evidence")
    ap.add_argument("--raw-transcript", default=None, dest="raw_transcript",
                    help="path to transcript.raw.json, for a speech policy")
    ap.add_argument("--policy", default="uniform", choices=sorted(POLICIES))
    ap.add_argument("--evidence", action="store_true",
                    help="run only the precursor this policy needs")
    ap.add_argument("--chunk-duration", type=float, default=20.0, dest="chunk_s",
                    help="uniform: the length; otherwise the maximum (default 20)")
    ap.add_argument("--min-chunk", type=float, default=5.0, dest="min_s")
    ap.add_argument("--max-chunk", type=float, default=None, dest="max_s",
                    help="defaults to --chunk-duration")
    ap.add_argument("--stride", type=int, default=scenes.DEFAULT_STRIDE,
                    help="scene: score every Nth frame (default 5). Higher is "
                         "cheaper and needs a HIGHER --threshold")
    ap.add_argument("--threshold", type=float, default=scenes.DEFAULT_THRESHOLD,
                    help="scene: content difference that opens a scene")
    ap.add_argument("--detect-width", type=int, default=scenes.DETECT_WIDTH,
                    dest="detect_width",
                    help="scene: width the frame is scaled to for detection "
                         "(default 320). A cut is a global property of the "
                         "frame, so full resolution buys only time")
    ap.add_argument("--silence", type=float, default=speech.DEFAULT_SILENCE_S,
                    help="vad: a gap this long or longer is a boundary")
    ap.add_argument("--retune", type=float, default=None, metavar="VALUE",
                    help="re-threshold the cached scores. Runs no model")
    ap.add_argument("--calibrate", action="store_true",
                    help="what every threshold would cost here. Reads the "
                         "cached scores; runs nothing")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        if args.calibrate:
            if args.cuts is None:
                print("error: --calibrate reads a cuts.json; pass --cuts")
                return 1
            cuts = load_cuts(args.cuts)
            rows = scenes.sweep(cuts, [5, 10, 15, 20, 27, 35, 45, 60, 80]
                                if cuts.source == "video"
                                else [0.25, 0.5, 0.65, 1.0, 1.5, 2.0, 3.0])
            current = cuts.params.get("threshold", cuts.params.get("silence_s"))
            print(f"{cuts.video_id}   {cuts.detector}   "
                  f"{len(cuts.scores['values'])} scored   currently {current}")
            print(f"  {'value':>10} {'cuts':>6} {'rate':>8} {'median gap':>12}")
            for r in rows:
                gap = "--" if r["median_gap_s"] is None else f"{r['median_gap_s']:g}s"
                mark = "  <- current" if r["threshold"] == current else ""
                print(f"  {r['threshold']:>10g} {r['cuts']:>6} {r['rate']:>8.1%} "
                      f"{gap:>12}{mark}")
            print("\n  A threshold is a property of the footage, not a default. "
                  "This reports; it does not choose.")
            return 0

        if args.retune is not None:
            if args.cuts is None:
                print("error: --retune re-thresholds a cuts.json; pass --cuts")
                return 1
            produced = retune(args.cuts, args.out, args.retune)
        elif args.evidence:
            produced = evidence(args.out, args.policy, media=args.media,
                                raw_transcript=args.raw_transcript,
                                stride=args.stride, threshold=args.threshold,
                                detect_width=args.detect_width,
                                silence_s=args.silence)
            if "evidence" in produced.skipped and not args.json:
                print(f"{produced.video_id}: policy {args.policy!r} needs no "
                      "evidence -- it is arithmetic over the container "
                      "duration")
                return 0
        else:
            produced = boundaries(args.media, args.out, args.policy,
                                  args.cuts, args.chunk_s, args.min_s,
                                  args.max_s)
    except (KeyError, ValueError, FileNotFoundError, scenes.NoPicture) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    if produced.component == "boundaries":
        print(f"{produced.video_id}   policy={s['policy']}  "
              f"derived_from={s['derived_from']}")
        print(f"  chunks       {s['chunks']}   over {s['duration_s']:g}s")
        print(f"  span         min {s['shortest_s']:g}s   max {s['longest_s']:g}s")
        if s["cuts_offered"]:
            print(f"  cuts offered {s['cuts_offered']}")
        print(f"  fingerprint  {s['fingerprint']}")
        print(f"\ntimeline -> {produced.artifacts['timeline']}")
    else:
        print(f"{produced.video_id}   {produced.component}")
        for key in ("frames_read", "frames_scored", "speech_spans", "elapsed_s",
                    "ms_per_frame", "threshold", "silence_s", "stride",
                    "cuts_found", "was"):
            if key in s and s[key] is not None:
                print(f"  {key:<14} {s[key]}")
        print(f"\ncuts -> {produced.artifacts['cuts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
