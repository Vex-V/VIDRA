"""Rebuild a frame store from a manifest and the video.

    video.recreate(manifest="d/manifest.json", video="shop.mp4", out="rebuilt",
                   verify="d/store")

A manifest names every frame a run kept by `index` (its position in a decode of
every frame) and records the source it came from. This decodes the video again,
writes those frames as the store did, and with `verify=` byte-compares them
against an existing store; a rebuilt store matches the original byte for byte.

Decodes with `av` and `cv2`, and imports nothing from the pipeline but the
error base: the manifest alone is enough. The file can be handed over with a
manifest and a video.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from vidra.shared.reporting.errors import Refused, VidraError

#: Degrees -> the cv2 rotation, for a source that records one.
_ROTATIONS = {90: "ROTATE_90_CLOCKWISE", 180: "ROTATE_180",
              270: "ROTATE_90_COUNTERCLOCKWISE"}


class Mismatch(VidraError, ValueError):
    """The video is not the one this manifest was built from."""


class MissingVideo(VidraError, FileNotFoundError):
    """No file at the path given as the video."""


def wanted_frames(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """index -> record, for every frame any sampler kept (each once)."""
    out: dict[int, dict[str, Any]] = {}
    for chunk in manifest.get("chunks", []):
        for block in chunk.get("samplers", {}).values():
            for record in block.get("frames", []):
                out.setdefault(int(record["index"]), record)
    return out


def check_source(manifest: dict[str, Any], path: Path) -> list[str]:
    """Everything about this file that disagrees with the manifest."""
    import av

    source = manifest.get("source", {})
    problems: list[str] = []
    with av.open(str(path)) as container:
        stream = next(iter(container.streams.video), None)
        if stream is None:
            return [f"{path} has no video stream"]
        checks = [
            ("width", stream.codec_context.width, source.get("width")),
            ("height", stream.codec_context.height, source.get("height")),
            ("time_base", str(stream.time_base), source.get("time_base")),
            ("frames", stream.frames or None, source.get("frames")),
        ]
        rate = float(stream.guessed_rate or stream.average_rate or 0) or None
        checks.append(("rate", rate, source.get("rate")))
    for name, found, expected in checks:
        if expected is not None and found is not None and found != expected:
            problems.append(f"{name}: video has {found!r}, manifest says {expected!r}")
    return problems


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare(rebuilt: Path, original: Path, names: set[str]) -> dict[str, Any]:
    """Byte-compare the rebuilt frames against the original store, over the
    frames this manifest names -- a store keeps earlier runs' frames too."""
    mine = {p.name: p for p in rebuilt.glob("*.jpg") if p.name in names}
    theirs = {p.name: p for p in Path(original).glob("*.jpg")}
    shared = sorted(set(mine) & set(theirs))
    identical = [n for n in shared if _digest(mine[n]) == _digest(theirs[n])]
    differing = sorted(set(shared) - set(identical))
    absent = sorted(set(mine) - set(theirs))
    return {
        "identical": not differing and not absent,
        "compared": len(shared),
        "matching": len(identical),
        # A frame both have whose bytes differ.
        "differing": differing,
        # A frame the manifest names that the original store lacks.
        "missing_from_original": absent,
        # Frames the store holds that this manifest does not name: not a
        # failure, a store keeps earlier runs' frames.
        "unnamed_in_original": len(set(theirs) - set(mine)),
    }


def recreate(*, manifest: str | Path, video: str | Path, out: str | Path,
             verify: Optional[str | Path] = None, force: bool = False,
             quality: int = 95) -> dict[str, Any]:
    """Decode `video` and write every frame `manifest` names into `out`, as the
    frame store did. Returns what was done:

        named, written, missing   frames the manifest names, those written, and
                                  any it names that the decode never reached
        problems                  where the video disagrees with the manifest
        verified                  with `verify=` (an existing store): whether
                                  every named frame is byte-identical, and what
                                  was compared

    A video whose width, height, time base, frame count or rate disagrees with
    the manifest is refused (`Mismatch`); `force=True` rebuilds anyway and
    lists the disagreements. `quality` must be the JPEG quality the store was
    written at (95) for a rebuild to be byte-identical.
    """
    import av
    import cv2

    where = Path(manifest)
    document = json.loads(where.read_text(encoding="utf-8"))
    path = Path(video)
    if not path.is_file():
        recorded = document.get("source", {}).get("path")
        raise MissingVideo(f"no video at {path}"
                           + (f"; the manifest was built from {recorded}" if recorded else ""))
    problems = check_source(document, path)
    if problems and not force:
        raise Mismatch(f"{path} is not the video {where.name} was built from: "
                       + "; ".join(problems) + ". force=True rebuilds anyway.")
    records = wanted_frames(document)
    if not records:
        raise Refused(f"{where} names no frames")
    if not 1 <= quality <= 100:
        raise Refused(f"quality is a JPEG quality, 1 to 100; got {quality}")

    rotation = _ROTATIONS.get(int(document.get("source", {}).get("rotation", 0) or 0))
    rotate = getattr(cv2, rotation) if rotation else None
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)

    written = 0
    index = 0
    highest = max(records)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(video=0):
            if index in records:
                image = frame.to_ndarray(format="bgr24")
                if rotate is not None:
                    image = cv2.rotate(image, rotate)
                cv2.imwrite(str(folder / f"{index:07d}.jpg"), image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                written += 1
            index += 1
            if index > highest:
                break

    done: dict[str, Any] = {
        "named": len(records), "written": written,
        "missing": sorted(set(records) - {int(p.stem) for p in folder.glob("*.jpg")}),
        "problems": problems, "out": str(folder), "video": str(path)}
    if verify is not None:
        done["verified"] = _compare(folder, Path(verify),
                                    {f"{i:07d}.jpg" for i in records})
    return done


__all__ = ["Mismatch", "MissingVideo", "recreate"]
