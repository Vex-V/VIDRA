"""STANDALONE: rebuild a frame store from a manifest and the video.

Needs only `av`, `opencv-python` and `numpy`, and imports nothing from the
pipeline. A rebuilt store should match the original byte for byte.

    python -m recovery.recreate data/out/<id>/manifest.json --out rebuilt/
    python -m recovery.recreate <manifest> --verify data/out/<id>/store

Frames are named by `index`, the reader's count over every frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

import av
import cv2

ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class Mismatch(RuntimeError):
    """The video is not the one this manifest was built from."""


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


def recreate(manifest_path: Path, out_dir: Path,
             video: Optional[Path] = None, force: bool = False,
             quality: int = 95) -> dict[str, Any]:
    """Decode the video and write every frame the manifest names."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source", {})
    path = Path(video) if video else Path(source.get("path", ""))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Pass --video to point at a moved copy.")

    problems = check_source(manifest, path)
    if problems and not force:
        raise Mismatch(
            f"{path} does not match this manifest:\n  "
            + "\n  ".join(problems) + "\n--force overrides.")

    records = wanted_frames(manifest)
    if not records:
        raise ValueError(f"{manifest_path} names no frames")

    rotation = int(source.get("rotation", 0) or 0)
    rotate = ROTATIONS.get(rotation)
    out_dir.mkdir(parents=True, exist_ok=True)

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
                target = out_dir / f"{index:07d}.jpg"
                cv2.imwrite(str(target), image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                written += 1
            index += 1
            if index > highest:
                break

    return {"named": len(records), "written": written,
            "missing": sorted(set(records) - {int(p.stem)
                                              for p in out_dir.glob("*.jpg")}),
            "out": str(out_dir), "video": str(path), "problems": problems}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(rebuilt: Path, original: Path,
           names: Optional[set[str]] = None) -> dict[str, Any]:
    """Byte-compare a rebuilt store against the original, over the frames `names`
    lists (what this manifest names).
    """
    mine = {p.name: p for p in rebuilt.glob("*.jpg")}
    if names is not None:
        mine = {n: path for n, path in mine.items() if n in names}
    theirs = {p.name: p for p in original.glob("*.jpg")}
    shared = sorted(set(mine) & set(theirs))
    identical = [n for n in shared if digest(mine[n]) == digest(theirs[n])]
    return {
        "compared": len(shared),
        "identical": identical,
        # A frame both have whose bytes differ.
        "differing": sorted(set(shared) - set(identical)),
        # A frame the manifest names that the original store lacks.
        "missing_from_original": sorted(set(mine) - set(theirs)),
        # Frames the store holds that this manifest does not name: not a failure,
        # a store keeps earlier runs' frames.
        "orphaned_in_original": sorted(set(theirs) - set(mine)),
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild a frame store from a manifest and the video.")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, default=Path("rebuilt"))
    ap.add_argument("--video", type=Path, default=None,
                    help="the source, if it has moved since the manifest")
    ap.add_argument("--verify", type=Path, default=None,
                    help="byte-compare against an existing store")
    ap.add_argument("--force", action="store_true",
                    help="recreate even if the video does not match")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args(argv)

    try:
        result = recreate(args.manifest, args.out, args.video, args.force,
                          args.quality)
    except (Mismatch, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    print(f"{result['video']}")
    print(f"  named        {result['named']}")
    print(f"  written      {result['written']}")
    if result["missing"]:
        print(f"  MISSING      {len(result['missing'])}: {result['missing'][:8]}")
    if result["problems"]:
        print(f"  forced past  {len(result['problems'])} mismatch(es)")
    print(f"  out          {result['out']}")

    if args.verify is not None:
        named = {f"{index:07d}.jpg"
                 for index in wanted_frames(json.loads(
                     args.manifest.read_text(encoding="utf-8")))}
        check = verify(args.out, args.verify, named)
        print()
        if check["differing"] or check["missing_from_original"]:
            print(f"  FAIL: {len(check['identical'])}/{check['compared']} "
                  f"identical, {len(check['differing'])} differ, "
                  f"{len(check['missing_from_original'])} absent from the store")
            return 1
        print(f"  PASS: {len(check['identical'])}/{check['compared']} "
              f"frames byte-identical")
        if check["orphaned_in_original"]:
            print(f"        ({len(check['orphaned_in_original'])} other frames "
                  f"in the store this manifest does not name -- an earlier "
                  f"run's, not a mismatch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
