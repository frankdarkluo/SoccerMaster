"""Encode data/GSR-2025/images/SNGS-*/img1 → videos/*.mp4 + annotation CSV template."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.utils.video import encode_video, get_video_codec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = ROOT / "data" / "GSR-2025" / "images"
DEFAULT_VIDEOS = ROOT / "data" / "GSR-2025" / "videos"
DEFAULT_CSV = ROOT / "data" / "GSR-2025" / "标注模版_SNGS001-200.csv"
MIN_BYTES = 1_000_000  # ponytail: skip gate for truncated kills; raise if real clips shrink


def _fps_map(images_root: Path) -> dict[str, float]:
    """Optional sequences_info.json → {SNGS-001: fps, ...}."""
    out: dict[str, float] = {}
    candidates = list(images_root.rglob("sequences_info.json"))
    top = images_root.parent / "sequences_info.json"
    if top.is_file():
        candidates.append(top)
    for path in candidates:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        for entries in data.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                name, fps = entry.get("name"), entry.get("fps")
                if name and fps:
                    out[str(name)] = float(fps)
    return out


def _encode_one(name: str, img1: Path, out: Path, fps: float) -> str:
    print(f"ENCODE {name} fps={fps} → {out}", flush=True)
    encode_video(img1, out, fps=fps)
    codec, _ = get_video_codec(out)
    if codec != "h264":
        raise RuntimeError(f"{out} codec={codec}, expected h264")
    return name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    p.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=200)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if not args.images.is_dir():
        raise SystemExit(f"images root missing: {args.images} (run Mac→Ads rsync first)")

    fps_by_name = _fps_map(args.images)
    args.videos.mkdir(parents=True, exist_ok=True)
    present: list[str] = []
    jobs: list[tuple[str, Path, Path, float]] = []
    skipped = missing = 0

    for i in range(args.start, args.end + 1):
        name = f"SNGS-{i:03d}"
        img1 = args.images / name / "img1"
        if not img1.is_dir() or not any(img1.glob("*.jpg")):
            missing += 1
            continue
        present.append(name)
        out = args.videos / f"{name}.mp4"
        if out.is_file() and out.stat().st_size >= MIN_BYTES and not args.force:
            skipped += 1
            print(f"SKIP {name} (exists)", flush=True)
            continue
        if out.is_file() and out.stat().st_size < MIN_BYTES:
            out.unlink()
        jobs.append((name, img1, out, fps_by_name.get(name, args.fps)))

    encoded = 0
    fails: list[tuple[str, str]] = []
    workers = max(1, min(args.workers, len(jobs))) if jobs else 1
    print(f"queue={len(jobs)} workers={workers} skipped={skipped}", flush=True)
    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_encode_one, name, img1, out, fps): name
                for name, img1, out, fps in jobs
            }
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    fut.result()
                    encoded += 1
                    print(f"OK {name} ({encoded}/{len(jobs)})", flush=True)
                except Exception as e:  # noqa: BLE001
                    fails.append((name, str(e)))
                    print(f"FAIL {name}: {e}", flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id,tactics,时间范围,判断依据"]
    lines.extend(f"{sid},,," for sid in present)
    args.csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"done encoded={encoded} skipped={skipped} missing={missing} fails={len(fails)} "
        f"csv_rows={len(present)} → {args.csv}",
        flush=True,
    )
    if fails:
        raise SystemExit(f"encode failed: {[n for n, _ in fails]}")


if __name__ == "__main__":
    main()
