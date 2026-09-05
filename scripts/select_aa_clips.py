#!/usr/bin/env python3
"""Select a non-overlapping, broadcast-usable subset of SoccerNet-AA clips.

Why this exists
---------------
`scripts/screen_soccernet_aa.py` screens clips one at a time and audits overlap
using `head_dhash` / `tail_dhash` only -- 25 frames at each end. Because
SoccerNet-AA is cut as a sliding window with a 5-second stride, consecutive
30-second clips share up to 25 seconds, and that detector cannot see it. Measured
on a 218-clip pool: 130 overlapping pairs, 218 clips collapsing to 125 distinct
contents (43% redundant). Example: valid/AA-271 and valid/AA-1166 share 19.8s at
a 10s offset, and both carry the same cutback annotation.

This script hashes whole clips, reconstructs each clip's position on its source
timeline, and then solves for the largest set of clips that do not overlap at all
-- the real ceiling on independent 30-second samples.

It also gates on watchability, because a clip is only useful to a human annotator
or a commercial VLM if enough of it is the tactical (main wide) camera:

  * reject a clip when non-broadcast footage (replay / close-up / crowd / graphics)
    exceeds --max-bad-seconds (default 15s)
  * require at least --min-run-seconds (default 10s) of *continuous* broadcast view

Pipeline (each stage is resumable; re-running skips clips already in the cache)
------------------------------------------------------------------------------
  scan     decode every clip once -> per-second camera features + whole-clip dHash
  gate     apply the watchability rules -> keep / reject with a reason
  timeline group clips into source timelines and recover each clip's start offset
  select   greedy interval scheduling per timeline -> maximum non-overlapping set

Usage
-----
    # stage 1: decode (slow, ~1-2s per clip; safe to interrupt and resume)
    python scripts/select_aa_clips.py scan --splits train valid test

    # stages 2-4: cheap, re-run freely while tuning thresholds
    python scripts/select_aa_clips.py select \
        --max-bad-seconds 15 --min-run-seconds 10 \
        --out-csv outputs/soccernet_aa_screening/aa_selected_clips.csv \
        --out-report outputs/soccernet_aa_screening/aa_selection_report.json

Camera-view classification
--------------------------
Deliberately not a VLM. The gate only needs "usable broadcast" vs "not", so the
script scores each second on cheap geometric features (grass fraction, player-blob
count and size, field-line energy) rather than trying to name the failure mode.
Telling a replay from a close-up would need a model; telling either from a wide
tactical shot does not.

The one case this misses is a *wide-angle* replay, which looks like live play.
`slowmo_score` and `logo_cut` are recorded per clip as a proxy so a later pass can
filter those, but they are NOT used in the gate -- they need calibration against
labelled replays first. See "Calibration" below.

Calibration
-----------
Thresholds in CAMERA_THRESHOLDS were set from a small manual read of AA-1/AA-2 and
have not been validated at scale. Before trusting the output:

    python scripts/select_aa_clips.py scan --splits valid --limit 60
    python scripts/select_aa_clips.py gate --emit-review-csv /tmp/gate_review.csv

then eyeball 60 clips against the predicted per-second labels and adjust. The
report prints the keep rate so a wildly wrong threshold is obvious.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = "SoccerNet-AA"
DEFAULT_DATA_ROOT = ROOT / "data" / "SoccerNet-AA"
DEFAULT_CACHE = ROOT / "outputs" / "soccernet_aa_screening" / "clip_scan.jsonl"
SPLITS = ("train", "valid", "test", "challenge")

CLIP_SECONDS = 30.0
SAMPLE_HZ = 5          # frames sampled per second, for both hashing and features
MIN_SHARED_HASHES = 3  # exact-hash collisions needed before aligning a pair
MIN_MATCHED = 10       # matched samples needed to call two clips overlapping

# Grass hue band in HSV. OpenCV hue is 0..179.
GRASS_LO = np.array([30, 40, 40], dtype=np.uint8)
GRASS_HI = np.array([90, 255, 255], dtype=np.uint8)

# Calibrated on 75 frames sampled from valid/AA-{1,5,63,271,1166}. On that sample
# frames with grass_ratio < 0.30 all had zero interior blobs, and frames with
# grass_ratio >= 0.45 had blobs p10=2, line_energy p10=0.005, max_blob p90=0.046.
# Small sample -- re-check with `gate --emit-review-csv` before trusting at scale.
CAMERA_THRESHOLDS = {
    # A broadcast tactical frame is mostly pitch, with several small player blobs.
    "min_grass": 0.35,
    "min_blobs": 2,
    "max_blob_area": 0.12,   # largest interior non-grass blob, fraction of frame
    "min_line_energy": 0.002,
}


def _number(path: Path) -> int:
    match = re.fullmatch(r"AA-(\d+)\.mp4", path.name)
    if not match:
        raise ValueError(f"unexpected AA filename: {path}")
    return int(match.group(1))


def _aa_videos(split_dir: Path) -> list[Path]:
    found = list(split_dir.glob("AA-*.mp4"))
    for sub in ("clean", "dup"):
        found.extend((split_dir / sub).glob("AA-*.mp4"))
    by_stem: dict[str, Path] = {}
    for path in found:
        previous = by_stem.get(path.stem)
        if previous is not None and previous.resolve() != path.resolve():
            raise ValueError(f"duplicate {path.name} under {split_dir}")
        by_stem[path.stem] = path
    return sorted(by_stem.values(), key=_number)


def iter_videos(data_root: Path, splits: Iterable[str]) -> list[tuple[str, Path]]:
    found = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing split directory: {split_dir}")
        for path in _aa_videos(split_dir):
            found.append((f"{split}/{path.stem}", path))
    return found


# --------------------------------------------------------------------------- scan


def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")


def _frame_features(frame: np.ndarray) -> dict[str, float]:
    """Cheap geometric cues that separate a wide tactical shot from everything else."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, GRASS_LO, GRASS_HI)
    grass_ratio = float(grass.mean() / 255.0)

    # Players and other clutter are the non-grass blobs sitting inside the pitch.
    non_grass = cv2.morphologyEx(
        cv2.bitwise_not(grass), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(non_grass, connectivity=8)
    height, width = frame.shape[:2]
    frame_area = float(height * width)
    # The stands are one huge non-grass region touching the top border. Including it
    # makes every wide shot look like a close-up, so score only interior blobs.
    interior = []
    for i in range(1, count):
        top = stats[i, cv2.CC_STAT_TOP]
        left = stats[i, cv2.CC_STAT_LEFT]
        right = left + stats[i, cv2.CC_STAT_WIDTH]
        if top == 0 or left == 0 or right >= width:
            continue
        interior.append(stats[i, cv2.CC_STAT_AREA] / frame_area)
    interior.sort(reverse=True)
    # A player at broadcast distance is a small blob; a close-up is one huge blob.
    blobs = sum(1 for a in interior if 0.0004 <= a <= 0.02)
    max_blob = interior[0] if interior else 0.0

    # Pitch markings give strong near-horizontal/vertical gradients on green.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    line_energy = float(cv2.bitwise_and(edges, edges, mask=grass).mean() / 255.0)

    return {
        "grass_ratio": grass_ratio,
        "blobs": float(blobs),
        "max_blob": float(max_blob),
        "line_energy": line_energy,
    }


def _is_broadcast(feat: dict[str, float]) -> bool:
    t = CAMERA_THRESHOLDS
    return (
        feat["grass_ratio"] >= t["min_grass"]
        and feat["blobs"] >= t["min_blobs"]
        and feat["max_blob"] <= t["max_blob_area"]
        and feat["line_energy"] >= t["min_line_energy"]
    )


def scan_clip(path: Path) -> dict[str, Any]:
    """One decode pass: whole-clip hashes plus per-second camera verdicts."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / SAMPLE_HZ))

    hashes: list[str] = []
    per_sample: list[bool] = []
    diffs: list[float] = []
    cuts: list[float] = []
    saturation_spikes = 0

    previous_gray = None
    previous_small = None
    index = 0
    last_cut = -10.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            time_s = index / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hashes.append(f"{_dhash(gray):016x}")
            per_sample.append(_is_broadcast(_frame_features(frame)))

            small = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
            if previous_small is not None:
                delta = float(cv2.absdiff(previous_small, small).mean() / 255.0)
                diffs.append(delta)
                if delta >= 0.18 and time_s - last_cut >= 0.5:
                    cuts.append(round(time_s, 3))
                    last_cut = time_s
                    # Replay wipes are usually saturated branded transitions.
                    sat = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
                    if sat > 110.0:
                        saturation_spikes += 1
            previous_small = small
            previous_gray = gray
        index += 1
    capture.release()

    if not hashes:
        raise RuntimeError(f"decoded zero frames: {path}")

    # Slow motion shows up as unusually small frame-to-frame change during play.
    motion = float(np.median(diffs)) if diffs else 0.0
    return {
        "clip_id": f"{path.parent.name}/{path.stem}",
        "split": path.parent.name,
        "fps": round(fps, 3),
        "duration_s": round(len(hashes) / SAMPLE_HZ, 2),
        "sample_hz": SAMPLE_HZ,
        "dhash": hashes,
        "broadcast_mask": "".join("1" if v else "0" for v in per_sample),
        "cuts": cuts,
        "slowmo_score": round(motion, 5),
        "logo_cut": saturation_spikes,
    }


def run_scan(args: argparse.Namespace) -> None:
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = {row["clip_id"] for row in read_cache(cache)}
    videos = iter_videos(Path(args.data_root), args.splits)
    todo = [(cid, p) for cid, p in videos if cid not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(videos)} clips, {len(done)} cached, {len(todo)} to scan", flush=True)

    with cache.open("a", encoding="utf-8") as handle:
        for n, (clip_id, path) in enumerate(todo, 1):
            try:
                row = scan_clip(path)
            except RuntimeError as exc:
                print(f"  skip {clip_id}: {exc}", file=sys.stderr)
                continue
            row["clip_id"] = clip_id
            row["split"] = clip_id.split("/", 1)[0]
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    print(f"wrote {cache}", flush=True)


def read_cache(cache: Path) -> list[dict[str, Any]]:
    if not cache.exists():
        return []
    return [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- gate


def gate_clip(row: dict[str, Any], max_bad_s: float, min_run_s: float) -> dict[str, Any]:
    mask = row["broadcast_mask"]
    hz = row.get("sample_hz", SAMPLE_HZ)
    bad_s = mask.count("0") / hz

    longest = 0
    current = 0
    for ch in mask:
        current = current + 1 if ch == "1" else 0
        longest = max(longest, current)
    run_s = longest / hz

    reasons = []
    if bad_s > max_bad_s:
        reasons.append(f"non_broadcast_{bad_s:.1f}s>{max_bad_s:g}s")
    if run_s < min_run_s:
        reasons.append(f"longest_run_{run_s:.1f}s<{min_run_s:g}s")

    return {
        "clip_id": row["clip_id"],
        "split": row["split"],
        "broadcast_s": round(mask.count("1") / hz, 1),
        "non_broadcast_s": round(bad_s, 1),
        "longest_run_s": round(run_s, 1),
        "cuts": len(row.get("cuts", [])),
        "slowmo_score": row.get("slowmo_score", 0.0),
        "logo_cut": row.get("logo_cut", 0),
        "keep": not reasons,
        "reject_reason": ";".join(reasons),
    }


# ----------------------------------------------------------------------- timeline




def exact_five_second_alignment(
    source: list[int], target: list[int], *, sample_hz: int = SAMPLE_HZ
) -> dict[str, float | int] | None:
    """Return the exact local timestamp shift from source to target."""
    stride = 5 * sample_hz
    best_exact, best_offset = -1, 0
    lower = -len(source) + MIN_MATCHED
    upper = len(target) - MIN_MATCHED
    for offset in range(lower, upper + 1):
        if offset % stride:
            continue
        exact = sum(
            source[index] == target[index + offset]
            for index in range(len(source))
            if 0 <= index + offset < len(target)
        )
        if exact > best_exact:
            best_exact, best_offset = exact, offset
    if best_exact < MIN_MATCHED:
        return None
    return {
        "shift_s": best_offset / sample_hz,
        "exact_samples": best_exact,
    }


def find_overlaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build exact dHash edges between identical or adjacent 5-second clips."""
    values = {r["clip_id"]: [int(h, 16) for h in r["dhash"]] for r in rows}

    index: dict[str, set[str]] = collections.defaultdict(set)
    for clip_id, row in ((r["clip_id"], r) for r in rows):
        for h in row["dhash"]:
            index[h].add(clip_id)

    shared: collections.Counter = collections.Counter()
    for clips in index.values():
        # A hash shared by very many clips is a blank or crowd frame, not evidence.
        if 1 < len(clips) <= 30:
            ordered = sorted(clips)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    shared[(ordered[i], ordered[j])] += 1

    hz = rows[0].get("sample_hz", SAMPLE_HZ)
    overlaps = []
    for (a, b), n in shared.items():
        if n < MIN_SHARED_HASHES:
            continue
        alignment = exact_five_second_alignment(values[a], values[b], sample_hz=hz)
        if alignment is None or abs(alignment["shift_s"]) not in {0.0, 5.0}:
            continue
        overlaps.append({
            "a": a,
            "b": b,
            "overlap_s": round(alignment["exact_samples"] / hz, 1),
            "offset_s": alignment["shift_s"],
        })
    overlaps.sort(key=lambda d: -d["overlap_s"])
    return overlaps


def build_timelines(
    clip_ids: list[str], overlaps: list[dict[str, Any]]
) -> dict[str, tuple[str, float]]:
    """Group clips into source timelines and place each one on its timeline.

    `offset_s` migrates a local timestamp onto b, so b's physical
    start is `offset_s` before a. A spanning walk turns those pairwise start
    deltas into absolute positions per component.
    """
    adj: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for o in overlaps:
        adj[o["a"]].append((o["b"], -o["offset_s"]))
        adj[o["b"]].append((o["a"], o["offset_s"]))

    placed: dict[str, tuple[str, float]] = {}
    for n, root in enumerate(clip_ids):
        if root in placed:
            continue
        timeline = f"T{n:05d}"
        placed[root] = (timeline, 0.0)
        stack = [root]
        while stack:
            node = stack.pop()
            _, base = placed[node]
            for neighbour, offset in adj[node]:
                if neighbour in placed:
                    if abs(placed[neighbour][1] - (base + offset)) > 0.21:
                        raise ValueError(
                            f"inconsistent timeline offsets: {node}, {neighbour}"
                        )
                    continue
                placed[neighbour] = (timeline, base + offset)
                stack.append(neighbour)
    return placed


def assign_timeline_aa_ids(
    placed: dict[str, tuple[str, float]], *, stride_s: float = 5.0
) -> dict[str, str]:
    """Map old clip_id -> {split}/AA-n along 5s timelines, concatenated in T##### order."""
    by_timeline: dict[str, list[tuple[float, str]]] = collections.defaultdict(list)
    split = None
    for clip_id, (timeline, start) in placed.items():
        this_split = clip_id.split("/", 1)[0]
        if split is None:
            split = this_split
        elif this_split != split:
            raise ValueError(f"mixed splits in placed: {split} and {this_split}")
        by_timeline[timeline].append((start, clip_id))

    mapping: dict[str, str] = {}
    n = 0
    for timeline in sorted(by_timeline):
        seq = sorted(by_timeline[timeline])
        for i, (start, clip_id) in enumerate(seq):
            if i and round(start - seq[i - 1][0], 1) != round(stride_s, 1):
                raise ValueError(
                    f"{clip_id} is not on a 5s chain: "
                    f"{seq[i - 1][1]} start={seq[i - 1][0]} -> {clip_id} start={start}"
                )
            n += 1
            mapping[clip_id] = f"{split}/AA-{n}"
    return mapping


# ------------------------------------------------------------------------ select


def select_non_overlapping(
    candidates: list[dict[str, Any]], clip_seconds: float
) -> list[dict[str, Any]]:
    """Select the earliest fixed-length interval, then walk forward.

    Equal-length earliest-finish: take the earliest clip, skip anything that
    overlaps it, repeat. Maximum-cardinality; phase is the source-timeline head.
    """
    by_timeline: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in candidates:
        by_timeline[row["timeline"]].append(row)

    chosen = []
    for timeline, rows in sorted(by_timeline.items()):
        rows.sort(key=lambda r: (r["start_s"], r["clip_id"]))
        cursor = float("-inf")
        for row in rows:
            if row["start_s"] >= cursor:
                chosen.append(row)
                cursor = row["start_s"] + clip_seconds
    return chosen


def run_select(args: argparse.Namespace) -> None:
    rows = read_cache(Path(args.cache))
    if not rows:
        raise SystemExit(f"empty cache: {args.cache} -- run `scan` first")

    gates = {r["clip_id"]: gate_clip(r, args.max_bad_seconds, args.min_run_seconds)
             for r in rows}
    kept = [g for g in gates.values() if g["keep"]]

    overlaps = find_overlaps(rows)
    placed = build_timelines(sorted(gates), overlaps)

    candidates = []
    for gate in kept:
        timeline, start = placed[gate["clip_id"]]
        candidates.append({**gate, "timeline": timeline, "start_s": round(start, 1)})

    selected = select_non_overlapping(candidates, args.clip_seconds)

    # The ceiling ignores the quality gate: how many independent samples exist at all.
    all_placed = [
        {"clip_id": cid, "timeline": t, "start_s": s} for cid, (t, s) in placed.items()
    ]
    ceiling = len(select_non_overlapping(all_placed, args.clip_seconds))

    report = {
        "dataset": DATASET,
        "clips_scanned": len(rows),
        "by_split": dict(collections.Counter(r["split"] for r in rows)),
        "gate": {
            "max_bad_seconds": args.max_bad_seconds,
            "min_run_seconds": args.min_run_seconds,
            "kept": len(kept),
            "rejected": len(rows) - len(kept),
            "reject_reasons": dict(
                collections.Counter(
                    reason
                    for g in gates.values()
                    if not g["keep"]
                    for reason in g["reject_reason"].split(";")
                )
            ),
        },
        "overlap": {
            "pairs": len(overlaps),
            "timelines": len({t for t, _ in placed.values()}),
        },
        "selected": len(selected),
        "independent_ceiling_ignoring_gate": ceiling,
        "selected_by_split": dict(collections.Counter(r["split"] for r in selected)),
        "limitations": [
            "Camera classification is geometric, not a VLM; a wide-angle replay is "
            "scored as usable play.",
            "CAMERA_THRESHOLDS are uncalibrated at scale -- see the module docstring.",
            "slowmo_score and logo_cut are recorded but not used in the gate.",
        ],
    }

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["clip_id", "split", "timeline", "start_s", "broadcast_s",
              "non_broadcast_s", "longest_run_s", "cuts", "slowmo_score", "logo_cut"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(selected, key=lambda r: (r["timeline"], r["start_s"])))

    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nwrote {out_csv}\nwrote {args.out_report}")


def run_gate(args: argparse.Namespace) -> None:
    rows = read_cache(Path(args.cache))
    if not rows:
        raise SystemExit(f"empty cache: {args.cache} -- run `scan` first")
    gates = [gate_clip(r, args.max_bad_seconds, args.min_run_seconds) for r in rows]
    kept = sum(1 for g in gates if g["keep"])
    print(f"{kept}/{len(gates)} kept ({kept / len(gates):.0%})")
    print(json.dumps(
        dict(collections.Counter(
            reason for g in gates if not g["keep"]
            for reason in g["reject_reason"].split(";")
        )), ensure_ascii=False, indent=2))
    if args.emit_review_csv:
        path = Path(args.emit_review_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(gates[0]))
            writer.writeheader()
            writer.writerows(gates)
        print(f"wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="decode clips into the feature cache")
    scan.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    scan.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    scan.add_argument("--limit", type=int, default=0, help="scan at most N new clips")
    scan.set_defaults(func=run_scan)

    gate = sub.add_parser("gate", help="apply watchability rules only")
    gate.add_argument("--max-bad-seconds", type=float, default=15.0)
    gate.add_argument("--min-run-seconds", type=float, default=10.0)
    gate.add_argument("--emit-review-csv")
    gate.set_defaults(func=run_gate)

    select = sub.add_parser("select", help="gate, dedup, and pick the final set")
    select.add_argument("--max-bad-seconds", type=float, default=15.0)
    select.add_argument("--min-run-seconds", type=float, default=10.0)
    select.add_argument("--clip-seconds", type=float, default=CLIP_SECONDS)
    select.add_argument(
        "--out-csv",
        default=str(ROOT / "outputs/soccernet_aa_screening/aa_selected_clips.csv"),
    )
    select.add_argument(
        "--out-report",
        default=str(ROOT / "outputs/soccernet_aa_screening/aa_selection_report.json"),
    )
    select.set_defaults(func=run_select)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
