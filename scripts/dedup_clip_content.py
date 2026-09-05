#!/usr/bin/env python3
"""Find clips that share video content, and pick one representative per content group.

Why this exists
---------------
`outputs/soccernet_aa_screening/inventory.jsonl` stores only `head_dhash` and
`tail_dhash` (25 frames at each end). `overlap_groups.jsonl` was built from those,
so every record it holds is a 25-frame `boundary_overlap`. That detector cannot see
overlap in the middle of a clip.

SoccerNet-AA is cut as a sliding window with a 5-second stride, so consecutive
30-second clips share up to 25 seconds. Measured on a 218-clip pool: 130 overlapping
pairs, 218 clips collapsing to 125 independent contents (43% redundant). Example:
valid/AA-271 and valid/AA-1166 share 19.8s at a 10s offset, and both were annotated
with the same cutback.

This script hashes the whole clip instead, so mid-clip overlap is visible.

Usage
-----
    # scan clip ids listed in a CSV column, write groups + a deduped CSV
    python scripts/dedup_clip_content.py \
        --csv outputs/soccernet_aa_tactics/aa_tacticqa_expansion_pool.csv \
        --id-column id \
        --video-root data/SoccerNet-AA \
        --out-groups outputs/soccernet_aa_screening/content_groups.jsonl \
        --out-csv outputs/soccernet_aa_tactics/pool_deduped.csv

    # or scan explicit paths
    python scripts/dedup_clip_content.py --paths data/SoccerNet-AA/valid/*.mp4 \
        --out-groups /tmp/groups.jsonl

Representative choice: rows earlier in the input CSV win, so sort the CSV by your
own quality order (e.g. double-model-agreement first) before running.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import pathlib
import sys

import cv2

FRAME_STEP = 5      # hash every Nth frame
HAMMING_MAX = 6     # per-frame match threshold
MIN_SHARED = 3      # candidate pairs need this many identical hashes
MIN_MATCHED = 10    # matched samples needed to call a pair overlapping


def dhash_sequence(path: str, step: int = FRAME_STEP) -> list[int]:
    """Return one 64-bit dHash per sampled frame."""
    cap = cv2.VideoCapture(path)
    hashes, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8))
            bits = "".join(
                "1" if g[r, c + 1] > g[r, c] else "0" for r in range(8) for c in range(8)
            )
            hashes.append(int(bits, 2))
        idx += 1
    cap.release()
    return hashes


def best_alignment(a: list[int], b: list[int]) -> tuple[int, int]:
    """Return (matched_samples, offset) for the offset that aligns a and b best."""
    best = (0, 0)
    for off in range(-len(a) + MIN_SHARED, len(b) - MIN_SHARED + 1):
        pairs = [(i, i + off) for i in range(len(a)) if 0 <= i + off < len(b)]
        if len(pairs) < MIN_SHARED:
            continue
        matched = sum(1 for i, j in pairs if (a[i] ^ b[j]).bit_count() <= HAMMING_MAX)
        if matched > best[0]:
            best = (matched, off)
    return best


def find_overlaps(hashes: dict[str, list[int]], fps: float, step: int) -> list[dict]:
    """Compare only clips that share at least one exact frame hash."""
    index = collections.defaultdict(set)
    for cid, hs in hashes.items():
        for h in hs:
            index[h].add(cid)

    shared = collections.Counter()
    for clips in index.values():
        # A hash shared by very many clips is a blank/crowd frame, not evidence.
        if 1 < len(clips) <= 30:
            for pair in itertools.combinations(sorted(clips), 2):
                shared[pair] += 1

    secs_per_sample = step / fps
    overlaps = []
    for (a, b), n in shared.items():
        if n < MIN_SHARED:
            continue
        matched, off = best_alignment(hashes[a], hashes[b])
        if matched >= MIN_MATCHED:
            overlaps.append({
                "a": a,
                "b": b,
                "overlap_s": round(matched * secs_per_sample, 1),
                "offset_s": round(off * secs_per_sample, 1),
                "matched_samples": matched,
            })
    overlaps.sort(key=lambda d: -d["overlap_s"])
    return overlaps


def group(clip_ids: list[str], overlaps: list[dict]) -> dict[str, str]:
    """Connected components over the overlap graph. Returns clip_id -> group_id."""
    adj = collections.defaultdict(set)
    for o in overlaps:
        adj[o["a"]].add(o["b"])
        adj[o["b"]].add(o["a"])

    seen, groups = set(), {}
    for n, cid in enumerate(clip_ids):
        if cid in seen:
            continue
        stack, comp = [cid], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x])
        for x in comp:
            groups[x] = f"G{n:05d}"
    return groups


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV holding clip ids to scan")
    src.add_argument("--paths", nargs="+", help="explicit video paths")
    p.add_argument("--id-column", default="id", help="id column in --csv (default: id)")
    p.add_argument("--video-root", default="data/SoccerNet-AA",
                   help="prefix for ids from --csv (default: data/SoccerNet-AA)")
    p.add_argument("--out-groups", required=True, help="JSONL output: one line per overlap pair")
    p.add_argument("--out-csv", help="write the input CSV keeping one row per content group")
    p.add_argument("--fps", type=float, default=25.0, help="clip fps (default: 25)")
    p.add_argument("--frame-step", type=int, default=FRAME_STEP)
    args = p.parse_args()

    if args.csv:
        rows = list(csv.DictReader(open(args.csv)))
        if args.id_column not in (rows[0] if rows else {}):
            print(f"error: column {args.id_column!r} not in {args.csv}", file=sys.stderr)
            return 1
        clip_ids = sorted({r[args.id_column] for r in rows})
        path_of = lambda cid: f"{args.video_root}/{cid}.mp4"
    else:
        rows = []
        clip_ids = sorted(args.paths)
        path_of = lambda cid: cid

    print(f"hashing {len(clip_ids)} clips ...", flush=True)
    hashes = {}
    for i, cid in enumerate(clip_ids, 1):
        path = path_of(cid)
        if not pathlib.Path(path).exists():
            print(f"  skip (missing): {path}", file=sys.stderr)
            continue
        hashes[cid] = dhash_sequence(path, args.frame_step)
        if i % 50 == 0:
            print(f"  {i}/{len(clip_ids)}", flush=True)

    overlaps = find_overlaps(hashes, args.fps, args.frame_step)
    groups = group(sorted(hashes), overlaps)
    n_groups = len(set(groups.values()))

    with open(args.out_groups, "w") as f:
        for o in overlaps:
            f.write(json.dumps({**o, "type": "content_overlap"}, ensure_ascii=False) + "\n")

    print(f"\noverlapping pairs : {len(overlaps)}")
    print(f"clips             : {len(hashes)}")
    print(f"content groups    : {n_groups}  ({len(hashes) - n_groups} redundant, "
          f"{(len(hashes) - n_groups) / max(1, len(hashes)):.0%})")
    print(f"wrote {args.out_groups}")

    if args.out_csv and rows:
        kept, seen_groups = [], set()
        for r in rows:
            g = groups.get(r[args.id_column])
            if g is None or g in seen_groups:
                continue
            seen_groups.add(g)
            kept.append({**r, "content_group": g})
        fields = list(rows[0].keys()) + ["content_group"]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(kept)
        print(f"wrote {args.out_csv}  ({len(kept)}/{len(rows)} rows kept)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
