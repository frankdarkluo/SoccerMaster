#!/usr/bin/env python3
"""Rename SoccerNet-AA clips to 5s dHash order (head = AA-1). Train/valid later."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.select_aa_clips import (  # noqa: E402
    _aa_videos,
    assign_timeline_aa_ids,
    build_timelines,
    find_overlaps,
    read_cache,
)

DATA = ROOT / "data" / "SoccerNet-AA"
OUT = ROOT / "outputs" / "soccernet_aa_screening"
SPLITS = {
    "test": OUT / "clip_scan.jsonl",
    "challenge": OUT / "scan_challenge.jsonl",
}
EXTRA_JSONL = (OUT / "scan_test.jsonl", OUT / "clip_scan_test.jsonl")


def mapping_for(split: str, cache: Path) -> tuple[dict[str, str], dict[str, tuple[str, float]]]:
    rows = [r for r in read_cache(cache) if r["split"] == split]
    ids = sorted(r["clip_id"] for r in rows)
    placed = build_timelines(ids, find_overlaps(rows))
    missing = [cid for cid in ids if cid not in placed]
    if missing:
        raise SystemExit(f"{split}: {len(missing)} clips not on a 5s chain, e.g. {missing[:3]}")
    mapping = assign_timeline_aa_ids(placed)
    if set(mapping) != set(ids):
        raise SystemExit(f"{split}: mapping size {len(mapping)} != scan {len(ids)}")
    return mapping, placed


def verify(split: str, mapping: dict[str, str], placed: dict[str, tuple[str, float]]) -> list[str]:
    inverse = {new: old for old, new in mapping.items()}
    n = len(mapping)
    head = f"{split}/AA-1"
    if head not in inverse:
        raise SystemExit(f"{split}: missing {head}")
    first_tl = min(placed[old][0] for old in mapping)
    head_old = min(
        (placed[old][1], old) for old in mapping if placed[old][0] == first_tl
    )[1]
    if mapping[head_old] != head:
        raise SystemExit(
            f"{split}: timeline-0 head {head_old} mapped to {mapping[head_old]}, not {head}"
        )
    boundaries = []
    for i in range(1, n):
        a = inverse[f"{split}/AA-{i}"]
        b = inverse[f"{split}/AA-{i + 1}"]
        tl_a, sa = placed[a]
        tl_b, sb = placed[b]
        if tl_a == tl_b:
            if round(sb - sa, 1) != 5.0:
                raise SystemExit(
                    f"{split}: AA-{i} -> AA-{i + 1} delta {sb - sa}s (want 5.0)"
                )
        else:
            boundaries.append(f"{split}/AA-{i + 1}")
    if not boundaries:
        raise SystemExit(f"{split}: expected a second timeline")
    stems = {p.stem for p in _aa_videos(DATA / split)}
    want = {old.split("/", 1)[1] for old in mapping}
    if stems != want:
        raise SystemExit(f"{split}: files {len(stems)} vs mapping {len(want)}")
    return boundaries


def two_phase_rename(split: str, mapping: dict[str, str]) -> None:
    by_stem = {p.stem: p for p in _aa_videos(DATA / split)}
    temps: list[tuple[Path, Path]] = []
    for old_id, new_id in mapping.items():
        src = by_stem[old_id.split("/", 1)[1]]
        dest = src.with_name(f"{new_id.split('/', 1)[1]}.mp4")
        tmp = src.with_name(f".remap-{src.stem}.mp4")
        src.rename(tmp)
        temps.append((tmp, dest))
    for tmp, dest in temps:
        tmp.rename(dest)


def rewrite_jsonl(path: Path, mapping: dict[str, str]) -> int:
    n = 0
    out_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        cid = row.get("clip_id")
        if cid in mapping:
            row["clip_id"] = mapping[cid]
            n += 1
        out_lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return n


def rehome(split: str, selected: set[str]) -> tuple[int, int]:
    split_dir = DATA / split
    (split_dir / "clean").mkdir(exist_ok=True)
    (split_dir / "dup").mkdir(exist_ok=True)
    moved = 0
    for path in _aa_videos(split_dir):
        cid = f"{split}/{path.stem}"
        dest = split_dir / ("clean" if cid in selected else "dup") / path.name
        if path.resolve() != dest.resolve():
            if dest.exists():
                raise SystemExit(f"rehome collision: {dest}")
            path.rename(dest)
            moved += 1
    root_left = list(split_dir.glob("AA-*.mp4"))
    if root_left:
        raise SystemExit(f"{split} root still has {len(root_left)} AA-*.mp4")
    return moved, len(selected)


def main() -> None:
    combined: dict[str, str] = {}
    records: list[dict] = []
    boundaries: dict[str, str] = {}
    for split, cache in SPLITS.items():
        mapping, placed = mapping_for(split, cache)
        boundaries[split] = ", ".join(verify(split, mapping, placed))
        combined.update(mapping)
        for old_id, new_id in mapping.items():
            timeline, start = placed[old_id]
            records.append({
                "split": split,
                "old_id": old_id,
                "new_id": new_id,
                "start_s": round(start, 1),
                "timeline": timeline,
            })
        print(f"{split}: {len(mapping)} clips, boundary {boundaries[split]}", flush=True)

    map_path = OUT / "id_remap_test_challenge.jsonl"
    records.sort(key=lambda r: (r["split"], int(r["new_id"].rsplit("-", 1)[1])))
    map_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
        encoding="utf-8",
    )
    print(f"wrote {map_path}", flush=True)

    for split in SPLITS:
        mapping = {k: v for k, v in combined.items() if k.startswith(f"{split}/")}
        two_phase_rename(split, mapping)
        print(f"renamed {split} files", flush=True)

    n_scan = rewrite_jsonl(OUT / "clip_scan.jsonl", combined)
    n_chal = rewrite_jsonl(OUT / "scan_challenge.jsonl", combined)
    extra = {p.name: rewrite_jsonl(p, combined) for p in EXTRA_JSONL if p.exists()}
    print(f"rewrote clip_scan.jsonl={n_scan} scan_challenge.jsonl={n_chal} extra={extra}")


if __name__ == "__main__":
    main()
