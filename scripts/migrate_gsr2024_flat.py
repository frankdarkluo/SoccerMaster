#!/usr/bin/env python3
"""Move SoccerNetGS v1.3 into the flat data/GSR-2024 layout safely.

The default mode is read-only. Use --apply to perform same-filesystem renames,
--verify for an independent integrity pass, or --rollback JOURNAL to restore the
legacy tree. The 15 system-arm clips are deliberately outside this migration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_ROOT = REPO_ROOT / "codes/sn-gamestate/datasets/SoccerNetGS"
DEFAULT_VIDEO_SOURCE = REPO_ROOT / "data/videos/SoccerNetGS_videos"
DEFAULT_TARGET = REPO_ROOT / "data/GSR-2024"
EXPECTED_COUNTS = {"train": 57, "validation": 58, "test": 49, "challenge": 36}
SOURCE_DIR = {"train": "train", "validation": "valid", "test": "test", "challenge": "challenge"}
REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "clip_uid",
    "legacy_clip_uids",
    "clip_id",
    "dataset_id",
    "dataset_version",
    "source_split",
    "benchmark_split",
    "original_frames_path",
    "frames_path",
    "video_path",
    "labels_path",
    "labels_available",
    "n_frames",
    "fps",
    "duration_s",
    "frame_tree_sha256",
    "video_sha256",
    "content_group_id",
    "match_id",
    "source_group_id",
    "preprocessing_refs",
}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_tree_sha256(clip_dir: Path) -> str:
    image_dir = clip_dir / "img1"
    files = sorted(image_dir.glob("*.jpg"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _digest(path: Path, kind: str) -> str:
    if kind == "frames":
        return _frame_tree_sha256(path)
    return _tree_sha256(path) if path.is_dir() else _sha256_file(path)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists():
        current = current.parent
    return current


def _assert_same_filesystem(*paths: Path) -> None:
    devices = {_existing_parent(path).stat().st_dev for path in paths}
    if len(devices) != 1:
        raise RuntimeError("Frame source, video source, and target must be on one filesystem")


def _metadata(legacy_root: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {"version": "1.3"}
    all_names: list[str] = []
    for split, directory in SOURCE_DIR.items():
        path = legacy_root / directory / "sequences_info.json"
        data = _read_json(path)
        entries = data.get(split, [])
        expected = EXPECTED_COUNTS[split]
        if len(entries) != expected:
            raise ValueError(f"{path}: expected {expected} {split} entries, found {len(entries)}")
        names = [entry.get("name") for entry in entries]
        disk_names = sorted(p.name for p in (legacy_root / directory).glob("SNGS-*") if p.is_dir())
        if sorted(names) != disk_names:
            raise ValueError(f"{directory}: metadata and clip directories differ")
        merged[split] = entries
        all_names.extend(names)
    expected_names = [f"SNGS-{number:03d}" for number in range(1, 201)]
    if sorted(all_names) != expected_names or len(set(all_names)) != 200:
        raise ValueError("Official splits must contain SNGS-001 through SNGS-200 exactly once")
    return merged


def _label_info(clip_dir: Path, split: str, expected_frames: int) -> dict[str, Any] | None:
    label_path = clip_dir / "Labels-GameState.json"
    if split == "challenge":
        if label_path.exists():
            raise ValueError(f"Challenge clip unexpectedly contains GameState GT: {label_path}")
        return None
    if not label_path.is_file():
        raise ValueError(f"Missing GameState GT: {label_path}")
    info = _read_json(label_path).get("info", {})
    if int(info.get("seq_length", -1)) != expected_frames:
        raise ValueError(f"Frame count mismatch in {label_path}")
    return info


def _build_journal(
    legacy_root: Path,
    video_source: Path,
    target: Path,
    journal_path: Path,
    *,
    hashes: bool,
) -> dict[str, Any]:
    metadata = _metadata(legacy_root)
    videos = sorted(video_source.glob("SNGS-*.mp4"))
    if [path.stem for path in videos] != [f"SNGS-{number:03d}" for number in range(1, 201)]:
        raise ValueError("Video source must contain SNGS-001.mp4 through SNGS-200.mp4 exactly")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Target is not empty: {target}")
    _assert_same_filesystem(legacy_root, video_source, target)

    operations: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for split in ("train", "validation", "test", "challenge"):
        source_dir = SOURCE_DIR[split]
        for entry in metadata[split]:
            clip_id = entry["name"]
            source_clip = legacy_root / source_dir / clip_id
            target_clip = target / "images" / clip_id
            video = video_source / f"{clip_id}.mp4"
            target_video = target / "videos" / video.name
            frame_count = len(list((source_clip / "img1").glob("*.jpg")))
            expected_frames = int(entry["n_frames"])
            if frame_count != expected_frames:
                raise ValueError(f"{clip_id}: expected {expected_frames} JPGs, found {frame_count}")
            info = _label_info(source_clip, split, expected_frames)
            frame_hash = _frame_tree_sha256(source_clip) if hashes else None
            video_hash = _sha256_file(video) if hashes else None
            operations.extend(
                [
                    {"kind": "frames", "src": str(source_clip.resolve()), "dst": str(target_clip.resolve()), "sha256": frame_hash, "done": False},
                    {"kind": "video", "src": str(video.resolve()), "dst": str(target_video.resolve()), "sha256": video_hash, "done": False},
                ]
            )
            game_id = str(info["game_id"]) if info and info.get("game_id") is not None else None
            fps = float(info["frame_rate"]) if info else 25.0
            preprocessing_refs = [
                _display_path(path)
                for path in (
                    REPO_ROOT / "outputs/preprocessing" / clip_id,
                    REPO_ROOT / "outputs/GSR-2025/preprocessing" / clip_id,
                )
                if path.is_dir()
            ]
            manifest.append(
                {
                    "schema_version": "grounded-tactics-clip-manifest-v1",
                    "clip_uid": f"gsr2024:{clip_id}",
                    "legacy_clip_uids": [f"soccernetgs:{clip_id}"],
                    "clip_id": clip_id,
                    "dataset_id": "GSR-2024",
                    "dataset_version": str(info["version"]) if info else metadata["version"],
                    "source_split": split,
                    "benchmark_split": None,
                    "original_frames_path": _display_path(source_clip),
                    "frames_path": _display_path(target_clip),
                    "video_path": _display_path(target_video),
                    "labels_path": _display_path(target_clip / "Labels-GameState.json") if info else None,
                    "labels_available": info is not None,
                    "n_frames": frame_count,
                    "fps": fps,
                    "duration_s": frame_count / fps,
                    "frame_tree_sha256": frame_hash,
                    "video_sha256": video_hash,
                    "content_group_id": f"soccernetgs-v1.3:{clip_id}",
                    "match_id": game_id,
                    "source_group_id": f"soccernetgs-v1.3:game:{game_id}" if game_id else None,
                    "preprocessing_refs": preprocessing_refs,
                }
            )

    for directory in SOURCE_DIR.values():
        for source in sorted((legacy_root / directory).iterdir()):
            if source.name.startswith("SNGS-"):
                continue
            operations.append(
                {
                    "kind": "metadata",
                    "src": str(source.resolve()),
                    "dst": str((target / "source_metadata" / directory / source.name).resolve()),
                    "sha256": _digest(source, "metadata") if hashes else None,
                    "done": False,
                }
            )
    return {
        "schema_version": "gsr2024-flat-migration-v1",
        "status": "planned",
        "legacy_root": str(legacy_root.resolve()),
        "video_source": str(video_source.resolve()),
        "target": str(target.resolve()),
        "journal": str(journal_path.resolve()),
        "metadata": metadata,
        "manifest": manifest,
        "operations": operations,
    }


def _write_outputs(payload: dict[str, Any]) -> dict[str, str]:
    target = Path(payload["target"])
    _write_json(target / "sequences_info.json", payload["metadata"])
    manifest_path = target / "clip_manifest.jsonl"
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in payload["manifest"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(manifest_path)
    return {
        "clip_manifest.jsonl": _sha256_file(manifest_path),
        "sequences_info.json": _sha256_file(target / "sequences_info.json"),
    }


def apply_migration(payload: dict[str, Any], journal_path: Path) -> None:
    target = Path(payload["target"])
    for path in (target / "images", target / "videos", target / "source_metadata"):
        path.mkdir(parents=True, exist_ok=True)
    payload["status"] = "moving"
    _write_json(journal_path, payload)
    for operation in payload["operations"]:
        source, destination = Path(operation["src"]), Path(operation["dst"])
        if destination.exists() and not source.exists():
            operation["done"] = True
        elif source.exists() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            operation["done"] = True
        else:
            raise RuntimeError(f"Ambiguous migration state: {source} -> {destination}")
        _write_json(journal_path, payload)
    payload["generated_sha256"] = _write_outputs(payload)
    _write_json(journal_path, payload)
    for directory in SOURCE_DIR.values():
        split_dir = Path(payload["legacy_root"]) / directory
        if split_dir.exists():
            split_dir.rmdir()
    verify_migration(target, Path(payload["legacy_root"]))
    payload["status"] = "complete"
    _write_json(journal_path, payload)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_migration(target: Path, legacy_root: Path) -> dict[str, Any]:
    metadata = _read_json(target / "sequences_info.json")
    manifest = _load_manifest(target / "clip_manifest.jsonl")
    if len(manifest) != 200 or len({row["clip_uid"] for row in manifest}) != 200:
        raise ValueError("Manifest must contain 200 unique clips")
    if any(REQUIRED_MANIFEST_FIELDS - row.keys() for row in manifest):
        raise ValueError("Manifest row is missing required fields")
    counts = {split: sum(row["source_split"] == split for row in manifest) for split in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected split counts: {counts}")
    if sum(bool(row["labels_available"]) for row in manifest) != 164:
        raise ValueError("Expected exactly 164 clips with GameState GT")
    total_frames = 0
    for row in manifest:
        clip_dir = target / "images" / row["clip_id"]
        video = target / "videos" / f"{row['clip_id']}.mp4"
        frames = len(list((clip_dir / "img1").glob("*.jpg")))
        if frames != row["n_frames"]:
            raise ValueError(f"{row['clip_id']}: frame count changed")
        if _frame_tree_sha256(clip_dir) != row["frame_tree_sha256"]:
            raise ValueError(f"{row['clip_id']}: frame digest changed")
        if _sha256_file(video) != row["video_sha256"]:
            raise ValueError(f"{row['clip_id']}: video digest changed")
        label = clip_dir / "Labels-GameState.json"
        if label.is_file() != row["labels_available"]:
            raise ValueError(f"{row['clip_id']}: label availability changed")
        total_frames += frames
    for split, count in EXPECTED_COUNTS.items():
        if len(metadata.get(split, [])) != count:
            raise ValueError(f"sequences_info.json has wrong {split} count")
    for directory in SOURCE_DIR.values():
        source_metadata = target / "source_metadata" / directory / "sequences_info.json"
        if not source_metadata.is_file():
            raise ValueError(f"Missing preserved source metadata: {source_metadata}")
    if any((legacy_root / directory).exists() for directory in SOURCE_DIR.values()):
        raise ValueError("Legacy split directories still exist")
    result = {"clips": 200, "frames": total_frames, "videos": 200, "labels": 164, "splits": counts}
    print(json.dumps(result, sort_keys=True))
    return result


def _preflight_rollback(payload: dict[str, Any]) -> None:
    target = Path(payload["target"])
    generated_hashes = payload.get("generated_sha256", {})
    for name in ("clip_manifest.jsonl", "sequences_info.json"):
        path = target / name
        if path.exists():
            expected = generated_hashes.get(name)
            if not expected or _sha256_file(path) != expected:
                raise RuntimeError(f"Refusing rollback because generated output changed: {path}")
    for operation in payload["operations"]:
        source, destination = Path(operation["src"]), Path(operation["dst"])
        if source.exists() == destination.exists():
            raise RuntimeError(f"Ambiguous rollback state: {destination} -> {source}")
        current = destination if destination.exists() else source
        expected = operation.get("sha256")
        if not expected or _digest(current, operation["kind"]) != expected:
            raise RuntimeError(f"Refusing rollback because content changed: {current}")


def rollback(journal_path: Path) -> None:
    payload = _read_json(journal_path)
    target = Path(payload["target"])
    _preflight_rollback(payload)
    for generated in (target / "clip_manifest.jsonl", target / "sequences_info.json"):
        if generated.exists():
            generated.unlink()
    for operation in reversed(payload["operations"]):
        source, destination = Path(operation["src"]), Path(operation["dst"])
        if source.exists() and not destination.exists():
            operation["done"] = False
            continue
        source.parent.mkdir(parents=True, exist_ok=True)
        destination.rename(source)
        operation["done"] = False
        _write_json(journal_path, payload)
    for directory in SOURCE_DIR.values():
        metadata_dir = target / "source_metadata" / directory
        if metadata_dir.exists():
            metadata_dir.rmdir()
    for path in (target / "source_metadata", target / "videos", target / "images", target):
        if path.exists():
            path.rmdir()
    payload["status"] = "rolled_back"
    _write_json(journal_path, payload)
    print(f"Rolled back migration recorded in {journal_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="validate and print the plan (default)")
    modes.add_argument("--apply", action="store_true", help="perform journaled atomic renames")
    modes.add_argument("--verify", action="store_true", help="verify the migrated flat layout")
    modes.add_argument("--rollback", type=Path, metavar="JOURNAL", help="restore paths recorded by JOURNAL")
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--video-source", type=Path, default=DEFAULT_VIDEO_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--journal", type=Path)
    args = parser.parse_args()

    if args.rollback:
        rollback(args.rollback.resolve())
        return 0
    legacy_root = args.legacy_root.resolve()
    video_source = args.video_source.resolve()
    target = args.target.resolve()
    journal = (args.journal or target.with_name(f"{target.name}.migration.json")).resolve()
    if args.verify:
        verify_migration(target, legacy_root)
        return 0
    if not args.apply and (target / "clip_manifest.jsonl").is_file():
        status = _read_json(journal).get("status") if journal.is_file() else None
        if status != "complete":
            raise RuntimeError(f"Migrated target has no complete journal: {journal}")
        print(json.dumps({
            "mode": "dry-run",
            "state": "already_complete",
            "clips": len(_load_manifest(target / "clip_manifest.jsonl")),
            "target": str(target),
        }, sort_keys=True))
        return 0
    if args.apply and journal.is_file() and _read_json(journal).get("status") != "rolled_back":
        payload = _read_json(journal)
        if (payload["legacy_root"], payload["video_source"], payload["target"]) != (
            str(legacy_root), str(video_source), str(target)
        ):
            raise ValueError(f"Journal paths do not match arguments: {journal}")
    else:
        payload = _build_journal(legacy_root, video_source, target, journal, hashes=args.apply)
    if not args.apply:
        print(json.dumps({"mode": "dry-run", "clips": len(payload["manifest"]), "operations": len(payload["operations"]), "target": str(target)}, sort_keys=True))
        return 0
    if args.apply and payload.get("status") == "complete":
        verify_migration(target, legacy_root)
        print(f"Migration already complete; rollback journal: {journal}")
        return 0
    apply_migration(payload, journal)
    print(f"Migration complete; rollback journal: {journal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
