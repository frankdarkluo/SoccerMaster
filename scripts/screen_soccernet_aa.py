#!/usr/bin/env python3
"""Staged, resumable SoccerNet-AA broadcast screening and overlap audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.utils.video import get_video_codec, get_video_info


DATASET = "SoccerNet-AA"
EXPECTED_COUNTS = {"train": 4587, "valid": 1169}
BIN_S = 2.5
BIN_COUNT = 12
MIN_CONFIDENCE = 70
BAD_LABELS = {
    "replay",
    "player_closeup",
    "coach_crowd",
    "score_animation",
    "other_unusable",
}
BROADCAST_LABELS = ("main_wide", *sorted(BAD_LABELS))
OBSERVABILITY = {"observable", "insufficient"}
WEAK_LABELS = {
    "continuous_passing",
    "possession_regain",
    "attacking_transition",
    "wing_progression",
    "second_ball_recovery",
    "low_block",
    "mid_block",
    "back_build_up",
    "set_piece",
}
PRIORITY_LABELS = {"low_block", "mid_block", "back_build_up", "set_piece"}
CORE_LABELS = WEAK_LABELS - PRIORITY_LABELS
AUXILIARY_PROMPT = """Inspect the 12 ordered center frames, one per 2.5-second
bin of a 30-second football clip. Set observability to observable only when the
target team, ball direction, and relevant defensive structure can be assessed;
otherwise use insufficient. Add only visually supported weak_labels from
continuous_passing, possession_regain, attacking_transition, wing_progression,
second_ball_recovery, low_block, mid_block, back_build_up, set_piece. Each label
may appear at most once and needs start_s, end_s, confidence, and a short
evidence string. Low/mid-block evidence that is not established inside these 30
seconds must remain insufficient or a weak candidate, never certain gold."""
EXTERNAL_PROMPT = """Review this 30-second football video for benchmark broadcast
compatibility. Divide it into 12 bins of 2.5 seconds. For each bin return exactly
one label: main_wide, replay, player_closeup, coach_crowd, score_animation, or
other_unusable, plus confidence 0..100. main_wide must reveal ball direction and
the relevant defensive structure. Return observability and visually grounded
weak labels using only: continuous_passing, possession_regain,
attacking_transition, wing_progression, second_ball_recovery, low_block,
mid_block, back_build_up, set_piece. JSON only."""


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


def split_sources(
    data_root: Path, split: str, expected_count: int | None = None
) -> list[dict[str, Any]]:
    split_dir = Path(data_root) / split
    videos = _aa_videos(split_dir)
    if expected_count is not None and len(videos) != expected_count:
        raise ValueError(f"{split}: expected {expected_count} MP4s, found {len(videos)}")
    numbers = [_number(path) for path in videos]
    if numbers != list(range(1, len(videos) + 1)):
        raise ValueError(f"{split}: AA numbering is not contiguous from 1")
    sidecar_path = split_dir / "Labels-ball.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecars = payload.get("videos")
    if not isinstance(sidecars, list) or len(sidecars) != len(videos):
        raise ValueError(f"{split}: MP4/sidecar count mismatch")
    rows = []
    for number, video, sidecar in zip(numbers, videos, sidecars):
        expected_paths = {f"clip_{number}/720p.mp4", f"AA-{number}.mp4"}
        if sidecar.get("path") not in expected_paths:
            raise ValueError(
                f"{split} AA-{number}: expected one of {sorted(expected_paths)}, "
                f"found {sidecar.get('path')}"
            )
        rows.append(
            {
                "video_id": f"AA-{number}",
                "clip_uid": f"soccernet-aa:{split}:AA-{number}",
                "dataset": DATASET,
                "source_split": split,
                "video_path": video.as_posix(),
                "sidecar": sidecar,
            }
        )
    return rows


def validate_screen(payload: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("screen response must be an object")
    bins = payload.get("bins")
    if not isinstance(bins, list) or len(bins) != BIN_COUNT:
        raise ValueError(f"screen response needs exactly {BIN_COUNT} bins")
    normalized_bins = []
    for expected_index, item in enumerate(bins):
        if not isinstance(item, dict) or item.get("index") != expected_index:
            raise ValueError("bin indexes must be consecutive 0..11")
        label = item.get("label")
        confidence = item.get("confidence")
        if label not in BROADCAST_LABELS:
            raise ValueError(f"invalid broadcast label: {label}")
        if not isinstance(confidence, int) or not 0 <= confidence <= 100:
            raise ValueError("bin confidence must be an integer in 0..100")
        normalized_bins.append(
            {"index": expected_index, "label": label, "confidence": confidence}
        )
    observability = payload.get("observability")
    if observability not in OBSERVABILITY:
        raise ValueError(f"invalid observability: {observability}")
    reason = str(payload.get("observability_reason", "")).strip()
    weak = []
    for item in payload.get("weak_labels", []):
        if not isinstance(item, dict) or item.get("label") not in WEAK_LABELS:
            raise ValueError(f"invalid weak label: {item}")
        start = float(item.get("start_s"))
        end = float(item.get("end_s"))
        confidence = int(item.get("confidence"))
        if not 0 <= start < end <= 30 or not 0 <= confidence <= 100:
            raise ValueError(f"invalid weak-label span/confidence: {item}")
        weak.append(
            {
                "label": item["label"],
                "start_s": start,
                "end_s": end,
                "confidence": confidence,
                "evidence": str(item.get("evidence", "")).strip(),
                "source": item.get("source") or source or "unspecified",
            }
        )
    return {
        "bins": normalized_bins,
        "observability": observability,
        "observability_reason": reason,
        "weak_labels": weak,
    }


def screen_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    bins = payload["bins"]
    bad_s = sum(item["label"] in BAD_LABELS for item in bins) * BIN_S
    run = longest = 0
    for item in bins:
        run = run + 1 if item["label"] == "main_wide" else 0
        longest = max(longest, run)
    low_confidence = any(item["confidence"] < MIN_CONFIDENCE for item in bins)
    if low_confidence:
        decision = "hold"
    elif bad_s > 15.0 or longest * BIN_S < 10.0:
        decision = "reject_candidate"
    else:
        decision = "keep"
    return {
        "bad_s": bad_s,
        "max_continuous_wide_s": longest * BIN_S,
        "local_decision": decision,
        "observability": payload["observability"],
        "eligible_positive": payload["observability"] == "observable",
    }


def resolve_screen(
    local: str | None, external: str | None, manual: str | None
) -> str:
    if manual in {"keep", "reject"}:
        return manual
    if local == "keep":
        return "keep"
    if local in {None, "hold"}:
        return "external_review_pending" if external is None else "manual_review_pending"
    if local == "reject_candidate":
        if external is None:
            return "external_review_pending"
        if external == "reject_candidate":
            return "reject"
        return "manual_review_pending"
    raise ValueError(f"invalid local decision: {local}")


def _global_ssim(left: np.ndarray, right: np.ndarray) -> float:
    left_f = left.astype(np.float64)
    right_f = right.astype(np.float64)
    mean_l, mean_r = left_f.mean(), right_f.mean()
    var_l, var_r = left_f.var(), right_f.var()
    covariance = ((left_f - mean_l) * (right_f - mean_r)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(
        ((2 * mean_l * mean_r + c1) * (2 * covariance + c2))
        / ((mean_l**2 + mean_r**2 + c1) * (var_l + var_r + c2))
    )


def boundary_match(left_tail: np.ndarray, right_head: np.ndarray) -> dict[str, Any]:
    if left_tail.shape != right_head.shape or len(left_tail) != 25:
        raise ValueError("boundary arrays must have the same shape and 25 frames")
    scores = [_global_ssim(a, b) for a, b in zip(left_tail, right_head)]
    high = sum(score >= 0.98 for score in scores)
    mean = float(np.mean(scores))
    status = (
        "confirmed"
        if high >= 23 and mean >= 0.985
        else "ambiguous"
        if high >= 20 or mean >= 0.90
        else "none"
    )
    return {"status": status, "high_frames": high, "mean_ssim": round(mean, 6)}


def _selection_score(ids: Iterable[str], by_id: dict[str, dict[str, Any]]) -> tuple:
    rows = [by_id[uid] for uid in ids]
    labels = [
        item["label"]
        for row in rows
        for item in row.get("weak_labels", [])
        if isinstance(item, dict)
    ]
    return (
        len(rows),
        sum(row.get("screen_status", "keep") == "keep" for row in rows),
        sum(row.get("observability") == "observable" for row in rows),
        sum(label in PRIORITY_LABELS for label in labels),
        sum(label in CORE_LABELS for label in labels),
        sum(float(row.get("max_continuous_wide_s", 0)) for row in rows),
        -sum(float(row.get("bad_s", 30)) for row in rows),
    )


def _better(
    left: tuple[str, ...], right: tuple[str, ...], by_id: dict[str, dict[str, Any]]
) -> tuple[str, ...]:
    left_score = _selection_score(left, by_id)
    right_score = _selection_score(right, by_id)
    if left_score != right_score:
        return left if left_score > right_score else right
    return left if tuple(sorted(left)) < tuple(sorted(right)) else right


def select_path(
    rows: list[dict[str, Any]], edges: list[tuple[str, str]]
) -> set[str]:
    by_id = {row["clip_uid"]: row for row in rows}
    adjacency = {uid: set() for uid in by_id}
    for left, right in edges:
        if left not in adjacency or right not in adjacency:
            raise ValueError("overlap edge references unknown clip")
        adjacency[left].add(right)
        adjacency[right].add(left)
    branches = sorted(
        uid for uid, neighbors in adjacency.items() if len(neighbors) > 2
    )
    if branches:
        raise ValueError(f"overlap graph branches at {branches[:10]}")
    chosen: set[str] = set()
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        component = set()
        stack = [start]
        while stack:
            uid = stack.pop()
            if uid in component:
                continue
            component.add(uid)
            stack.extend(adjacency[uid] - component)
        unseen -= component
        if len(component) == 1:
            chosen |= component
            continue
        endpoints = sorted(uid for uid in component if len(adjacency[uid]) == 1)
        if not endpoints:
            raise ValueError(
                f"overlap graph cycle in {sorted(component)[:10]}"
            )
        ordered = []
        previous = None
        current = endpoints[0]
        while current is not None:
            ordered.append(current)
            following = adjacency[current] - ({previous} if previous else set())
            previous, current = current, next(iter(following), None)
        without: tuple[str, ...] = ()
        with_current: tuple[str, ...] = (ordered[0],)
        for uid in ordered[1:]:
            old_without = without
            without = _better(without, with_current, by_id)
            with_current = (*old_without, uid)
        chosen.update(_better(without, with_current, by_id))
    return chosen


def blind_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "video_id",
        "clip_uid",
        "dataset",
        "source_split",
        "video_path",
        "fps",
        "resolution",
        "width",
        "height",
        "duration_s",
        "n_frames",
        "codec",
        "sha256",
        "content_group_id",
    )
    return {key: row[key] for key in fields if key in row}


def weak_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "clip_uid": row["clip_uid"],
        "eligible_positive": bool(row["eligible_positive"]),
        "observability": row.get("observability"),
        "observability_reason": row.get("observability_reason", ""),
        "weak_labels": row.get("weak_labels", []),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_boundary(path: Path, tail: bool) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = []
    base = 25.0 if tail else 0.0
    for index in range(25):
        capture.set(cv2.CAP_PROP_POS_MSEC, (base + (index + 0.5) * 0.2) * 1000)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"cannot read boundary frame {index}: {path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA))
    capture.release()
    return np.stack(frames)


def _dhash(frame: np.ndarray) -> int:
    small = cv2.resize(frame, (9, 8), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")


def _hashes(frames: np.ndarray) -> list[str]:
    return [f"{_dhash(frame):016x}" for frame in frames]


def _overlap_candidates(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    index: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    heads = {}
    for row in rows:
        values = [int(value, 16) for value in row["head_dhash"]]
        heads[row["clip_uid"]] = values
        for position, value in enumerate(values):
            for chunk in range(8):
                index[(position, chunk, (value >> (chunk * 8)) & 0xFF)].add(
                    row["clip_uid"]
                )
    candidates = set()
    for row in rows:
        hits: dict[str, set[int]] = defaultdict(set)
        for position, value_hex in enumerate(row["tail_dhash"]):
            value = int(value_hex, 16)
            possible = set()
            for chunk in range(8):
                possible |= index[
                    (position, chunk, (value >> (chunk * 8)) & 0xFF)
                ]
            for other in possible:
                if other == row["clip_uid"]:
                    continue
                if (value ^ heads[other][position]).bit_count() <= 6:
                    hits[other].add(position)
        for other, positions in hits.items():
            if len(positions) >= 20:
                candidates.add((row["clip_uid"], other))
    return candidates


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --resume to replace it explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(_json_line(row) + "\n" for row in rows)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(body, encoding="utf-8")
    temp.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json_line(row) + "\n")



def _write_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --resume to replace it explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --resume to replace it explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _event_summary(sidecar: dict[str, Any]) -> dict[str, Any]:
    annotations = sidecar.get("annotations", {})
    observation = annotations.get("observation", [])
    anticipation = annotations.get("anticipation", [])
    return {
        "observed_events": observation,
        "anticipated_events": anticipation,
        "observed_event_counts": dict(sorted(Counter(
            item.get("label") for item in observation
        ).items())),
        "anticipated_event_counts": dict(sorted(Counter(
            item.get("label") for item in anticipation
        ).items())),
    }


def run_inventory(args: argparse.Namespace) -> None:
    output = args.output_dir
    inventory_path = output / "inventory.jsonl"
    overlap_path = output / "overlap_groups.jsonl"
    if not args.resume:
        for path in (inventory_path, overlap_path):
            if path.exists():
                raise FileExistsError(f"{path} exists; pass --resume to reuse it")
    existing_rows = _read_jsonl(inventory_path) if args.resume else []
    existing = {row["clip_uid"]: row for row in existing_rows}
    sources = []
    for split, expected in EXPECTED_COUNTS.items():
        sources.extend(split_sources(args.data_root, split, expected))
    rows = []
    for position, source in enumerate(sources, 1):
        cached = existing.get(source["clip_uid"])
        if cached:
            rows.append(cached)
            continue
        path = Path(source["video_path"])
        info = get_video_info(path)
        codec, pix_fmt = get_video_codec(path)
        if (
            abs(info["fps"] - 25) > 1e-6
            or abs(info["duration_s"] - 30) > 0.01
            or info["width"] != 1280
            or info["height"] != 720
            or info["total_frames"] != 750
        ):
            raise ValueError(f"unexpected AA media properties: {source['clip_uid']} {info}")
        head = _read_boundary(path, tail=False)
        tail = _read_boundary(path, tail=True)
        row = {
            key: value for key, value in source.items() if key != "sidecar"
        }
        row.update(
            {
                "fps": info["fps"],
                "resolution": f"{info['width']}x{info['height']}",
                "width": info["width"],
                "height": info["height"],
                "duration_s": info["duration_s"],
                "n_frames": info["total_frames"],
                "codec": codec,
                "pix_fmt": pix_fmt,
                "sha256": _sha256(path),
                "head_dhash": _hashes(head),
                "tail_dhash": _hashes(tail),
                **_event_summary(source["sidecar"]),
            }
        )
        rows.append(row)
        _append_jsonl(inventory_path, row)
        if position % 100 == 0:
            print(f"inventory {position}/{len(sources)}", flush=True)
    _write_jsonl(inventory_path, rows, overwrite=True)

    exact = defaultdict(list)
    for row in rows:
        exact[row["sha256"]].append(row["clip_uid"])
    exact_records = [
        {
            "type": "exact_duplicate",
            "status": "confirmed",
            "sha256": digest,
            "members": sorted(members),
        }
        for digest, members in sorted(exact.items())
        if len(members) > 1
    ]
    old_overlap = _read_jsonl(overlap_path) if args.resume else []
    boundary = {
        (row["left_uid"], row["right_uid"]): row
        for row in old_overlap
        if row.get("type") == "boundary_overlap"
    }
    overlap = exact_records + [
        boundary[key] for key in sorted(boundary)
    ]
    _write_jsonl(overlap_path, overlap, overwrite=True)

    by_id = {row["clip_uid"]: row for row in rows}
    candidates = sorted(_overlap_candidates(rows))
    for position, (left, right) in enumerate(candidates, 1):
        if (left, right) in boundary:
            continue
        result = {
            "type": "boundary_overlap",
            "left_uid": left,
            "right_uid": right,
            **boundary_match(
                _read_boundary(Path(by_id[left]["video_path"]), tail=True),
                _read_boundary(Path(by_id[right]["video_path"]), tail=False),
            ),
        }
        boundary[(left, right)] = result
        _append_jsonl(overlap_path, result)
        if position % 100 == 0:
            print(f"overlap verify {position}/{len(candidates)}", flush=True)
    overlap = exact_records + [boundary[key] for key in sorted(boundary)]
    _write_jsonl(overlap_path, overlap, overwrite=True)
    print(f"wrote {len(rows)} inventory rows and {len(overlap)} overlap records")


def _scan_frames(path: Path) -> tuple[list[np.ndarray], list[float]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    centers = [(index + 0.5) * BIN_S for index in range(BIN_COUNT)]
    samples: list[np.ndarray | None] = [None] * BIN_COUNT
    cuts = []
    previous = None
    last_cut = -10.0
    frame_index = 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        time_s = frame_index / fps
        gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54),
            interpolation=cv2.INTER_AREA,
        )
        if previous is not None:
            delta = float(cv2.absdiff(previous, gray).mean() / 255.0)
            if delta >= 0.18 and time_s - last_cut >= 0.5:
                cuts.append(round(time_s, 3))
                last_cut = time_s
        previous = gray
        for index, center in enumerate(centers):
            if samples[index] is None and time_s >= center:
                samples[index] = frame.copy()
        frame_index += 1
    capture.release()
    if any(frame is None for frame in samples):
        raise RuntimeError(f"failed to sample all 12 bins: {path}")
    return [frame for frame in samples if frame is not None], cuts




def _extract_required(text: str, required: set[str]) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and required <= value.keys():
            return value
    raise ValueError(f"model returned no object with {sorted(required)}: {text[:500]}")


def _qwen_review(path: Path, model_path: str) -> tuple[dict[str, Any], str, list[float]]:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not hasattr(_qwen_review, "_cache"):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
        _qwen_review._cache = (model, processor)
    model, processor = _qwen_review._cache

    def generate(conversations: list[list[dict[str, Any]]], tokens: int) -> list[str]:
        rendered = []
        images = []
        videos = []
        for messages in conversations:
            rendered.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
            image_inputs, video_inputs = process_vision_info(messages)
            images.extend(image_inputs or [])
            videos.extend(video_inputs or [])
        inputs = processor(
            text=rendered,
            images=images or None,
            videos=videos or None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=tokens)
        trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        return processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    frames, cuts = _scan_frames(path)
    with tempfile.TemporaryDirectory(prefix="aa-screen-") as directory:
        image_paths = []
        for index, frame in enumerate(frames):
            image_path = Path(directory) / f"{index:02d}.jpg"
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"failed to write sample frame: {image_path}")
            image_paths.append(image_path)
        bin_prompt = """Classify this one football broadcast frame. main_wide is
allowed only when a broad field view shows enough players to infer team shape
and ball direction. One to four large people is player_closeup even if grass is
visible. Use exactly one label from main_wide, replay, player_closeup,
coach_crowd, score_animation, other_unusable. Return only
{"label":"player_closeup","confidence":90} with the appropriate values."""
        conversations = [
            [{"role": "user", "content": [
                {"type": "image", "image": image_path.as_uri()},
                {"type": "text", "text": bin_prompt},
            ]}]
            for image_path in image_paths
        ]
        bin_raw = generate(conversations, 80)
        bins = []
        for index, raw in enumerate(bin_raw):
            item = _extract_required(raw, {"label", "confidence"})
            bins.append({
                "index": index,
                "label": item["label"],
                "confidence": item["confidence"],
            })

        weak_schema = {
            "type": "object",
            "properties": {
                key: value
                for key, value in _review_schema()["properties"].items()
                if key != "bins"
            },
            "required": ["observability", "observability_reason", "weak_labels"],
            "additionalProperties": False,
        }
        content = [
            {"type": "image", "image": image_path.as_uri()}
            for image_path in image_paths
        ]
        content.append({"type": "text", "text": (
            AUXILIARY_PROMPT
            + "\nReturn no bins in this call. Return only observability, "
            "observability_reason, and at most nine unique weak_labels. Schema:\n"
            + json.dumps(weak_schema, ensure_ascii=False)
        )})
        weak_raw = generate([[{"role": "user", "content": content}]], 1200)[0]
        auxiliary = _extract_required(
            weak_raw, {"observability", "observability_reason", "weak_labels"}
        )
    payload = validate_screen(
        {"bins": bins, **auxiliary}, source="local_qwen"
    )
    raw = json.dumps(
        {"bin_responses": bin_raw, "auxiliary_response": weak_raw},
        ensure_ascii=False,
    )
    return payload, raw, cuts


def _review_schema() -> dict[str, Any]:
    bin_schema = {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "minimum": 0, "maximum": 11},
            "label": {"type": "string", "enum": list(BROADCAST_LABELS)},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["index", "label", "confidence"],
        "additionalProperties": False,
    }
    weak_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": sorted(WEAK_LABELS)},
            "start_s": {"type": "number", "minimum": 0, "maximum": 30},
            "end_s": {"type": "number", "minimum": 0, "maximum": 30},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "evidence": {"type": "string"},
        },
        "required": ["label", "start_s", "end_s", "confidence", "evidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "bins": {
                "type": "array",
                "minItems": BIN_COUNT,
                "maxItems": BIN_COUNT,
                "items": bin_schema,
            },
            "observability": {"type": "string", "enum": sorted(OBSERVABILITY)},
            "observability_reason": {"type": "string"},
            "weak_labels": {
                "type": "array",
                "maxItems": len(WEAK_LABELS),
                "items": weak_schema,
            },
        },
        "required": [
            "bins", "observability", "observability_reason", "weak_labels"
        ],
        "additionalProperties": False,
    }


def _selected_inventory(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _read_jsonl(args.output_dir / "inventory.jsonl")
    if not rows:
        raise FileNotFoundError("run inventory first")
    requested = set(args.clip_uid or [])
    unknown = requested - {row["clip_uid"] for row in rows}
    if unknown:
        raise ValueError(f"unknown clip UIDs: {sorted(unknown)}")
    return [row for row in rows if not requested or row["clip_uid"] in requested]


def _run_calibration(args: argparse.Namespace) -> None:
    path = args.output_dir / "calibration_reviews.jsonl"
    if path.exists() and not args.resume:
        raise FileExistsError(f"{path} exists; pass --resume to reuse it")
    specs = (
        ("aa-train-aa-1", Path(args.data_root) / "train" / "AA-1.mp4", "keep"),
        (
            "aa-train-aa-2",
            Path(args.data_root) / "train" / "AA-2.mp4",
            "reject_candidate",
        ),
        (
            "gsr-sngs-001",
            ROOT / "data" / "GSR-2025" / "videos" / "SNGS-001.mp4",
            "keep",
        ),
    )
    reviews = _read_jsonl(path) if args.resume else []
    by_sample = {row["sample_id"]: row for row in reviews}
    for sample_id, video_path, expected in specs:
        review = by_sample.get(sample_id)
        if review is None:
            if not video_path.exists():
                raise FileNotFoundError(f"calibration video missing: {video_path}")
            payload, raw, cuts = _qwen_review(video_path.resolve(), args.local_model)
            review = {
                "sample_id": sample_id,
                "video_path": video_path.as_posix(),
                "expected_decision": expected,
                "provider": "local_qwen",
                "payload": payload,
                "metrics": screen_metrics(payload),
                "scene_cuts_s": cuts,
                "raw_response": raw,
            }
            reviews.append(review)
            by_sample[sample_id] = review
            _write_jsonl(path, reviews, overwrite=True)
        actual = review.get("metrics", {}).get("local_decision")
        if actual != expected:
            raise RuntimeError(
                f"calibration failed for {sample_id}: expected {expected}, got {actual}"
            )


def run_local_screen(args: argparse.Namespace) -> None:
    path = args.output_dir / "local_reviews.jsonl"
    if path.exists() and not args.resume:
        raise FileExistsError(f"{path} exists; pass --resume to reuse it")
    _run_calibration(args)
    existing_rows = _read_jsonl(path) if args.resume else []
    existing = {row["clip_uid"] for row in existing_rows}
    targets = [row for row in _selected_inventory(args) if row["clip_uid"] not in existing]
    for position, row in enumerate(targets, 1):
        try:
            payload, raw, cuts = _qwen_review(
                Path(row["video_path"]).resolve(), args.local_model
            )
            review = {
                "clip_uid": row["clip_uid"],
                "provider": "local_qwen",
                "payload": payload,
                "metrics": screen_metrics(payload),
                "scene_cuts_s": cuts,
                "raw_response": raw,
            }
        except Exception as exc:
            review = {
                "clip_uid": row["clip_uid"],
                "provider": "local_qwen",
                "error": f"{type(exc).__name__}: {exc}",
            }
        _append_jsonl(path, review)
        print(f"local-screen {position}/{len(targets)} {row['clip_uid']}", flush=True)


def run_external_review(args: argparse.Namespace) -> None:
    if args.provider != "doubao":
        raise ValueError("only the reviewed doubao adapter is supported")
    local = {row["clip_uid"]: row for row in _read_jsonl(
        args.output_dir / "local_reviews.jsonl"
    )}
    targets = []
    for row in _selected_inventory(args):
        review = local.get(row["clip_uid"])
        decision = review.get("metrics", {}).get("local_decision") if review else None
        if decision != "keep":
            targets.append(row)
    if not os.environ.get("ARK_API_KEY"):
        print(f"ARK_API_KEY missing; {len(targets)} reviews remain external_review_pending")
        return
    if not args.allow_external_upload:
        raise PermissionError("pass --allow-external-upload to upload videos to Doubao")
    from pipeline.video_models import generate_json

    path = args.output_dir / "external_reviews.jsonl"
    if path.exists() and not args.resume:
        raise FileExistsError(f"{path} exists; pass --resume to reuse it")
    existing_rows = _read_jsonl(path) if args.resume else []
    existing = {row["clip_uid"] for row in existing_rows}
    for position, row in enumerate(
        [item for item in targets if item["clip_uid"] not in existing], 1
    ):
        try:
            payload, usage, attempts = generate_json(
                "doubao",
                EXTERNAL_PROMPT,
                _review_schema(),
                video_path=Path(row["video_path"]).resolve(),
                fps=2.0,
                temperature=0.0,
            )
            validated = validate_screen(payload, source="external_doubao")
            review = {
                "clip_uid": row["clip_uid"],
                "provider": "doubao",
                "payload": validated,
                "metrics": screen_metrics(validated),
                "usage": usage,
                "attempts": attempts,
            }
        except Exception as exc:
            review = {
                "clip_uid": row["clip_uid"],
                "provider": "doubao",
                "error": f"{type(exc).__name__}: {exc}",
            }
        _append_jsonl(path, review)
        print(f"external-review {position}/{len(targets)} {row['clip_uid']}", flush=True)


def _manual_decisions(path: Path) -> dict[tuple[str, str, str], str]:
    if not path.exists():
        return {}
    decisions = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            decision = row.get("decision", "").strip()
            if not decision:
                continue
            key = (
                row.get("review_type", "").strip(),
                row.get("clip_uid", "").strip(),
                row.get("other_clip_uid", "").strip(),
            )
            if key in decisions and decisions[key] != decision:
                raise ValueError(f"conflicting manual decisions for {key}")
            decisions[key] = decision
    return decisions


def _component_ids(uids: Iterable[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    adjacency = {uid: set() for uid in uids}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    result = {}
    unseen = set(adjacency)
    while unseen:
        stack = [min(unseen)]
        component = set()
        while stack:
            uid = stack.pop()
            if uid in component:
                continue
            component.add(uid)
            stack.extend(adjacency[uid] - component)
        unseen -= component
        digest = hashlib.sha256("\n".join(sorted(component)).encode()).hexdigest()[:16]
        group_id = f"aa-content-{digest}"
        result.update({uid: group_id for uid in component})
    return result


def _representative(members: list[str], by_id: dict[str, dict[str, Any]]) -> str:
    best = (members[0],)
    for uid in members[1:]:
        best = _better(best, (uid,), by_id)
    return best[0]


def run_finalize(args: argparse.Namespace) -> None:
    output = args.output_dir
    inventory = _read_jsonl(output / "inventory.jsonl")
    if len(inventory) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"expected 5756 inventory rows, found {len(inventory)}")
    local_rows = _read_jsonl(output / "local_reviews.jsonl")
    external_rows = _read_jsonl(output / "external_reviews.jsonl")
    local = {row["clip_uid"]: row for row in local_rows}
    external = {row["clip_uid"]: row for row in external_rows}
    manual = _manual_decisions(args.manual_reviews)
    queue = []
    manifest = []
    by_id = {}
    for row in inventory:
        uid = row["clip_uid"]
        local_review = local.get(uid, {})
        external_review = external.get(uid, {})
        local_metrics = local_review.get("metrics", {})
        external_metrics = external_review.get("metrics", {})
        manual_decision = manual.get(("screening", uid, ""))
        if manual_decision and manual_decision not in {"keep", "reject"}:
            raise ValueError(f"invalid manual screening decision: {manual_decision}")
        status = resolve_screen(
            local_metrics.get("local_decision"),
            external_metrics.get("local_decision"),
            manual_decision,
        )
        if status.endswith("pending"):
            queue.append(
                {
                    "review_type": "screening",
                    "clip_uid": uid,
                    "other_clip_uid": "",
                    "model_status": status,
                    "decision": "",
                    "notes": local_review.get("error")
                    or external_review.get("error")
                    or "",
                }
            )
        payload = local_review.get("payload") or external_review.get("payload") or {}
        metrics = local_metrics or external_metrics
        record = {
            **{key: value for key, value in row.items() if key not in {
                "head_dhash", "tail_dhash", "observed_events", "anticipated_events"
            }},
            "screen_status": status,
            "observability": payload.get("observability"),
            "observability_reason": payload.get("observability_reason", ""),
            "weak_labels": payload.get("weak_labels", []),
            "bad_s": metrics.get("bad_s"),
            "max_continuous_wide_s": metrics.get("max_continuous_wide_s"),
            "scene_cuts_s": local_review.get("scene_cuts_s", []),
            "eligible_positive": (
                status == "keep" and payload.get("observability") == "observable"
            ),
        }
        manifest.append(record)
        by_id[uid] = record

    overlap_rows = _read_jsonl(output / "overlap_groups.jsonl")
    duplicate_map = {}
    exact_edges = []
    for item in overlap_rows:
        if item["type"] != "exact_duplicate":
            continue
        members = item["members"]
        representative = _representative(members, by_id)
        for uid in members:
            duplicate_map[uid] = representative
            if uid != representative:
                exact_edges.append((representative, uid))

    temporal_edges = []
    for item in overlap_rows:
        if item["type"] != "boundary_overlap" or item["status"] == "none":
            continue
        left, right = item["left_uid"], item["right_uid"]
        decision = manual.get(("overlap", left, right))
        if decision is None:
            queue.append(
                {
                    "review_type": "overlap",
                    "clip_uid": left,
                    "other_clip_uid": right,
                    "model_status": item["status"],
                    "decision": "",
                    "notes": (
                        f"high_frames={item['high_frames']};"
                        f"mean_ssim={item['mean_ssim']}"
                    ),
                }
            )
            continue
        if decision not in {"confirm_overlap", "not_overlap"}:
            raise ValueError(f"invalid manual overlap decision: {decision}")
        if decision == "confirm_overlap":
            mapped = (duplicate_map.get(left, left), duplicate_map.get(right, right))
            if mapped[0] != mapped[1]:
                temporal_edges.append(mapped)

    eligible_rows = [
        row for row in manifest
        if row["screen_status"] == "keep"
        and duplicate_map.get(row["clip_uid"], row["clip_uid"]) == row["clip_uid"]
    ]
    eligible_ids = {row["clip_uid"] for row in eligible_rows}
    active_edges = [
        edge for edge in temporal_edges
        if edge[0] in eligible_ids and edge[1] in eligible_ids
    ]
    try:
        selected = select_path(eligible_rows, active_edges)
    except ValueError as exc:
        selected = set()
        queue.append(
            {
                "review_type": "overlap_graph",
                "clip_uid": "",
                "other_clip_uid": "",
                "model_status": "manual_review_pending",
                "decision": "",
                "notes": str(exc),
            }
        )
    group_ids = _component_ids(
        [row["clip_uid"] for row in manifest], exact_edges + temporal_edges
    )
    for row in manifest:
        row["content_group_id"] = group_ids[row["clip_uid"]]
        representative = duplicate_map.get(row["clip_uid"], row["clip_uid"])
        row["duplicate_of"] = (
            representative if representative != row["clip_uid"] else None
        )
        row["selected_candidate"] = (
            row["clip_uid"] in selected if not queue else None
        )
        row["selection_status"] = (
            "pending"
            if queue
            else "selected"
            if row["clip_uid"] in selected
            else "exact_duplicate"
            if representative != row["clip_uid"]
            else row["screen_status"]
            if row["screen_status"] != "keep"
            else "confirmed_overlap_excluded"
        )


    _write_jsonl(output / "screening_manifest.jsonl", manifest, args.resume)
    _write_jsonl(
        output / "model_reviews.jsonl",
        sorted(local_rows + external_rows, key=lambda row: (
            row["clip_uid"], row.get("provider", "")
        )),
        args.resume,
    )
    queue_fields = [
        "review_type", "clip_uid", "other_clip_uid", "model_status",
        "decision", "notes",
    ]
    resolved_rows = [
        {
            "review_type": key[0],
            "clip_uid": key[1],
            "other_clip_uid": key[2],
            "model_status": "manual_resolved",
            "decision": decision,
            "notes": "",
        }
        for key, decision in sorted(manual.items())
    ]
    _write_csv(
        output / "manual_review_queue.csv",
        resolved_rows + queue,
        queue_fields,
        args.resume,
    )

    pending = Counter(row["screen_status"] for row in manifest)
    summary = {
        "dataset": DATASET,
        "inventory_count": len(inventory),
        "screen_status_counts": dict(sorted(pending.items())),
        "overlap_record_count": len(overlap_rows),
        "manual_review_pending_count": len(queue),
        "candidate_count": len(selected) if not queue else None,
        "selection_sha256": (
            hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest()
            if not queue
            else None
        ),
        "limitations": [
            "Camera rules were calibrated only on AA-1 and AA-2.",
            "The full-pool false-rejection rate was not estimated.",
            "Action labels are weak labels, not benchmark gold.",
            "Low/mid-block evidence may be insufficient in a 30-second clip.",
            "Source timelines are absent; overlap chains are content-reconstructed.",
        ],
    }
    _write_json(output / "summary.json", summary, args.resume)
    if queue:
        stale = [
            output / "candidates_blind.csv",
            output / "candidates_blind.jsonl",
            output / "weak_labels_after_round1.jsonl",
        ]
        if any(path.exists() for path in stale):
            raise RuntimeError(
                "pending reviews coexist with prior final candidates; use a clean output dir"
            )
        print(f"finalize blocked: {len(queue)} unresolved review rows")
        return

    candidates = sorted(
        (blind_row(by_id[uid]) for uid in selected),
        key=lambda row: row["clip_uid"],
    )
    _write_jsonl(output / "candidates_blind.jsonl", candidates, args.resume)
    candidate_fields = list(candidates[0]) if candidates else [
        "video_id", "clip_uid", "dataset", "source_split", "video_path",
        "fps", "resolution",
    ]
    _write_csv(
        output / "candidates_blind.csv", candidates, candidate_fields, args.resume
    )
    _write_jsonl(
        output / "weak_labels_after_round1.jsonl",
        [weak_row(by_id[uid]) for uid in sorted(selected)],
        args.resume,
    )
    print(f"wrote {len(candidates)} zero-overlap blind candidates")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/SoccerNet-AA"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/soccernet_aa_screening"),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse completed rows and explicitly replace stage artifacts",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory")

    local = subparsers.add_parser("local-screen")
    local.add_argument(
        "--local-model",
        default="/data/zhangzheng/pretrained_models/jn/Qwen2.5-VL-7B-Instruct",
    )
    local.add_argument("--clip-uid", action="append")

    external = subparsers.add_parser("external-review")
    external.add_argument("--provider", choices=["doubao"], required=True)
    external.add_argument("--allow-external-upload", action="store_true")
    external.add_argument("--clip-uid", action="append")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--manual-reviews", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    commands = {
        "inventory": run_inventory,
        "local-screen": run_local_screen,
        "external-review": run_external_review,
        "finalize": run_finalize,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
