"""Recover missing track-level jersey numbers from a few best-view crops."""

import argparse
import json
import pickle
import re
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image


def load_state(path, video_id):
    with zipfile.ZipFile(path) as archive:
        detections = pickle.load(archive.open(f"{video_id}.pkl"))
        images = pickle.load(archive.open(f"{video_id}_image.pkl"))
    return detections, images


def crop_image(row, scale=4):
    image = cv2.imread(row.file_path)
    left, top, width, height = map(float, row.bbox_ltwh)
    x1, y1 = max(0, round(left)), max(0, round(top))
    x2 = min(image.shape[1], round(left + width))
    y2 = min(image.shape[0], round(top + height))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def select_frames(track, image_paths, count=9, min_gap=5):
    candidates = []
    for index, row in track.iterrows():
        crop = crop_image(SimpleNamespace(file_path=image_paths[int(row.image_id)], bbox_ltwh=row.bbox_ltwh), 1)
        if crop is None:
            continue
        sharpness = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        area = float(row.bbox_ltwh[2] * row.bbox_ltwh[3])
        candidates.append((area * np.log1p(sharpness), int(row.image_id), index))
    selected = []
    for score, image_id, index in sorted(candidates, reverse=True):
        frame = image_id % 1_000_000
        if all(abs(frame - old[1] % 1_000_000) >= min_gap for old in selected):
            selected.append((score, image_id, index))
        if len(selected) == count:
            break
    return selected


def parse_number(text):
    numbers = re.findall(r"(?<!\d)\d{1,2}(?!\d)", text)
    return numbers[0] if len(numbers) == 1 else None


def choose_vote(numbers, total):
    votes = Counter(number for number in numbers if number is not None)
    if not votes:
        return None, votes
    winner, count = votes.most_common(1)[0]
    runner_up = votes.most_common(2)[1][1] if len(votes) > 1 else 0
    accepted = count >= 3 and count > runner_up and count / sum(votes.values()) >= 0.6 and count / total >= 1 / 3
    return winner if accepted else None, votes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-id", default="003")
    parser.add_argument("--model", default="/data/zhangzheng/pretrained_models/jn/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--audit-track", type=int, action="append", default=[2])
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    detections, images = load_state(args.state, args.video_id)
    players = detections[detections.role == "player"]
    missing = [int(track_id) for track_id, track in players.groupby("track_id") if track.jersey_number.isna().all()]
    targets = sorted(set(missing + args.audit_track))
    image_paths = {int(image_id): path for image_id, path in zip(images.id, images.file_path)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evidence = {}
    crops = []
    crop_refs = []
    for track_id in targets:
        track = players[players.track_id == track_id]
        evidence[str(track_id)] = {"audit_only": track_id not in missing, "frames": []}
        track_dir = args.output_dir / "crops" / f"track_{track_id}"
        track_dir.mkdir(parents=True, exist_ok=True)
        for score, image_id, index in select_frames(track, image_paths, args.frames):
            row = players.loc[index]
            crop = crop_image(SimpleNamespace(file_path=image_paths[image_id], bbox_ltwh=row.bbox_ltwh))
            crop_path = track_dir / f"frame_{image_id % 1_000_000:06d}.jpg"
            cv2.imwrite(str(crop_path), crop)
            item = {"frame": image_id % 1_000_000, "score": round(float(score), 2), "crop": str(crop_path)}
            evidence[str(track_id)]["frames"].append(item)
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            crop_refs.append((track_id, item))

    if not args.select_only:
        import transformers
        from transformers import Qwen2_5_VLForConditionalGeneration

        # ponytail: remove this compatibility shim after Transformers provides Qwen3-VL.
        transformers.Qwen2_5VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration
        transformers.Qwen3VLMoeForConditionalGeneration = Qwen2_5_VLForConditionalGeneration
        from sn_gamestate.jersey_and_role.qwen2_5vl_jn_and_role_api import QWEN2_5VL_JN_AND_ROLE_BATCH

        cfg = SimpleNamespace(
            model_path=args.model,
            use_vllm=False,
            save_jersey_number_full_detection=False,
            use_legibility_filter=False,
            image_width=1920,
            image_height=1080,
            downsample_factor=2,
            max_new_tokens=16,
        )
        model = QWEN2_5VL_JN_AND_ROLE_BATCH(cfg, batch_size=4, device="cuda")
        prompt = "Read the player's jersey number. Reply with only a number from 0 to 99, or No if it is not clearly visible. Do not guess."
        for start in range(0, len(crops), 4):
            messages = [[{"role": "user", "content": [{"type": "image", "image": crop}, {"type": "text", "text": prompt}]}] for crop in crops[start : start + 4]]
            for (track_id, item), response in zip(crop_refs[start : start + 4], model._generate_texts(messages)):
                item["response"] = response.strip()
                item["number"] = parse_number(response)

        predictions = json.loads(args.predictions.read_text())
        accepted = {}
        for track_id in targets:
            item = evidence[str(track_id)]
            winner, votes = choose_vote([frame.get("number") for frame in item["frames"]], len(item["frames"]))
            item["votes"] = dict(votes)
            item["accepted"] = winner
            if winner is not None and not item["audit_only"]:
                accepted[track_id] = winner
        for annotation in predictions["annotations"]:
            track_id = int(annotation["track_id"])
            if track_id in accepted and annotation["attributes"].get("role") == "player" and not annotation["attributes"].get("jersey"):
                annotation["attributes"]["jersey"] = accepted[track_id]
        (args.output_dir / "predictions_jersey_repair.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False))

    (args.output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
    print(json.dumps({track_id: {"votes": item.get("votes", {}), "accepted": item.get("accepted")} for track_id, item in evidence.items()}, indent=2))


if __name__ == "__main__":
    main()
