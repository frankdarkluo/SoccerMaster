#!/usr/bin/env python3
"""
pitch_distances.py — 计算足球场上球员之间、球员到场地边界的物理距离（米）。

两种输入模式：
  1) GT 模式 (--mode gt)：直接读 SoccerNet GSR 的 Labels-GameState.json 里
     每个球员的 bbox_pitch（已是米为单位的球场坐标）。用于在 GSR 上验证。
  2) Calib 模式 (--mode calib)：没有 bbox_pitch 时（你自己的视频），用相机标定
     得到的 homography 把 bbox_image 的脚点投影到球场平面。需要一个
     {image_id: 3x3 H} 的 json（H 把球场坐标[米] 映射到图像像素）。

坐标系约定（从 GSR 数据反推，原点=中圈点）：
    x = 长轴，球门线在 x = ±PITCH_LENGTH/2
    y = 宽轴，边线在   y = ±PITCH_WIDTH/2
若你的标定输出用了别的约定，改 PITCH_* 常量和 boundary 计算即可。

示例：
  # 单个序列
  python pitch_distances.py \\
    codes/sn-gamestate/datasets/SoccerNetGS/train/SNGS-021/Labels-GameState.json

  # 整个 train split（默认输出到 outputs/pitch_distances/train/）
  python pitch_distances.py --split train

  # calib 模式：需提供 homography json，或从同一份 GSR labels 自动估计（验证用）
  python pitch_distances.py \\
    codes/sn-gamestate/datasets/SoccerNetGS/train/SNGS-061/Labels-GameState.json \\
    --mode calib --homography-from-labels
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PITCH_LENGTH = 105.0   # m, 长边（goal line 到 goal line）
PITCH_WIDTH = 68.0     # m, 短边（touch line 到 touch line）

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "codes/sn-gamestate/datasets/SoccerNetGS"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/pitch_distances"

PLAYER_ROLES = {"player", "goalkeeper"}   # 算球员间距时纳入的角色


# ---------- 几何 ----------
def image_to_pitch(u, v, H):
    """图像点 (u,v) -> 球场坐标 (x,y) 米。H: 球场->图像 的 3x3 homography。"""
    a, b, c = H
    det = (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0])
           + a[2]*(b[0]*c[1]-b[1]*c[0]))
    inv = [
        [(b[1]*c[2]-b[2]*c[1])/det, (a[2]*c[1]-a[1]*c[2])/det, (a[1]*b[2]-a[2]*b[1])/det],
        [(b[2]*c[0]-b[0]*c[2])/det, (a[0]*c[2]-a[2]*c[0])/det, (a[2]*b[0]-a[0]*b[2])/det],
        [(b[0]*c[1]-b[1]*c[0])/det, (a[1]*c[0]-a[0]*c[1])/det, (a[0]*b[1]-a[1]*b[0])/det],
    ]
    x = inv[0][0]*u + inv[0][1]*v + inv[0][2]
    y = inv[1][0]*u + inv[1][1]*v + inv[1][2]
    w = inv[2][0]*u + inv[2][1]*v + inv[2][2]
    return x/w, y/w


def boundary_distances(x, y, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """球场点到四条边界的距离（米），原点在中心。"""
    d_goal = length/2 - abs(x)
    d_touch = width/2 - abs(y)
    return {
        "to_goal_line": round(d_goal, 3),
        "to_touch_line": round(d_touch, 3),
        "to_nearest_boundary": round(min(d_goal, d_touch), 3),
        "is_off_pitch": (d_goal < 0) or (d_touch < 0),
    }


def euclidean(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])


def foot_pitch_coord(ann, mode, H_by_image):
    """返回 (x, y) 球场坐标，米；取不到返回 None。"""
    if mode == "gt":
        bp = ann.get("bbox_pitch")
        if not bp:
            return None
        return bp["x_bottom_middle"], bp["y_bottom_middle"]
    bi = ann.get("bbox_image")
    if not bi:
        return None
    u = bi.get("x_center", bi["x"] + bi["w"]/2.0)
    vfoot = bi["y"] + bi["h"]
    H = H_by_image.get(str(ann["image_id"]))
    if H is None:
        return None
    return image_to_pitch(u, vfoot, H)


def build_homography_from_labels(doc, min_points=4):
    """从 GSR labels 的 bbox_image / bbox_pitch 脚点对应关系估计每帧 H（pitch->image）。"""
    pitch_pts_by_image = defaultdict(list)
    image_pts_by_image = defaultdict(list)

    for ann in doc["annotations"]:
        if ann.get("supercategory") not in (None, "object"):
            continue
        bi = ann.get("bbox_image")
        bp = ann.get("bbox_pitch")
        if not bi or not bp:
            continue
        u = bi.get("x_center", bi["x"] + bi["w"] / 2.0)
        vfoot = bi["y"] + bi["h"]
        x = bp["x_bottom_middle"]
        y = bp["y_bottom_middle"]
        iid = str(ann["image_id"])
        pitch_pts_by_image[iid].append([x, y])
        image_pts_by_image[iid].append([u, vfoot])

    H_by_image = {}
    skipped = 0
    for iid in pitch_pts_by_image:
        pitch_pts = np.float32(pitch_pts_by_image[iid])
        image_pts = np.float32(image_pts_by_image[iid])
        if len(pitch_pts) < min_points:
            skipped += 1
            continue
        H, _ = cv2.findHomography(pitch_pts, image_pts, method=0)
        if H is not None:
            H_by_image[iid] = H.tolist()

    print(
        f"Built homography for {len(H_by_image)} frames "
        f"({skipped} frames skipped: <{min_points} correspondences)"
    )
    return H_by_image


def load_homography_map(args, labels_path=None):
    if args.homography_from_labels:
        if not labels_path:
            ap = argparse.ArgumentParser()
            ap.error("--homography-from-labels 需要配合 labels 文件路径")
        with open(labels_path, encoding="utf-8") as f:
            doc = json.load(f)
        return build_homography_from_labels(doc)

    if not args.homography:
        raise SystemExit(
            "calib 模式需要 --homography <path.json>，"
            "或对 GSR labels 使用 --homography-from-labels"
        )

    homography_path = Path(args.homography)
    if not homography_path.is_file():
        raise SystemExit(f"homography 文件不存在: {homography_path}")

    with open(homography_path, encoding="utf-8") as f:
        return json.load(f)


def discover_label_files(split_dir):
    """Find Labels-GameState.json under SNGS-* sequence folders."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    label_files = sorted(split_dir.glob("SNGS-*/Labels-GameState.json"))
    if label_files:
        return label_files

    # Fallback: any direct child with labels file
    label_files = sorted(split_dir.glob("*/Labels-GameState.json"))
    if label_files:
        return label_files

    raise FileNotFoundError(
        f"No Labels-GameState.json found under {split_dir}/SNGS-*/"
    )


def sequence_name_from_labels(labels_path):
    labels_path = Path(labels_path)
    if labels_path.parent.name.startswith("SNGS-"):
        return labels_path.parent.name
    with open(labels_path, encoding="utf-8") as f:
        doc = json.load(f)
    return doc.get("info", {}).get("name") or labels_path.parent.name


def process_labels(labels_path, out_prefix, mode, H_by_image, length, width, sequence=None):
    labels_path = Path(labels_path)
    sequence = sequence or sequence_name_from_labels(labels_path)

    with open(labels_path, encoding="utf-8") as f:
        doc = json.load(f)
    anns = doc["annotations"]

    frames = defaultdict(list)
    for a in anns:
        role = (a.get("attributes") or {}).get("role")
        if role is None:
            continue
        xy = foot_pitch_coord(a, mode, H_by_image)
        if xy is None:
            continue
        frames[str(a["image_id"])].append({
            "track_id": a.get("track_id"),
            "role": role,
            "team": (a.get("attributes") or {}).get("team"),
            "jersey": (a.get("attributes") or {}).get("jersey"),
            "x": round(xy[0], 3), "y": round(xy[1], 3),
        })

    rows = []
    pair_rows = []
    for image_id in sorted(frames, key=lambda s: int(s)):
        objs = frames[image_id]
        players = [o for o in objs if o["role"] in PLAYER_ROLES]
        for o in objs:
            bd = boundary_distances(o["x"], o["y"], length, width)
            nearest_opp = None
            if o["role"] in PLAYER_ROLES and o["team"]:
                opp = [p for p in players
                       if p["team"] and p["team"] != o["team"]
                       and p["track_id"] != o["track_id"]]
                if opp:
                    nearest_opp = round(min(euclidean((o["x"], o["y"]), (p["x"], p["y"]))
                                            for p in opp), 3)
            rows.append({
                "sequence": sequence,
                "image_id": image_id, "track_id": o["track_id"], "role": o["role"],
                "team": o["team"], "jersey": o["jersey"], "x": o["x"], "y": o["y"],
                **bd, "nearest_opponent_m": nearest_opp,
            })
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                a_, b_ = players[i], players[j]
                pair_rows.append({
                    "sequence": sequence,
                    "image_id": image_id,
                    "track_a": a_["track_id"], "team_a": a_["team"],
                    "track_b": b_["track_id"], "team_b": b_["team"],
                    "distance_m": round(euclidean((a_["x"], a_["y"]), (b_["x"], b_["y"])), 3),
                })

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_player_csv = out_prefix.with_name(out_prefix.name + "_per_player.csv")
    pairs_csv = out_prefix.with_name(out_prefix.name + "_pairs.csv")

    if rows:
        with open(per_player_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        per_player_csv.write_text("sequence,image_id,track_id,role,team,jersey,x,y,"
                                  "to_goal_line,to_touch_line,to_nearest_boundary,"
                                  "is_off_pitch,nearest_opponent_m\n")

    if pair_rows:
        with open(pairs_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            w.writerows(pair_rows)
    else:
        pairs_csv.write_text("sequence,image_id,track_a,team_a,track_b,team_b,distance_m\n")

    print(f"[{sequence}] labels: {labels_path}")
    print(f"[{sequence}] frames with data : {len(frames)}")
    print(f"[{sequence}] player rows       : {len(rows)} -> {per_player_csv}")
    print(f"[{sequence}] pairwise rows     : {len(pair_rows)} -> {pairs_csv}")

    if frames:
        fid = sorted(frames, key=lambda s: int(s))[0]
        print(f"[{sequence}] sample frame {fid}: {len(frames[fid])} objects")
        for o in frames[fid][:4]:
            bd = boundary_distances(o["x"], o["y"], length, width)
            jersey = o["jersey"] if o["jersey"] is not None else "-"
            print(f"  #{str(jersey):>2} {o['role']:<10} {o['team'] or '-':<5} "
                  f"pitch=({o['x']:7.2f},{o['y']:7.2f})  "
                  f"goal_line={bd['to_goal_line']:6.2f}m touch_line={bd['to_touch_line']:6.2f}m")

    return per_player_csv, pairs_csv


def main():
    ap = argparse.ArgumentParser(description="Per-frame player distances on a soccer pitch.")
    ap.add_argument(
        "labels",
        nargs="?",
        help="Labels-GameState.json（单序列模式）",
    )
    ap.add_argument(
        "--split",
        choices=["train", "valid", "test", "challenge", "sn500"],
        help="批量处理 {dataset_root}/{split}/ 下所有 SNGS-* 序列",
    )
    ap.add_argument(
        "--split-dir",
        type=Path,
        help="批量处理指定目录（优先于 --split）",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"GSR 数据集根目录（默认: {DEFAULT_DATASET_ROOT}）",
    )
    ap.add_argument("--mode", choices=["gt", "calib"], default="gt")
    ap.add_argument(
        "--homography",
        metavar="PATH",
        help="calib 模式: {image_id: 3x3 H} 的 json 文件路径（H: 球场米坐标 -> 图像像素）",
    )
    ap.add_argument(
        "--homography-from-labels",
        action="store_true",
        help="calib 模式: 从同一份 GSR labels 的 bbox_image/bbox_pitch 自动估计 H（验证用，非推理场景）",
    )
    ap.add_argument(
        "--write-homography",
        type=Path,
        help="将估计/加载的 homography 写到 json 后退出（需配合 --homography-from-labels）",
    )
    ap.add_argument(
        "--out-prefix",
        help="单序列输出前缀（默认: {out_dir}/distances/<sequence>/distances）",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"批量输出根目录（默认: {DEFAULT_OUT_DIR}）",
    )
    ap.add_argument("--length", type=float, default=PITCH_LENGTH)
    ap.add_argument("--width", type=float, default=PITCH_WIDTH)
    args = ap.parse_args()

    H_by_image = {}
    if args.mode == "calib":
        if args.split_dir or args.split:
            ap.error("calib 模式暂不支持 --split 批量；请单序列运行并提供 homography")
        if not args.labels:
            ap.error("calib 模式请提供 labels 路径")
        H_by_image = load_homography_map(args, labels_path=args.labels)
        if args.write_homography:
            args.write_homography.parent.mkdir(parents=True, exist_ok=True)
            with open(args.write_homography, "w", encoding="utf-8") as f:
                json.dump(H_by_image, f)
            print(f"Wrote homography map -> {args.write_homography}")
            return

    if args.split_dir:
        split_dir = args.split_dir
        split_name = split_dir.name
    elif args.split:
        split_dir = args.dataset_root / args.split
        split_name = args.split
    elif args.labels:
        labels_path = Path(args.labels)
        sequence = sequence_name_from_labels(labels_path)
        out_prefix = Path(args.out_prefix) if args.out_prefix else args.out_dir / sequence / "distances"
        process_labels(
            labels_path, out_prefix, args.mode, H_by_image, args.length, args.width, sequence=sequence
        )
        return
    else:
        ap.error("请提供 labels 路径，或使用 --split / --split-dir 批量处理")

    label_files = discover_label_files(split_dir)
    out_root = args.out_dir / split_name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(label_files)} sequences under {split_dir}")
    print(f"Writing outputs to {out_root}")

    for labels_path in label_files:
        sequence = sequence_name_from_labels(labels_path)
        out_prefix = out_root / sequence / "distances"
        process_labels(
            labels_path, out_prefix, args.mode, H_by_image, args.length, args.width, sequence=sequence
        )
        print()


if __name__ == "__main__":
    main()
