#!/usr/bin/env python3
"""
viz_minimap.py — 俯视小地图距离可视化（转播式动态焦点）。

消费 pitch_distances.py 输出的 <seq>_per_player.csv，产出：
  1) <seq>.mp4                     —— 整段俯视小地图动画，逐帧跟拍持球者
  2) <seq>_boundary_distances.csv  —— 同目录、全量、每帧每人到边界的距离
  3) <seq>_frame_<id>.png          —— 可选，--image-id 抽检单帧

不修改 pitch_distances.py，不依赖原始帧图像。坐标系约定与 pitch_distances.py 一致：
原点=中圈点，x=长轴（球门线 ±length/2），y=宽轴（边线 ±width/2）。

示例：
  # 动态焦点（默认）：每帧跟离球最近的球员
  python viz_minimap.py outputs/pitch_distances/SNGS-060/distances_per_player.csv

  # 固定焦点：整段跟某个 track_id
  python viz_minimap.py distances_per_player.csv --focus 7

  # 控体积/速度：每 2 帧取一帧，15fps
  python viz_minimap.py distances_per_player.csv --stride 2 --fps 15

  # 只抽检某一帧出 PNG
  python viz_minimap.py distances_per_player.csv --image-id 1060000001
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "codes/sn-gamestate/datasets/SoccerNetGS"
DEFAULT_PITCH_DIST_DIR = REPO_ROOT / "outputs/pitch_distances"

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PLAYER_ROLES = {"player", "goalkeeper"}

TEAM_COLOR = {"left": "#d6342c", "right": "#2c6fd6"}
REF_COLOR = "#e8902e"
BALL_COLOR = "#ffffff"
LINE_BOUNDARY = "#ffd23f"   # 黄：到边界
LINE_OPP = "#ffffff"        # 白：到最近对手
FIELD_GREEN = "#2e8b3d"

# boundary CSV 想要的列（spec §5.2）。sequence 若存在则带上，方便后期合并。
BOUNDARY_COLS = ["sequence", "image_id", "track_id", "role", "team", "jersey",
                 "x", "y", "to_goal_line", "to_touch_line", "to_nearest_boundary"]


# ---------- 几何 ----------
def nearest_boundary(x, y, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """返回 (边界点坐标(bx,by), 距离米)。x/y 在以中圈为原点的米坐标系。"""
    d_goal = length / 2 - abs(x)
    d_touch = width / 2 - abs(y)
    if d_goal <= d_touch:
        return (math.copysign(length / 2, x), y), d_goal
    return (x, math.copysign(width / 2, y)), d_touch


def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


# ---------- 读 CSV ----------
def load_frames(csv_path):
    """读 _per_player.csv -> (frames: {image_id:[obj,...]}, sorted_ids, header_has_sequence)。"""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        frames = defaultdict(list)
        for r in reader:
            try:
                x = float(r["x"]); y = float(r["y"])
            except (KeyError, ValueError, TypeError):
                continue
            frames[r["image_id"]].append({
                "sequence": r.get("sequence", ""),
                "image_id": r["image_id"],
                "track_id": r.get("track_id", ""),
                "role": r.get("role", ""),
                "team": (r.get("team") or "").strip() or None,
                "jersey": (r.get("jersey") or "").strip(),
                "x": x, "y": y,
                "to_goal_line": r.get("to_goal_line", ""),
                "to_touch_line": r.get("to_touch_line", ""),
                "to_nearest_boundary": r.get("to_nearest_boundary", ""),
            })
    sorted_ids = sorted(frames, key=lambda s: int(s) if s.isdigit() else s)
    return frames, sorted_ids, ("sequence" in fields)


# ---------- 写同目录 boundary CSV ----------
def write_boundary_csv(frames, sorted_ids, out_path, has_sequence):
    cols = BOUNDARY_COLS if has_sequence else BOUNDARY_COLS[1:]
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for image_id in sorted_ids:
            for o in frames[image_id]:
                w.writerow({c: o.get(c, "") for c in cols})
                n += 1
    return n


# ---------- 焦点选择 ----------
def pick_focus(objs, mode_focus_track, last_focus_track):
    """返回 (focus_obj 或 None, focus_track_id 或 None)。
       mode_focus_track: 固定模式给的 track_id（字符串）或 None。"""
    players = [o for o in objs if o["role"] in PLAYER_ROLES]
    if not players:
        return None, last_focus_track

    # 固定焦点模式
    if mode_focus_track is not None:
        for o in players:
            if str(o["track_id"]) == str(mode_focus_track):
                return o, mode_focus_track
        return None, mode_focus_track  # 本帧不在场

    # 动态焦点：离球最近
    ball = next((o for o in objs if o["role"] == "ball"), None)
    if ball is not None:
        focus = min(players, key=lambda o: dist((o["x"], o["y"]), (ball["x"], ball["y"])))
        return focus, focus["track_id"]

    # 本帧无球：carry-forward 上一帧持球者
    if last_focus_track is not None:
        for o in players:
            if str(o["track_id"]) == str(last_focus_track):
                return o, last_focus_track
    # 整段开头就无球：用第一个球员
    return players[0], players[0]["track_id"]


# ---------- 画场地 ----------
def draw_pitch(ax, length=PITCH_LENGTH, width=PITCH_WIDTH):
    hl, hw = length / 2, width / 2
    ax.add_patch(Rectangle((-hl - 4, -hw - 4), length + 8, width + 8,
                           fc=FIELD_GREEN, ec="none", zorder=0))
    ax.add_patch(Rectangle((-hl, -hw), length, width, fill=False, ec="white", lw=1.6, zorder=1))
    ax.plot([0, 0], [-hw, hw], color="white", lw=1.4, zorder=1)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, ec="white", lw=1.4, zorder=1))
    ax.plot(0, 0, "o", color="white", ms=3, zorder=1)
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * hl, -20.16), -s * 16.5, 40.32, fill=False, ec="white", lw=1.3, zorder=1))
        ax.add_patch(Rectangle((s * hl, -9.16), -s * 5.5, 18.32, fill=False, ec="white", lw=1.3, zorder=1))
        ax.add_patch(Rectangle((s * hl, -3.66), s * 2.0, 7.32, fill=False, ec="white", lw=1.3, zorder=1))
        ax.plot(s * (hl - 11), 0, "o", color="white", ms=2, zorder=1)
    ax.set_xlim(-hl - 5, hl + 5)
    ax.set_ylim(-hw - 5, hw + 5)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------- 画一帧 ----------
def render_frame(ax, objs, focus, frame_no, length, width):
    ax.clear()
    draw_pitch(ax, length, width)

    players = [o for o in objs if o["role"] in PLAYER_ROLES]
    focus_id = str(focus["track_id"]) if focus else None

    # 所有对象
    for o in objs:
        if o["role"] == "ball":
            ax.plot(o["x"], o["y"], "o", color=BALL_COLOR, ms=7, mec="black", mew=0.8, zorder=6)
            continue
        color = TEAM_COLOR.get(o["team"], REF_COLOR if o["role"] == "referee" else "#888888")
        is_focus = focus_id is not None and str(o["track_id"]) == focus_id
        if is_focus:
            continue  # 焦点最后单独高亮画
        ax.plot(o["x"], o["y"], "o", color=color, ms=8, alpha=0.32, zorder=2)

    # 焦点高亮 + 距离线
    hud = f"frame {frame_no}"
    if focus is not None:
        fcolor = TEAM_COLOR.get(focus["team"], REF_COLOR)
        # 到最近边界（黄）
        (bx, by), bd = nearest_boundary(focus["x"], focus["y"], length, width)
        ax.plot([focus["x"], bx], [focus["y"], by], color=LINE_BOUNDARY, lw=2.4, zorder=4)
        ax.text((focus["x"] + bx) / 2, (focus["y"] + by) / 2, f"{bd:.1f} m",
                color=LINE_BOUNDARY, fontsize=9, fontweight="bold",
                ha="center", va="bottom", zorder=7)
        # 到最近对手（白）
        opps = [p for p in players if p["team"] and focus["team"]
                and p["team"] != focus["team"] and str(p["track_id"]) != focus_id]
        if opps:
            nopp = min(opps, key=lambda p: dist((p["x"], p["y"]), (focus["x"], focus["y"])))
            dopp = dist((nopp["x"], nopp["y"]), (focus["x"], focus["y"]))
            ax.plot([focus["x"], nopp["x"]], [focus["y"], nopp["y"]],
                    color=LINE_OPP, lw=1.8, zorder=4)
            ax.text((focus["x"] + nopp["x"]) / 2, (focus["y"] + nopp["y"]) / 2,
                    f"{dopp:.1f} m", color=LINE_OPP, fontsize=8, fontweight="bold",
                    ha="center", va="bottom", zorder=7)
        # 焦点点
        ax.plot(focus["x"], focus["y"], "o", color=fcolor, ms=15, mec="white", mew=2, zorder=8)
        if focus["jersey"]:
            ax.text(focus["x"], focus["y"], focus["jersey"], color="white",
                    ha="center", va="center", fontsize=8, fontweight="bold", zorder=9)
        jn = focus["jersey"] or focus["track_id"]
        hud = f"frame {frame_no}   focus #{jn} ({focus['team'] or '-'})   boundary {bd:.1f} m"

    # HUD
    ax.text(0.01, 0.99, hud, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color="#111",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75), zorder=10)


# ---------- 批量发现 ----------
def discover_per_player_csvs(split_dir, pitch_dist_dir, split_name):
    """从数据集 split 目录或 pitch_distances 输出目录发现 *_per_player.csv。"""
    split_dir = Path(split_dir)
    pitch_dist_dir = Path(pitch_dist_dir)

    # pitch_distances 批量输出布局: {pitch_dist_dir}/{split}/{SNGS-xxx}/distances_per_player.csv
    if split_dir.is_dir() and sorted(split_dir.glob("SNGS-*/distances_per_player.csv")):
        return sorted(split_dir.glob("SNGS-*/distances_per_player.csv"))

    # 数据集 split 目录: 按 SNGS-* 子目录拼 pitch_distances 路径
    seq_dirs = sorted(split_dir.glob("SNGS-*"))
    if not seq_dirs:
        raise FileNotFoundError(f"No SNGS-* sequences under {split_dir}")

    csvs = []
    missing = []
    for seq_dir in seq_dirs:
        csv_path = pitch_dist_dir / split_name / seq_dir.name / "distances_per_player.csv"
        if csv_path.is_file():
            csvs.append(csv_path)
        else:
            missing.append(csv_path)
    if missing and not csvs:
        raise FileNotFoundError(
            f"No distances_per_player.csv found. Example missing: {missing[0]}\n"
            f"Run: python pitch_distances.py --split {split_name}"
        )
    for p in missing:
        print(f"[skip] missing input: {p}", file=sys.stderr)
    return csvs


# ---------- 单序列处理 ----------
def process_csv(csv_path, out_dir, fps, stride, focus, image_id, length, width):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, sorted_ids, has_seq = load_frames(csv_path)
    if not sorted_ids:
        raise ValueError(f"{csv_path} 里没有可用数据")

    seq = ""
    for o in frames[sorted_ids[0]]:
        if o.get("sequence"):
            seq = o["sequence"]
            break
    if not seq:
        seq = csv_path.parent.name if csv_path.parent.name.startswith("SNGS-") else (
            csv_path.stem.replace("_per_player", "") or "sequence"
        )

    bcsv = out_dir / f"{seq}_boundary_distances.csv"
    n = write_boundary_csv(frames, sorted_ids, bcsv, has_seq)
    print(f"[{seq}] boundary CSV: {n} rows -> {bcsv}")

    fig, ax = plt.subplots(figsize=(10, 6.8), dpi=120)

    if image_id is not None:
        if image_id not in frames:
            raise ValueError(f"--image-id {image_id} 不在 {seq} 数据中")
        objs = frames[image_id]
        foc, _ = pick_focus(objs, focus, None)
        render_frame(ax, objs, foc, image_id, length, width)
        png = out_dir / f"{seq}_frame_{image_id}.png"
        fig.tight_layout()
        fig.savefig(png, facecolor="white")
        plt.close(fig)
        print(f"[{seq}] PNG -> {png}")
        return

    anim_ids = sorted_ids[::max(1, stride)]
    state = {"last_focus": None}

    def update(i):
        fid = anim_ids[i]
        objs = frames[fid]
        foc, ft = pick_focus(objs, focus, state["last_focus"])
        state["last_focus"] = ft
        render_frame(ax, objs, foc, fid, length, width)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(anim_ids), blit=False)
    mp4 = out_dir / f"{seq}.mp4"
    writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
    print(f"[{seq}] rendering {len(anim_ids)} frames @ {fps}fps (stride={stride}) ...")
    anim.save(str(mp4), writer=writer)
    plt.close(fig)
    print(f"[{seq}] MP4 -> {mp4}")


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="俯视小地图距离可视化（转播式动态焦点）。")
    ap.add_argument("per_player_csv", nargs="?", help="pitch_distances.py 输出的 *_per_player.csv")
    ap.add_argument("--out-dir", type=Path, default=None, help="输出目录（默认: 输入 CSV 同目录）")
    ap.add_argument("--fps", type=int, default=25, help="MP4 帧率（默认 25）")
    ap.add_argument("--stride", type=int, default=1, help="每 N 帧取一帧（默认 1）")
    ap.add_argument("--focus", default=None, help="固定焦点的 track_id；不给走动态持球者")
    ap.add_argument("--image-id", default=None, help="只渲染该帧出 PNG 抽检")
    ap.add_argument("--length", type=float, default=PITCH_LENGTH)
    ap.add_argument("--width", type=float, default=PITCH_WIDTH)
    ap.add_argument(
        "--split",
        choices=["train", "valid", "test", "challenge", "sn500"],
        help="批量处理 {dataset_root}/{split}/ 下所有 SNGS-*（需先有 pitch_distances 输出）",
    )
    ap.add_argument(
        "--split-dir",
        type=Path,
        help="批量处理指定目录（数据集 split 或 pitch_distances/{split} 输出目录）",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"GSR 数据集根目录（默认: {DEFAULT_DATASET_ROOT}）",
    )
    ap.add_argument(
        "--pitch-distances-dir",
        type=Path,
        default=DEFAULT_PITCH_DIST_DIR,
        help=f"pitch_distances.py 输出根目录（默认: {DEFAULT_PITCH_DIST_DIR}）",
    )
    args = ap.parse_args()

    render_kw = dict(
        fps=args.fps, stride=args.stride, focus=args.focus, image_id=args.image_id,
        length=args.length, width=args.width,
    )

    if args.split_dir or args.split:
        if args.split_dir:
            split_dir = args.split_dir
            split_name = split_dir.name
        else:
            split_dir = args.dataset_root / args.split
            split_name = args.split

        csv_files = discover_per_player_csvs(split_dir, args.pitch_distances_dir, split_name)
        print(f"Processing {len(csv_files)} sequences")
        failed = []
        for csv_path in csv_files:
            out_dir = args.out_dir or csv_path.parent
            try:
                process_csv(csv_path, out_dir, **render_kw)
            except Exception as e:
                print(f"[ERROR] {csv_path}: {e}", file=sys.stderr)
                failed.append((csv_path, e))
            print()
        if failed:
            print(f"Failed {len(failed)}/{len(csv_files)} sequences", file=sys.stderr)
            sys.exit(1)
        return

    if not args.per_player_csv:
        ap.error("请提供 per_player_csv，或使用 --split / --split-dir 批量处理")

    csv_path = Path(args.per_player_csv)
    out_dir = args.out_dir or csv_path.parent
    process_csv(csv_path, out_dir, **render_kw)


if __name__ == "__main__":
    main()