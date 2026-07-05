import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle, Polygon, Rectangle
from scipy.spatial import ConvexHull, QhullError

from .io_gamestate import Detection, load_detections
from .lines import detect_lines
from .pitch import HALF_LENGTH, HALF_WIDTH, canonicalize
from .pipeline import infer_attack_dirs

TEAM_COLOR = {"left": "#d6342c", "right": "#2c6fd6"}
BALL_COLOR = "#ffffff"
FIELD_GREEN = "#2e8b3d"


def sequence_name_from_labels(labels_path):
    labels_path = Path(labels_path)
    return labels_path.parent.name or labels_path.stem


def parse_keyframes(value):
    if value is None:
        return ()
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def default_topology_path(labels_path):
    sequence_name = sequence_name_from_labels(labels_path)
    return Path("outputs") / "formation_topology" / sequence_name / "topo.json"


def _format_second(second):
    if float(second).is_integer():
        return f"{int(second):02d}"
    return str(second).rstrip("0").rstrip(".")


def output_paths(output_dir, sequence_name, keyframes):
    output_dir = Path(output_dir)
    video_path = output_dir / f"{sequence_name}_topology_topdown.mp4"
    png_paths = [
        output_dir / f"{sequence_name}_topology_t{_format_second(second)}s.png"
        for second in keyframes
    ]
    return video_path, png_paths


def records_at_time(records, t):
    """Pick the topology record whose window center is nearest to time t per team."""
    by_team = defaultdict(list)
    for record in records:
        by_team[record["team"]].append(record)

    selected = {}
    for team, rows in by_team.items():
        selected[team] = min(
            rows,
            key=lambda row: abs(((row["t_start"] + row["t_end"]) / 2.0) - t),
        )
    return selected


def group_depth_lines(raw_points, attack_dir, gap_delta=7.0, max_lines=3):
    """Return raw-coordinate point groups ordered from deepest to highest."""
    raw_points = np.asarray(raw_points, dtype=float).reshape(-1, 2)
    if len(raw_points) == 0:
        return []

    canonical = canonicalize(raw_points, attack_dir)
    line_count, _ = detect_lines(canonical[:, 0], gap_delta=gap_delta, max_lines=max_lines)
    if line_count <= 1:
        return [raw_points]

    order = np.argsort(canonical[:, 0])
    sorted_canonical_x = canonical[order, 0]
    gaps = np.diff(sorted_canonical_x)
    candidates = [(gap, idx) for idx, gap in enumerate(gaps) if gap > gap_delta]
    candidates.sort(reverse=True)
    split_idx = sorted(idx for _, idx in candidates[: line_count - 1])

    groups = []
    start = 0
    for idx in split_idx:
        groups.append(raw_points[order[start : idx + 1]])
        start = idx + 1
    groups.append(raw_points[order[start:]])
    return groups


def detections_by_frame(detections):
    frames = defaultdict(list)
    for det in detections:
        frames[det.frame].append(det)
    return frames


def draw_pitch(ax):
    ax.add_patch(
        Rectangle(
            (-HALF_LENGTH - 4, -HALF_WIDTH - 4),
            HALF_LENGTH * 2 + 8,
            HALF_WIDTH * 2 + 8,
            fc=FIELD_GREEN,
            ec="none",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (-HALF_LENGTH, -HALF_WIDTH),
            HALF_LENGTH * 2,
            HALF_WIDTH * 2,
            fill=False,
            ec="white",
            lw=1.6,
            zorder=1,
        )
    )
    ax.plot([0, 0], [-HALF_WIDTH, HALF_WIDTH], color="white", lw=1.3, zorder=1)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, ec="white", lw=1.3, zorder=1))
    for side in (-1, 1):
        ax.add_patch(
            Rectangle(
                (side * HALF_LENGTH, -20.16),
                -side * 16.5,
                40.32,
                fill=False,
                ec="white",
                lw=1.2,
                zorder=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (side * HALF_LENGTH, -9.16),
                -side * 5.5,
                18.32,
                fill=False,
                ec="white",
                lw=1.2,
                zorder=1,
            )
        )
    ax.set_xlim(-HALF_LENGTH - 5, HALF_LENGTH + 5)
    ax.set_ylim(-HALF_WIDTH - 5, HALF_WIDTH + 5)
    ax.set_aspect("equal")
    ax.axis("off")


def _player_label(det):
    if det.jersey_number is not None:
        return str(det.jersey_number)
    return str(det.track_id)


def _draw_team_shape(ax, team, players, attack_dir, record, gap_delta):
    if len(players) == 0:
        return

    color = TEAM_COLOR.get(team, "#888888")
    pts = np.array([[det.x, det.y] for det in players], dtype=float)

    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            ax.add_patch(
                Polygon(
                    pts[hull.vertices],
                    closed=True,
                    fc=color,
                    ec=color,
                    alpha=0.14,
                    lw=1.4,
                    zorder=2,
                )
            )
        except QhullError:
            pass

    centroid = pts.mean(axis=0)
    ax.plot(centroid[0], centroid[1], "x", color=color, ms=9, mew=2.2, zorder=5)

    if record and record.get("line_count") and not record.get("low_confidence"):
        for group in group_depth_lines(pts, attack_dir=attack_dir, gap_delta=gap_delta):
            if len(group) == 0:
                continue
            x = float(np.mean(group[:, 0]))
            y0 = float(np.min(group[:, 1]))
            y1 = float(np.max(group[:, 1]))
            if abs(y1 - y0) < 3.0:
                y0 -= 2.0
                y1 += 2.0
            ax.plot([x, x], [y0, y1], color=color, lw=3.0, alpha=0.85, zorder=4)


def draw_frame(ax, detections, topology_records, attack_dirs, frame_no, fps, gap_delta):
    ax.clear()
    draw_pitch(ax)

    t = frame_no / fps
    records = records_at_time(topology_records, t)
    players_by_team = defaultdict(list)
    balls = []
    for det in detections:
        if det.role == "player" and det.team in ("left", "right"):
            players_by_team[det.team].append(det)
        elif det.role == "ball":
            balls.append(det)

    for team, players in players_by_team.items():
        _draw_team_shape(
            ax,
            team,
            players,
            attack_dirs.get(team, 1 if team == "left" else -1),
            records.get(team),
            gap_delta,
        )

    for det in detections:
        if det.role == "ball":
            ax.plot(det.x, det.y, "o", color=BALL_COLOR, ms=7, mec="black", mew=0.8, zorder=8)
            continue
        if det.role not in ("player", "goalkeeper"):
            continue
        color = TEAM_COLOR.get(det.team, "#888888")
        marker = "s" if det.role == "goalkeeper" else "o"
        ax.plot(det.x, det.y, marker, color=color, ms=8, mec="white", mew=0.7, zorder=6)
        ax.text(
            det.x,
            det.y,
            _player_label(det),
            color="white",
            ha="center",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            zorder=7,
        )

    lines = [f"frame {frame_no}  t={t:.1f}s"]
    for team in ("left", "right"):
        record = records.get(team)
        if not record:
            continue
        gaps = ",".join(f"{gap:.1f}" for gap in record["inter_line_gaps_m"]) or "-"
        lines.append(
            f"{team}: cov {record['coverage_n']}  h {record['block_height_m']:.1f}m  "
            f"d {record['block_depth_m']:.1f}m  w {record['block_width_m']:.1f}m  "
            f"lines {record['line_count']} gaps {gaps} poss {record['possession_tag']}"
        )
    ax.text(
        0.01,
        0.99,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.82),
        zorder=20,
    )


def _figure_to_bgr(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    rgb = rgba[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def render_demo(
    labels_path,
    output_dir,
    topology_path=None,
    sequence_name=None,
    fps=25.0,
    video_stride=2,
    gap_delta=7.0,
    keyframes=(5, 8, 22, 27),
    codec="h264",
):
    labels_path = Path(labels_path)
    sequence_name = sequence_name or sequence_name_from_labels(labels_path)
    topology_path = Path(topology_path) if topology_path else default_topology_path(labels_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = load_detections(str(labels_path))
    frame_map = detections_by_frame(detections)
    frames = sorted(frame_map)
    topology_records = json.loads(topology_path.read_text())
    attack_dirs = infer_attack_dirs(detections)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
    first = frames[0]
    draw_frame(ax, frame_map[first], topology_records, attack_dirs, first, fps, gap_delta)
    sample = _figure_to_bgr(fig)
    height, width = sample.shape[:2]

    video_path, png_paths = output_paths(output_dir, sequence_name, keyframes)
    raw_video_path = (
        output_dir / f"{sequence_name}_topology_topdown.tmp.mp4"
        if codec == "h264"
        else video_path
    )
    writer = cv2.VideoWriter(
        str(raw_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / video_stride,
        (width, height),
    )
    for frame_no in frames[::video_stride]:
        draw_frame(ax, frame_map[frame_no], topology_records, attack_dirs, frame_no, fps, gap_delta)
        writer.write(_figure_to_bgr(fig))
    writer.release()

    if codec == "h264":
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw_video_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-crf",
                "20",
                "-preset",
                "medium",
                str(video_path),
            ],
            check=True,
        )
        raw_video_path.unlink(missing_ok=True)
    elif codec != "mp4v":
        raise ValueError(f"Unsupported codec: {codec}")

    for second, png_path in zip(keyframes, png_paths):
        frame_no = min(frames, key=lambda frame: abs(frame - int(round(second * fps))))
        draw_frame(ax, frame_map[frame_no], topology_records, attack_dirs, frame_no, fps, gap_delta)
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return video_path, png_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render a top-down formation topology demo.")
    parser.add_argument("--labels", required=True, help="Labels-GameState.json path")
    parser.add_argument(
        "--topology",
        help="topology records JSON path; defaults to outputs/formation_topology/{sequence}/topo.json",
    )
    parser.add_argument("--output-dir", required=True, help="directory for MP4 and PNG outputs")
    parser.add_argument("--sequence-name", help="sequence name for output files; defaults to labels parent")
    parser.add_argument("--fps", type=float, default=25.0, help="source video FPS")
    parser.add_argument("--video-stride", type=int, default=2, help="render every Nth frame")
    parser.add_argument("--gap-delta", type=float, default=7.0, help="line split gap threshold")
    parser.add_argument(
        "--keyframes",
        default="5,8,22,27",
        help="comma-separated keyframe seconds, e.g. 5,8,22,27",
    )
    parser.add_argument("--codec", choices=("h264", "mp4v"), default="h264", help="output video codec")
    args = parser.parse_args(argv)

    video_path, png_paths = render_demo(
        labels_path=args.labels,
        topology_path=args.topology,
        output_dir=args.output_dir,
        sequence_name=args.sequence_name,
        fps=args.fps,
        video_stride=args.video_stride,
        gap_delta=args.gap_delta,
        keyframes=parse_keyframes(args.keyframes),
        codec=args.codec,
    )
    print(f"wrote video: {video_path}")
    for png_path in png_paths:
        print(f"wrote keyframe: {png_path}")


if __name__ == "__main__":
    main()
