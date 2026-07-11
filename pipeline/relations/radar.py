"""Render relations.json snapshots as top-view radar PNGs for the LLM."""
from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

from pipeline.config import PITCH_LENGTH, PITCH_WIDTH

SCALE = 6  # px per meter -> 630x408
W, H = int(PITCH_LENGTH * SCALE), int(PITCH_WIDTH * SCALE)
GREEN = (34, 120, 52)
LINE = (240, 240, 240)
TEAM_COLOR = {"left": (40, 90, 220), "right": (220, 60, 50)}
BALL = (250, 220, 40)


def _to_px(x: float, y: float) -> tuple[int, int]:
    return (int((x + PITCH_LENGTH / 2) * SCALE),
            int((y + PITCH_WIDTH / 2) * SCALE))


def _draw_pitch(d: ImageDraw.ImageDraw) -> None:
    d.rectangle([0, 0, W - 1, H - 1], outline=LINE, width=2)
    d.line([W // 2, 0, W // 2, H], fill=LINE, width=2)
    radius = int(9.15 * SCALE)
    d.ellipse([W // 2 - radius, H // 2 - radius,
               W // 2 + radius, H // 2 + radius], outline=LINE, width=2)
    for side in (-1, 1):
        x0, _ = _to_px(side * PITCH_LENGTH / 2, 0)
        depth = int(16.5 * SCALE) * (-side)
        half_width = int(40.32 / 2 * SCALE)
        d.rectangle([min(x0, x0 + depth), H // 2 - half_width,
                     max(x0, x0 + depth), H // 2 + half_width],
                    outline=LINE, width=2)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_radar_frames(relations: dict, out_dir: Path,
                        hz: float = 1.0) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _font(14)
    small = _font(11)

    snaps = relations["snapshots"]
    duration = relations.get("video_info", {}).get("duration_s")
    if duration is None:
        step = snaps[1]["t"] - snaps[0]["t"] if len(snaps) > 1 else 1.0 / hz
        duration = snaps[-1]["t"] + step
    n_frames = max(1, int(duration * hz))
    selected = [min(snaps, key=lambda snap: abs(snap["t"] - i / hz))
                for i in range(n_frames)]

    paths = []
    for snap in selected:
        img = Image.new("RGB", (W, H), GREEN)
        d = ImageDraw.Draw(img)
        _draw_pitch(d)
        for player in snap["players"]:
            cx, cy = _to_px(player["x"], player["y"])
            color = TEAM_COLOR.get(player["team"], (128, 128, 128))
            radius = 10
            if player["role"] == "goalkeeper":
                d.rectangle([cx - radius, cy - radius, cx + radius, cy + radius],
                            fill=color, outline=LINE)
            else:
                d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                          fill=color, outline=LINE)
            if player["jersey"]:
                d.text((cx, cy), player["jersey"], fill=(255, 255, 255),
                       font=small, anchor="mm")
        bx, by = _to_px(snap["ball"]["x"], snap["ball"]["y"])
        d.ellipse([bx - 5, by - 5, bx + 5, by + 5], fill=BALL, outline=(0, 0, 0))
        d.text((8, 6), f"t={snap['t']:.1f}s", fill=(255, 255, 60), font=font)
        path = out_dir / f"radar_{snap['t']:06.2f}.png"
        img.save(path)
        paths.append(path)
    return paths
