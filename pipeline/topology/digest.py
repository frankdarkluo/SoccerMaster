"""Compress a visible-topology JSON into a compact per-window fact table for prompts."""

from __future__ import annotations

from typing import Any

_ROLE_KEYS = (
    ("attacking_forward", "forward"),
    ("attacking_midfield", "midfield"),
    ("defending_defensive", "last_line"),
)


def build_digest(topology: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate each renderable line/zone in a window into a compact row.

    Pure aggregation over fields `analyze_visible_topology` already computes
    (mean/round/select) — no new geometry, no ball-relative inference.
    """
    rows = []
    for window in topology["windows"]:
        row: dict[str, Any] = {"t": window["start_s"], "attacking_team": window["attacking_team"]}
        by_role = {line["role"]: line for line in window["lines"] if line["status"] == "renderable"}
        for role, key in _ROLE_KEYS:
            line = by_role.get(role)
            if line and line["members"]:
                xs = [member["x_m"] for member in line["members"]]
                row[key] = {"team": line["team"], "x_m": round(sum(xs) / len(xs), 1), "span_m": line["lateral_span_m"]}
        gap_zone = next((zone for zone in window.get("zones", []) if zone["type"] == "inter_line_space"), None)
        if gap_zone:
            xs = [point[0] for point in gap_zone["polygon_pitch"]]
            row["inter_line_gap_m"] = round(max(xs) - min(xs), 1)
        if window.get("ball"):
            row["ball"] = window["ball"]
        gaps = sorted({gap for line in window["lines"] for gap in line["evidence_gaps"]})
        if gaps:
            row["evidence_gaps"] = gaps
        rows.append(row)
    return rows


def window_digest(topology: dict[str, Any], start_s: float, end_s: float, *, time_scale: float = 1.0) -> dict[str, Any]:
    """Restrict a topology dict to [start_s, end_s], re-zeroed to match a trimmed clip's timeline."""
    windows = [
        {**window, "start_s": round((max(window["start_s"], start_s) - start_s) * time_scale, 3), "end_s": round((min(window["end_s"], end_s) - start_s) * time_scale, 3)}
        for window in topology["windows"]
        if window["start_s"] < end_s and window["end_s"] > start_s
    ]
    return {**topology, "windows": windows}


def render_digest(topology: dict[str, Any]) -> str:
    """Render a digest as one compact text line per window, for prompt injection."""
    rows = build_digest(topology)
    header = f"position_source={topology['position_source']}"
    if not rows:
        return f"{header}\nno renderable topology windows (insufficient ball/goalkeeper/possession evidence)"
    lines = [header]
    for row in rows:
        parts = [f"t={row['t']:.1f}s"]
        if row["attacking_team"]:
            parts.append(f"attack={row['attacking_team']}")
        for _, key in _ROLE_KEYS:
            if key in row:
                value = row[key]
                parts.append(f"{key}[{value['team']}]=x{value['x_m']}m/span{value['span_m']}m")
        if "inter_line_gap_m" in row:
            parts.append(f"gap={row['inter_line_gap_m']}m")
        if "ball" in row:
            ball = row["ball"]
            parts.append(f"ball=x{ball['x_m']}m/y{ball['y_m']}m/move{ball['displacement_m']}m")
        if row.get("evidence_gaps"):
            parts.append(f"gaps={'|'.join(row['evidence_gaps'])}")
        lines.append(" ".join(parts))
    return "\n".join(lines)
