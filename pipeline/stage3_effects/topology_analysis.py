"""Formation-topology analysis: derive per-window team-shape metrics from
predictions.json and write topo.json for Stage 4 commentary context."""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.topology.analysis import analyze
from pipeline.topology.io_gamestate import load_detections


def run_topology_analysis(
    predictions_json_path: Path,
    output_path: Path,
    fps: float,
    win_s: float = 3.0,
    stride_s: float = 1.0,
) -> Path:
    """Load GameState detections, compute team-shape records, write JSON."""
    predictions_json_path = Path(predictions_json_path)
    output_path = Path(output_path)
    if not predictions_json_path.exists():
        raise FileNotFoundError(f"predictions.json not found: {predictions_json_path}")

    detections = load_detections(str(predictions_json_path))
    records = analyze(detections, fps=fps, win_s=win_s, stride_s=stride_s)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
