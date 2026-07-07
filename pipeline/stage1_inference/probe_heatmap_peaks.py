#!/usr/bin/env python3
"""Probe raw HRNet heatmap peaks on sample frames (H2 diagnostic).

Compares max local-maxima scores before thresholding on valid vs invalid frames.
If peaks on invalid frames are consistently below kp threshold (0.1449), lowering
threshold will not recover calibration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
GSR = REPO / "codes" / "sn-gamestate"
MODEL_DIR = GSR / "pretrained_models"
FRAMES = GSR / "datasets" / "SoccerNetGS" / "test" / "SNGS-148" / "img1"

KP_THRESH = 0.1449
LINE_THRESH = 0.2983


def _load_models(device: str):
    sys.path.insert(0, str(GSR / "plugins" / "calibration"))
    from nbjw_calib.model.cls_hrnet import get_cls_net
    from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l

    cfg_path = GSR / "sn_gamestate" / "sn_gamestate" / "configs" / "modules" / "pitch" / "nbjw_calib.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())["cfg"]
    cfg_l = yaml.safe_load(cfg_path.read_text())["cfg_l"]

    kp_ckpt = MODEL_DIR / "calibration" / "SV_kp"
    l_ckpt = MODEL_DIR / "calibration" / "SV_lines"

    model = get_cls_net(cfg)
    model.load_state_dict(torch.load(kp_ckpt, map_location=device))
    model.to(device).eval()

    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load(l_ckpt, map_location=device))
    model_l.to(device).eval()

    tfms = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    return model, model_l, tfms


def _max_local_maxima(heatmap: torch.Tensor) -> float:
    """Max score among spatial local maxima (pre-threshold)."""
    hm = heatmap[:, :-1, :, :]  # drop background channel
    kernel = 3
    pad = 1
    padded = torch.nn.functional.pad(hm, (pad, pad, pad, pad), mode="constant", value=1.0)
    pooled = torch.nn.functional.max_pool2d(padded, kernel, stride=1, padding=0)
    local_max = pooled == hm
    masked = hm * local_max
    return float(masked.max().item())


def probe_frame(model, model_l, tfms, frame_idx: int, device: str) -> dict:
    # frame N → img1/{N+1:06d}.jpg (0-based frame column in pklz)
    img_path = FRAMES / f"{frame_idx + 1:06d}.jpg"
    img = tfms(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        hm_kp = model(img)
        hm_l = model_l(img)
    return {
        "frame": frame_idx,
        "kp_max_peak": _max_local_maxima(hm_kp),
        "line_max_peak": _max_local_maxima(hm_l),
        "kp_above_thresh": _max_local_maxima(hm_kp) >= KP_THRESH,
        "line_above_thresh": _max_local_maxima(hm_l) >= LINE_THRESH,
    }


def main():
    sys.path.insert(0, str(REPO))
    from pipeline.stage1_inference.torch_compat import patch_torch_load
    patch_torch_load()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_l, tfms = _load_models(device)

    # valid-adjacent + invalid samples from handoff
    samples = [125, 126, 127, 155, 156, 325, 326, 350, 449, 515, 564, 565]
    print(f"device={device}  kp_thresh={KP_THRESH}  line_thresh={LINE_THRESH}")
    print(f"{'frame':>6}  {'kp_peak':>8}  {'line_peak':>9}  kp_ok  line_ok")
    for f in samples:
        r = probe_frame(model, model_l, tfms, f, device)
        print(
            f"{r['frame']:6d}  {r['kp_max_peak']:8.4f}  {r['line_max_peak']:9.4f}  "
            f"{'Y' if r['kp_above_thresh'] else 'N':>5}  {'Y' if r['line_above_thresh'] else 'N':>6}"
        )


if __name__ == "__main__":
    main()
