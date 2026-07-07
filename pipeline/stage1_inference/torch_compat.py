"""PyTorch 2.6+ compatibility helpers for legacy checkpoints (YOLO, etc.)."""
from __future__ import annotations


def patch_torch_load() -> None:
    """Default ``weights_only=False`` so ultralytics / older ckpts still load.

    PyTorch >= 2.6 changed ``torch.load`` default to ``weights_only=True``,
    which breaks YOLO and other full-pickle checkpoints used by GSR.
    """
    import torch

    if getattr(torch.load, "_soccermaster_patched", False):
        return

    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._soccermaster_patched = True  # type: ignore[attr-defined]
    torch.load = _load  # type: ignore[assignment]
