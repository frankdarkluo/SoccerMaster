"""Thin wrapper around the vendored CosyVoice model."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
COSYVOICE_ROOT = REPO_ROOT / "codes" / "CosyVoice"
MODEL_DIR = REPO_ROOT / "codes" / "sn-gamestate" / "pretrained_models" / "Fun-CosyVoice3-0.5B"
DEFAULT_PROMPT_WAV = COSYVOICE_ROOT / "asset" / "zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>"
    "希望你以后能够做的比我还好呦。"
)
CLONE_PROMPT_WAV = REPO_ROOT / "voice_sample.wav"
MODEL_ARTIFACTS = ("cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt")

for path in (COSYVOICE_ROOT, COSYVOICE_ROOT / "third_party" / "Matcha-TTS"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@contextmanager
def _without_wetext():
    """Force CosyVoice's documented no-frontend path without leaking import state."""
    saved = {
        name: module for name, module in sys.modules.items()
        if name == "wetext" or name.startswith("wetext.")
    }
    sys.modules["wetext"] = None
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "wetext" or name.startswith("wetext."):
                del sys.modules[name]
        sys.modules.update(saved)


class CosyVoiceSynthesizer:
    def __init__(self, model_dir: Path):
        missing = [name for name in MODEL_ARTIFACTS if not (model_dir / name).is_file()]
        if not model_dir.is_dir() or missing:
            detail = ", ".join(missing) if missing else str(model_dir)
            raise FileNotFoundError(f"CosyVoice model is missing or incomplete: {detail}")
        from cosyvoice.cli.cosyvoice import AutoModel

        with _without_wetext():
            self.model = AutoModel(model_dir=str(model_dir))

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        prompt_wav: Optional[Path] = None,
        prompt_text: Optional[str] = None,
    ) -> Path:
        import torch
        import torchaudio

        reference = prompt_wav or (
            DEFAULT_PROMPT_WAV if voice == "default" else CLONE_PROMPT_WAV
        )
        if not reference.is_file():
            raise FileNotFoundError(f"CosyVoice reference audio not found: {reference}")
        if prompt_text:
            chunks = self.model.inference_zero_shot(
                text, prompt_text, str(reference), stream=False
            )
        else:
            chunks = self.model.inference_cross_lingual(
                f"You are a helpful assistant.<|endofprompt|>{text}",
                str(reference), stream=False,
            )
        tensors = [chunk["tts_speech"] for chunk in chunks]
        if not tensors:
            raise RuntimeError("CosyVoice returned no audio")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            str(output_path), torch.cat(tensors, dim=1), self.model.sample_rate
        )
        return output_path
