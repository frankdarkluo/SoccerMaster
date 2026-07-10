"""CosyVoice3 local TTS adapter (Fun-CosyVoice3-0.5B, instruct-mode voice clone).

Clones the commentator's voice from a reference wav — instruct mode needs NO
transcript of the reference — and controls emotion per energy tier via
instruct_text plus a speed multiplier. Runs on cuda when available, else cpu
(CosyVoice resolves the device internally).

Env vars (all optional, defaults in parentheses):
    COSYVOICE_REPO        path to the cloned FunAudioLLM/CosyVoice repo
                          (<repo_root>/codes/CosyVoice)
    COSYVOICE_MODEL_DIR   model weights dir
                          (<repo_root>/pretrained_models/Fun-CosyVoice3-0.5B)
    COSYVOICE_PROMPT_WAV  reference voice wav (<repo_root>/voice_sample.wav)

One-time setup: bash scripts/setup_cosyvoice.sh
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from pipeline.config import REPO_ROOT
from pipeline.stage5_tts.tts_adapter import (
    ENERGY_ENGAGED,
    ENERGY_EXCITED,
    ENERGY_EXPLOSIVE,
    ENERGY_NORMAL,
    TTSAdapter,
)

log = logging.getLogger(__name__)

INSTRUCT_ZH = {
    ENERGY_NORMAL: "You are a helpful assistant. 用平稳、自然的中文足球解说语气播报。<|endofprompt|>",
    ENERGY_ENGAGED: "You are a helpful assistant. 用投入而期待的中文足球解说语气播报，语调略微上扬。<|endofprompt|>",
    ENERGY_EXCITED: "You are a helpful assistant. 用激动、急促的中文足球解说语气播报，情绪高涨。<|endofprompt|>",
    ENERGY_EXPLOSIVE: "You are a helpful assistant. 用极度亢奋的中文足球解说语气呐喊播报，像进球瞬间一样高亢有力！<|endofprompt|>",
}
INSTRUCT_EN = {
    ENERGY_NORMAL: "You are a helpful assistant. Speak as a calm, natural English football commentator.<|endofprompt|>",
    ENERGY_ENGAGED: "You are a helpful assistant. Speak as an engaged football commentator, tone lifting with anticipation.<|endofprompt|>",
    ENERGY_EXCITED: "You are a helpful assistant. Speak as a thrilled football commentator, fast and full of excitement.<|endofprompt|>",
    ENERGY_EXPLOSIVE: "You are a helpful assistant. Shout like a football commentator at the moment of a goal, ecstatic and powerful!<|endofprompt|>",
}
SPEED = {
    ENERGY_NORMAL: 1.0,
    ENERGY_ENGAGED: 1.05,
    ENERGY_EXCITED: 1.15,
    ENERGY_EXPLOSIVE: 1.2,
}


def instruct_for(energy: str, language: str) -> str:
    table = INSTRUCT_ZH if language == "zh" else INSTRUCT_EN
    return table.get(energy, table[ENERGY_NORMAL])


class CosyVoiceAdapter(TTSAdapter):
    """Fun-CosyVoice3-0.5B via inference_instruct2.

    The model and reference audio load lazily on first synthesize() so that
    constructing the adapter (e.g. for CLI --help) costs nothing.
    """

    def __init__(
        self,
        language: str = "zh",
        model_dir: str | Path | None = None,
        prompt_wav: str | Path | None = None,
    ) -> None:
        self.language = language
        self.model_dir = Path(
            model_dir
            or os.environ.get(
                "COSYVOICE_MODEL_DIR",
                REPO_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B",
            )
        )
        self.prompt_wav = Path(
            prompt_wav
            or os.environ.get("COSYVOICE_PROMPT_WAV", REPO_ROOT / "voice_sample.wav")
        )
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        repo = Path(os.environ.get("COSYVOICE_REPO", REPO_ROOT / "codes" / "CosyVoice"))
        if not repo.is_dir():
            raise RuntimeError(
                f"CosyVoice repo not found: {repo} — run scripts/setup_cosyvoice.sh"
            )
        if not self.model_dir.is_dir():
            raise RuntimeError(
                f"CosyVoice model dir not found: {self.model_dir} — run scripts/setup_cosyvoice.sh"
            )
        if not self.prompt_wav.is_file():
            raise RuntimeError(f"Reference voice wav not found: {self.prompt_wav}")
        for path in (str(repo), str(repo / "third_party" / "Matcha-TTS")):
            if path not in sys.path:
                sys.path.insert(0, path)
        from cosyvoice.cli.cosyvoice import AutoModel

        log.info("Loading CosyVoice3 from %s ...", self.model_dir)
        self._model = AutoModel(model_dir=str(self.model_dir))

    def synthesize(
        self,
        text: str,
        output_path: Path,
        energy: str = ENERGY_NORMAL,
    ) -> Path:
        self._load()
        import torch
        import torchaudio

        output_path.parent.mkdir(parents=True, exist_ok=True)
        instruct = instruct_for(energy, self.language)
        speed = SPEED.get(energy, 1.0)

        chunks = [
            out["tts_speech"]
            for out in self._model.inference_instruct2(
                text, instruct, str(self.prompt_wav), stream=False, speed=speed
            )
        ]
        wav_path = output_path.with_suffix(".wav")
        torchaudio.save(str(wav_path), torch.cat(chunks, dim=1), self._model.sample_rate)

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(wav_path),
                "-c:a", "libmp3lame", "-q:a", "2", str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg wav to mp3 failed: {result.stderr[-400:]}")
        wav_path.unlink()
        return output_path
