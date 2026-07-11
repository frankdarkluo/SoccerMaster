"""Free Microsoft Edge TTS adapter (no voice cloning, good for testing)."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from pipeline.stage3_tts.tts_adapter import (
    ENERGY_ENGAGED,
    ENERGY_EXCITED,
    ENERGY_EXPLOSIVE,
    ENERGY_NORMAL,
    TTSAdapter,
)

log = logging.getLogger(__name__)

_VOICES = {
    "zh": "zh-CN-YunxiNeural",
    "en": "en-US-GuyNeural",
}

# Prosody per energy tier (grill-me option B).
_PROSODY = {
    ENERGY_NORMAL: {"rate": "+15%", "pitch": "+0Hz", "volume": "+0%"},
    ENERGY_ENGAGED: {"rate": "+20%", "pitch": "+4Hz", "volume": "+5%"},
    ENERGY_EXCITED: {"rate": "+25%", "pitch": "+8Hz", "volume": "+10%"},
    ENERGY_EXPLOSIVE: {"rate": "+35%", "pitch": "+15Hz", "volume": "+20%"},
}


class EdgeTTSAdapter(TTSAdapter):
    """Uses Microsoft Edge TTS (free, no API key).

    Produces decent quality but cannot clone a specific voice.
    Energy tiers are expressed via rate / pitch / volume.
    """

    def __init__(self, language: str = "zh") -> None:
        self.voice = _VOICES.get(language, _VOICES["zh"])

    def synthesize(
        self,
        text: str,
        output_path: Path,
        energy: str = ENERGY_NORMAL,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prosody = _PROSODY.get(energy, _PROSODY[ENERGY_NORMAL])

        async def _run() -> None:
            import edge_tts

            comm = edge_tts.Communicate(
                text,
                self.voice,
                rate=prosody["rate"],
                pitch=prosody["pitch"],
                volume=prosody["volume"],
            )
            await comm.save(str(output_path))

        asyncio.run(_run())
        log.info(
            "EdgeTTS [%s] → %s (%d bytes)",
            energy, output_path, output_path.stat().st_size,
        )
        return output_path
