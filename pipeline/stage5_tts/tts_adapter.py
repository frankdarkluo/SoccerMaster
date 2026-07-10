"""Abstract TTS adapter interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

# Energy tiers for football commentary delivery.
ENERGY_NORMAL = "normal"        # calm narration
ENERGY_ENGAGED = "engaged"      # something developing — tone lifts
ENERGY_EXCITED = "excited"      # shots, dangerous attacks, won duels
ENERGY_EXPLOSIVE = "explosive"  # goals


class TTSAdapter(ABC):
    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: Path,
        energy: str = ENERGY_NORMAL,
    ) -> Path:
        """Synthesize *text* to an audio file at *output_path*.

        *energy* is one of ``normal`` / ``engaged`` / ``excited`` / ``explosive`` and
        controls delivery intensity (rate/pitch/volume or emotion prompts).

        Returns *output_path* on success.
        """
