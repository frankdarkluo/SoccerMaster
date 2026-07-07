"""Abstract LLM adapter interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union


class LLMAdapter(ABC):
    @abstractmethod
    def supports_video(self) -> bool:
        """Whether this backend accepts video input directly."""

    @abstractmethod
    def generate(self, prompt: str, visual_input: Union[Path, List[Path], None] = None) -> str:
        """Generate text from prompt + optional visual input."""

    def prepare_visual_input(
        self,
        visual_input: Union[Path, List[Path], None],
    ) -> Union[Path, List[Path], None]:
        """If video given but not supported, extract frames."""
        if visual_input is None:
            return None
        if (
            isinstance(visual_input, Path)
            and visual_input.suffix == ".mp4"
            and not self.supports_video()
        ):
            from pipeline.utils.video import extract_frames

            frame_dir = visual_input.parent / f"{visual_input.stem}_frames_for_llm"
            extract_frames(visual_input, frame_dir)
            return sorted(frame_dir.glob("*.jpg"))
        return visual_input
