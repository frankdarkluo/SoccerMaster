"""Local Qwen2.5-VL adapter."""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

from pipeline.stage4_commentary.llm_adapter import LLMAdapter


class QwenLocalAdapter(LLMAdapter):
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._processor = None

    def supports_video(self) -> bool:
        return True

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map=self.device,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)

    def generate(self, prompt: str, visual_input: Union[Path, List[Path], None] = None) -> str:
        self._load_model()
        visual_input = self.prepare_visual_input(visual_input)

        messages = [{"role": "user", "content": []}]
        if visual_input is not None:
            if isinstance(visual_input, Path):
                messages[0]["content"].append({"type": "video", "video": str(visual_input)})
            elif isinstance(visual_input, list):
                for img_path in visual_input[:30]:
                    messages[0]["content"].append({"type": "image", "image": str(img_path)})
        messages[0]["content"].append({"type": "text", "text": prompt})

        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        output_ids = self._model.generate(**inputs, max_new_tokens=4096)
        trimmed = output_ids[0][len(inputs.input_ids[0]):]
        return self._processor.decode(trimmed, skip_special_tokens=True)
