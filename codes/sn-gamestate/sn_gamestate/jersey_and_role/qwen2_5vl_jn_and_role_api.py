import logging

import numpy as np
import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
except ImportError:  # pragma: no cover
    Qwen3VLMoeForConditionalGeneration = None

from tracklab.pipeline.detectionlevel_module import DetectionLevelModule
from tracklab.utils.collate import Unbatchable, default_collate

log = logging.getLogger(__name__)


class QWEN2_5VL_JN_AND_ROLE_BATCH(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = [
        "jersey_number_detection",
        "jersey_number_confidence",
        "role_detection",
        "role_confidence",
    ]
    collate_fn = default_collate

    def __init__(self, cfg, batch_size, device, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        import os

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.cfg = cfg
        self.model_path = self.cfg.model_path
        self.use_vllm = bool(getattr(cfg, "use_vllm", False))
        self.save_jersey_number_full_detection = cfg.save_jersey_number_full_detection
        if self.save_jersey_number_full_detection:
            self.output_columns.append("jersey_number_full_detection")
        self.use_legibility_filter = cfg.use_legibility_filter
        if self.use_legibility_filter:
            self.input_columns.append("legibility_score")
            self.legibility_filter_threshold = cfg.legibility_filter_threshold

        self.NUM_BEAMS = 1
        self.TEMPERATURE = 0.0
        self.MAX_NEW_TOKENS = int(getattr(cfg, "max_new_tokens", 64))
        self.USE_CACHE = True
        self.DO_SAMPLE = self.TEMPERATURE > 0

        processor_path = (
            getattr(cfg, "vllm_model_path", None) or self.model_path
            if self.use_vllm
            else self.model_path
        )
        max_pixels = (
            int(cfg.image_width * cfg.image_height / cfg.downsample_factor / cfg.downsample_factor)
            // (28 * 28)
            * (28 * 28)
        )
        self.processor = AutoProcessor.from_pretrained(
            processor_path, max_pixels=max_pixels, use_fast=False
        )
        self.processor.tokenizer.padding_side = "left"
        self.batch_size = batch_size
        self.device = device

        self.jn_text_prompt = (
            "Analyze this image and determine if the player is facing away from the camera. "
            "If the player is facing away, output the jersey number on their back. "
            "If the player is not facing away from the camera, output 'No'."
        )
        self.role_text_prompt = """
            There are an image and a crop from it.
            Analyze this image and determine the role of the person in this image.
            Respond ONLY with a single word in ['player', 'referee', 'goalkeeper', 'other'].
            If there is no person in the image, or the person is not an athlete on the pitch, respond 'other'.
            """
        self.role_list = ["player", "referee", "goalkeeper", "other"]

        if self.use_vllm:
            try:
                self._init_vllm(cfg)
                self.model = None
                log.info("jersey_and_role backend: vLLM (%s)", self.vllm_model_path)
            except Exception:
                self.use_vllm = False
                log.warning(
                    "vLLM initialization failed; falling back to HuggingFace backend for this run.",
                    exc_info=True,
                )
                self._init_hf()
                log.info(
                    "jersey_and_role backend: HuggingFace (%s) (fallback)",
                    self.model_path,
                )
        else:
            self._init_hf()
            log.info("jersey_and_role backend: HuggingFace (%s)", self.model_path)

    def _init_hf(self):
        if "2.5" in self.model_path:
            model_api = Qwen2_5_VLForConditionalGeneration
        elif "3" in self.model_path:
            if Qwen3VLMoeForConditionalGeneration is None:
                raise ImportError("transformers does not provide Qwen3VLMoeForConditionalGeneration. Upgrade transformers to a version that supports Qwen3 models.")
            model_api = Qwen3VLMoeForConditionalGeneration
        else:
            raise ValueError(f"Model path {self.model_path} is not supported")

        self.model = model_api.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        self.vllm_llm = None
        self.vllm_sampling = None
        self.vllm_use_tqdm = False

    def _init_vllm(self, cfg):
        from vllm import LLM, SamplingParams

        self.vllm_model_path = getattr(cfg, "vllm_model_path", None) or self.model_path
        quantization = getattr(cfg, "vllm_quantization", None)
        if quantization in (None, "", "null", "none"):
            quantization = "awq" if "awq" in self.vllm_model_path.lower() else None
        elif str(quantization).lower() == "awq" and "awq" not in self.vllm_model_path.lower():
            log.warning(
                "Ignoring vllm_quantization=awq for non-AWQ model %s",
                self.vllm_model_path,
            )
            quantization = None

        llm_kwargs = {
            "model": self.vllm_model_path,
            "gpu_memory_utilization": float(getattr(cfg, "vllm_gpu_memory_utilization", 0.90)),
            "max_model_len": int(getattr(cfg, "vllm_max_model_len", 2048)),
            "trust_remote_code": True,
            "limit_mm_per_prompt": {"image": 2},
            "enforce_eager": bool(getattr(cfg, "vllm_enforce_eager", True)),
        }
        if quantization:
            llm_kwargs["quantization"] = quantization

        self.vllm_llm = LLM(**llm_kwargs)
        self.vllm_sampling = SamplingParams(
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_NEW_TOKENS,
        )
        # vLLM prints per-batch tqdm ("Rendering prompts" / "Processed prompts").
        # TrackLab already shows one progress bar per pipeline module.
        self.vllm_use_tqdm = bool(getattr(cfg, "vllm_use_tqdm", False))

    def no_jersey_number(self):
        return None, 0

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((10, 10, 3), dtype=np.uint8)

        batch = {
            "images": Unbatchable([image]),
            "crops": Unbatchable([crop]),
        }

        if self.use_legibility_filter:
            batch["legibility_score"] = detection.legibility_score

        return batch

    def extract_numbers(self, text):
        if text.strip() == "?":
            return None
        number = ""
        for char in text:
            if char.isdigit():
                number += char
        return number if number != "" else None

    def extract_role(self, text):
        text = text.lower()
        if text in self.role_list:
            return text
        return None

    def _build_vllm_request(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        request = {"prompt": text}
        mm_data = {}
        if image_inputs:
            mm_data["image"] = image_inputs[0] if len(image_inputs) == 1 else image_inputs
        if video_inputs:
            mm_data["video"] = video_inputs
        if mm_data:
            request["multi_modal_data"] = mm_data
        return request

    def _generate_texts(self, messages_batch):
        if self.use_vllm:
            vllm_inputs = [self._build_vllm_request(messages) for messages in messages_batch]
            outputs = self.vllm_llm.generate(
                vllm_inputs, self.vllm_sampling, use_tqdm=self.vllm_use_tqdm
            )
            return [output.outputs[0].text for output in outputs]

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(
            **inputs,
            num_beams=self.NUM_BEAMS,
            temperature=self.TEMPERATURE,
            max_new_tokens=self.MAX_NEW_TOKENS,
            use_cache=self.USE_CACHE,
            do_sample=self.DO_SAMPLE,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        real_bs = len(batch["crops"])

        jersey_number_detection = [None] * real_bs
        jersey_number_confidence = [0.0] * real_bs
        jersey_number_full_detection = [""] * real_bs
        role_detection = [None] * real_bs
        role_confidence = [0.0] * real_bs

        if self.use_legibility_filter:
            jn_idxs = [
                i
                for i, score in enumerate(batch["legibility_score"])
                if score >= self.legibility_filter_threshold
            ]
        else:
            jn_idxs = list(range(len(batch["crops"])))

        if jn_idxs:
            crops = [batch["crops"][idx].cpu().numpy() for idx in jn_idxs]
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": Image.fromarray(crop)},
                            {"type": "text", "text": self.jn_text_prompt},
                        ],
                    }
                ]
                for crop in crops
            ]
            output_texts = self._generate_texts(messages)
            for idx, output_text in zip(jn_idxs, output_texts):
                jersey_number = self.extract_numbers(output_text)
                jersey_number_detection[idx] = jersey_number
                jersey_number_confidence[idx] = 1.0 if jersey_number is not None else 0.0
                if self.save_jersey_number_full_detection:
                    jersey_number_full_detection[idx] = output_text

        role_idxs = list(range(len(batch["crops"])))
        if role_idxs:
            images = [batch["images"][idx].cpu().numpy() for idx in role_idxs]
            crops = [batch["crops"][idx].cpu().numpy() for idx in role_idxs]
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": Image.fromarray(image)},
                            {"type": "image", "image": Image.fromarray(crop)},
                            {"type": "text", "text": self.role_text_prompt},
                        ],
                    }
                ]
                for image, crop in zip(images, crops)
            ]
            output_texts = self._generate_texts(messages)
            for idx, output_text in zip(role_idxs, output_texts):
                role = self.extract_role(output_text)
                role_detection[idx] = role
                role_confidence[idx] = 1.0 if role is not None else 0.0

        detections["jersey_number_detection"] = jersey_number_detection
        detections["jersey_number_confidence"] = jersey_number_confidence
        if self.save_jersey_number_full_detection:
            detections["jersey_number_full_detection"] = jersey_number_full_detection
        detections["role_detection"] = role_detection
        detections["role_confidence"] = role_confidence

        return detections
