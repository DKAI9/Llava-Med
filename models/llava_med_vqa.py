"""LLaVA(-Med) inference wrapper."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
from PIL import Image
from models.llava_official import load_llava_official
from utils.generation_constraints import get_open_begin_suppress_tokens
from utils.prompting import build_prompt, build_user_text_open
from utils.text_norm import extract_first_non_empty_line

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Container for model generation output.

    Attributes:
        raw_generation: Raw decoded model output.
        normalized_generation: Normalized answer string for evaluation.
    """

    raw_generation: str
    normalized_generation: str


class LlavaMedVQA:
    """Wrapper for LLaVA-family VQA inference."""

    def __init__(
        self,
        model_id: str,
        fallback_model_id: Optional[str] = None,
        device_map: str = "auto",
        use_flash_attention: bool = False,
        conv_mode: str = "mistral_instruct",
    ) -> None:
        """Initialize the inference wrapper.

        Args:
            model_id: Primary model identifier/path.
            fallback_model_id: Optional fallback model identifier/path.
            device_map: Device map specifier.
            use_flash_attention: Whether to enable flash attention if supported.
            conv_mode: Conversation template key.
        """
        self.model_id = model_id
        self.fallback_model_id = fallback_model_id
        self.device_map = device_map
        self.use_flash_attention = use_flash_attention
        self.conv_mode = conv_mode
        self.dtype = self._choose_dtype()
        self.processor, self.model = self._load_model()
        self.mm_use_im_start_end = bool(getattr(self.model.config, "mm_use_im_start_end", False))

    @staticmethod
    def _choose_dtype() -> torch.dtype:
        """Pick the preferred autocast dtype based on hardware.

        Returns:
            Selected torch dtype for autocast.
        """
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _load_model(self):
        """Load the official LLaVA model with fallback support.

        Returns:
            Tuple of (processor, model).

        Raises:
            RuntimeError: If no model can be loaded.
        """
        last_error: Exception | None = None
        for model_id in filter(None, [self.model_id, self.fallback_model_id]):
            try:
                processor, model = load_llava_official(
                    model_id,
                    device_map=self.device_map,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                model.eval()
                logger.info("Loaded official LLaVA model %s", model_id)
                return processor, model
            except Exception as exc:  # pragma: no cover - dependent on llava
                last_error = exc
                logger.warning("Failed to load official LLaVA model %s: %s", model_id, exc)
        raise RuntimeError("Unable to load LLaVA model") from last_error

    def _format_prompt(self, question: str) -> str:
        """Format a VQA prompt with the correct image token placement.

        Args:
            question: Raw question string.

        Returns:
            Prompt string suitable for LLaVA generation.
        """
        user_text = build_user_text_open({"question": question})
        return build_prompt(
            conv_mode=self.conv_mode,
            user_text=user_text,
            with_image=True,
            answer_text=None,
            mm_use_im_start_end=self.mm_use_im_start_end,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize a raw model output to a short answer string.

        Args:
            text: Raw model output string.

        Returns:
            Normalized answer string.
        """
        first_line = extract_first_non_empty_line(text)
        lowered = " ".join(first_line.lower().strip().split())
        stripped = lowered.strip("\"'`.,;:!?()[]{}")
        if stripped in {"yes", "y", "true"}:
            return "yes"
        if stripped in {"no", "n", "false"}:
            return "no"
        return stripped

    @staticmethod
    def _prepare_generate_inputs(model: torch.nn.Module, inputs: dict) -> dict:
        """Align input field names for official LLaVA generate() calls.

        Args:
            model: LLaVA model instance.
            inputs: Input dict with token/image fields.

        Returns:
            Updated inputs dict with aligned key names.
        """
        if "inputs" not in inputs and "input_ids" in inputs:
            inputs["inputs"] = inputs["input_ids"]
        if "images" not in inputs and "pixel_values" in inputs:
            inputs["images"] = inputs["pixel_values"]
        if "images" in inputs and "pixel_values" in inputs:
            inputs.pop("pixel_values", None)
        return inputs

    def generate(
        self,
        images: Sequence[Image.Image],
        questions: Sequence[str],
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        do_sample: bool = False,
    ) -> List[GenerationResult]:
        """Generate answers for a batch of image/question pairs.

        Args:
            images: List of input images.
            questions: List of question strings aligned to images.
            max_new_tokens: Maximum generation length.
            temperature: Sampling temperature (0 for deterministic).
            top_p: Nucleus sampling probability.
            num_beams: Beam width for decoding.
            do_sample: Whether to sample stochastically.

        Returns:
            List of GenerationResult objects in the same order as inputs.

        Raises:
            ValueError: If input lengths mismatch or official processor missing.
        """
        if len(images) != len(questions):
            raise ValueError("Images and questions must have the same length")

        prompts = [self._format_prompt(q) for q in questions]
        if not getattr(self.processor, "is_llava_official", False):
            raise ValueError("Official LLaVA processor not available for VQA generation.")
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token

        tokenizer = self.processor.tokenizer
        begin_suppress_tokens = get_open_begin_suppress_tokens(tokenizer)
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        input_ids_list = []
        for prompt in prompts:
            ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            if ids is None:
                logger.error(
                    "tokenizer_image_token returned None. processor=%s prompt=%r",
                    type(self.processor).__name__,
                    prompt,
                )
                raise ValueError("tokenizer_image_token returned None for prompt.")
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long)
            ids = ids.squeeze(0)
            input_ids_list.append(ids)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = input_ids.ne(pad_token_id).long()
        image_tensor = process_images(
            list(images),
            self.processor.image_processor,
            self.model.config,
        )
        inputs = {
            "input_ids": input_ids.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
            "images": image_tensor.to(self.model.device),
        }
        inputs = self._prepare_generate_inputs(self.model, inputs)
        if inputs.get("input_ids") is None:
            logger.error("Official LLaVA inputs missing input_ids. keys=%s", list(inputs.keys()))
            raise ValueError("Missing input_ids in official LLaVA inputs.")

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if begin_suppress_tokens:
            gen_kwargs["begin_suppress_tokens"] = begin_suppress_tokens

        with torch.inference_mode():
            with torch.autocast(
                device_type=self.model.device.type,
                dtype=self.dtype,
                enabled=self.model.device.type == "cuda",
            ):
                output_ids = self.model.generate(**inputs, **gen_kwargs)

        raw_outputs = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        results: List[GenerationResult] = []
        for output in raw_outputs:
            normalized = self._normalize(output)
            results.append(GenerationResult(output, normalized))
        return results

    def generate_single(
        self,
        image: Image.Image,
        question: str,
        **gen_kwargs,
    ) -> GenerationResult:
        """Generate a single answer for one image/question pair.

        Args:
            image: Input image.
            question: Question text.
            **gen_kwargs: Generation overrides forwarded to ``generate``.

        Returns:
            GenerationResult for the single sample.
        """
        return self.generate([image], [question], **gen_kwargs)[0]

    def prompt_has_single_image_token(self, question: str) -> bool:
        """Return True if the formatted prompt contains a single <image> token.

        Args:
            question: Question text.

        Returns:
            True if exactly one image token is present.
        """
        prompt = self._format_prompt(question)
        return prompt.count("<image>") == 1

    def batch_prompt_has_single_image_token(self, questions: Iterable[str]) -> bool:
        """Return True if all prompts contain a single <image> token.

        Args:
            questions: Iterable of question strings.

        Returns:
            True if all prompts have exactly one image token.
        """
        return all(self.prompt_has_single_image_token(q) for q in questions)
