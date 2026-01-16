"""Trainable LLaVA-Med wrapper with OPEN/CLOSED helpers."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image

from models.llava_official import is_unsupported_model_error, load_llava_official
from utils.closed_candidates import length_normalize_score
from utils.closed_router import build_closed_candidates
from utils.generation_constraints import get_open_begin_suppress_tokens
from utils.prompting import (
    build_prompt,
    build_triples_context,
    build_user_text_closed,
    build_user_text_open,
    resolve_stop_strings,
)
from utils.lora_utils import match_lora_target_modules, parse_lora_target_modules
from utils.text_norm import extract_first_non_empty_line, normalize_answer

logger = logging.getLogger(__name__)


def left_pad_1d_tensors(
    sequences: Sequence[torch.Tensor],
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pad 1D tensors to a common length with an attention mask.

    Args:
        sequences: Sequence of 1D token id tensors (lengths may vary).
        pad_token_id: Token id used for left padding.

    Returns:
        Tuple of (input_ids, attention_mask) tensors with shape (B, T).
    """
    if not sequences:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long)
    max_len = max(int(seq.shape[0]) for seq in sequences)
    batch_size = len(sequences)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for idx, seq in enumerate(sequences):
        seq_len = int(seq.shape[0])
        input_ids[idx, -seq_len:] = seq
        attention_mask[idx, -seq_len:] = 1
    return input_ids, attention_mask


def slice_generated_tokens(sequences: torch.Tensor, input_len: int) -> torch.Tensor:
    """Slice generated-only tokens from left-padded sequences.

    Args:
        sequences: Left-padded token id tensor with shape (B, T).
        input_len: Prompt length to slice from.

    Returns:
        Tensor containing only generated tokens.
    """
    return sequences[:, input_len:]


class LlavaMedTrainable(torch.nn.Module):
    """Trainable LLaVA-Med wrapper for SLAKE."""

    def __init__(
        self,
        model_id: str,
        fallback_model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        model_base: Optional[str] = None,
        device_map: str = "auto",
        use_flash_attention: bool = False,
        dtype: Optional[torch.dtype] = None,
        conv_mode: str = "mistral_instruct",
    ) -> None:
        """Initialize the trainable wrapper.

        Args:
            model_id: Primary LLaVA model identifier/path.
            fallback_model_id: Fallback model identifier/path.
            model_base: Optional base model identifier for adapters.
            device_map: Device map specifier for loading.
            use_flash_attention: Whether to enable flash attention if supported.
            dtype: Optional override for model dtype.
            conv_mode: Conversation template key.
        """
        super().__init__()
        self.model_id = model_id
        self.fallback_model_id = fallback_model_id
        self.model_base = model_base
        self.device_map = device_map
        self.use_flash_attention = use_flash_attention
        self.dtype = dtype or self._choose_dtype()
        self.conv_mode = conv_mode
        self.processor, self.model = self._load_model()
        self.mm_use_im_start_end = bool(getattr(self.model.config, "mm_use_im_start_end", False))

    @staticmethod
    def _choose_dtype() -> torch.dtype:
        """Pick the preferred autocast dtype based on hardware."""
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
        for model_id, model_base in [
            (self.model_id, self.model_base),
            (self.fallback_model_id, None),
        ]:
            try:
                processor, model = load_llava_official(
                    model_id,
                    model_base=model_base,
                    device_map=self.device_map,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                logger.info("Loaded official LLaVA model %s", model_id)
                return processor, model
            except Exception as exc:  # pragma: no cover - dependent on HF configs
                last_error = exc
                if is_unsupported_model_error(exc):
                    logger.warning("Unsupported LLaVA model %s: %s", model_id, exc)
                else:
                    logger.warning("Failed to load official LLaVA model %s: %s", model_id, exc)
        raise RuntimeError("Unable to load LLaVA model") from last_error

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[dict] = None) -> None:
        """Enable gradient checkpointing on the underlying model."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                if gradient_checkpointing_kwargs is None:
                    self.model.gradient_checkpointing_enable()
                else:
                    self.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                    )
            except TypeError:
                self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        if not any(param.requires_grad for param in self.model.parameters()):
            logger.warning("Gradient checkpointing enabled, but no trainable parameters found.")

    def _resolve_vision_tower(self) -> Optional[torch.nn.Module]:
        """Resolve the vision tower module across model variants."""
        vision = getattr(self.model, "vision_tower", None)
        if vision is not None:
            return vision
        vision = getattr(self.model, "vision_model", None)
        if vision is not None:
            return vision
        get_vision = getattr(self.model, "get_vision_tower", None)
        if callable(get_vision):
            return get_vision()
        inner_model = getattr(self.model, "model", None)
        if inner_model is not None:
            return getattr(inner_model, "vision_tower", None) or getattr(inner_model, "vision_model", None)
        return None

    def _resolve_mm_projector(self) -> Optional[torch.nn.Module]:
        """Resolve the multimodal projector module across model variants."""
        projector = getattr(self.model, "multi_modal_projector", None)
        if projector is not None:
            return projector
        projector = getattr(self.model, "mm_projector", None)
        if projector is not None:
            return projector
        inner_model = getattr(self.model, "model", None)
        if inner_model is not None:
            return getattr(inner_model, "mm_projector", None) or getattr(
                inner_model, "multi_modal_projector", None
            )
        return None

    def _resolve_language_model(self) -> Optional[torch.nn.Module]:
        """Resolve the language model module across model variants."""
        llm = getattr(self.model, "language_model", None)
        if llm is not None:
            return llm
        llm = getattr(self.model, "model", None)
        if llm is not None:
            return llm
        return None

    def freeze_vision_tower(self) -> None:
        """Freeze the vision tower parameters."""
        vision = self._resolve_vision_tower()
        if vision is None:
            logger.warning("No vision tower found to freeze")
            return
        for param in vision.parameters():
            param.requires_grad = False

    def set_mm_projector_trainable(self) -> None:
        """Mark multimodal projector parameters as trainable."""
        projector = self._resolve_mm_projector()
        if projector is None:
            logger.warning("No multimodal projector found")
            return
        for param in projector.parameters():
            param.requires_grad = True

    def apply_lora_if_enabled(
        self,
        use_lora: bool,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules_raw: Optional[str] = None,
        lora_target_modules: Optional[Union[str, List[str]]] = None,
    ) -> Optional[Dict[str, object]]:
        """Inject LoRA adapters when enabled.

        Args:
            use_lora: Whether to apply LoRA.
            lora_r: LoRA rank.
            lora_alpha: LoRA alpha scaling.
            lora_dropout: LoRA dropout rate.
            lora_target_modules_raw: Raw target module string.
            lora_target_modules: Parsed target module list.

        Returns:
            Summary dict of trainable parameters, or ``None`` if disabled.

        Raises:
            RuntimeError: If peft is unavailable.
            ValueError: If no target modules match or trainable params are zero.
        """
        if not use_lora:
            return None
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("peft is required for --use_lora") from exc

        parsed_target_modules = lora_target_modules
        if parsed_target_modules is None:
            parsed_target_modules = parse_lora_target_modules(lora_target_modules_raw)
        target_modules = parsed_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        module_names = [name for name, _ in self.model.named_modules() if name]
        matched_modules = match_lora_target_modules(module_names, target_modules)
        match_count = len(matched_modules)
        if match_count == 0:
            example_modules = [name for name in module_names if "proj" in name][:30]
            raise ValueError(
                "LoRA target_modules matched zero modules. "
                f"raw='{lora_target_modules_raw}', parsed={target_modules}. "
                f"Example module names (first 30 with 'proj'): {example_modules}. "
                "Suggested defaults for Mistral: "
                "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj."
            )
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_names = [name for name, param in self.model.named_parameters() if param.requires_grad]
        if trainable_params <= 0:
            raise ValueError("LoRA injection resulted in zero trainable parameters.")
        if not any("lora_" in name for name in trainable_names):
            raise ValueError("LoRA injection did not create any trainable lora_ parameters.")
        summary = {
            "raw_target_modules": lora_target_modules_raw,
            "parsed_target_modules": target_modules,
            "parsed_target_modules_type": "regex" if isinstance(target_modules, str) else "list",
            "matched_module_count": match_count,
            "matched_modules": matched_modules[:10],
            "trainable_params": trainable_params,
        }
        logger.info("LoRA summary: %s", summary)
        self.model.print_trainable_parameters()
        return summary

    def unfreeze_last_llm_layers(self, num_layers: int) -> None:
        """Unfreeze the last N transformer layers of the language model.

        Args:
            num_layers: Number of trailing layers to unfreeze.
        """
        if num_layers <= 0:
            return
        llm = self._resolve_language_model()
        if llm is None:
            logger.warning("Unable to locate language model for unfreezing")
            return
        layers = getattr(llm, "layers", None)
        if layers is None:
            layers = getattr(llm, "decoder", None)
        if layers is None:
            logger.warning("Unable to locate transformer layers for unfreezing")
            return
        target_layers = list(layers)[-num_layers:]
        for layer in target_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, pixel_values, labels):
        """Forward pass through the underlying LLaVA model.

        Args:
            input_ids: Token ids tensor with shape (B, T).
            attention_mask: Attention mask tensor with shape (B, T).
            pixel_values: Image tensor with shape (B, C, H, W).
            labels: Label tensor for supervised training.

        Returns:
            Model outputs from the underlying LLaVA model.
        """
        if getattr(self.processor, "is_llava_official", False):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                labels=labels,
            )
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )

    def count_trainable_params(self) -> Dict[str, int]:
        """Count total and trainable parameters by component.

        Returns:
            Dict of parameter counts keyed by component name.
        """
        totals = {"total": 0, "vision_tower": 0, "mm_projector": 0, "lora": 0, "llm": 0}
        vision = self._resolve_vision_tower()
        projector = self._resolve_mm_projector()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            numel = param.numel()
            totals["total"] += numel
            if vision is not None and (name.startswith("vision_tower") or name.startswith("vision_model")):
                totals["vision_tower"] += numel
            elif projector is not None and ("mm_projector" in name or "multi_modal_projector" in name):
                totals["mm_projector"] += numel
            elif "lora_" in name:
                totals["lora"] += numel
            else:
                totals["llm"] += numel
        return totals

    def get_param_norms(self, tags: Sequence[str]) -> Dict[str, float]:
        """Compute L2 norms for parameter groups matching tags.

        Args:
            tags: Iterable of component tags to measure.

        Returns:
            Mapping of tag -> L2 norm.
        """
        norms: Dict[str, float] = {}
        for tag in tags:
            params: List[torch.Tensor] = []
            if tag == "mm_projector":
                projector = self._resolve_mm_projector()
                if projector is not None:
                    params = [p for p in projector.parameters() if p.requires_grad]
            elif tag == "lora":
                params = [
                    p
                    for name, p in self.model.named_parameters()
                    if p.requires_grad and "lora_" in name
                ]
            elif tag == "vision_tower":
                vision = self._resolve_vision_tower()
                if vision is not None:
                    params = [p for p in vision.parameters() if p.requires_grad]
            elif tag == "llm":
                params = [
                    p
                    for name, p in self.model.named_parameters()
                    if p.requires_grad and "lora_" not in name and "mm_projector" not in name
                ]
            if not params:
                norms[tag] = 0.0
                continue
            total = torch.zeros((), device=params[0].device)
            for param in params:
                total += param.detach().float().norm(2) ** 2
            norms[tag] = float(torch.sqrt(total).item())
        return norms

    def build_prompt(
        self,
        user_text: str,
        answer_text: Optional[str],
        add_generation_prompt: bool,
    ) -> str:
        """Build a LLaVA prompt for SFT or generation.

        Args:
            user_text: User content including question and context.
            answer_text: Optional answer text to append.
            add_generation_prompt: Unused placeholder for interface compatibility.

        Returns:
            Prompt string with the correct image token placement.
        """
        del add_generation_prompt
        return build_prompt(
            conv_mode=self.conv_mode,
            user_text=user_text,
            with_image=True,
            answer_text=answer_text,
            mm_use_im_start_end=self.mm_use_im_start_end,
        )

    @staticmethod
    def _prepare_generate_inputs(model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Align input field names for official LLaVA generate() calls.

        Args:
            model: LLaVA model instance.
            inputs: Input dict with token/image fields.

        Returns:
            Updated input dict with aligned key names.
        """
        if "inputs" not in inputs and "input_ids" in inputs:
            inputs["inputs"] = inputs["input_ids"]
        if "images" not in inputs and "pixel_values" in inputs:
            inputs["images"] = inputs["pixel_values"]
        if "images" in inputs and "pixel_values" in inputs:
            inputs.pop("pixel_values", None)
        return inputs

    @staticmethod
    def _ensure_left_padding(tokenizer) -> int:
        """Ensure tokenizer padding is left-aligned for generation.

        Args:
            tokenizer: Tokenizer instance to update in-place.

        Returns:
            Pad token id used for left padding.
        """
        # Decoder-only batched generation requires left padding to avoid misaligned outputs.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            # Some LLaVA tokenizers lack a pad token; fall back to unk/eos for safe padding.
            if tokenizer.unk_token_id is not None:
                pad_token_id = tokenizer.unk_token_id
                pad_token = tokenizer.unk_token
            else:
                pad_token_id = tokenizer.eos_token_id
                pad_token = tokenizer.eos_token
            if pad_token_id is None:
                pad_token_id = 0
                pad_token = None
            tokenizer.pad_token_id = pad_token_id
            if tokenizer.pad_token is None and pad_token is not None:
                tokenizer.pad_token = pad_token
        return tokenizer.pad_token_id

    @staticmethod
    def _is_rank_zero() -> bool:
        """Return True if running on the main process.

        Returns:
            True if current rank is zero or distributed is uninitialized.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def generate_open(
        self,
        images: Sequence[Image.Image],
        user_texts: Sequence[str],
        max_new_tokens: int = 16,
        stop_strings: Optional[Sequence[str]] = None,
        return_metadata: bool = False,
        min_new_tokens: int = 1,
        temperature: float = 0.0,
        do_sample: bool = False,
        bad_words_ids: Optional[Sequence[Sequence[int]]] = None,
    ) -> (
        List[Dict[str, str]]
        | Tuple[List[Dict[str, str]], List[Dict[str, int | str | None]]]
    ):
        """Generate OPEN answers for a batch of images and prompts.

        Args:
            images: List of PIL images.
            user_texts: List of user prompts aligned to images.
            max_new_tokens: Maximum generation length.
            stop_strings: Optional list of stop strings to truncate output.
            return_metadata: Whether to return per-sample metadata.
            min_new_tokens: Minimum number of tokens to generate.
            temperature: Sampling temperature (0 for deterministic).
            do_sample: Whether to sample stochastically.
            bad_words_ids: Optional token-id sequences to block.

        Returns:
            Outputs with raw/extracted/normalized text, and optional metadata.

        Raises:
            ValueError: If input lengths mismatch or tokenizer fails.
        """
        if len(images) != len(user_texts):
            raise ValueError("Images and questions length mismatch")
        prompts = [
            self.build_prompt(user_text=prompt_text, answer_text=None, add_generation_prompt=True)
            for prompt_text in user_texts
        ]
        tokenizer = self.processor.tokenizer
        begin_suppress_tokens = get_open_begin_suppress_tokens(tokenizer)
        pad_token_id = self._ensure_left_padding(tokenizer)
        if getattr(self.processor, "is_llava_official", False):
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.mm_utils import tokenizer_image_token

            image_inputs = self.processor(images=list(images), return_tensors="pt")
            input_ids_list = []
            for prompt in prompts:
                ids = tokenizer_image_token(
                    prompt,
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors=None,
                )
                if ids is None:
                    logger.error(
                        "tokenizer_image_token returned None. processor=%s prompt=%r",
                        type(self.processor).__name__,
                        prompt,
                    )
                    raise ValueError("tokenizer_image_token returned None for prompt.")
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                input_ids_list.append(torch.tensor(ids, dtype=torch.long))

            input_ids, attention_mask = left_pad_1d_tensors(input_ids_list, pad_token_id)
            text_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            inputs = {**text_inputs, **image_inputs}
            if inputs.get("input_ids") is None:
                logger.error(
                    "Official LLaVA inputs missing input_ids. keys=%s",
                    list(inputs.keys()),
                )
                raise ValueError("Missing input_ids in official LLaVA inputs.")
        else:
            inputs = self.processor(
                images=list(images),
                text=prompts,
                padding=True,
                return_tensors="pt",
            )
            if inputs.get("input_ids") is None:
                text_inputs = tokenizer(
                    prompts,
                    padding=True,
                    return_tensors="pt",
                )
                inputs.update(text_inputs)
            if inputs.get("attention_mask") is None and inputs.get("input_ids") is not None:
                inputs["attention_mask"] = (inputs["input_ids"] != pad_token_id).long()
            if inputs.get("input_ids") is None:
                logger.error(
                    "Processor returned no input_ids for multimodal generation. "
                    "processor=%s keys=%s sample_prompt=%r",
                    type(self.processor).__name__,
                    list(inputs.keys()),
                    prompts[0] if prompts else "",
                )
                raise ValueError("Missing input_ids after processor/tokenizer fallback.")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if v is not None}
        inputs = self._prepare_generate_inputs(self.model, inputs)
        stopping_criteria = None
        use_stop_criteria = False
        if stop_strings is None:
            stop_strings = resolve_stop_strings(self.conv_mode)
        if stop_strings:
            stop_strings = [s for s in stop_strings if s and s != "\n"]
            if stop_strings:
                try:
                    from transformers.generation.stopping_criteria import StopStringCriteria, StoppingCriteriaList

                    stopping_criteria = StoppingCriteriaList(
                        [StopStringCriteria(stop_strings=list(stop_strings), tokenizer=self.processor.tokenizer)]
                    )
                    use_stop_criteria = True
                except Exception:  # pragma: no cover - optional dependency
                    stopping_criteria = None
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": int(min_new_tokens),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
        }
        if begin_suppress_tokens:
            gen_kwargs["begin_suppress_tokens"] = begin_suppress_tokens
        if bad_words_ids:
            gen_kwargs["bad_words_ids"] = [list(seq) for seq in bad_words_ids if seq]
        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = stopping_criteria
        input_len_padded = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.model.device.type,
                dtype=self.dtype,
                enabled=self.model.device.type == "cuda",
            ):
                if return_metadata or use_stop_criteria:
                    output_ids = self.model.generate(
                        **inputs,
                        **gen_kwargs,
                        return_dict_in_generate=True,
                        output_scores=return_metadata,
                    )
                else:
                    output_ids = self.model.generate(**inputs, **gen_kwargs)
        sequences = output_ids.sequences if hasattr(output_ids, "sequences") else output_ids
        gen_ids = slice_generated_tokens(sequences, input_len_padded)
        outputs = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        cleaned: List[Dict[str, str]] = []
        metadata: List[Dict[str, int | str | None]] = []
        eos_token_id = tokenizer.eos_token_id
        for idx, (output, tokens) in enumerate(zip(outputs, gen_ids)):
            raw_text = output
            extracted_text = extract_first_non_empty_line(raw_text)
            normalized_text = normalize_answer(extracted_text)
            cleaned.append(
                {
                    "raw_text": raw_text,
                    "extracted_text": extracted_text,
                    "normalized_text": normalized_text,
                }
            )
            first_new_token_id = None
            if sequences.shape[1] > input_len_padded:
                first_new_token_id = int(sequences[idx, input_len_padded].item())
            if return_metadata:
                gen_len = int(tokens.shape[0])
                stopped_by = "max_new_tokens"
                if eos_token_id is not None and eos_token_id in tokens:
                    stopped_by = "eos"
                if stop_strings and any(stop in output for stop in stop_strings):
                    stopped_by = "stop_string"
                metadata.append(
                    {
                        "generated_token_len": gen_len,
                        "stopped_by": stopped_by,
                        "first_new_token_id": first_new_token_id,
                    }
                )
        if return_metadata:
            return cleaned, metadata
        return cleaned

    def score_closed_candidates(
        self,
        image: Image.Image,
        prompt: str,
        candidates: Sequence[str],
    ) -> Dict[str, float]:
        """Score CLOSED candidates by mean log-probability.

        Args:
            image: PIL image for the sample.
            prompt: Prompt text including image token.
            candidates: Candidate label strings to score.

        Returns:
            Mapping of candidate -> length-normalized score.

        Raises:
            ValueError: If candidates are empty or image tensor is missing.
        """
        if not candidates:
            return {}
        tokenizer = self.processor.tokenizer
        prompt_ids = self.encode_prompt_with_image_token(prompt)
        prompt_len = int(prompt_ids.shape[1])
        if tokenizer.bos_token_id is not None and prompt_ids[0, 0].item() != tokenizer.bos_token_id:
            bos = torch.tensor([[tokenizer.bos_token_id]], device=prompt_ids.device)
            prompt_ids = torch.cat([bos, prompt_ids], dim=1)
            prompt_len += 1

        scores: Dict[str, float] = {}
        image_inputs = self.processor(images=[image], return_tensors="pt")
        image_inputs = {
            key: value.to(self.model.device)
            for key, value in image_inputs.items()
            if value is not None
        }
        if getattr(self.processor, "is_llava_official", False):
            image_key = "images"
            image_tensor = image_inputs.get(image_key) or image_inputs.get("pixel_values")
            image_kwargs = {image_key: image_tensor}
        else:
            image_key = "pixel_values"
            image_tensor = image_inputs.get(image_key) or image_inputs.get("images")
            image_kwargs = {image_key: image_tensor}
        if image_tensor is None:
            logger.error(
                "Processor returned no image tensor for scoring. processor=%s keys=%s",
                type(self.processor).__name__,
                list(image_inputs.keys()),
            )
            raise ValueError("Missing image tensor for closed candidate scoring.")
        if torch.is_floating_point(image_tensor):
            image_tensor = image_tensor.to(dtype=self.dtype)
            image_kwargs = {image_key: image_tensor}

        for cand in candidates:
            cand_ids = tokenizer(cand, add_special_tokens=False, return_tensors="pt").input_ids.to(
                self.model.device
            )
            eos = None
            if tokenizer.eos_token_id is not None:
                eos = torch.tensor([[tokenizer.eos_token_id]], device=self.model.device)
            seq_parts = [prompt_ids.to(self.model.device), cand_ids]
            if eos is not None:
                seq_parts.append(eos)
            input_ids = torch.cat(seq_parts, dim=1)
            attention_mask = torch.ones_like(input_ids)
            labels = torch.full_like(input_ids, -100)
            labels[:, prompt_len : prompt_len + cand_ids.shape[1]] = cand_ids
            with torch.inference_mode():
                with torch.autocast(
                    device_type=self.model.device.type,
                    dtype=self.dtype,
                    enabled=self.model.device.type == "cuda",
                ):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        **image_kwargs,
                    )
            logits = outputs.logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            total_logprob = 0.0
            for idx in range(cand_ids.shape[1]):
                token_id = cand_ids[0, idx].item()
                logit_index = prompt_len - 1 + idx
                total_logprob += float(log_probs[logit_index, token_id].item())
            scores[cand] = length_normalize_score(total_logprob, int(cand_ids.shape[1]))
        return scores

    def score_candidate_variants(
        self,
        prompt_ids: torch.Tensor,
        image: Image.Image,
        variant_texts: Sequence[str],
    ) -> List[float]:
        """Score candidate variants by mean log-probability.

        Args:
            prompt_ids: Tokenized prompt ids of shape (1, T).
            image: PIL image for the sample.
            variant_texts: Candidate variant strings to score.

        Returns:
            List of scores aligned to ``variant_texts``.
        """
        if not variant_texts:
            return []
        tokenizer = self.processor.tokenizer
        prompt_ids = prompt_ids.to(self.model.device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_len = int(prompt_ids.shape[1])
        if tokenizer.bos_token_id is not None and prompt_ids[0, 0].item() != tokenizer.bos_token_id:
            bos = torch.tensor([[tokenizer.bos_token_id]], device=prompt_ids.device)
            prompt_ids = torch.cat([bos, prompt_ids], dim=1)
            prompt_len += 1

        image_inputs = self.processor(images=[image], return_tensors="pt")
        image_inputs = {
            key: value.to(self.model.device)
            for key, value in image_inputs.items()
            if value is not None
        }
        if getattr(self.processor, "is_llava_official", False):
            image_key = "images"
            image_tensor = image_inputs.get(image_key) or image_inputs.get("pixel_values")
            image_kwargs = {image_key: image_tensor}
        else:
            image_key = "pixel_values"
            image_tensor = image_inputs.get(image_key) or image_inputs.get("images")
            image_kwargs = {image_key: image_tensor}
        if image_tensor is None:
            logger.error(
                "Processor returned no image tensor for scoring. processor=%s keys=%s",
                type(self.processor).__name__,
                list(image_inputs.keys()),
            )
            raise ValueError("Missing image tensor for closed candidate scoring.")
        if torch.is_floating_point(image_tensor):
            image_tensor = image_tensor.to(dtype=self.dtype)
            image_kwargs = {image_key: image_tensor}

        scores: List[float] = []
        for variant in variant_texts:
            cand_ids = tokenizer(variant, add_special_tokens=False, return_tensors="pt").input_ids.to(
                self.model.device
            )
            input_ids = torch.cat([prompt_ids, cand_ids], dim=1)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                with torch.autocast(
                    device_type=self.model.device.type,
                    dtype=self.dtype,
                    enabled=self.model.device.type == "cuda",
                ):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **image_kwargs,
                    )
            logits = outputs.logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            total_logprob = 0.0
            for idx in range(cand_ids.shape[1]):
                token_id = cand_ids[0, idx].item()
                logit_index = prompt_len - 1 + idx
                total_logprob += float(log_probs[logit_index, token_id].item())
            scores.append(total_logprob / max(int(cand_ids.shape[1]), 1))
        return scores

    def score_closed(
        self,
        sample: Dict,
        image: Image.Image,
        closed_vocab: Sequence[str],
        cfg: Dict,
    ) -> tuple[str, Dict]:
        """Score CLOSED candidates and return the best prediction.

        Args:
            sample: Dataset record with question/answer metadata.
            image: PIL image for the sample.
            closed_vocab: Vocabulary fallback list.
            cfg: Routing config flags and prompt styles.

        Returns:
            Tuple of (predicted_label, debug_info).
        """
        triples_str = build_triples_context(sample, cfg.get("triples_mode", "real_only"))
        options = None
        candidates_info = build_closed_candidates(sample, closed_vocab, cfg)
        route = candidates_info.get("route", "vocab")
        labels = candidates_info.get("labels", [])
        variants = candidates_info.get("variants", {})
        if route == "options":
            options = labels
        user_text = build_user_text_closed(
            sample,
            options=options,
            triples_str=triples_str,
            closed_style=cfg.get("closed_style", "minimal"),
        )
        prompt = build_prompt(
            conv_mode=self.conv_mode,
            user_text=user_text,
            with_image=True,
            answer_text=None,
            mm_use_im_start_end=self.mm_use_im_start_end,
        )
        prompt_ids = self.encode_prompt_with_image_token(prompt)

        label_scores: Dict[str, float] = {}
        best_variants: Dict[str, str] = {}
        for label in labels:
            variant_texts = variants.get(label, [label])
            scores = self.score_candidate_variants(prompt_ids, image, variant_texts)
            if not scores:
                continue
            best_idx = max(range(len(scores)), key=scores.__getitem__)
            label_scores[label] = scores[best_idx]
            best_variants[label] = variant_texts[best_idx]

        sorted_scores = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        best_label = sorted_scores[0][0] if sorted_scores else ""
        best_score = sorted_scores[0][1] if sorted_scores else 0.0
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else best_score
        chosen_variant = best_variants.get(best_label, best_label)
        debug = {
            "route": route,
            "chosen_variant": chosen_variant,
            "best_score": best_score,
            "second_score": second_score,
            "margin": best_score - second_score,
            "candidate_count": len(labels),
        }
        return best_label, debug

    def extract_open_answer(self, output: str) -> str:
        """Extract a normalized OPEN answer from raw output.

        Args:
            output: Raw model output string.

        Returns:
            Normalized answer string.
        """
        extracted = extract_first_non_empty_line(output)
        return normalize_answer(extracted)

    def encode_prompt_with_image_token(self, prompt: str) -> torch.Tensor:
        """Encode a prompt containing an image token.

        Args:
            prompt: Prompt string with image token marker.

        Returns:
            Tensor of token ids with shape (1, T).
        """
        if getattr(self.processor, "is_llava_official", False):
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.mm_utils import tokenizer_image_token

            ids = tokenizer_image_token(
                prompt,
                self.processor.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            if ids is None:
                raise ValueError("tokenizer_image_token returned None for prompt.")
            if isinstance(ids, list):
                ids = torch.tensor([ids], dtype=torch.long)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            return ids.to(self.model.device)
        ids = self.processor.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids.to(self.model.device)
