"""Helpers for official LLaVA model loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch


UNSUPPORTED_MODEL_SUBSTRINGS = (
    "llava_mistral",
    "does not recognize",
    "unknown model type",
    "not supported",
)


def is_unsupported_model_error(exc: Exception) -> bool:
    """Check if an exception indicates an unsupported LLaVA model.

    Args:
        exc: Exception from a model load attempt.

    Returns:
        True if the error message matches known unsupported substrings.
    """
    message = str(exc).lower()
    return any(token in message for token in UNSUPPORTED_MODEL_SUBSTRINGS)


class ProcessorShim:
    """Minimal processor wrapper for official LLaVA loaders."""

    def __init__(self, tokenizer, image_processor, model_config) -> None:
        """Initialize the processor shim.

        Args:
            tokenizer: HF tokenizer instance.
            image_processor: HF image processor instance.
            model_config: Model config containing vision settings.
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.is_llava_official = True

    def __call__(self, images: Sequence, return_tensors: str = "pt") -> dict:
        """Process images into pixel values for LLaVA.

        Args:
            images: Sequence of PIL images.
            return_tensors: Tensor type specifier (only ``"pt"`` is supported).

        Returns:
            Dict with a ``pixel_values`` tensor.

        Raises:
            ValueError: If unsupported ``return_tensors`` is provided.
        """
        from llava.mm_utils import process_images

        pixel_values = process_images(images, self.image_processor, self.model_config)
        if return_tensors not in {"pt", None}:
            raise ValueError("ProcessorShim only supports return_tensors='pt' for images.")
        return {"pixel_values": pixel_values}

    def batch_decode(self, ids, skip_special_tokens: bool = True) -> list[str]:
        """Decode token ids into text strings.

        Args:
            ids: Token id sequences.
            skip_special_tokens: Whether to drop special tokens.

        Returns:
            List of decoded strings.
        """
        return self.tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)


def load_llava_official(
    model_path: str,
    *,
    model_base: str | None = None,
    device_map: str,
    device: str | torch.device,
) -> tuple[ProcessorShim, torch.nn.Module]:
    """Load an official LLaVA model and processor shim.

    Args:
        model_path: HF model identifier or local path.
        model_base: Optional base model for LLaVA adapters.
        device_map: Device map specifier for model loading.
        device: Device string or torch.device instance.

    Returns:
        Tuple of (ProcessorShim, model).
    """
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    name_source = model_base or model_path
    model_name = get_model_name_from_path(name_source)
    if model_base is None:
        config_path = Path(model_path) / "config.json"
        if config_path.is_file():
            try:
                with config_path.open("r", encoding="utf-8") as handle:
                    config = json.load(handle)
                model_type = str(config.get("model_type", "")).lower()
            except json.JSONDecodeError:
                model_type = ""
            if "llava" in model_type and "mistral" in model_type and "mistral" not in model_name.lower():
                model_name = "llava-mistral"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_4bit=False,
        load_8bit=False,
        device_map=device_map,
        device=device,
    )
    processor = ProcessorShim(tokenizer, image_processor, model.config)
    return processor, model
