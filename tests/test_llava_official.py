"""Sanity checks for official LLaVA loading helpers."""
from __future__ import annotations

import pytest
import torch

from data.collate_llava_sft import build_prompt
from utils.prompting import build_user_text_open
from models.llava_med_trainable import LlavaMedTrainable


def test_official_tokenization_inserts_image_token() -> None:
    pytest.importorskip("llava")
    pytest.importorskip("transformers")

    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception:
        pytest.skip("Tokenizer not available for offline tests.")

    user_text = build_user_text_open({"question": "What is shown?"})
    prompt, _ = build_prompt(
        user_text,
        "lung",
        conv_mode="mistral_instruct",
        mm_use_im_start_end=False,
    )
    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
    assert IMAGE_TOKEN_INDEX in ids


def test_forward_runs_with_official_images_key() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for forward sanity check.")

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_kwargs = None

        def forward(self, **kwargs):
            self.last_kwargs = kwargs
            return {"loss": torch.tensor(0.0, device=kwargs["input_ids"].device)}

    dummy_model = DummyModel().to("cuda")
    processor = type("Proc", (), {"is_llava_official": True})()

    trainable = LlavaMedTrainable.__new__(LlavaMedTrainable)
    trainable.model = dummy_model
    trainable.processor = processor

    input_ids = torch.zeros((1, 4), dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.zeros((1, 3, 224, 224), device="cuda")
    labels = torch.zeros_like(input_ids)
    trainable.forward(input_ids, attention_mask, pixel_values, labels)

    assert dummy_model.last_kwargs is not None
    assert "images" in dummy_model.last_kwargs
    assert "pixel_values" not in dummy_model.last_kwargs
