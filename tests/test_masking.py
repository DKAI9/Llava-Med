"""Tests for label masking in collator."""
from __future__ import annotations

from typing import Dict, List

import torch

from data.collate_llava_sft import LlavaSFTCollator, build_prompt
from utils.prompting import build_user_text_open


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False, max_length: int = 512) -> Dict:
        tokens = [ord(ch) % 100 + 2 for ch in text]
        if truncation:
            tokens = tokens[:max_length]
        return {"input_ids": tokens}


class DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = DummyTokenizer()

    def __call__(self, images: List, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        batch = len(images)
        return {"pixel_values": torch.zeros((batch, 3, 224, 224))}


def test_prompt_tokens_masked() -> None:
    processor = DummyProcessor()
    collator = LlavaSFTCollator(
        processor=processor,
        max_prompt_len=128,
        max_answer_len=16,
        conv_mode="mistral_instruct",
        mm_use_im_start_end=False,
    )
    sample = {
        "qid": 1,
        "question": "What is shown?",
        "answer": "lung",
        "answer_type": "OPEN",
        "image_path": "missing.jpg",
    }
    batch = collator([sample])
    user_text = build_user_text_open(sample)
    prompt_no_answer, _ = build_prompt(
        user_text,
        sample["answer"],
        conv_mode="mistral_instruct",
        mm_use_im_start_end=False,
    )
    prompt_ids = processor.tokenizer(prompt_no_answer, truncation=True, max_length=128)["input_ids"]
    labels = batch["labels"][0].tolist()

    assert all(label == -100 for label in labels[: len(prompt_ids)])
    assert any(label != -100 for label in labels[len(prompt_ids) :])
