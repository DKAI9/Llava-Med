"""Tests for collator payloads used in tensorboard logging."""
from __future__ import annotations

from typing import Dict, List

import torch

from data.collate_llava_sft import LlavaSFTCollator, build_prompt
from utils.prompting import build_user_text_open
from utils.text_norm import normalize_answer


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict:
        tokens = [ord(ch) % 100 + 2 for ch in text]
        return {"input_ids": tokens}


class DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = DummyTokenizer()

    def __call__(self, images: List, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        batch = len(images)
        return {"pixel_values": torch.zeros((batch, 3, 224, 224))}


def test_collator_payload_keys_and_lengths() -> None:
    processor = DummyProcessor()
    collator = LlavaSFTCollator(
        processor=processor,
        max_prompt_len=5,
        max_answer_len=3,
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
    assert set(batch.keys()) >= {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "labels",
        "qid",
        "answer_type",
        "prompt_len",
        "answer_len",
        "prompt_truncated",
        "answer_truncated",
        "used_placeholder_image",
    }
    assert batch["input_ids"].shape[0] == 1
    assert batch["pixel_values"].shape == (1, 3, 224, 224)

    user_text = build_user_text_open(sample)
    prompt_no_answer, prompt_with_answer = build_prompt(
        user_text,
        sample["answer"],
        conv_mode="mistral_instruct",
        mm_use_im_start_end=False,
    )
    prompt_ids_full = processor.tokenizer(prompt_no_answer)["input_ids"]
    full_ids = processor.tokenizer(prompt_with_answer)["input_ids"]
    answer_ids_full = full_ids[len(prompt_ids_full) :]
    expected_prompt_len = min(len(prompt_ids_full), 5)
    expected_answer_len = min(len(answer_ids_full), 3)

    assert batch["prompt_len"].item() == expected_prompt_len
    assert batch["answer_len"].item() == expected_answer_len
    labels = batch["labels"][0].tolist()
    assert all(label == -100 for label in labels[:expected_prompt_len])


def test_prompt_contains_single_image_token() -> None:
    user_text = build_user_text_open({"question": "What is shown?"})
    prompt_no_answer, _ = build_prompt(
        user_text,
        "lung",
        conv_mode="mistral_instruct",
        mm_use_im_start_end=False,
    )
    assert prompt_no_answer.count("<image>") == 1


def test_normalize_answer_roundtrip() -> None:
    assert normalize_answer(" Yes ") == "yes"
