"""Tests for OPEN answer extraction and token-level F1."""
from __future__ import annotations

import pytest

from utils.text_norm import extract_first_non_empty_line, open_token_f1


def test_extract_first_non_empty_line_handles_answer_prefix() -> None:
    text = "\nAnswer: yes"
    assert extract_first_non_empty_line(text) == "yes"


def test_open_token_f1_overlap() -> None:
    assert open_token_f1("left lung", "lung left") == 1.0


def test_open_token_f1_partial_overlap() -> None:
    score = open_token_f1("left lung apex", "left lung")
    assert 0.0 < score < 1.0


def test_encode_prompt_with_image_token_includes_image_index() -> None:
    pytest.importorskip("llava")
    pytest.importorskip("transformers")

    from llava.constants import IMAGE_TOKEN_INDEX
    from transformers import AutoTokenizer

    from models.llava_med_trainable import LlavaMedTrainable

    try:
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception:
        pytest.skip("Tokenizer not available for offline tests.")

    processor = type("Proc", (), {"is_llava_official": True, "tokenizer": tokenizer})()
    model = type("Dummy", (), {"device": "cpu"})()
    trainable = LlavaMedTrainable.__new__(LlavaMedTrainable)
    trainable.processor = processor
    trainable.model = model

    prompt = "USER: <image>\nWhat is shown?\nASSISTANT:"
    ids = trainable.encode_prompt_with_image_token(prompt)
    assert IMAGE_TOKEN_INDEX in ids.squeeze(0).tolist()
