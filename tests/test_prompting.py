"""Tests for prompt formatting."""
import pytest

pytest.importorskip("llava")

from utils.prompting import build_prompt, build_user_text_open, build_v6_open_user_text


def test_prompt_has_single_image_token() -> None:
    user_text = build_user_text_open({"question": "What is shown?"})
    prompt = build_prompt(
        conv_mode="mistral_instruct",
        user_text=user_text,
        with_image=True,
        answer_text=None,
        mm_use_im_start_end=False,
    )
    assert prompt.count("<image>") == 1


def test_open_prompt_has_constraint() -> None:
    user_text = build_user_text_open({"question": "What is shown?"})
    assert user_text == build_v6_open_user_text("What is shown?")
