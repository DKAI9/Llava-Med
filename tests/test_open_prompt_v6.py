"""Tests for restoring v6 OPEN prompt formatting."""
from __future__ import annotations

import pytest

pytest.importorskip("llava")

from llava.conversation import conv_templates

from utils.prompting import build_image_question_text, build_v6_open_user_text


def test_v6_open_user_text_exact() -> None:
    expected = (
        "You are answering a medical VQA question.\n"
        "Return ONLY the final answer.\n"
        "- Keep it short: 1 to 4 words.\n"
        "- No explanations, no full sentences.\n"
        "- If yes/no: output exactly 'yes' or 'no'.\n"
        "- If a number: output only the number (and unit only if needed).\n"
        "Question: What modality is used?"
    )
    assert build_v6_open_user_text("What modality is used?") == expected


def test_v6_open_user_content_prefixes() -> None:
    expected = build_v6_open_user_text("What modality is used?")
    with_start_end = build_image_question_text(expected, mm_use_im_start_end=True)
    assert with_start_end.startswith("<im_start><image><im_end>\n")
    assert with_start_end[len("<im_start><image><im_end>\n") :] == expected

    without_start_end = build_image_question_text(expected, mm_use_im_start_end=False)
    assert without_start_end.startswith("<image>\n")
    assert without_start_end[len("<image>\n") :] == expected


def test_v6_open_prompt_render() -> None:
    conv_mode = "mistral_instruct"
    expected = build_v6_open_user_text("What modality is used?")
    user_content = build_image_question_text(expected, mm_use_im_start_end=False)
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_content)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)
