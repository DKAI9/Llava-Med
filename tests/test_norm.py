"""Tests for normalization and answer extraction."""
from utils.text_norm import extract_short_answer, normalize_answer


def test_yes_no_canonicalization() -> None:
    assert normalize_answer("YES") == "yes"
    assert normalize_answer("no") == "no"
    assert normalize_answer("True") == "yes"


def test_answer_prefix_extraction() -> None:
    text = "Answer: lung"
    assert extract_short_answer(text) == "lung"


def test_first_line_extraction() -> None:
    text = "liver\nBecause it is visible."
    assert extract_short_answer(text) == "liver"
