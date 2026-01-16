"""Tests for parsing options from questions."""
from utils.closed_candidates import parse_options_from_question


def test_parse_options_comma_or() -> None:
    question = "Which is bigger, lung,liver or heart?"
    options = parse_options_from_question(question)
    assert options == ["lung", "liver", "heart"]


def test_parse_options_slash() -> None:
    question = "Choose A/B/C for the best option."
    options = parse_options_from_question(question)
    assert options == ["A", "B", "C"]


def test_parse_options_missing() -> None:
    question = "What does the image show?"
    options = parse_options_from_question(question)
    assert options == []
