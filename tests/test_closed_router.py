"""Tests for CLOSED routing utilities."""
from utils.closed_router import build_closed_vocab_from_train, is_yesno, parse_explicit_options


def test_parse_explicit_options_lettered() -> None:
    question = "Which is correct? A. Liver B. Heart C. Lung"
    options = parse_explicit_options(question)
    assert options == ["liver", "heart", "lung"]


def test_parse_explicit_options_or_phrase() -> None:
    question = "Is it left kidney or right kidney?"
    options = parse_explicit_options(question)
    assert options == ["left kidney", "right kidney"]


def test_is_yesno_detects_aux() -> None:
    sample = {"question": "Is there a fracture?", "answer": "Yes"}
    assert is_yesno(sample)


def test_closed_vocab_builder_excludes_yesno_and_non_train() -> None:
    records = [
        {"answer_type": "CLOSED", "answer": "Yes", "split": "train"},
        {"answer_type": "CLOSED", "answer": "No", "split": "train"},
        {"answer_type": "CLOSED", "answer": "Liver", "split": "train"},
        {"answer_type": "CLOSED", "answer": "Heart", "split": "validation"},
    ]
    vocab = build_closed_vocab_from_train(records, topk=10)
    assert vocab == ["Liver"]
