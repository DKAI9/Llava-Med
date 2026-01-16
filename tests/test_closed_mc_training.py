"""Tests for CLOSED MC training helpers."""
from utils.closed_router import canonicalize_yesno, find_option_target, sample_negatives


def test_sample_negatives_deterministic() -> None:
    vocab = ["Liver", "Heart", "Lung", "Kidney"]
    first = sample_negatives(vocab, k=2, seed=7, key="qid-1", exclude="Liver")
    second = sample_negatives(vocab, k=2, seed=7, key="qid-1", exclude="Liver")
    assert first == second


def test_canonicalize_yesno() -> None:
    assert canonicalize_yesno("yes") == "Yes"
    assert canonicalize_yesno("No") == "No"


def test_find_option_target() -> None:
    options = ["Left kidney", "Right kidney", "Liver"]
    assert find_option_target(options, "Right kidney") == 1
