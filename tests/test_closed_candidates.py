"""Tests for closed-candidate utilities."""
from utils.closed_candidates import length_normalize_score, make_candidate_variants


def test_make_candidate_variants_yes() -> None:
    variants = make_candidate_variants("yes")
    assert "yes" in variants
    assert "Yes" in variants
    assert " yes" in variants
    assert "Yes." in variants


def test_length_normalize_prefers_mean() -> None:
    short_score = length_normalize_score(-1.0, 1)
    long_score = length_normalize_score(-1.5, 2)
    assert long_score > short_score
