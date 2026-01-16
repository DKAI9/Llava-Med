"""Tests for triple serialization."""
import pytest

pytest.importorskip("llava")

from utils.prompting import format_triples


def test_triple_includes_tail() -> None:
    triple = [["vhead", "belong to", "respiratory system"]]
    output = format_triples(triple)
    assert "respiratory system" in output


def test_triple_filters_placeholder() -> None:
    triple = [["vhead", "_", "_"], ["vhead", "belong to", "digestive system"]]
    output = format_triples(triple)
    assert "digestive system" in output
    assert "_)" not in output
