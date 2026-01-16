"""Generation constraint helpers for OPEN answers."""
from __future__ import annotations

from typing import List


def get_open_begin_suppress_tokens(tokenizer) -> List[int]:
    """Return token ids to suppress at generation start for OPEN answers.

    Args:
        tokenizer: Tokenizer with ``__call__`` returning ``input_ids``.

    Returns:
        List of token ids to suppress at position 0.
    """
    cached = getattr(tokenizer, "_open_begin_suppress_tokens", None)
    if cached is not None:
        return list(cached)
    variants = ["Image", "image", " Image", " image"]
    ids = set()
    for text in variants:
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if token_ids:
            ids.add(token_ids[0])
    result = sorted(ids)
    setattr(tokenizer, "_open_begin_suppress_tokens", result)
    return result


def get_open_bad_words_ids(tokenizer) -> List[List[int]]:
    """Return bad_words_ids to avoid image-index answers in OPEN decoding.

    Args:
        tokenizer: Tokenizer with ``__call__`` returning ``input_ids``.

    Returns:
        List of token-id sequences to block during generation.
    """
    cached = getattr(tokenizer, "_open_bad_words_ids", None)
    if cached is not None:
        return [list(item) for item in cached]
    variants = ["Image", "image", " Image", " image"]
    ids = set()
    for text in variants:
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if token_ids:
            ids.add(tuple(token_ids))
    result = [list(token_tuple) for token_tuple in sorted(ids)]
    setattr(tokenizer, "_open_bad_words_ids", result)
    return result
