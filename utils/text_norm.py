"""Text normalization helpers for SLAKE/LLaVA evaluation."""
from __future__ import annotations

from collections import Counter
import re
from typing import Optional


def normalize_answer(text: str) -> str:
    """Normalize answer strings for exact-match comparison.

    Applies lowercase, strip, whitespace collapse, punctuation trimming, and
    yes/no canonicalization.

    Args:
        text: Raw answer string.

    Returns:
        Normalized answer string for metrics.
    """
    lowered = " ".join(text.lower().strip().split())
    stripped = lowered.strip("\"'`.,;:!?()[]{}")
    if stripped in {"yes", "y", "true"}:
        return "yes"
    if stripped in {"no", "n", "false"}:
        return "no"
    return stripped


def extract_first_non_empty_line(text: str) -> str:
    """Extract the first non-empty line after an optional Answer: prefix.

    Args:
        text: Raw model output.

    Returns:
        First non-empty line after stripping any ``Answer:`` prefix.
    """
    if not text:
        return ""
    span = text.lstrip()
    match = re.search(r"answer:\s*", span, flags=re.IGNORECASE)
    if match:
        span = span[match.end() :]
    for line in span.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def extract_short_answer(text: str, answer_prefix: str = "Answer:") -> str:
    """Extract and normalize a short answer span from model output.

    Args:
        text: Raw model output text.
        answer_prefix: Prefix to strip before extracting the short answer.

    Returns:
        Normalized short answer string.
    """
    if not text:
        return ""
    match = re.search(re.escape(answer_prefix), text, flags=re.IGNORECASE)
    if match:
        span = text[match.end() :]
    else:
        span = text
    first_line = extract_first_non_empty_line(span)
    return normalize_answer(first_line)


def extract_first_line(text: str) -> str:
    """Return the first line without normalization (for debugging).

    Args:
        text: Input text string.

    Returns:
        First line of text, or empty string when input is blank.
    """
    return text.strip().splitlines()[0] if text.strip() else ""


def canonicalize_yes_no(text: str) -> Optional[str]:
    """Return normalized yes/no answer if applicable.

    Args:
        text: Raw answer string.

    Returns:
        ``"yes"``/``"no"`` if canonicalizable, otherwise ``None``.
    """
    normalized = normalize_answer(text)
    if normalized in {"yes", "no"}:
        return normalized
    return None


def open_token_f1(pred: str, gt: str) -> float:
    """Compute token-level F1 using normalized multiset overlap.

    Args:
        pred: Predicted answer string.
        gt: Ground-truth answer string.

    Returns:
        Token-level F1 score in [0, 1].
    """
    pred_tokens = normalize_answer(pred).split()
    gt_tokens = normalize_answer(gt).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
