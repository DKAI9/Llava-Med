"""Candidate utilities for CLOSED-answer evaluation."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

from utils.text_norm import normalize_answer

YESNO_KEYWORDS = {"is", "are", "was", "were", "do", "does", "did", "has", "have", "had"}


def is_yesno_question(question: str, answer_type: str, gt_answer: str | None = None) -> bool:
    """Heuristic to detect yes/no questions.

    Args:
        question: Raw question string.
        answer_type: Answer type label (e.g., OPEN/CLOSED).
        gt_answer: Optional ground-truth answer for overriding heuristics.

    Returns:
        True if the question should be treated as yes/no.
    """
    if answer_type.upper() == "CLOSED" and normalize_answer(gt_answer or "") in {"yes", "no"}:
        return True
    first_token = question.strip().split(maxsplit=1)[0].lower() if question.strip() else ""
    return first_token in YESNO_KEYWORDS


def parse_options_from_question(question: str) -> List[str]:
    """Parse explicit options such as "A, B or C" or "A/B/C".

    Args:
        question: Raw question string containing possible options.

    Returns:
        List of parsed option strings, or empty list when none found.
    """
    slash_match = re.search(
        r"\b([A-Za-z0-9][\w-]*(?:\s*/\s*[A-Za-z0-9][\w-]*)+)\b",
        question,
    )
    if slash_match:
        option_block = slash_match.group(1)
        parts = [part.strip() for part in option_block.split("/") if part.strip()]
        return parts if len(parts) > 1 else []

    comma_or_match = re.search(
        r"(?:,\s*)([A-Za-z0-9][\w\-/ ]*(?:\s*,\s*[A-Za-z0-9][\w\-/ ]*)+\s*(?:or|and)\s*[A-Za-z0-9][\w\-/ ]+)",
        question,
        flags=re.IGNORECASE,
    )
    if not comma_or_match:
        comma_or_match = re.search(
            r"^([A-Za-z0-9][\w\-/ ]*(?:\s*,\s*[A-Za-z0-9][\w\-/ ]*)+\s*(?:or|and)\s*[A-Za-z0-9][\w\-/ ]+)",
            question.strip(),
            flags=re.IGNORECASE,
        )
    if comma_or_match:
        option_block = comma_or_match.group(1)
        parts = re.split(r"\s*(?:,|or|and)\s*", option_block, flags=re.IGNORECASE)
        parts = [part.strip() for part in parts if part.strip()]
        return parts if len(parts) > 1 else []

    or_match = re.search(
        r"\b([A-Za-z0-9][\w\-/ ]+)\s+or\s+([A-Za-z0-9][\w\-/ ]+)\b",
        question,
        flags=re.IGNORECASE,
    )
    if or_match:
        parts = [or_match.group(1).strip(), or_match.group(2).strip()]
        if parts[0].split()[0].lower() in YESNO_KEYWORDS | {"what", "which"}:
            return []
        return parts if len(parts) > 1 else []
    return []


def make_candidate_variants(label: str) -> List[str]:
    """Generate candidate surface-form variants for scoring.

    Args:
        label: Base candidate label string.

    Returns:
        List of surface-form variants (case, leading space, punctuation).
    """
    base = label.strip()
    base_forms = [base, base.lower(), base.capitalize()]
    seen = set()
    variants: List[str] = []
    for form in base_forms:
        if not form:
            continue
        for entry in (form, f" {form}"):
            if entry not in seen:
                seen.add(entry)
                variants.append(entry)
    for form in base_forms:
        if not form:
            continue
        for suffix in [".", ",", ":"]:
            for entry in (f"{form}{suffix}", f" {form}{suffix}"):
                if entry not in seen:
                    seen.add(entry)
                    variants.append(entry)
    return variants


def length_normalize_score(total_logprob: float, token_count: int) -> float:
    """Normalize log-probability by token count.

    Args:
        total_logprob: Sum of per-token log probabilities.
        token_count: Number of tokens in the candidate.

    Returns:
        Length-normalized mean log probability.
    """
    return total_logprob / max(int(token_count), 1)


def build_closed_vocab_from_train(records: Iterable[dict]) -> List[str]:
    """Build closed-answer vocab from train-only CLOSED samples.

    Args:
        records: Iterable of dataset records.

    Returns:
        Answer strings ordered by descending frequency.
    """
    counter = Counter()
    for record in records:
        if str(record.get("answer_type", "")).upper() != "CLOSED":
            continue
        answer = normalize_answer(str(record.get("answer", "")))
        if answer:
            counter[answer] += 1
    return [item for item, _ in counter.most_common()]


def save_closed_vocab(vocab: Sequence[str], path: Path) -> None:
    """Persist the CLOSED vocabulary to disk as JSON."""
    path.write_text(json.dumps(list(vocab), indent=2), encoding="utf-8")


def load_closed_vocab(path: Path) -> List[str]:
    """Load CLOSED vocabulary from a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def select_topk_vocab_candidates(question: str, vocab: Sequence[str], k: int = 50) -> List[str]:
    """Select top-K candidates by token overlap with question.

    Args:
        question: Raw question text.
        vocab: Candidate vocabulary list.
        k: Number of candidates to return.

    Returns:
        Ranked candidate list by token overlap (descending), then lexicographic.
    """
    if not vocab:
        return []
    q_tokens = set(normalize_answer(question).split())
    scored = []
    for cand in vocab:
        tokens = set(cand.split())
        score = len(tokens & q_tokens)
        scored.append((score, cand))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [cand for _, cand in scored[:k]]
    return selected
