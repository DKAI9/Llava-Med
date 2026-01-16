"""Routing and candidate building for CLOSED questions."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import hashlib
import random

from utils.text_norm import normalize_answer

YESNO_AUX_VERBS = {"is", "are", "do", "does", "has", "have", "can", "was", "were"}
YESNO_VARIANTS = {
    "Yes": ["Yes", "yes", " yes", "Yes.", " yes.", "YES"],
    "No": ["No", "no", " no", "No.", " no.", "NO"],
}
OPTION_MARKER_RE = re.compile(r"\b(?:or|either|vs|versus)\b|/", flags=re.IGNORECASE)
QUESTION_LEAD_TOKENS = YESNO_AUX_VERBS | {
    "a",
    "an",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "either",
    "has",
    "have",
    "how",
    "is",
    "it",
    "modality",
    "plane",
    "side",
    "should",
    "the",
    "there",
    "this",
    "that",
    "these",
    "those",
    "was",
    "were",
    "what",
    "whether",
    "which",
    "view",
    "who",
    "why",
}
OPTION_STOPWORDS = QUESTION_LEAD_TOKENS | {"correct", "answer", "option", "options", "or", "and", "of", "to"}
MEDICAL_OPTION_TOKENS = {
    "ct",
    "mri",
    "mr",
    "ultrasound",
    "x-ray",
    "xray",
    "pet",
    "pet-ct",
    "axial",
    "coronal",
    "sagittal",
    "hyperdense",
    "hypodense",
    "hyperintense",
    "hypointense",
    "t1",
    "t2",
    "flair",
    "dwi",
    "diffusion",
    "contrast",
    "postcontrast",
    "post-contrast",
    "precontrast",
    "pre-contrast",
}
SOFT_OPTION_WORD_LIMIT = 10


def _normalize_text(text: str) -> str:
    """Normalize text input to a stripped string.

    Args:
        text: Input text or None-like value.

    Returns:
        Stripped string representation.
    """
    return str(text or "").strip()


def has_option_markers(question: str) -> bool:
    """Check if a question contains explicit option markers.

    Args:
        question: Raw question text.

    Returns:
        True if option markers (or/vs/slash) are present.
    """
    return bool(OPTION_MARKER_RE.search(question or ""))


def _strip_leading_question_tokens(text: str) -> str:
    """Remove generic question-leading tokens from an option candidate.

    Args:
        text: Candidate option text.

    Returns:
        Text with leading question tokens removed.
    """
    tokens = text.split()
    while tokens and tokens[0].lower() in QUESTION_LEAD_TOKENS:
        tokens = tokens[1:]
    return " ".join(tokens)


def _is_medical_like(text: str) -> bool:
    """Heuristic for option strings that look like medical jargon.

    Args:
        text: Candidate option text.

    Returns:
        True if the option resembles medical modality/terminology.
    """
    lowered = text.lower()
    if any(token in lowered for token in MEDICAL_OPTION_TOKENS):
        return True
    if re.search(r"\d", text):
        return True
    if "-" in text:
        return True
    return False


def _option_is_informative(text: str) -> bool:
    """Return True if an option contains meaningful content tokens.

    Args:
        text: Normalized option text.

    Returns:
        True if the option has non-stopword content.
    """
    if not text:
        return False
    tokens = text.split()
    if not tokens:
        return False
    if all(token in OPTION_STOPWORDS for token in tokens):
        return False
    if len(text) <= 1:
        return False
    return True


def _normalize_option_text(text: str) -> str:
    """Normalize option text for comparison and deduplication.

    Args:
        text: Raw option text.

    Returns:
        Normalized option string for comparisons.
    """
    cleaned = re.sub(r"^[\s\-\:\.\,\;\(\)\[\]]+|[\s\-\:\.\,\;\(\)\[\]]+$", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _strip_leading_question_tokens(cleaned)
    return normalize_answer(cleaned)


def _finalize_options(options: Sequence[str]) -> List[str]:
    """Filter, normalize, and deduplicate parsed options.

    Args:
        options: Candidate option strings from parsing rules.

    Returns:
        Normalized, deduplicated list of options.
    """
    deduped: List[str] = []
    seen = set()
    for option in options:
        normalized = _normalize_option_text(option)
        if not _option_is_informative(normalized):
            continue
        word_count = len(normalized.split())
        if word_count > SOFT_OPTION_WORD_LIMIT and not _is_medical_like(option):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def parse_explicit_options(question: str) -> List[str]:
    """Parse explicit options like 'A) ... B) ...', 'A or B', or 'A/B', 'A vs B'.

    Args:
        question: Raw question text possibly containing explicit options.

    Returns:
        Normalized list of option strings (length >= 2) or empty list.
    """
    question = str(question or "").strip()
    if not question:
        return []

    labeled_matches = list(
        re.finditer(r"(?:^|[\s\(\[])([A-Z]|\d+)[\.\):]\s*", question)
    )
    if len(labeled_matches) >= 2:
        options: List[str] = []
        for idx, match in enumerate(labeled_matches):
            start = match.end()
            end = labeled_matches[idx + 1].start() if idx + 1 < len(labeled_matches) else len(question)
            option_text = question[start:end].strip(" ;:,.")
            if option_text:
                options.append(option_text)
        finalized = _finalize_options(options)
        if len(finalized) >= 2:
            return finalized

    slash_question = re.sub(r"\s*/\s*", "/", question)
    slash_matches = re.findall(r"\b[^\s/]+(?:/[^\s/]+)+\b", slash_question)
    if slash_matches:
        split_options: List[str] = []
        for match in slash_matches:
            cleaned = match.strip(" ,;:?.")
            split_options.extend(segment for segment in cleaned.split("/") if segment)
        finalized = _finalize_options(split_options)
        if len(finalized) >= 2:
            return finalized

    versus_match = re.search(
        r"\b([A-Za-z0-9][\w\-/]*(?:\s+[A-Za-z0-9][\w\-/]*){0,9})\s+"
        r"(?:vs|versus)\s+"
        r"([A-Za-z0-9][\w\-/]*(?:\s+[A-Za-z0-9][\w\-/]*){0,9})\b",
        question,
        flags=re.IGNORECASE,
    )
    if versus_match:
        finalized = _finalize_options([versus_match.group(1), versus_match.group(2)])
        if len(finalized) >= 2:
            return finalized

    or_match = re.search(
        r"\b([A-Za-z0-9][\w\-/]*(?:\s+[A-Za-z0-9][\w\-/]*){0,9})\s+or\s+"
        r"([A-Za-z0-9][\w\-/]*(?:\s+[A-Za-z0-9][\w\-/]*){0,9})\b",
        question,
        flags=re.IGNORECASE,
    )
    if or_match:
        finalized = _finalize_options([or_match.group(1), or_match.group(2)])
        if len(finalized) >= 2:
            return finalized

    lower_question = question.lower()
    if " or " in lower_question and "," in question:
        or_index = lower_question.rfind(" or ")
        left = question[:or_index]
        right = question[or_index + 4 :]
        left_parts = [part.strip() for part in left.split(",") if part.strip()]
        candidates = left_parts + [right.strip(" ?;:.")]
        finalized = _finalize_options(candidates)
        if len(finalized) >= 2:
            return finalized

    return []


def is_yesno(sample: Dict) -> bool:
    """Determine whether a sample should be routed to yes/no.

    Args:
        sample: Record containing ``question`` and ``answer`` fields.

    Returns:
        True if the sample appears to be yes/no and lacks explicit options.
    """
    answer = _normalize_text(sample.get("answer", "")).lower()
    if answer in {"yes", "no"}:
        return True
    question = _normalize_text(sample.get("question", ""))
    if has_option_markers(question):
        return False
    first_token = question.split(maxsplit=1)[0].lower() if question else ""
    return first_token in YESNO_AUX_VERBS


def canonicalize_yesno(text: str) -> str:
    """Canonicalize yes/no answers to title case.

    Args:
        text: Raw answer string.

    Returns:
        "Yes"/"No" when applicable, otherwise original stripped text.
    """
    normalized = _normalize_text(text).lower()
    if normalized == "yes":
        return "Yes"
    if normalized == "no":
        return "No"
    return _normalize_text(text)


def sample_negatives(
    vocab: Sequence[str],
    k: int,
    seed: int,
    key: object,
    exclude: str,
) -> List[str]:
    """Sample deterministic negatives for MC training.

    Args:
        vocab: Vocabulary of possible answers.
        k: Number of negatives to sample.
        seed: Base seed for determinism.
        key: Stable key used to diversify samples per record.
        exclude: Ground-truth label to exclude from negatives.

    Returns:
        List of sampled negative labels.
    """
    if k <= 0 or not vocab:
        return []
    candidates = [item for item in vocab if item != exclude]
    if not candidates:
        return []
    key_bytes = str(key).encode("utf-8")
    key_hash = int(hashlib.md5(key_bytes).hexdigest(), 16)
    # Combine seed and key hash for deterministic per-sample negatives.
    rng = random.Random(int(seed) + key_hash)
    if k >= len(candidates):
        rng.shuffle(candidates)
        return candidates
    return rng.sample(candidates, k)


def find_option_target(options: Sequence[str], answer: str) -> int:
    """Find the index of a normalized answer within options.

    Args:
        options: Option label list.
        answer: Ground-truth answer string.

    Returns:
        Index of matching option, or -1 if not found.
    """
    answer = normalize_answer(answer)
    for idx, option in enumerate(options):
        if normalize_answer(option) == answer:
            return idx
    return -1


def build_closed_candidates(
    sample: Dict,
    closed_vocab: Sequence[str],
    cfg: Dict,
) -> Dict[str, object]:
    """Build CLOSED candidate labels and variants using routing precedence.

    Args:
        sample: Record with ``question`` and ``answer`` fields.
        closed_vocab: Vocabulary fallback list.
        cfg: Routing config flags (e.g. closed_yesno_variants).

    Returns:
        Dict containing:
            - route: "options" | "yesno" | "vocab"
            - labels: candidate labels for scoring
            - variants: mapping of base label -> surface forms
    """
    options = parse_explicit_options(sample.get("question", ""))
    if options:
        # Explicit option list takes precedence over yes/no or vocab fallbacks.
        labels = options
        variants: Dict[str, List[str]] = {}
        for opt in labels:
            base = normalize_answer(opt)
            if not base:
                continue
            variants[base] = list(
                dict.fromkeys([base, f" {base}", f"{base}.", f" {base}."])
            )
        return {"route": "options", "labels": labels, "variants": variants}
    if is_yesno(sample):
        labels = ["Yes", "No"]
        variants = YESNO_VARIANTS if cfg.get("closed_yesno_variants", True) else {k: [k] for k in labels}
        return {"route": "yesno", "labels": labels, "variants": variants}
    if not closed_vocab and not cfg.get("closed_use_vocab_fallback", True):
        return {"route": "vocab", "labels": [], "variants": {}}
    labels = list(closed_vocab)
    variants = {label: [label, f" {label}"] for label in labels if _normalize_text(label)}
    return {"route": "vocab", "labels": labels, "variants": variants}


def build_closed_vocab_from_train(records: Iterable[dict], topk: int) -> List[str]:
    """Build a CLOSED-answer vocabulary from train split records.

    Args:
        records: Iterable of dataset records (may include split metadata).
        topk: Optional cap on the vocabulary size.

    Returns:
        List of answer strings ordered by frequency.
    """
    counter = Counter()
    for record in records:
        split = str(record.get("split", "train")).lower()
        if split and split != "train":
            continue
        if str(record.get("answer_type", "")).upper() != "CLOSED":
            continue
        answer = _normalize_text(record.get("answer", ""))
        if not answer:
            continue
        if answer.strip().lower() in {"yes", "no"}:
            continue
        counter[answer] += 1
    vocab = [item for item, _ in counter.most_common()]
    return vocab[: int(topk)] if topk else vocab


def save_closed_vocab(vocab: Sequence[str], path: Path) -> None:
    """Persist CLOSED vocabulary to JSON on disk.

    Args:
        vocab: Vocabulary list to save.
        path: Output path.
    """
    path.write_text(json.dumps(list(vocab), indent=2), encoding="utf-8")


def load_closed_vocab(path: Path) -> List[str]:
    """Load CLOSED vocabulary from a JSON file.

    Args:
        path: Input JSON file path.

    Returns:
        List of vocabulary items.
    """
    return json.loads(path.read_text(encoding="utf-8"))
