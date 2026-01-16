"""Prompt builders for LLaVA-Med training and evaluation."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from llava.conversation import SeparatorStyle, conv_templates

def build_v6_open_user_text(question: str) -> str:
    """Build the OPEN-question prompt template.

    Args:
        question: Raw question string.

    Returns:
        Prompt text instructing a short, direct answer.
    """
    return (
        "You are answering a medical VQA question.\n"
        "Return ONLY the final answer.\n"
        "- Keep it short: 1 to 4 words.\n"
        "- No explanations, no full sentences.\n"
        "- If yes/no: output exactly 'yes' or 'no'.\n"
        "- If a number: output only the number (and unit only if needed).\n"
        f"Question: {question}"
    )


def triple_is_placeholder(field: object) -> bool:
    """Return True if a triple field represents a placeholder value.

    Args:
        field: Triple field value.

    Returns:
        True if the field is empty or a placeholder token.
    """
    text = str(field or "").strip().lower()
    return text in {"", "_", "none", "null"}


def _is_meaningful_field(field: object) -> bool:
    """Filter out empty/placeholder KG fields.

    Args:
        field: Triple field value.

    Returns:
        True if the field contains meaningful content.
    """
    text = str(field or "").strip()
    if not text:
        return False
    if triple_is_placeholder(text):
        return False
    return text.lower() != "vhead"


def triple_is_real(triple: Sequence[object]) -> bool:
    """Return True if a triple contains at least two meaningful fields.

    Args:
        triple: Sequence representing a triple entry.

    Returns:
        True if at least two fields are meaningful.
    """
    fields = [str(item or "").strip() for item in triple]
    meaningful = [field for field in fields if _is_meaningful_field(field)]
    if len(meaningful) >= 2:
        return True
    if len(fields) >= 3 and _is_meaningful_field(fields[1]) and _is_meaningful_field(fields[2]):
        return True
    return False


def _normalize_triples(triples: Sequence) -> list[list[str]]:
    """Normalize triples into a list-of-list string structure.

    Args:
        triples: Sequence of triple entries or flat fields.

    Returns:
        Normalized list of triples where each triple is a list of strings.
    """
    if not triples:
        return []
    if all(isinstance(item, str) for item in triples):
        if len(triples) == 3:
            return [list(triples)]
        return [[item] for item in triples]
    normalized: list[list[str]] = []
    for item in triples:
        if isinstance(item, (list, tuple)):
            normalized.append([str(part) for part in item])
        else:
            normalized.append([str(item)])
    return normalized


def _format_triple_parts(triple_parts: Sequence[str]) -> str:
    """Format a single triple into a parenthesized string.

    Args:
        triple_parts: Parts of a triple entry.

    Returns:
        Parenthesized triple string or empty string if no parts are meaningful.
    """
    parts = [str(part).strip() for part in triple_parts if _is_meaningful_field(part)]
    if not parts:
        return ""
    return f"({', '.join(parts)})"


def format_triples(triples: Sequence) -> str:
    """Format a sequence of triples into a semicolon-separated string.

    Args:
        triples: Sequence of triple entries.

    Returns:
        Semicolon-separated formatted triple string.
    """
    normalized = _normalize_triples(triples)
    formatted = [_format_triple_parts(item) for item in normalized if triple_is_real(item)]
    formatted = [item for item in formatted if item]
    return "; ".join(formatted)


def build_triples_context(sample: dict, triples_mode: str) -> str:
    """Build a context string for triples based on the configured mode.

    Args:
        sample: Sample dict containing ``triple`` and metadata fields.
        triples_mode: ``off``, ``real_only``, or ``kvqa_real_only``.

    Returns:
        Formatted triples string (may be empty).
    """
    if triples_mode == "off":
        return ""
    if triples_mode == "kvqa_real_only" and str(sample.get("base_type", "")).lower() != "kvqa":
        return ""
    triples_str = format_triples(sample.get("triple", []))
    return triples_str


def build_question_block(question: str, triples_str: str) -> str:
    """Build the question block with optional triple context.

    Args:
        question: Raw question text.
        triples_str: Formatted triples context.

    Returns:
        Question block string containing optional context.
    """
    question = question.strip()
    if triples_str:
        return f"Context triples: {triples_str}\nQuestion: {question}"
    return f"Question: {question}"


def build_user_text_open(sample: dict, triples_str: str = "", open_style: str = "short") -> str:
    """Build the user text for OPEN questions.

    Args:
        sample: Record with ``question`` field.
        triples_str: Optional formatted triples context.
        open_style: Prompt style identifier (currently ``short``).

    Returns:
        Prompt text for the OPEN question.

    Raises:
        ValueError: If an unsupported ``open_style`` is provided.
    """
    if open_style != "short":
        raise ValueError(f"Unsupported open_style: {open_style}")
    return build_v6_open_user_text(sample.get("question", ""))


def build_user_text_closed(
    sample: dict,
    options: Optional[Iterable[str]] = None,
    triples_str: str = "",
    closed_style: str = "minimal",
) -> str:
    """Build the user text for CLOSED questions with optional options list.

    Args:
        sample: Record with ``question`` field.
        options: Explicit options to include in the prompt.
        triples_str: Optional formatted triples context.
        closed_style: Prompt style identifier (currently ``minimal``).

    Returns:
        Prompt text for the CLOSED question.

    Raises:
        ValueError: If an unsupported ``closed_style`` is provided.
    """
    question_block = build_question_block(sample.get("question", ""), triples_str)
    if closed_style != "minimal":
        raise ValueError(f"Unsupported closed_style: {closed_style}")
    if options:
        options_text = "; ".join(str(opt).strip() for opt in options if str(opt).strip())
        if options_text:
            return f"{question_block}\nOptions: {options_text}\nAnswer:"
    return f"{question_block}\nAnswer:"


def build_image_question_text(user_text: str, mm_use_im_start_end: bool) -> str:
    """Prefix user text with the appropriate image token wrapper.

    Args:
        user_text: User content (question + context).
        mm_use_im_start_end: Whether to use <im_start>/<im_end> tokens.

    Returns:
        Prompt text with image token prefix.
    """
    if mm_use_im_start_end:
        prefix = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
    else:
        prefix = DEFAULT_IMAGE_TOKEN
    return f"{prefix}\n{user_text}"


def build_prompt(
    conv_mode: str,
    user_text: str,
    with_image: bool = True,
    answer_text: Optional[str] = None,
    mm_use_im_start_end: bool = False,
) -> str:
    """Build a full conversation prompt using a LLaVA template.

    Args:
        conv_mode: Conversation template key.
        user_text: User content (question + context).
        with_image: Whether to prefix the image token.
        answer_text: Optional assistant response to append (for SFT).
        mm_use_im_start_end: Whether to use <im_start>/<im_end> tokens.

    Returns:
        Prompt string for the model.
    """
    conv = conv_templates[conv_mode].copy()
    qs = build_image_question_text(user_text, mm_use_im_start_end) if with_image else user_text
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], answer_text)
    return conv.get_prompt()


def build_prompt_pair(
    conv_mode: str,
    user_text: str,
    answer_text: str,
    mm_use_im_start_end: bool,
) -> tuple[str, str]:
    """Build prompt variants without and with the answer text.

    Args:
        conv_mode: Conversation template key.
        user_text: User content (question + context).
        answer_text: Ground-truth answer text.
        mm_use_im_start_end: Whether to use <im_start>/<im_end> tokens.

    Returns:
        Tuple of ``(prompt_without_answer, prompt_with_answer)`` strings.
    """
    prompt_no_answer = build_prompt(
        conv_mode=conv_mode,
        user_text=user_text,
        with_image=True,
        answer_text=None,
        mm_use_im_start_end=mm_use_im_start_end,
    )
    prompt_with_answer = build_prompt(
        conv_mode=conv_mode,
        user_text=user_text,
        with_image=True,
        answer_text=answer_text,
        mm_use_im_start_end=mm_use_im_start_end,
    )
    return prompt_no_answer, prompt_with_answer


def resolve_stop_strings(conv_mode: str) -> list[str]:
    """Resolve the stop string for a conversation template.

    Args:
        conv_mode: Conversation template key.

    Returns:
        List containing the stop string (or empty list if none).
    """
    conv = conv_templates[conv_mode]
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if not stop_str:
        stop_str = conv.sep2
    return [stop_str] if stop_str else []
