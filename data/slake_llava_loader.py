"""SLAKE dataset loader for LLaVA-style evaluation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class SlakeSample:
    qid: int
    question: str
    answer: str
    answer_type: str
    img_name: str
    img_id: int
    q_lang: str
    modality: str
    location: str
    content_type: str
    base_type: str
    triple: Sequence[str]


def _append_triple_context(question: str, triple: Sequence[str]) -> str:
    if not triple:
        return question
    triples = "; ".join(f"({item})" for item in triple)
    return f"Context triples: {triples}. Question: {question}"


def _normalize_record(record: Dict) -> SlakeSample:
    return SlakeSample(
        qid=int(record["qid"]),
        question=str(record["question"]),
        answer=str(record["answer"]),
        answer_type=str(record["answer_type"]),
        img_name=str(record["img_name"]),
        img_id=int(record["img_id"]),
        q_lang=str(record.get("q_lang", "en")),
        modality=str(record.get("modality", "")),
        location=str(record.get("location", "")),
        content_type=str(record.get("content_type", "")),
        base_type=str(record.get("base_type", "")),
        triple=list(record.get("triple", []) or []),
    )


def _load_local(split_path: Path) -> List[SlakeSample]:
    with split_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [_normalize_record(item) for item in raw]


def load_slake_dataset(
    split: str,
    data_root: Optional[Path] = None,
    use_hf: bool = False,
    use_triple_context: bool = False,
) -> List[SlakeSample]:
    """Load SLAKE dataset.

    Args:
        split: train/validation/test.
        data_root: local root containing {split}.json and images.
        use_hf: use HuggingFace datasets instead of local files.
        use_triple_context: append triple context to question.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("Split must be train/validation/test")

    if use_hf:
        dataset = load_dataset("BoKelvin/SLAKE", split=split)
        records = [_normalize_record(item) for item in dataset]
    else:
        if data_root is None:
            raise ValueError("data_root is required for local SLAKE loading")
        split_path = data_root / f"{split}.json"
        records = _load_local(split_path)

    if use_triple_context:
        updated: List[SlakeSample] = []
        for sample in records:
            question = _append_triple_context(sample.question, sample.triple)
            updated.append(
                SlakeSample(
                    qid=sample.qid,
                    question=question,
                    answer=sample.answer,
                    answer_type=sample.answer_type,
                    img_name=sample.img_name,
                    img_id=sample.img_id,
                    q_lang=sample.q_lang,
                    modality=sample.modality,
                    location=sample.location,
                    content_type=sample.content_type,
                    base_type=sample.base_type,
                    triple=sample.triple,
                )
            )
        records = updated

    logger.info("Loaded %d samples from SLAKE %s", len(records), split)
    return records


def resolve_image_path(data_root: Path, img_name: str) -> Path:
    return data_root / "img" / img_name
