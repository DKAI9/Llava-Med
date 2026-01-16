"""SLAKE dataset utilities for LLaVA-Med training/evaluation."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image, UnidentifiedImageError

from utils.mask_preprocess import apply_mask, load_mask
from utils.prompting import format_triples

logger = logging.getLogger(__name__)

SLAKE_FIELDS = {
    "img_name",
    "location",
    "answer",
    "modality",
    "base_type",
    "answer_type",
    "question",
    "qid",
    "content_type",
    "triple",
    "img_id",
    "q_lang",
}


def _normalize_record(record: Dict) -> Dict:
    """Normalize a raw SLAKE record into a stable schema.

    Args:
        record: Raw record loaded from JSON or datasets library.

    Returns:
        Dict with all SLAKE_FIELDS populated, defaulting missing values.
    """
    normalized = {
        "img_name": str(record.get("img_name", "")),
        "location": str(record.get("location", "")),
        "answer": str(record.get("answer", "")),
        "modality": str(record.get("modality", "")),
        "base_type": str(record.get("base_type", "")),
        "answer_type": str(record.get("answer_type", "")),
        "question": str(record.get("question", "")),
        "qid": int(record.get("qid", 0)),
        "content_type": str(record.get("content_type", "")),
        "triple": list(record.get("triple", []) or []),
        "img_id": int(record.get("img_id", 0)),
        "q_lang": str(record.get("q_lang", "en")),
    }
    return normalized


def _append_triple_context(
    question: str,
    triple: Sequence,
    triple_k: int = 3,
    max_triples: int | None = None,
) -> str:
    """Append formatted knowledge triples to a question prompt.

    Args:
        question: Natural language question.
        triple: Iterable of triple fields, each sequence-like element.
        triple_k: Maximum fields to keep per triple element.
        max_triples: Optional cap on the number of triples to include.

    Returns:
        Question text with a ``Context triples: ...`` prefix when applicable.
    """
    if not triple:
        return question
    triples = list(triple)
    if max_triples is not None:
        triples = triples[: max(0, int(max_triples))]
    trimmed: List[Sequence] = []
    for item in triples:
        if isinstance(item, (list, tuple)):
            trimmed.append(list(item)[: int(triple_k)])
        else:
            trimmed.append(item)
    formatted = format_triples(trimmed)
    if not formatted:
        return question
    return f"Context triples: {formatted}. Question: {question}"


def load_slake_records(
    source: str,
    split: str,
    slake_root: Path | None = None,
    use_triple_context: bool = False,
    triple_k: int = 3,
    max_triples: int | None = None,
) -> List[Dict]:
    """Load SLAKE records from HF or local files.

    Args:
        source: ``hf`` for HuggingFace datasets or ``local`` for JSON files.
        split: Dataset split name (train/validation/test).
        slake_root: Root directory containing local SLAKE JSON files.
        use_triple_context: Whether to inject KG triples into the question text.
        triple_k: Maximum fields to keep per triple element.
        max_triples: Optional cap on the number of triples to include.

    Returns:
        List of normalized records sorted by ``qid`` for deterministic iteration.

    Raises:
        ValueError: If ``split`` or ``source`` are unsupported, or ``slake_root`` is missing.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("Split must be train/validation/test")
    if source not in {"hf", "local"}:
        raise ValueError("source must be hf or local")

    if source == "hf":
        from datasets import load_dataset

        dataset = load_dataset("BoKelvin/SLAKE", split=split)
        raw_records = [_normalize_record(item) for item in dataset]
    else:
        if slake_root is None:
            raise ValueError("slake_root is required for local source")
        split_path = slake_root / f"{split}.json"
        with split_path.open("r", encoding="utf-8") as handle:
            raw_records = [_normalize_record(item) for item in json.load(handle)]

    raw_records = [rec for rec in raw_records if rec.get("q_lang") == "en"]

    if use_triple_context:
        for record in raw_records:
            record["question"] = _append_triple_context(
                record["question"],
                record.get("triple", []),
                triple_k=triple_k,
                max_triples=max_triples,
            )

    records = sorted(raw_records, key=lambda rec: rec["qid"])
    logger.info("Loaded %d SLAKE records (%s/%s)", len(records), source, split)
    return records


def resolve_image_path(slake_root: Path, img_name: str) -> Path:
    """Resolve the image path for a SLAKE record.

    Args:
        slake_root: SLAKE root directory containing the ``img/`` folder.
        img_name: Image filename.

    Returns:
        Absolute path to the image file.
    """
    return slake_root / "img" / img_name


class SlakeDataset(torch.utils.data.Dataset):
    """SLAKE dataset wrapper that loads images and optional masks."""

    def __init__(
        self,
        records: List[Dict],
        slake_root: Path,
        mask_mode: str = "none",
        mask_pad_ratio: float = 0.1,
        mask_threshold: int = 1,
        mask_union_mode: str = "prefer_disease_for_abnormality",
    ) -> None:
        """Initialize the dataset.

        Args:
            records: Pre-loaded SLAKE records (preferably qid-sorted).
            slake_root: Root directory for local image/mask files.
            mask_mode: ``none``, ``masked``, or ``crop``; applies to images only.
            mask_pad_ratio: Padding ratio for crop masks.
            mask_threshold: Pixel threshold for binary masks.
            mask_union_mode: How to union multiple masks (e.g. disease-first).
        """
        self.records = records
        self.slake_root = slake_root
        self.mask_mode = mask_mode
        self.mask_pad_ratio = float(mask_pad_ratio)
        self.mask_threshold = int(mask_threshold)
        self.mask_union_mode = mask_union_mode

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        """Return a sample with loaded image and optional mask application.

        Args:
            idx: Index into the qid-sorted records list.

        Returns:
            Dict containing record fields plus ``image`` (PIL Image) and
            ``image_path`` (str). ``used_placeholder_image`` is set when file
            loading fails.
        """
        record = dict(self.records[idx])
        image_path = resolve_image_path(self.slake_root, record["img_name"])
        record["image_path"] = str(image_path)
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            # Keep training/evaluation deterministic even when images are missing.
            image = Image.new("RGB", (224, 224), color=0)
            record["used_placeholder_image"] = True
        if self.mask_mode != "none":
            mask = load_mask(record, self.slake_root, self.mask_threshold, self.mask_union_mode)
            image = apply_mask(image, mask, self.mask_mode, self.mask_pad_ratio)
        record["image"] = image
        return record
