"""Mask preprocessing utilities for SLAKE.

SLAKE masks can appear in two layouts:
1) A single segmentation map (class-id mask) stored in a dedicated directory
   (e.g. ``mask/``, ``masks/``, ``seg/``). Each pixel value is a class id; a
   ``mask.txt`` file at the dataset root maps ids to labels.
2) A legacy layout with multiple binary ``*mask*.png`` files stored alongside
   the image.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_MASK_DIR_CANDIDATES: Sequence[str] = (
    "mask",
    "masks",
    "seg",
    "segmentation",
    "annotations",
    "img",
)
_MASK_NAME_SUFFIXES: Sequence[str] = ("_mask.png", ".mask.png", "_seg.png")
_DISEASE_KEYWORDS: Sequence[str] = (
    "cancer",
    "tumor",
    "edema",
    "mass",
    "lesion",
    "pneumothorax",
    "effusion",
    "atelectasis",
)


def _load_mask_file(path: Path, threshold: int) -> np.ndarray:
    """Load a binary mask PNG and apply a threshold.

    Args:
        path: Path to a mask image file.
        threshold: Pixel intensity threshold to treat as foreground.

    Returns:
        Boolean mask array with shape (H, W).
    """
    with Image.open(path) as img:
        mask = np.array(img.convert("L"))
    return mask > int(threshold)


def _collect_legacy_mask_paths(image_path: Path) -> List[Path]:
    """Collect legacy per-region mask files near the image.

    Args:
        image_path: Path to the image file.

    Returns:
        Sorted list of ``*mask*.png`` files in the same directory.
    """
    if not image_path.exists():
        return []
    parent = image_path.parent
    return sorted([path for path in parent.iterdir() if path.suffix.lower() == ".png" and "mask" in path.name])


def _select_masks_for_abnormality(mask_paths: Iterable[Path]) -> List[Path]:
    """Prefer disease-labeled masks for abnormality questions.

    Args:
        mask_paths: Iterable of candidate mask paths.

    Returns:
        Subset of masks containing "disease" in the filename, or all masks.
    """
    disease_masks = [path for path in mask_paths if "disease" in path.name.lower()]
    return disease_masks or list(mask_paths)


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    """Deduplicate while preserving original order.

    Args:
        values: Iterable of string values.

    Returns:
        List of unique values in first-seen order.
    """
    seen = set()
    deduped: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


@functools.lru_cache(maxsize=8192)
def resolve_segmentation_mask_path(root: str, img_name: str) -> Optional[Path]:
    """Resolve a single segmentation-map path for an image.

    Uses directory and filename heuristics to find class-id masks in common
    locations (``mask/``, ``masks/``, ``seg/``, ``segmentation/``, ``annotations/``,
    plus ``img/`` for backward compatibility).

    Args:
        root: SLAKE dataset root directory.
        img_name: Image filename to resolve.

    Returns:
        Path to a segmentation map, or ``None`` if not found.
    """
    if not img_name:
        return None
    root_path = Path(root)
    img_path = Path(img_name)
    stem = img_path.stem
    suffix = img_path.suffix.lower()

    names: List[str] = []
    if suffix == ".png":
        names.append(img_path.name)
    else:
        names.append(f"{stem}.png")
    names.extend(f"{stem}{suffix_name}" for suffix_name in _MASK_NAME_SUFFIXES)
    names = _dedupe_preserve_order(names)

    for dir_name in _MASK_DIR_CANDIDATES:
        mask_dir = root_path / dir_name
        for name in names:
            candidate = mask_dir / name
            if dir_name == "img" and name == img_path.name:
                continue
            if candidate.exists():
                return candidate
    return None


@functools.lru_cache(maxsize=128)
def parse_mask_txt(root: Path) -> Dict[int, str]:
    """Parse SLAKE ``mask.txt`` mapping class ids to labels.

    Args:
        root: SLAKE dataset root directory.

    Returns:
        Mapping from integer class id to label string.
    """
    mask_txt = root / "mask.txt"
    if not mask_txt.exists():
        return {}
    label_map: Dict[int, str] = {}
    with mask_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            try:
                idx = int(key.strip())
            except ValueError:
                continue
            label_map[idx] = value.strip()
    return label_map


def infer_disease_ids(label_map: Dict[int, str]) -> set[int]:
    """Infer disease-related class ids based on keyword matches.

    Args:
        label_map: Mapping of class ids to label strings.

    Returns:
        Set of ids whose labels contain disease keywords.
    """
    disease_ids: set[int] = set()
    for idx, name in label_map.items():
        lower = name.casefold()
        if any(keyword in lower for keyword in _DISEASE_KEYWORDS):
            disease_ids.add(idx)
    return disease_ids


def load_segmentation_map(path: Path) -> np.ndarray:
    """Load a segmentation map as integer class ids.

    Args:
        path: Path to segmentation map PNG.

    Returns:
        Integer array of class ids with shape (H, W).
    """
    with Image.open(path) as img:
        arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _segmentation_to_mask(seg_map: np.ndarray, threshold: int) -> np.ndarray:
    """Convert a segmentation map into a binary mask.

    Args:
        seg_map: Segmentation map array (H, W) with class ids.
        threshold: Threshold for binary maps (0/255).

    Returns:
        Boolean mask array of shape (H, W).
    """
    unique_vals = np.unique(seg_map)
    if set(unique_vals.tolist()).issubset({0, 255}):
        return seg_map > int(threshold)
    if set(unique_vals.tolist()).issubset({0, 1}):
        return seg_map > 0
    return seg_map != 0


def load_mask(
    sample: dict,
    root: Path,
    threshold: int,
    union_mode: str,
) -> Optional[np.ndarray]:
    """Load or build a binary mask for a SLAKE sample.

    Args:
        sample: Sample dict containing ``img_name`` and ``content_type``.
        root: SLAKE dataset root directory.
        threshold: Pixel threshold for binary mask conversion.
        union_mode: Mask union strategy (e.g. disease-first for abnormality).

    Returns:
        Boolean mask array or ``None`` if no mask is available.
    """
    img_name = str(sample.get("img_name", ""))
    seg_path = resolve_segmentation_mask_path(str(root), img_name)
    if seg_path is not None:
        seg_map = load_segmentation_map(seg_path)
        if (
            union_mode == "prefer_disease_for_abnormality"
            and str(sample.get("content_type", "")).lower() == "abnormality"
        ):
            # Prefer disease-class pixels when abnormality is the target content type.
            label_map = parse_mask_txt(root)
            disease_ids = infer_disease_ids(label_map)
            if disease_ids:
                return np.isin(seg_map, list(disease_ids))
        return _segmentation_to_mask(seg_map, threshold)

    image_path = root / "img" / img_name
    mask_paths = _collect_legacy_mask_paths(image_path)
    if not mask_paths:
        return None
    if (
        union_mode == "prefer_disease_for_abnormality"
        and str(sample.get("content_type", "")).lower() == "abnormality"
    ):
        mask_paths = _select_masks_for_abnormality(mask_paths)
    masks = [_load_mask_file(path, threshold) for path in mask_paths]
    if not masks:
        return None
    union = masks[0]
    for mask in masks[1:]:
        union = np.logical_or(union, mask)
    return union


def apply_mask(
    image_pil: Image.Image,
    mask: Optional[np.ndarray],
    mode: str,
    pad_ratio: float,
) -> Image.Image:
    """Apply a binary mask to an image via masking or cropping.

    Args:
        image_pil: Input image as a PIL Image.
        mask: Binary mask array aligned to the image.
        mode: ``none`` to bypass, ``masked`` to zero out background, ``crop`` to crop.
        pad_ratio: Padding ratio applied around crop bounding box.

    Returns:
        PIL image with mask applied according to mode.
    """
    if mode == "none" or mask is None:
        return image_pil
    image = image_pil.convert("RGB")
    img_arr = np.array(image)
    if mask.shape[:2] != img_arr.shape[:2]:
        # Resize mask with nearest-neighbor to preserve binary structure.
        mask_img = Image.fromarray(mask.astype("uint8") * 255)
        mask = np.array(mask_img.resize(image.size, resample=Image.NEAREST)) > 0
    if mode == "masked":
        img_arr[~mask] = 0
        return Image.fromarray(img_arr)
    if mode == "crop":
        coords = np.argwhere(mask)
        if coords.size == 0:
            return image
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        height, width = img_arr.shape[:2]
        box_h = max(1, y_max - y_min + 1)
        box_w = max(1, x_max - x_min + 1)
        pad_h = int(box_h * float(pad_ratio))
        pad_w = int(box_w * float(pad_ratio))
        y_min = max(0, y_min - pad_h)
        y_max = min(height - 1, y_max + pad_h)
        x_min = max(0, x_min - pad_w)
        x_max = min(width - 1, x_max + pad_w)
        return image.crop((x_min, y_min, x_max + 1, y_max + 1))
    return image
