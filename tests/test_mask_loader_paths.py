"""Tests for segmentation map mask loading and resolution."""
from pathlib import Path

import numpy as np
from PIL import Image

from utils.mask_preprocess import apply_mask, load_mask


def _write_sample_files(root: Path) -> None:
    (root / "img").mkdir(parents=True, exist_ok=True)
    (root / "mask").mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (10, 10), color=(255, 255, 255))
    image.save(root / "img" / "sample.jpg")

    seg_map = np.zeros((10, 10), dtype=np.uint8)
    seg_map[2:5, 2:5] = 150
    seg_map[6:8, 6:9] = 30
    Image.fromarray(seg_map).save(root / "mask" / "sample.png")

    (root / "mask.txt").write_text("30:Left Humerus Head\n150:Kidney Cancer\n", encoding="utf-8")


def test_segmentation_map_union_and_disease_filter(tmp_path: Path) -> None:
    _write_sample_files(tmp_path)
    sample = {"img_name": "sample.jpg", "content_type": "Abnormality"}

    union_mask = load_mask(sample, tmp_path, threshold=1, union_mode="union")
    assert union_mask is not None
    assert union_mask.sum() == 3 * 3 + 2 * 3

    disease_mask = load_mask(
        sample,
        tmp_path,
        threshold=1,
        union_mode="prefer_disease_for_abnormality",
    )
    assert disease_mask is not None
    assert disease_mask.sum() == 3 * 3

    image = Image.open(tmp_path / "img" / "sample.jpg").convert("RGB")
    cropped = apply_mask(image, union_mask, mode="crop", pad_ratio=0.0)
    assert cropped.size[0] < image.size[0]
    assert cropped.size[1] < image.size[1]
