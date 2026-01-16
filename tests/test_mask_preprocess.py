"""Tests for mask preprocessing."""
import numpy as np
from PIL import Image

from utils.mask_preprocess import apply_mask


def test_crop_expands_bbox_with_pad() -> None:
    image = Image.new("RGB", (10, 10), color=(255, 255, 255))
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    cropped = apply_mask(image, mask, mode="crop", pad_ratio=0.25)
    assert cropped.size == (6, 6)
