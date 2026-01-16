"""TensorBoard logging helpers."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class SafeSummaryWriter:
    """Wrapper around SummaryWriter with guards for missing values."""

    def __init__(self, log_dir: str, **kwargs: Any) -> None:
        """Initialize the summary writer.

        Args:
            log_dir: TensorBoard log directory path.
            **kwargs: Forwarded to ``SummaryWriter``.
        """
        self.writer = SummaryWriter(log_dir=log_dir, **kwargs)

    def safe_add_scalar(self, tag: str, value: Any, step: int) -> None:
        """Safely add a scalar to TensorBoard, ignoring invalid values.

        Args:
            tag: TensorBoard tag name.
            value: Scalar-like value to log.
            step: Global step.
        """
        if value is None:
            return
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return
        if np.isnan(scalar) or np.isinf(scalar):
            return
        self.writer.add_scalar(tag, scalar, step)

    def safe_add_histogram(self, tag: str, values: Any, step: int) -> None:
        """Safely add a histogram to TensorBoard, ignoring empty values.

        Args:
            tag: TensorBoard tag name.
            values: Array-like values to log.
            step: Global step.
        """
        if values is None:
            return
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values)
        if values.size == 0:
            return
        self.writer.add_histogram(tag, values, step)

    def safe_add_text(self, tag: str, text: str, step: int) -> None:
        """Safely add a text entry to TensorBoard.

        Args:
            tag: TensorBoard tag name.
            text: Text content to log.
            step: Global step.
        """
        if not text:
            return
        self.writer.add_text(tag, text, step)

    def safe_add_image(self, tag: str, image: Any, step: int, dataformats: str = "HWC") -> None:
        """Safely add an image to TensorBoard.

        Args:
            tag: TensorBoard tag name.
            image: Image data as PIL, numpy array, or torch tensor.
            step: Global step.
            dataformats: TensorBoard dataformat string.
        """
        if image is None:
            return
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.size == 0:
            return
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            max_val = float(np.max(image)) if image.size else 1.0
            if max_val > 1.0:
                image = image / 255.0
            image = np.clip(image, 0.0, 1.0)
        self.writer.add_image(tag, image, step, dataformats=dataformats)

    def flush(self) -> None:
        """Flush pending TensorBoard events."""
        self.writer.flush()

    def close(self) -> None:
        """Close the underlying TensorBoard writer."""
        self.writer.close()


def format_kv_table(entries: Dict[str, Any]) -> str:
    """Format a key/value mapping into a monospaced table string.

    Args:
        entries: Mapping of key/value pairs to display.

    Returns:
        Formatted table string, or empty string for no entries.
    """
    if not entries:
        return ""
    max_key = max(len(str(key)) for key in entries)
    lines = []
    for key, value in entries.items():
        lines.append(f"{str(key).ljust(max_key)} : {value}")
    return "\n".join(lines)
