"""Resize and render normalized class activation maps."""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_heatmap(
    heatmap: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Bilinearly resize a 2D heatmap to ``(height, width)``."""
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        raise ValueError("Expected a non-empty 2D heatmap")
    if not np.isfinite(values).all():
        raise ValueError("Heatmap values must be finite")
    height, width = size
    if height <= 0 or width <= 0:
        raise ValueError("Heatmap dimensions must be positive")

    resized = Image.fromarray(values, mode="F").resize(
        (width, height), resample=Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32).clip(0.0, 1.0)


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Convert a normalized 2D heatmap into a red-yellow RGB heatmap."""
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        raise ValueError("Expected a non-empty 2D heatmap")
    if not np.isfinite(values).all():
        raise ValueError("Heatmap values must be finite")
    values = np.clip(values, 0.0, 1.0)

    red = np.clip(2.0 * values, 0.0, 1.0)
    green = np.clip(2.0 * values - 0.5, 0.0, 1.0)
    blue = np.clip(4.0 * values - 3.0, 0.0, 1.0)
    return np.rint(np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a heatmap over a grayscale or RGB image, returning uint8 RGB."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    base = np.asarray(image)
    if base.ndim == 2:
        base = np.repeat(base[:, :, np.newaxis], 3, axis=2)
    if base.ndim != 3 or base.shape[2] not in (3, 4) or not base.size:
        raise ValueError("Expected a non-empty grayscale or RGB image")
    base = base[:, :, :3]
    if not np.isfinite(base).all():
        raise ValueError("Image values must be finite")
    if np.issubdtype(base.dtype, np.floating):
        base = np.rint(np.clip(base, 0.0, 1.0) * 255)
    base = base.astype(np.uint8)

    resized = resize_heatmap(heatmap, base.shape[:2])
    colored = colorize_heatmap(resized)
    blended = (1.0 - alpha) * base.astype(np.float32) + alpha * colored
    return np.rint(blended).clip(0, 255).astype(np.uint8)
