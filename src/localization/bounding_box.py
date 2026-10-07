"""Convert activated CAM clusters into image-coordinate boxes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.localization.clustering import Cluster

Box = tuple[float, float, float, float]


def filter_clusters_two_sigma(clusters: Sequence[Cluster]) -> list[Cluster]:
    """Keep cluster pixel counts within the per-image mean +/- 2 population SD.

    This configurable area-outlier rule is an explicit operationalization of
    the paper's 2-standard-deviation post-processing description; exact details
    are not specified in the supplied paper summary.
    """
    if not clusters:
        return []
    sizes = np.asarray([len(cluster) for cluster in clusters], dtype=np.float64)
    mean = float(sizes.mean())
    standard_deviation = float(sizes.std(ddof=0))
    lower = max(0.0, mean - 2.0 * standard_deviation)
    upper = mean + 2.0 * standard_deviation
    return [
        cluster
        for cluster, size in zip(clusters, sizes)
        if lower <= size <= upper
    ]


def clusters_to_boxes(
    clusters: Sequence[Cluster],
    *,
    map_size: tuple[int, int],
    image_size: tuple[int, int],
    apply_two_sigma: bool = True,
) -> list[Box]:
    """Convert ``(y, x)`` clusters to ``(x, y, width, height)`` image boxes."""
    map_height, map_width = map_size
    image_height, image_width = image_size
    if min(map_height, map_width, image_height, image_width) <= 0:
        raise ValueError("Map and image dimensions must be positive")
    selected = (
        filter_clusters_two_sigma(clusters) if apply_two_sigma else list(clusters)
    )
    scale_x = image_width / map_width
    scale_y = image_height / map_height
    boxes: list[Box] = []
    for cluster in selected:
        if not cluster:
            continue
        ys = [point[0] for point in cluster]
        xs = [point[1] for point in cluster]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        x = min_x * scale_x
        y = min_y * scale_y
        width = (max_x + 1 - min_x) * scale_x
        height = (max_y + 1 - min_y) * scale_y
        boxes.append((x, y, width, height))
    return boxes


def activated_mask(cam: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold a normalized CAM using an inclusive configurable cutoff."""
    if cam.ndim != 2:
        raise ValueError(f"Expected a 2D CAM, got shape {cam.shape}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be greater than 0 and at most 1")
    if not np.isfinite(cam).all():
        raise ValueError("CAM values must be finite")
    return cam >= threshold
