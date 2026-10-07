"""Depth-first connected-component clustering for thresholded CAMs."""

from __future__ import annotations

import numpy as np

Pixel = tuple[int, int]
Cluster = list[Pixel]


def find_clusters(mask: np.ndarray) -> list[Cluster]:
    """Return 8-connected activated pixel clusters using iterative DFS.

    Pixel coordinates in each cluster are ``(y, x)``. Iterative traversal
    avoids Python recursion limits on large activated regions.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D activation mask, got shape {mask.shape}")
    active = np.asarray(mask, dtype=bool)
    visited = np.zeros(active.shape, dtype=bool)
    height, width = active.shape
    clusters: list[Cluster] = []

    for start_y, start_x in np.argwhere(active):
        y0, x0 = int(start_y), int(start_x)
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        stack = [(y0, x0)]
        cluster: Cluster = []
        while stack:
            y, x = stack.pop()
            cluster.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    neighbor_y, neighbor_x = y + dy, x + dx
                    if (
                        0 <= neighbor_y < height
                        and 0 <= neighbor_x < width
                        and active[neighbor_y, neighbor_x]
                        and not visited[neighbor_y, neighbor_x]
                    ):
                        visited[neighbor_y, neighbor_x] = True
                        stack.append((neighbor_y, neighbor_x))
        clusters.append(cluster)
    return clusters
