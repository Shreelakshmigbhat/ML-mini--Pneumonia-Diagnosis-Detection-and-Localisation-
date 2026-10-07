"""Intersection-over-Union and multi-box image-level evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from src.localization.bounding_box import Box


def box_iou(first: Box, second: Box) -> float:
    """Compute continuous-coordinate IoU for ``(x, y, width, height)`` boxes."""
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    first_right, first_bottom = first_x + first_width, first_y + first_height
    second_right, second_bottom = second_x + second_width, second_y + second_height
    intersection_width = max(0.0, min(first_right, second_right) - max(first_x, second_x))
    intersection_height = max(0.0, min(first_bottom, second_bottom) - max(first_y, second_y))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_width) * max(0.0, first_height)
    second_area = max(0.0, second_width) * max(0.0, second_height)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def image_iou(
    predicted_boxes: Sequence[Box], ground_truth_boxes: Sequence[Box]
) -> float:
    """Compute an image-level score with greedy one-to-one box matching.

    Candidate pairs are considered in descending IoU order; each prediction
    and each ground-truth box can be matched at most once. The score is the
    sum of matched IoUs divided by the larger box count, so unmatched boxes
    contribute zero and both missed and extra regions are penalized.
    """
    denominator = max(len(predicted_boxes), len(ground_truth_boxes))
    if denominator == 0:
        return 0.0
    pairs = sorted(
        (
            (box_iou(prediction, truth), prediction_index, truth_index)
            for prediction_index, prediction in enumerate(predicted_boxes)
            for truth_index, truth in enumerate(ground_truth_boxes)
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matched_iou = 0.0
    for overlap, prediction_index, truth_index in pairs:
        if prediction_index in used_predictions or truth_index in used_truths:
            continue
        used_predictions.add(prediction_index)
        used_truths.add(truth_index)
        matched_iou += overlap
        if len(used_predictions) == denominator or len(used_truths) == denominator:
            break
    return matched_iou / denominator
