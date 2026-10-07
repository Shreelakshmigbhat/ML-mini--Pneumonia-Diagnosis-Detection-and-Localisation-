"""Class activation maps from the final convolutional feature maps."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from src.models.cnn import Custom10LayerCNN


def compute_cam(
    model: Custom10LayerCNN,
    images: torch.Tensor,
    *,
    class_index: int = 1,
    output_size: tuple[int, int] = (128, 128),
    feature_maps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-image ReLU CAMs normalized to [0, 1] at ``output_size``.

    The map is the positive part of the weighted final feature-map sum,
    normalized independently per image, then bilinearly resized. No box labels
    or annotations are accepted by this function.
    """
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("Expected grayscale input shaped (batch, 1, height, width)")
    if not 0 <= class_index < model.num_classes:
        raise ValueError(f"class_index must be in [0, {model.num_classes - 1}]")
    if len(output_size) != 2 or min(output_size) <= 0:
        raise ValueError("output_size must contain two positive dimensions")

    model.eval()
    with torch.inference_mode():
        if feature_maps is None:
            _logits, feature_maps = model.forward_with_features(images)
        if (
            feature_maps.ndim != 4
            or feature_maps.shape[0] != images.shape[0]
            or feature_maps.shape[1] != model.cam_weights().shape[1]
        ):
            raise ValueError(
                "feature_maps must match the input batch and model CAM channels"
            )
        class_weights = model.cam_weights()[class_index]
        raw_cam = torch.relu(torch.einsum("k,bkhw->bhw", class_weights, feature_maps))
        minimum = raw_cam.amin(dim=(-2, -1), keepdim=True)
        maximum = raw_cam.amax(dim=(-2, -1), keepdim=True)
        normalized = (raw_cam - minimum) / (maximum - minimum).clamp_min(
            torch.finfo(raw_cam.dtype).eps
        )
        resized = functional.interpolate(
            normalized.unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return resized.clamp_(0.0, 1.0)
