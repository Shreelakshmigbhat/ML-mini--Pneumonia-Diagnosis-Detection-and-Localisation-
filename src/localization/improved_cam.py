"""Grad-CAM++ maps for the frozen Stage 8 classifier."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from src.models.cnn import Custom10LayerCNN


def compute_gradcam_plus_plus(
    model: Custom10LayerCNN,
    images: torch.Tensor,
    *,
    class_index: int = 1,
    output_size: tuple[int, int] = (128, 128),
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute normalized Grad-CAM++ maps from image inputs and frozen weights.

    The Stage 8 classifier is a linear layer after global average pooling.
    For the exponential class score, its first three feature-map derivatives
    therefore have closed forms in the classifier weights. Common positive
    factors cancel in alpha and in per-image map normalization.
    """
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("Expected grayscale input shaped (batch, 1, height, width)")
    if not 0 <= class_index < model.num_classes:
        raise ValueError(f"class_index must be in [0, {model.num_classes - 1}]")
    if len(output_size) != 2 or min(output_size) <= 0:
        raise ValueError("output_size must contain two positive dimensions")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not torch.isfinite(images).all():
        raise ValueError("Input images must contain only finite values")

    model.eval()
    with torch.no_grad():
        feature_maps = model.forward_features(images)
        class_weights = model.cam_weights()[class_index].view(1, -1, 1, 1)
        spatial_size = feature_maps.shape[-2] * feature_maps.shape[-1]

        second_derivatives = class_weights.square()
        third_derivatives = class_weights.pow(3)
        denominator = (
            2.0 * second_derivatives
            + (feature_maps * third_derivatives).sum(
                dim=(-2, -1), keepdim=True
            )
            / spatial_size
        )
        safe_denominator = torch.where(
            denominator.abs() < epsilon,
            torch.full_like(denominator, epsilon),
            denominator,
        )
        alpha = second_derivatives / safe_denominator
        channel_weights = (alpha * class_weights.relu()).sum(dim=(-2, -1))
        raw_maps = torch.relu(
            torch.einsum("bk,bkhw->bhw", channel_weights, feature_maps)
        )
        maxima = raw_maps.amax(dim=(-2, -1), keepdim=True)
        normalized = raw_maps / maxima.clamp_min(epsilon)
        resized = functional.interpolate(
            normalized.unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        maps = resized.clamp(0.0, 1.0)
        if not torch.isfinite(maps).all():
            raise FloatingPointError("Grad-CAM++ produced a non-finite map")
        return maps
