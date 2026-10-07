"""CAM-compatible 10-convolution-layer CNN for grayscale chest X-rays."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


INPUT_SIZE = (128, 128)
CHANNELS = (32, 32, 64, 64, 96, 96, 128, 128, 128, 128)
NUM_CLASSES = 2


class Custom10LayerCNN(nn.Module):
    """Ten 3x3 convolutions, global average pooling, and one linear classifier.

    ``forward_features`` exposes the final post-ReLU convolutional maps, and
    ``classifier.weight`` provides the linear class weights needed for CAM.
    """

    def __init__(
        self,
        channels: Iterable[int] = CHANNELS,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        channel_sizes = tuple(channels)
        if len(channel_sizes) != 10:
            raise ValueError("The CNN must be configured with exactly 10 convolutions")
        if any(width <= 0 for width in channel_sizes):
            raise ValueError("Convolution channel sizes must be positive")
        if num_classes != 2:
            raise ValueError("This baseline is defined for exactly 2 classes")

        layers: list[nn.Module] = []
        in_channels = 1
        for index, out_channels in enumerate(channel_sizes, start=1):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    nn.ReLU(inplace=False),
                ]
            )
            if index in {2, 4, 6, 8}:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channel_sizes[-1], num_classes)
        self.channels = channel_sizes
        self.num_classes = num_classes

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return final post-ReLU convolutional feature maps for CAM."""
        return self.features(images)

    def forward_with_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.forward_features(images)
        pooled = self.global_average_pool(feature_maps).flatten(start_dim=1)
        return self.classifier(pooled), feature_maps

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(images)
        return logits

    def cam_weights(self) -> torch.Tensor:
        """Return the single final linear layer's class-by-channel weights."""
        return self.classifier.weight

    def verify_cam_compatibility(self) -> dict[str, Any]:
        """Check that exposed feature maps and classifier weights align."""
        device = next(self.parameters()).device
        with torch.no_grad():
            logits, feature_maps = self.forward_with_features(
                torch.zeros(2, 1, *INPUT_SIZE, device=device)
            )
        weights = self.cam_weights()
        convolution_count = sum(
            isinstance(module, nn.Conv2d) for module in self.modules()
        )
        linear_layers = [
            module for module in self.modules() if isinstance(module, nn.Linear)
        ]
        if convolution_count != 10:
            raise AssertionError(f"Expected 10 convolutions, found {convolution_count}")
        if len(linear_layers) != 1 or linear_layers[0] is not self.classifier:
            raise AssertionError("Expected exactly one final linear classifier")
        if feature_maps.shape[1] != weights.shape[1]:
            raise AssertionError("Feature-map channels do not align with FC weights")
        if logits.shape != (2, NUM_CLASSES):
            raise AssertionError(f"Unexpected classifier output shape: {logits.shape}")
        return {
            "convolution_count": convolution_count,
            "final_feature_map_shape": list(feature_maps.shape[1:]),
            "classifier_weight_shape": list(weights.shape),
            "fully_connected_layer_count": len(linear_layers),
            "cam_compatible": True,
        }

    def architecture_config(self) -> dict[str, Any]:
        return {
            "input_channels": 1,
            "input_size": list(INPUT_SIZE),
            "channels": list(self.channels),
            "num_classes": self.num_classes,
            "kernel_size": 3,
            "padding": 1,
            "activation": "ReLU",
            "pooling": "2x2 max pooling after convolutional layers 2, 4, 6, and 8",
            "global_pooling": "Adaptive Average Pooling to 1x1",
            "classifier": "one linear layer",
        }

    def summary(self) -> list[dict[str, int | str]]:
        """Print and return module shapes and parameter counts for one input."""
        if self.training:
            was_training = True
            self.eval()
        else:
            was_training = False

        rows: list[dict[str, int | str]] = []
        hooks: list[Any] = []

        def record(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                return
            params = sum(parameter.numel() for parameter in module.parameters(recurse=False))
            rows.append(
                {
                    "layer": module.__class__.__name__,
                    "output_shape": str(tuple(output.shape)),
                    "parameters": params,
                }
            )

        for module in self.modules():
            if module is not self and not any(module.children()):
                hooks.append(module.register_forward_hook(record))

        try:
            with torch.no_grad():
                self(torch.zeros(1, 1, *INPUT_SIZE, device=next(self.parameters()).device))
        finally:
            for hook in hooks:
                hook.remove()
            if was_training:
                self.train()

        total = sum(parameter.numel() for parameter in self.parameters())
        print(f"{'Layer':<18} {'Output shape':<24} {'Parameters':>12}")
        print("-" * 56)
        for row in rows:
            print(
                f"{row['layer']:<18} {row['output_shape']:<24} "
                f"{row['parameters']:>12,}"
            )
        print("-" * 56)
        print(f"Trainable parameters: {total:,}")
        print(f"Convolutional layers: {sum(isinstance(m, nn.Conv2d) for m in self.modules())}")
        print(f"Fully connected layers: {sum(isinstance(m, nn.Linear) for m in self.modules())}")
        return rows


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
