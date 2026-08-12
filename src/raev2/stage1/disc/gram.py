from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torchvision import models

__all__ = ["GramLoss"]


class VGG19StyleFeatureExtractor(nn.Module):
    """VGG19 features for Gatys-style Gram losses."""

    LAYER_TO_INDEX = {
        "conv1_1": 0,
        "relu1_1": 1,
        "conv1_2": 2,
        "relu1_2": 3,
        "conv2_1": 5,
        "relu2_1": 6,
        "conv2_2": 7,
        "relu2_2": 8,
        "conv3_1": 10,
        "relu3_1": 11,
        "conv3_2": 12,
        "relu3_2": 13,
        "conv3_3": 14,
        "relu3_3": 15,
        "conv3_4": 16,
        "relu3_4": 17,
        "conv4_1": 19,
        "relu4_1": 20,
        "conv4_2": 21,
        "relu4_2": 22,
        "conv4_3": 23,
        "relu4_3": 24,
        "conv4_4": 25,
        "relu4_4": 26,
        "conv5_1": 28,
        "relu5_1": 29,
        "conv5_2": 30,
        "relu5_2": 31,
        "conv5_3": 32,
        "relu5_3": 33,
        "conv5_4": 34,
        "relu5_4": 35,
    }

    def __init__(
        self,
        layers: Iterable[str],
        requires_grad: bool = False,
        pretrained: bool = True,
        weights_path: str | None = None,
        pooling: str = "max",
    ) -> None:
        super().__init__()
        self.layers = tuple(layers)
        if pooling not in {"max", "avg"}:
            raise ValueError("VGG19StyleFeatureExtractor pooling must be 'max' or 'avg'.")
        unknown = sorted(set(self.layers) - set(self.LAYER_TO_INDEX))
        if unknown:
            raise ValueError(f"Unsupported VGG19 style layers: {unknown}. Valid layers: {list(self.LAYER_TO_INDEX)}")

        features = models.vgg19(weights=None).features
        if pooling == "avg":
            for index, layer in enumerate(features):
                if isinstance(layer, nn.MaxPool2d):
                    features[index] = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)

        if pretrained:
            if weights_path is None:
                raise ValueError("Gatys GramLoss requires an explicit VGG19 weights_path.")
            path = Path(weights_path)
            if not path.is_file():
                raise FileNotFoundError(f"VGG19 weights not found: {path}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            feature_state = {
                key.removeprefix("features."): value
                for key, value in state.items()
                if key.startswith("features.")
            }
            features.load_state_dict(feature_state, strict=True)

        self.features = features[: max(self.LAYER_TO_INDEX[layer] for layer in self.layers) + 1]
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        capture = {self.LAYER_TO_INDEX[layer]: layer for layer in self.layers}
        h = tensor
        for index, layer in enumerate(self.features):
            h = layer(h)
            name = capture.get(index)
            if name is not None:
                outputs[name] = h
        return outputs


def _named_vgg_outputs(outputs) -> dict[str, torch.Tensor]:
    return {name: value for name, value in zip(outputs._fields, outputs)}


class GramLoss(nn.Module):
    """VGG feature Gram-matrix loss.

    ``legacy`` preserves the original project behavior: reuse LPIPS VGG16 features
    and compare normalized Gram matrices with L1.

    ``gatys`` follows the style-loss formulation used by Gatys et al.: VGG19
    ImageNet-normalized features and
    sum((G - A)^2) / (4 * channels^2 * spatial_size^2) per layer.
    """

    DEFAULT_LAYERS = ("relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3")
    GATYS_LAYERS = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")

    def __init__(
        self,
        layers: Iterable[str] | None = None,
        mode: str = "legacy",
        input_range: str = "minus_one_one",
        clamp_input: bool = False,
        weights_path: str | None = None,
        layer_weights: Iterable[float] | None = None,
        pooling: str = "max",
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in {"legacy", "gatys"}:
            raise ValueError("GramLoss mode must be 'legacy' or 'gatys'.")
        self.input_range = input_range
        if self.input_range not in {"minus_one_one", "zero_one"}:
            raise ValueError("GramLoss input_range must be 'minus_one_one' or 'zero_one'.")
        self.clamp_input = bool(clamp_input)
        self.uses_lpips_features = self.mode == "legacy"
        default_layers = self.DEFAULT_LAYERS if self.mode == "legacy" else self.GATYS_LAYERS
        self.layers = tuple(layers) if layers is not None else default_layers
        if not self.layers:
            raise ValueError("GramLoss requires at least one VGG layer.")
        if layer_weights is None:
            self.layer_weights = tuple(1.0 for _ in self.layers)
            if self.mode == "gatys":
                self.layer_weights = tuple(1.0 / len(self.layers) for _ in self.layers)
        else:
            self.layer_weights = tuple(float(weight) for weight in layer_weights)
            if len(self.layer_weights) != len(self.layers):
                raise ValueError("GramLoss layer_weights must have the same length as layers.")

        if self.mode == "legacy":
            unknown = sorted(set(self.layers) - set(self.DEFAULT_LAYERS))
            if unknown:
                raise ValueError(f"Unsupported Gram VGG layers: {unknown}. Valid layers: {list(self.DEFAULT_LAYERS)}")
            self.net = None
        else:
            self.net = VGG19StyleFeatureExtractor(
                self.layers,
                pretrained=True,
                requires_grad=False,
                weights_path=weights_path,
                pooling=pooling,
            )
            self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
            self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    @staticmethod
    def gram_matrix(feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        flat = feature.float().reshape(batch, channels, height * width)
        gram = torch.bmm(flat, flat.transpose(1, 2))
        return gram / float(channels * height * width)

    @staticmethod
    def gatys_style_loss(input_feature: torch.Tensor, target_feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = input_feature.shape
        spatial_size = height * width
        input_flat = input_feature.float().reshape(batch, channels, spatial_size)
        target_flat = target_feature.float().reshape(batch, channels, spatial_size)
        input_gram = torch.bmm(input_flat, input_flat.transpose(1, 2))
        target_gram = torch.bmm(target_flat, target_flat.transpose(1, 2))
        denom = 4.0 * float(channels ** 2) * float(spatial_size ** 2)
        return torch.sum((input_gram - target_gram) ** 2, dim=(1, 2)) / denom

    def _normalize_for_vgg(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.input_range == "minus_one_one":
            tensor = (tensor + 1.0) * 0.5
        if self.clamp_input:
            tensor = tensor.clamp(0.0, 1.0)
        return (tensor - self.imagenet_mean) / self.imagenet_std

    def forward_from_features(self, feats_input, feats_target, reduction: str = "mean") -> torch.Tensor:
        if self.mode != "legacy":
            raise RuntimeError("forward_from_features is only available for legacy GramLoss.")
        input_by_name = _named_vgg_outputs(feats_input)
        target_by_name = _named_vgg_outputs(feats_target)
        losses = []
        for layer, weight in zip(self.layers, self.layer_weights):
            input_gram = self.gram_matrix(input_by_name[layer])
            target_gram = self.gram_matrix(target_by_name[layer])
            losses.append(weight * torch.mean(torch.abs(input_gram - target_gram), dim=(1, 2)))
        value = torch.stack(losses, dim=0).sum(dim=0)
        if reduction == "none":
            return value
        if reduction == "sum":
            return torch.sum(value)
        if reduction == "mean":
            return torch.mean(value)
        raise ValueError(f"Unsupported reduction '{reduction}'")

    def forward(self, input: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        if self.mode == "legacy":
            raise RuntimeError("legacy GramLoss expects LPIPS features via forward_from_features.")
        input_feats = self.net(self._normalize_for_vgg(input))
        target_feats = self.net(self._normalize_for_vgg(target))
        losses = []
        for layer, weight in zip(self.layers, self.layer_weights):
            losses.append(weight * self.gatys_style_loss(input_feats[layer], target_feats[layer]))
        value = torch.stack(losses, dim=0).sum(dim=0)
        if reduction == "none":
            return value
        if reduction == "sum":
            return torch.sum(value)
        if reduction == "mean":
            return torch.mean(value)
        raise ValueError(f"Unsupported reduction '{reduction}'")
