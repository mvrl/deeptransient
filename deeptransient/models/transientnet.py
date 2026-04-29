"""TransientNet and CloudyNet models for predicting transient scene attributes.

Both architectures share the same CNN backbone → dropout → linear head design.
Multiple backbones are supported so users can reproduce the original AlexNet
results or experiment with more powerful modern networks.

Supported backbones
-------------------
- ``alexnet``        – original paper architecture (Krizhevsky et al., 2012)
- ``resnet18``       – lightweight ResNet variant
- ``resnet50``       – ResNet-50 (recommended modern baseline)
- ``efficientnet_b0`` – EfficientNet-B0
- ``vit_b_16``       – Vision Transformer ViT-B/16

Supported weight initialisations
---------------------------------
- ``imagenet``   – ImageNet-1K pretrained (default torchvision weights)
- ``places365``  – Places365 pretrained; weights are downloaded on first use
                   from http://places2.csail.mit.edu/models_places365/ and
                   cached in ``~/.cache/deeptransient/``
- ``random``     – random (Xavier) initialisation
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    AlexNet_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
)

# ---------------------------------------------------------------------------
# Attribute metadata
# ---------------------------------------------------------------------------

#: The 40 transient scene attributes defined by Laffont et al., SIGGRAPH 2014.
TRANSIENT_ATTRIBUTES: list[str] = [
    "clouds",
    "fog",
    "frost",
    "glowing",
    "haze",
    "ice",
    "is_night",
    "lush",
    "mist",
    "overcast",
    "rain",
    "sand",
    "snow",
    "storm",
    "sunny",
    "warm",
    "wet",
    "autumn",
    "beautiful",
    "bright",
    "calm",
    "cold",
    "colorful",
    "dark",
    "day",
    "dry",
    "dusk",
    "flowers",
    "fresh",
    "grass",
    "hot",
    "humid",
    "mossy",
    "muddy",
    "open",
    "rugged",
    "shiny",
    "spring",
    "summer",
    "windy",
]

NUM_ATTRIBUTES: int = len(TRANSIENT_ATTRIBUTES)
assert NUM_ATTRIBUTES == 40, "Attribute list must contain exactly 40 entries."

#: Names accepted by the ``backbone`` argument.
BACKBONES: tuple[str, ...] = (
    "alexnet",
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "vit_b_16",
)

#: Names accepted by the ``pretrained`` argument.
PRETRAINS: tuple[str, ...] = ("imagenet", "places365", "random")

# URLs for Places365 pretrained weights (CSAILVision/places365)
_PLACES365_URLS: dict[str, str] = {
    "alexnet": (
        "http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar"
    ),
    "resnet18": (
        "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
    ),
    "resnet50": (
        "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
    ),
}

_CACHE_DIR = Path.home() / ".cache" / "deeptransient"


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------


def build_backbone(name: str, pretrained: str = "imagenet") -> tuple[nn.Module, int]:
    """Instantiate a backbone and return ``(backbone, feature_dim)``.

    The classification head of the pretrained network is removed so that
    the returned module outputs a flat feature vector of size ``feature_dim``.

    Args:
        name: Backbone architecture.  One of :data:`BACKBONES`.
        pretrained: Weight initialisation.  One of :data:`PRETRAINS`.
            ``"places365"`` is handled separately via
            :func:`_load_places365_weights` after construction.

    Returns:
        A tuple ``(backbone_module, feature_dim)``.
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}. Choose from {BACKBONES}.")
    if pretrained not in PRETRAINS:
        raise ValueError(
            f"Unknown pretrained init {pretrained!r}. Choose from {PRETRAINS}."
        )

    # For places365 we construct the net without ImageNet weights and load
    # Places365 weights afterwards.
    imagenet_weights: dict[str, Optional[object]] = {
        "alexnet": AlexNet_Weights.DEFAULT,
        "resnet18": ResNet18_Weights.DEFAULT,
        "resnet50": ResNet50_Weights.DEFAULT,
        "efficientnet_b0": EfficientNet_B0_Weights.DEFAULT,
        "vit_b_16": ViT_B_16_Weights.DEFAULT,
    }
    w = imagenet_weights[name] if pretrained == "imagenet" else None

    if name == "alexnet":
        net = models.alexnet(weights=w)
        feat_dim = 4096
        # Drop the final fc1000 layer; keep everything up to fc7 (relu + drop)
        net.classifier = nn.Sequential(*list(net.classifier.children())[:-1])

    elif name in ("resnet18", "resnet50"):
        constructor = models.resnet18 if name == "resnet18" else models.resnet50
        net = constructor(weights=w)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()

    elif name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=w)
        feat_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()

    else:  # vit_b_16
        net = models.vit_b_16(weights=w)
        feat_dim = net.heads.head.in_features
        net.heads = nn.Identity()

    if pretrained == "places365":
        net = _load_places365_weights(net, name)

    return net, feat_dim


def _load_places365_weights(net: nn.Module, backbone: str) -> nn.Module:
    """Download (if needed) and load Places365 pretrained weights.

    Weights are cached in ``~/.cache/deeptransient/``.
    Supported backbones: ``alexnet``, ``resnet18``, ``resnet50``.
    """
    if backbone not in _PLACES365_URLS:
        raise ValueError(
            f"Places365 weights are not available for {backbone!r}. "
            f"Supported: {sorted(_PLACES365_URLS)}."
        )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{backbone}_places365.pth.tar"

    if not cache_path.exists():
        url = _PLACES365_URLS[backbone]
        print(f"Downloading Places365 weights for {backbone} from {url} …")
        urllib.request.urlretrieve(url, cache_path)

    checkpoint = torch.load(cache_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Strip "module." prefix added by DataParallel during original training
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # Load only matching keys; the task head will be randomly re-initialised
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if unexpected:
        # Only warn if keys beyond the final classification layer are missing
        task_head_keys = {"classifier.6.weight", "classifier.6.bias", "fc.weight", "fc.bias"}
        non_head_unexpected = [k for k in unexpected if k not in task_head_keys]
        if non_head_unexpected:
            print(f"Warning: unexpected keys in Places365 checkpoint: {non_head_unexpected}")
    return net


# ---------------------------------------------------------------------------
# TransientNet
# ---------------------------------------------------------------------------


class TransientNet(nn.Module):
    """Predicts the 40 transient scene attributes of a single RGB image.

    The model outputs a 40-dimensional vector of values in ``[0, 1]``,
    representing the degree to which each attribute is present.

    Training loss: Mean Squared Error (MSE) between predictions and
    crowd-sourced mean annotations.

    Args:
        backbone: CNN backbone.  One of :data:`BACKBONES`.
        pretrained: Weight initialisation.  One of :data:`PRETRAINS`.
        dropout: Dropout probability applied before the prediction head.

    Example::

        model = TransientNet(backbone="resnet50", pretrained="imagenet")
        scores = model(image_batch)  # (B, 40), values in [0, 1]
    """

    def __init__(
        self,
        backbone: str = "alexnet",
        pretrained: str = "imagenet",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.pretrained = pretrained

        self.backbone, feat_dim = build_backbone(backbone, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, NUM_ATTRIBUTES),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Attribute scores of shape ``(B, 40)`` with values in ``[0, 1]``.
        """
        return self.head(self.backbone(x))

    @property
    def config(self) -> dict:
        """Return a serialisable configuration dict."""
        return {
            "task": "transient_attrs",
            "backbone": self.backbone_name,
            "pretrained": self.pretrained,
            "num_attributes": NUM_ATTRIBUTES,
            "attributes": TRANSIENT_ATTRIBUTES,
        }

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> "TransientNet":
        return cls(
            backbone=config["backbone"],
            pretrained=config.get("pretrained", "random"),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# CloudyNet
# ---------------------------------------------------------------------------


class CloudyNet(nn.Module):
    """Classifies an outdoor scene image as sunny (0) or cloudy (1).

    Training loss: Cross-entropy.
    Evaluation metric: normalised accuracy = max((acc − 0.5) / 0.5, 0),
    as defined by Lu et al. (2014).

    Args:
        backbone: CNN backbone.  One of :data:`BACKBONES`.
        pretrained: Weight initialisation.  One of :data:`PRETRAINS`.
        dropout: Dropout probability applied before the classification head.

    Example::

        model = CloudyNet(backbone="resnet50", pretrained="imagenet")
        logits = model(image_batch)  # (B, 2)
        pred = logits.argmax(dim=1)  # 0 = sunny, 1 = cloudy
    """

    CLASSES: list[str] = ["sunny", "cloudy"]

    def __init__(
        self,
        backbone: str = "alexnet",
        pretrained: str = "imagenet",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.pretrained = pretrained

        self.backbone, feat_dim = build_backbone(backbone, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Class logits of shape ``(B, 2)``.
        """
        return self.head(self.backbone(x))

    @property
    def config(self) -> dict:
        """Return a serialisable configuration dict."""
        return {
            "task": "two_class_weather",
            "backbone": self.backbone_name,
            "pretrained": self.pretrained,
            "classes": self.CLASSES,
        }

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> "CloudyNet":
        return cls(
            backbone=config["backbone"],
            pretrained=config.get("pretrained", "random"),
            **kwargs,
        )
