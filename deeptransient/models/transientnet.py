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
- ``vit_b_16``       – Vision Transformer ViT-B/16 (ImageNet-21k)
- ``clip_vit_b32``   – CLIP ViT-B/32 visual encoder (OpenAI, 512-d)
- ``clip_vit_l14``   – CLIP ViT-L/14 visual encoder (OpenAI, 768-d)

Supported weight initialisations
---------------------------------
- ``imagenet``   – ImageNet-1K pretrained (default torchvision weights)
- ``places365``  – Places365 pretrained; weights are downloaded on first use
                   from http://places2.csail.mit.edu/models_places365/ and
                   cached in ``~/.cache/deeptransient/``
- ``random``     – random (Xavier) initialisation
- ``clip``       – CLIP pretrained (only valid for ``clip_vit_*`` backbones;
                   set automatically when a ``clip_vit_*`` backbone is chosen)

CLIP backbones require ``open-clip-torch``::

    pip install open-clip-torch
"""

from __future__ import annotations

import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
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

#: The 40 transient scene attributes defined by Laffont et al., SIGGRAPH 2014,
#: in the *exact* column order used by the dataset's ``annotations.tsv``.
#: This must match ``annotations/attributes.txt`` in Brown's distribution
#: byte-for-byte — the i-th score predicted by the model corresponds to the
#: i-th name here.
TRANSIENT_ATTRIBUTES: list[str] = [
    "dirty",
    "daylight",
    "night",
    "sunrisesunset",
    "dawndusk",
    "sunny",
    "clouds",
    "fog",
    "storm",
    "snow",
    "warm",
    "cold",
    "busy",
    "beautiful",
    "flowers",
    "spring",
    "summer",
    "autumn",
    "winter",
    "glowing",
    "colorful",
    "dull",
    "rugged",
    "midday",
    "dark",
    "bright",
    "dry",
    "moist",
    "windy",
    "rain",
    "ice",
    "cluttered",
    "soothing",
    "stressful",
    "exciting",
    "sentimental",
    "mysterious",
    "boring",
    "gloomy",
    "lush",
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
    "clip_vit_b32",
    "clip_vit_l14",
)

#: Names accepted by the ``pretrained`` argument.
PRETRAINS: tuple[str, ...] = ("imagenet", "places365", "random", "clip")

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

# open_clip model name → (arch_string, feature_dim)
_CLIP_MODELS: dict[str, tuple[str, int]] = {
    "clip_vit_b32": ("ViT-B-32", 512),
    "clip_vit_l14": ("ViT-L-14", 768),
}

# CLIP image normalisation constants (differ from ImageNet)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_CACHE_DIR = Path.home() / ".cache" / "deeptransient"


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------


def get_transform(backbone: str, split: str = "train") -> transforms.Compose:
    """Return the correct image pre-processing pipeline for *backbone*.

    CLIP backbones require a different normalisation than ImageNet-pretrained
    networks; this function provides the correct transform for each.

    Args:
        backbone: Backbone name.  One of :data:`BACKBONES`.
        split: ``"train"`` (random crop + flip) or ``"test"``/``"val"``
               (deterministic centre crop).

    Returns:
        A :class:`torchvision.transforms.Compose` pipeline.
    """
    is_clip = backbone in _CLIP_MODELS
    mean = _CLIP_MEAN if is_clip else _IMAGENET_MEAN
    std = _CLIP_STD if is_clip else _IMAGENET_STD

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------


def build_backbone(name: str, pretrained: str = "imagenet") -> tuple[nn.Module, int]:
    """Instantiate a backbone and return ``(backbone, feature_dim)``.

    The classification head of the pretrained network is removed so that
    the returned module outputs a flat feature vector of size ``feature_dim``.

    For CLIP backbones (``clip_vit_b32``, ``clip_vit_l14``) the *pretrained*
    argument is ignored – the CLIP weights are always used.  Pass
    ``pretrained="clip"`` to make this explicit.

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

    # ------------------------------------------------------------------
    # CLIP backbones
    # ------------------------------------------------------------------
    if name in _CLIP_MODELS:
        return _build_clip_backbone(name)

    # ------------------------------------------------------------------
    # Standard torchvision backbones
    # ------------------------------------------------------------------
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


def _build_clip_backbone(name: str) -> tuple[nn.Module, int]:
    """Construct and return a CLIP visual encoder.

    Requires ``open-clip-torch`` (``pip install open-clip-torch``).

    The visual encoder is used with its projection head intact so that it
    outputs compact CLIP embeddings (512-d for ViT-B/32, 768-d for ViT-L/14).
    These representations are well-aligned with semantic concepts and work
    very well as frozen features for a lightweight linear head.

    Args:
        name: One of ``"clip_vit_b32"`` or ``"clip_vit_l14"``.

    Returns:
        ``(visual_encoder, feature_dim)`` where the visual encoder accepts
        ``(B, 3, 224, 224)`` tensors pre-processed with :data:`_CLIP_MEAN`
        / :data:`_CLIP_STD` normalisation.
    """
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open-clip-torch is required for CLIP backbones:\n"
            "    pip install open-clip-torch"
        ) from exc

    arch, feat_dim = _CLIP_MODELS[name]
    with warnings.catch_warnings():
        # Suppress the benign QuickGELU activation-mismatch warning that
        # open_clip emits when loading OpenAI weights into a default config.
        warnings.filterwarnings("ignore", message="QuickGELU mismatch")
        clip_model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained="openai"
        )

    # Use the visual encoder with its projection head intact. This outputs
    # compact CLIP embeddings of size feat_dim that are already semantically
    # rich and well-suited to a downstream linear probe or fine-tuning.
    visual = clip_model.visual
    return visual, feat_dim


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
            For ``clip_vit_*`` backbones this is ignored (CLIP weights
            are always used).
        dropout: Dropout probability applied before the prediction head.

    Example::

        model = TransientNet(backbone="resnet50", pretrained="imagenet")
        scores = model(image_batch)  # (B, 40), values in [0, 1]

        # CLIP variant – requires open-clip-torch
        model = TransientNet(backbone="clip_vit_b32")
        scores = model(clip_preprocessed_batch)
    """

    def __init__(
        self,
        backbone: str = "alexnet",
        pretrained: str = "imagenet",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        # CLIP backbones always use CLIP pretraining
        self.pretrained = "clip" if backbone in _CLIP_MODELS else pretrained

        self.backbone, feat_dim = build_backbone(backbone, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, NUM_ATTRIBUTES),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Image tensor of shape ``(B, 3, H, W)``.  For CLIP backbones
               apply :func:`get_transform` with ``backbone="clip_vit_b32"``
               (or ``"clip_vit_l14"``) for correct normalisation.

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
            For ``clip_vit_*`` backbones this is ignored (CLIP weights
            are always used).
        dropout: Dropout probability applied before the classification head.

    Example::

        model = CloudyNet(backbone="resnet50", pretrained="imagenet")
        logits = model(image_batch)  # (B, 2)
        pred = logits.argmax(dim=1)  # 0 = sunny, 1 = cloudy

        # CLIP variant – requires open-clip-torch
        model = CloudyNet(backbone="clip_vit_b32")
        logits = model(clip_preprocessed_batch)
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
        self.pretrained = "clip" if backbone in _CLIP_MODELS else pretrained

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
