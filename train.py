#!/usr/bin/env python3
"""Train TransientNet or CloudyNet.

Quick-start examples
--------------------
Replicate TransientNet-I (AlexNet, ImageNet init)::

    python train.py \\
        --task transient_attrs \\
        --backbone alexnet \\
        --pretrained imagenet \\
        --data-root /path/to/transient_attrs \\
        --output-dir runs/transientnet_alexnet_imagenet

Replicate CloudyNet-H (AlexNet, Hybrid Places365+ImageNet) fold 0::

    python train.py \\
        --task two_class_weather \\
        --backbone alexnet \\
        --pretrained places365 \\
        --data-root /path/to/weather \\
        --fold 0 \\
        --output-dir runs/cloudynet_alexnet_places365_fold0

Train a modern ResNet-50 TransientNet::

    python train.py \\
        --task transient_attrs \\
        --backbone resnet50 \\
        --pretrained imagenet \\
        --data-root /path/to/transient_attrs \\
        --output-dir runs/transientnet_resnet50

Load from a YAML config file::

    python train.py --config configs/transientnet_resnet50.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deeptransient.data import TransientAttributesDataset, TwoClassWeatherDataset
from deeptransient.metrics import mean_attribute_error, normalised_accuracy
from deeptransient.models import CloudyNet, TransientNet, get_transform

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TransientNet or CloudyNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- config file (optional shortcut) --------------------------------
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="YAML",
        help="Path to a YAML config file.  CLI flags override config values.",
    )

    # ---- task & model ---------------------------------------------------
    parser.add_argument(
        "--task",
        choices=["transient_attrs", "two_class_weather"],
        default="transient_attrs",
    )
    parser.add_argument(
        "--backbone",
        choices=["alexnet", "resnet18", "resnet50", "efficientnet_b0", "vit_b_16",
                 "clip_vit_b32", "clip_vit_l14"],
        default="resnet50",
    )
    parser.add_argument(
        "--pretrained",
        choices=["imagenet", "places365", "random", "clip"],
        default="imagenet",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Head dropout probability."
    )

    # ---- data -----------------------------------------------------------
    parser.add_argument("--data-root", type=str, required=True, metavar="PATH")
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        choices=range(5),
        metavar="[0-4]",
        help="Random split fold (two_class_weather only).",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="imageLD",
        help="Image sub-directory name (transient_attrs only).",
    )

    # ---- optimisation ---------------------------------------------------
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=10,
        help="StepLR step size (epochs).",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        help="StepLR decay factor.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision (requires CUDA).",
    )

    # ---- misc -----------------------------------------------------------
    parser.add_argument(
        "--workers", type=int, default=4, help="DataLoader worker processes."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=str, default="runs/default", metavar="PATH"
    )
    parser.add_argument(
        "--resume", type=str, default=None, metavar="CKPT", help="Resume from checkpoint."
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the best checkpoint to HuggingFace Hub after training.",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        metavar="USER/REPO",
        help="HuggingFace Hub repository id (required with --push-to-hub).",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Freeze all backbone parameters and only train the task head. "
            "Highly recommended for CLIP backbones to replicate the minimal "
            "fine-tuning experiment."
        ),
    )

    args = parser.parse_args(argv)

    # Merge YAML config (CLI flags take priority)
    if args.config is not None:
        args = _merge_yaml_config(args)

    return args


def _merge_yaml_config(args: argparse.Namespace) -> argparse.Namespace:
    """Load a YAML config and let CLI flags override its values."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for --config: pip install pyyaml") from exc

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    # Only set values that were not explicitly supplied on the CLI
    defaults = vars(parse_args([]))  # parser defaults
    cli_overrides = {
        k for k, v in vars(args).items() if v != defaults.get(k) and k != "config"
    }
    for key, value in cfg.items():
        dest = key.replace("-", "_")
        if dest not in cli_overrides:
            setattr(args, dest, value)
    return args


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
) -> float:
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
) -> tuple[float, float]:
    """Return ``(val_loss, primary_metric)``.

    For *transient_attrs* the primary metric is mean attribute error (%).
    For *two_class_weather* it is normalised accuracy (in [0, 1]).
    """
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        total_loss += criterion(outputs, targets).item()
        all_preds.append(outputs.cpu())
        all_targets.append(targets.cpu())

    preds = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)

    if task == "transient_attrs":
        metric = mean_attribute_error(preds, targets_cat)
    else:
        metric = normalised_accuracy(preds, targets_cat)

    return total_loss / len(loader), metric


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metric: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metric": metric,
            "config": model.config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metric", float("inf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    train_transform = get_transform(args.backbone, split="train")
    val_transform = get_transform(args.backbone, split="val")

    if args.task == "transient_attrs":
        train_ds = TransientAttributesDataset(
            args.data_root, split="train", image_dir=args.image_dir,
            transform=train_transform,
        )
        val_ds = TransientAttributesDataset(
            args.data_root, split="test", image_dir=args.image_dir,
            transform=val_transform,
        )
    else:
        train_ds = TwoClassWeatherDataset(
            args.data_root, split="train", fold=args.fold,
            transform=train_transform,
        )
        val_ds = TwoClassWeatherDataset(
            args.data_root, split="test", fold=args.fold,
            transform=val_transform,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(
        f"Train: {len(train_ds)} samples | "
        f"Val: {len(val_ds)} samples | "
        f"Task: {args.task}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.task == "transient_attrs":
        model = TransientNet(
            backbone=args.backbone,
            pretrained=args.pretrained,
            dropout=args.dropout,
        )
        criterion = nn.MSELoss()
    else:
        model = CloudyNet(
            backbone=args.backbone,
            pretrained=args.pretrained,
            dropout=args.dropout,
        )
        criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    # ------------------------------------------------------------------
    # Optionally freeze backbone (only train the task head)
    # ------------------------------------------------------------------
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        n_frozen = sum(p.numel() for p in model.backbone.parameters())
        n_trainable = sum(p.numel() for p in model.head.parameters())
        print(
            f"Backbone frozen ({n_frozen:,} params). "
            f"Training head only ({n_trainable:,} params)."
        )

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    scaler = (
        torch.amp.GradScaler() if args.amp and device.type == "cuda" else None
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    # For transient_attrs lower is better; for two_class_weather higher is better
    best_metric = float("inf") if args.task == "transient_attrs" else -float("inf")

    if args.resume is not None:
        start_epoch, best_metric = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    metric_label = (
        "mean_attr_err(%)" if args.task == "transient_attrs" else "norm_acc"
    )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_metric = evaluate(
            model, val_loader, criterion, device, args.task
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch + 1:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"{metric_label}={val_metric:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # Save latest checkpoint
        save_checkpoint(
            output_dir / "checkpoint_last.pth",
            model, optimizer, scheduler, epoch + 1, val_metric,
        )

        # Save best checkpoint
        is_best = (
            val_metric < best_metric
            if args.task == "transient_attrs"
            else val_metric > best_metric
        )
        if is_best:
            best_metric = val_metric
            save_checkpoint(
                output_dir / "checkpoint_best.pth",
                model, optimizer, scheduler, epoch + 1, val_metric,
            )
            print(f"  ↳ New best {metric_label}={best_metric:.4f}")

    print(f"\nTraining complete. Best {metric_label}: {best_metric:.4f}")
    print(f"Checkpoints saved to: {output_dir}")

    # ------------------------------------------------------------------
    # Push to HuggingFace Hub
    # ------------------------------------------------------------------
    if args.push_to_hub:
        if args.hub_repo is None:
            raise ValueError("--hub-repo USER/REPO is required with --push-to-hub.")
        from deeptransient.hub import push_to_hub

        push_to_hub(
            checkpoint_path=output_dir / "checkpoint_best.pth",
            repo_id=args.hub_repo,
        )


if __name__ == "__main__":
    main()
