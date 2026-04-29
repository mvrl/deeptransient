#!/usr/bin/env python3
"""Evaluate a trained TransientNet or CloudyNet checkpoint.

Examples
--------
Evaluate TransientNet on the test holdout::

    python evaluate.py \\
        --checkpoint runs/transientnet_resnet50/checkpoint_best.pth \\
        --data-root /path/to/transient_attrs

Evaluate CloudyNet across all five folds (requires five checkpoints)::

    python evaluate.py \\
        --task two_class_weather \\
        --checkpoints \\
            runs/cloudynet_fold0/checkpoint_best.pth \\
            runs/cloudynet_fold1/checkpoint_best.pth \\
            runs/cloudynet_fold2/checkpoint_best.pth \\
            runs/cloudynet_fold3/checkpoint_best.pth \\
            runs/cloudynet_fold4/checkpoint_best.pth \\
        --data-root /path/to/weather

Load a model from HuggingFace Hub::

    python evaluate.py \\
        --hub-repo mvrl/deeptransient-transientnet-resnet50 \\
        --data-root /path/to/transient_attrs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from deeptransient.data import TransientAttributesDataset, TwoClassWeatherDataset
from deeptransient.metrics import (
    five_fold_normalised_accuracy,
    mean_attribute_error,
    normalised_accuracy,
)
from deeptransient.models import CloudyNet, TransientNet, TRANSIENT_ATTRIBUTES


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepTransient checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- model source (one of the following) ----------------------------
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        type=str,
        metavar="PATH",
        help="Single checkpoint file (transient_attrs or one CloudyNet fold).",
    )
    source.add_argument(
        "--hub-repo",
        type=str,
        metavar="USER/REPO",
        help="HuggingFace Hub repository containing a 'checkpoint_best.pth' file.",
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        metavar="PATH",
        default=None,
        help="Five CloudyNet checkpoints (one per fold) for full evaluation.",
    )

    # ---- task & data ----------------------------------------------------
    parser.add_argument(
        "--task",
        choices=["transient_attrs", "two_class_weather"],
        default=None,
        help="Task.  Inferred from checkpoint config when omitted.",
    )
    parser.add_argument("--data-root", type=str, required=True, metavar="PATH")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="imageLD",
        help="Image sub-directory (transient_attrs only).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--per-attribute",
        action="store_true",
        help="Print per-attribute errors (transient_attrs only).",
    )
    return parser.parse_args(argv)


def load_model_from_checkpoint(ckpt_path: str | Path) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    task = config["task"]
    if task == "transient_attrs":
        model = TransientNet.from_config(config)
    else:
        model = CloudyNet.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def load_model_from_hub(repo_id: str) -> tuple[torch.nn.Module, dict]:
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id=repo_id, filename="checkpoint_best.pth")
    return load_model_from_checkpoint(ckpt_path)


@torch.no_grad()
def eval_transient(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    per_attribute: bool = False,
) -> dict:
    model.eval()
    all_preds, all_targets = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        preds = model(images).cpu()
        all_preds.append(preds)
        all_targets.append(targets)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    per_attr_mse = ((preds - targets) ** 2).mean(dim=0).numpy()
    mean_err = float(per_attr_mse.mean()) * 100.0

    result = {"mean_attr_error_pct": mean_err}

    if per_attribute:
        result["per_attribute"] = {
            name: float(mse) * 100.0
            for name, mse in zip(TRANSIENT_ATTRIBUTES, per_attr_mse)
        }

    return result


@torch.no_grad()
def eval_cloudy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds, all_targets = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).cpu()
        all_preds.append(logits)
        all_targets.append(targets)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    acc = (preds.argmax(dim=1) == targets).float().mean().item()
    norm_acc = max((acc - 0.5) / 0.5, 0.0) * 100.0
    return {"accuracy": acc * 100.0, "normalised_accuracy": norm_acc}


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # ------------------------------------------------------------------
    # Determine task
    # ------------------------------------------------------------------
    # Build checkpoint list
    if args.checkpoints is not None:
        ckpt_list = args.checkpoints
    elif args.checkpoint is not None:
        ckpt_list = [args.checkpoint]
    else:
        ckpt_list = None  # hub

    # Infer task from first checkpoint when not given
    task = args.task
    if task is None:
        if ckpt_list:
            _, cfg = load_model_from_checkpoint(ckpt_list[0])
            task = cfg["task"]
        elif args.hub_repo:
            _, cfg = load_model_from_hub(args.hub_repo)
            task = cfg["task"]

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    if task == "transient_attrs":
        val_ds = TransientAttributesDataset(
            args.data_root, split="test", image_dir=args.image_dir
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )

        if args.hub_repo:
            model, _ = load_model_from_hub(args.hub_repo)
        else:
            model, _ = load_model_from_checkpoint(ckpt_list[0])

        model = model.to(device)
        results = eval_transient(model, val_loader, device, args.per_attribute)

        print(f"\n{'='*50}")
        print(f"TransientNet evaluation on {len(val_ds)} test images")
        print(f"{'='*50}")
        print(f"Mean attribute error: {results['mean_attr_error_pct']:.2f}%")
        print(f"  (paper reference: TransientNet-H = 3.83%)")

        if args.per_attribute and "per_attribute" in results:
            print("\nPer-attribute errors:")
            for attr, err in sorted(
                results["per_attribute"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {attr:20s}: {err:.2f}%")

    else:  # two_class_weather
        if args.hub_repo:
            # Single model evaluation
            model, _ = load_model_from_hub(args.hub_repo)
            model = model.to(device)
            val_ds = TwoClassWeatherDataset(args.data_root, split="test", fold=0)
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
            )
            res = eval_cloudy(model, val_loader, device)
            print(f"\nCloudyNet evaluation (single fold)")
            print(f"Accuracy: {res['accuracy']:.2f}%")
            print(f"Normalised accuracy: {res['normalised_accuracy']:.2f}%")
        else:
            # Five-fold evaluation
            folds = ckpt_list if len(ckpt_list) > 1 else ckpt_list * 5
            fold_accs = []
            for fold_idx, ckpt in enumerate(folds[:5]):
                model, _ = load_model_from_checkpoint(ckpt)
                model = model.to(device)
                val_ds = TwoClassWeatherDataset(
                    args.data_root, split="test", fold=fold_idx
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                )
                res = eval_cloudy(model, val_loader, device)
                fold_accs.append(res["accuracy"] / 100.0)
                print(
                    f"  Fold {fold_idx}: acc={res['accuracy']:.2f}%  "
                    f"norm_acc={res['normalised_accuracy']:.2f}%"
                )

            mean_na, std_na = five_fold_normalised_accuracy(fold_accs)
            print(f"\n{'='*50}")
            print(f"CloudyNet 5-fold evaluation")
            print(f"{'='*50}")
            print(f"Normalised accuracy: {mean_na:.1f} ± {std_na:.1f}%")
            print(f"  (paper reference: CloudyNet-H = 87.1 ± 0.3%)")


if __name__ == "__main__":
    main()
