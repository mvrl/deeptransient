#!/usr/bin/env python3
"""Run TransientNet or CloudyNet on one or more images.

Examples
--------
Run on a single image using a local checkpoint::

    python predict.py \\
        --checkpoint runs/transientnet_resnet50/checkpoint_best.pth \\
        --image /path/to/image.jpg

Run on multiple images and save results to a JSON file::

    python predict.py \\
        --checkpoint runs/transientnet_resnet50/checkpoint_best.pth \\
        --image /path/to/img1.jpg /path/to/img2.jpg \\
        --output predictions.json

Load a pretrained model directly from HuggingFace Hub::

    python predict.py \\
        --hub-repo mvrl/deeptransient-transientnet-resnet50 \\
        --image /path/to/image.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from deeptransient.models import (
    CloudyNet,
    TransientNet,
    TRANSIENT_ATTRIBUTES,
    get_transform,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict transient scene attributes or sunny/cloudy class",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        type=str,
        metavar="PATH",
        help="Path to a trained checkpoint (.pth).",
    )
    source.add_argument(
        "--hub-repo",
        type=str,
        metavar="USER/REPO",
        help="HuggingFace Hub repository id.",
    )

    parser.add_argument(
        "--image",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more image paths.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional JSON output file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top attributes to display (transient_attrs only).",
    )
    return parser.parse_args(argv)


def load_model(checkpoint: Optional[str], hub_repo: Optional[str]):
    if hub_repo is not None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(repo_id=hub_repo, filename="checkpoint_best.pth")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    task = config["task"]

    if task == "transient_attrs":
        model = TransientNet.from_config(config)
    else:
        model = CloudyNet.from_config(config)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = get_transform(config["backbone"], split="val")
    return model, task, transform


def predict_transient(
    model: torch.nn.Module,
    transform,
    image_paths: list[str],
    top_k: int,
) -> list[dict]:
    results = []
    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0)
            scores = model(tensor).squeeze(0)

            attrs = {
                name: float(score)
                for name, score in zip(TRANSIENT_ATTRIBUTES, scores.tolist())
            }
            top = sorted(attrs.items(), key=lambda x: x[1], reverse=True)[:top_k]

            results.append(
                {
                    "image": str(path),
                    "attributes": attrs,
                    "top_attributes": dict(top),
                }
            )
    return results


def predict_cloudy(
    model: torch.nn.Module,
    transform,
    image_paths: list[str],
) -> list[dict]:
    results = []
    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0)
            logits = model(tensor).squeeze(0)
            probs = torch.softmax(logits, dim=0)
            pred_idx = int(probs.argmax())
            results.append(
                {
                    "image": str(path),
                    "prediction": CloudyNet.CLASSES[pred_idx],
                    "probabilities": {
                        cls: float(p)
                        for cls, p in zip(CloudyNet.CLASSES, probs.tolist())
                    },
                }
            )
    return results


def main() -> None:
    args = parse_args()

    model, task, transform = load_model(args.checkpoint, args.hub_repo)

    if task == "transient_attrs":
        results = predict_transient(model, transform, args.image, args.top_k)
        for r in results:
            print(f"\nImage: {r['image']}")
            print(f"Top {args.top_k} attributes:")
            for attr, score in r["top_attributes"].items():
                bar = "█" * int(score * 20)
                print(f"  {attr:20s} {score:.3f}  {bar}")
    else:
        results = predict_cloudy(model, transform, args.image)
        for r in results:
            print(f"\nImage: {r['image']}")
            print(f"  Prediction: {r['prediction']}")
            for cls, prob in r["probabilities"].items():
                bar = "█" * int(prob * 20)
                print(f"  {cls:10s} {prob:.3f}  {bar}")

    if args.output:
        output_path = Path(args.output)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
