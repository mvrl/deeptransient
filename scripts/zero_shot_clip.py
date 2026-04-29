#!/usr/bin/env python3
"""Zero-shot CLIP baselines for TransientNet and CloudyNet.

No training, no checkpoints — just compute image embeddings and compare
them to text-prompt embeddings.  Runs on a single GPU.

Examples
--------
TransientNet (40-dim regression, official 81/20 holdout)::

    python scripts/zero_shot_clip.py \\
        --task transient_attrs \\
        --model ViT-B-32 \\
        --data-root /data/jacobsn/transient_attrs

CloudyNet (5-fold normalised accuracy)::

    python scripts/zero_shot_clip.py \\
        --task two_class_weather \\
        --model ViT-B-32 \\
        --data-root /data/jacobsn/two_class_weather/weather_database

The TransientNet path uses *contrast prompts* per attribute — for each
attribute we compare ``"a [attribute] outdoor scene"`` to its negation
``"a non-[attribute] outdoor scene"`` and softmax the two similarities to
get a probability in [0, 1] which is then compared to the ground-truth
crowd-sourced score with MSE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import open_clip

from deeptransient.data import TransientAttributesDataset, TwoClassWeatherDataset
from deeptransient.metrics import (
    five_fold_normalised_accuracy,
    mean_attribute_error,
)
from deeptransient.models import TRANSIENT_ATTRIBUTES

# Many attribute names are abbreviations or compounds — give CLIP a more
# natural phrasing to embed.  Keys must match TRANSIENT_ATTRIBUTES exactly.
ATTRIBUTE_PHRASES: dict[str, str] = {
    "dirty": "dirty",
    "daylight": "in daylight",
    "night": "at night",
    "sunrisesunset": "at sunrise or sunset",
    "dawndusk": "at dawn or dusk",
    "sunny": "sunny",
    "clouds": "with clouds",
    "fog": "foggy",
    "storm": "stormy",
    "snow": "snowy",
    "warm": "warm",
    "cold": "cold",
    "busy": "busy",
    "beautiful": "beautiful",
    "flowers": "with flowers",
    "spring": "in spring",
    "summer": "in summer",
    "autumn": "in autumn",
    "winter": "in winter",
    "glowing": "glowing",
    "colorful": "colorful",
    "dull": "dull",
    "rugged": "rugged",
    "midday": "at midday",
    "dark": "dark",
    "bright": "bright",
    "dry": "dry",
    "moist": "moist",
    "windy": "windy",
    "rain": "rainy",
    "ice": "icy",
    "cluttered": "cluttered",
    "soothing": "soothing",
    "stressful": "stressful",
    "exciting": "exciting",
    "sentimental": "sentimental",
    "mysterious": "mysterious",
    "boring": "boring",
    "gloomy": "gloomy",
    "lush": "lush",
}


def _device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )


def load_clip(model_name: str, device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device).eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_text(model, tokenizer, prompts: list[str], device) -> torch.Tensor:
    tokens = tokenizer(prompts).to(device)
    embeds = model.encode_text(tokens)
    return F.normalize(embeds, dim=-1)


@torch.no_grad()
def encode_images(model, loader: DataLoader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (image_embeds, targets), L2-normalised image embeddings."""
    all_embeds, all_targets = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        emb = model.encode_image(images)
        emb = F.normalize(emb, dim=-1)
        all_embeds.append(emb.cpu())
        all_targets.append(targets)
    return torch.cat(all_embeds), torch.cat(all_targets)


# ---------------------------------------------------------------------------
# TransientNet zero-shot
# ---------------------------------------------------------------------------


def zero_shot_transient(
    model_name: str,
    data_root: str,
    image_dir: str,
    batch_size: int,
    workers: int,
) -> dict:
    device = _device()
    model, preprocess, tokenizer = load_clip(model_name, device)

    pos_prompts = [f"a photo of a scene that is {ATTRIBUTE_PHRASES[a]}"
                   for a in TRANSIENT_ATTRIBUTES]
    neg_prompts = [f"a photo of a scene that is not {ATTRIBUTE_PHRASES[a]}"
                   for a in TRANSIENT_ATTRIBUTES]

    pos_emb = encode_text(model, tokenizer, pos_prompts, device)  # (40, D)
    neg_emb = encode_text(model, tokenizer, neg_prompts, device)  # (40, D)

    val_ds = TransientAttributesDataset(
        data_root, split="test", image_dir=image_dir, transform=preprocess,
    )
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)

    image_emb, targets = encode_images(model, loader, device)
    image_emb = image_emb.to(device)

    # Cosine similarities (D-dim → scalar per (image, attribute))
    sim_pos = image_emb @ pos_emb.T   # (N, 40)
    sim_neg = image_emb @ neg_emb.T   # (N, 40)

    # Softmax with the model's learned logit_scale → calibrated probabilities.
    scale = model.logit_scale.exp().detach()
    logits = scale * torch.stack([sim_pos, sim_neg], dim=-1)  # (N, 40, 2)
    probs = logits.softmax(dim=-1)[..., 0]                    # (N, 40)
    probs = probs.cpu()

    err = mean_attribute_error(probs, targets)
    per_attr = ((probs - targets) ** 2).mean(dim=0).numpy() * 100.0
    return {
        "model": model_name,
        "n_images": probs.shape[0],
        "mean_attr_error_pct": err,
        "per_attribute": dict(zip(TRANSIENT_ATTRIBUTES, per_attr.tolist())),
    }


# ---------------------------------------------------------------------------
# CloudyNet zero-shot (single-fold and 5-fold)
# ---------------------------------------------------------------------------


SUNNY_PROMPTS = [
    "a photo of a sunny day",
    "a photograph of a clear sunny sky",
    "an outdoor scene under bright sunlight",
    "a sunlit landscape with blue sky",
    "a photo of clear weather with sunshine",
]
CLOUDY_PROMPTS = [
    "a photo of a cloudy day",
    "a photograph of a cloudy overcast sky",
    "an outdoor scene under heavy clouds",
    "a landscape with a grey overcast sky",
    "a photo of overcast weather with no sunshine",
]


def zero_shot_cloudy_one_fold(
    model_name: str,
    data_root: str,
    fold: int,
    batch_size: int,
    workers: int,
    model=None,
    preprocess=None,
    tokenizer=None,
) -> dict:
    """5-prompt ensemble per class — averaged & re-normalised text embedding.

    Prompt ensembling is a well-known trick for CLIP zero-shot
    classification (Radford et al., 2021): the mean of several phrasings
    of the same concept reduces prompt-specific bias.
    """
    device = _device()
    if model is None:
        model, preprocess, tokenizer = load_clip(model_name, device)

    sunny_emb = encode_text(model, tokenizer, SUNNY_PROMPTS, device).mean(0)
    cloudy_emb = encode_text(model, tokenizer, CLOUDY_PROMPTS, device).mean(0)
    text_emb = F.normalize(torch.stack([sunny_emb, cloudy_emb]), dim=-1)

    val_ds = TwoClassWeatherDataset(
        data_root, split="test", fold=fold, transform=preprocess,
    )
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)

    image_emb, targets = encode_images(model, loader, device)
    image_emb = image_emb.to(device)

    logits = image_emb @ text_emb.T  # (N, 2): [sunny_score, cloudy_score]
    pred = logits.argmax(dim=1).cpu()
    acc = (pred == targets).float().mean().item()
    norm_acc = max((acc - 0.5) / 0.5, 0.0) * 100.0
    return {"fold": fold, "acc": acc * 100.0, "norm_acc": norm_acc}


def zero_shot_cloudy_5fold(
    model_name: str,
    data_root: str,
    batch_size: int,
    workers: int,
) -> dict:
    device = _device()
    model, preprocess, tokenizer = load_clip(model_name, device)
    fold_accs = []
    per_fold = []
    for fold in range(5):
        res = zero_shot_cloudy_one_fold(
            model_name, data_root, fold, batch_size, workers,
            model=model, preprocess=preprocess, tokenizer=tokenizer,
        )
        fold_accs.append(res["acc"] / 100.0)
        per_fold.append(res)
    mean_na, std_na = five_fold_normalised_accuracy(fold_accs)
    return {
        "model": model_name,
        "per_fold": per_fold,
        "norm_acc_mean": mean_na,
        "norm_acc_std": std_na,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["transient_attrs", "two_class_weather"],
                        required=True)
    parser.add_argument("--model", default="ViT-B-32",
                        help="open_clip architecture (e.g. ViT-B-32, ViT-L-14).")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-dir", default="imageAlignedLD")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-attribute", action="store_true",
                        help="Print per-attribute MSE for transient_attrs.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON output path; collect_results.py reads "
                             "runs/zero_shot_*.json automatically.")
    args = parser.parse_args(argv)

    if args.task == "transient_attrs":
        res = zero_shot_transient(
            args.model, args.data_root, args.image_dir,
            args.batch_size, args.workers,
        )
        print(f"\nZero-shot CLIP {res['model']} on TransientNet "
              f"({res['n_images']} test images)")
        print(f"Mean attribute error: {res['mean_attr_error_pct']:.2f}%")
        print(f"  (paper TransientNet-H = 3.83%)")
        if args.per_attribute:
            print("\nPer-attribute errors (sorted, worst first):")
            for attr, err in sorted(res["per_attribute"].items(),
                                    key=lambda x: x[1], reverse=True):
                print(f"  {attr:18s} {err:.2f}%")

    else:  # two_class_weather
        res = zero_shot_cloudy_5fold(
            args.model, args.data_root, args.batch_size, args.workers,
        )
        print(f"\nZero-shot CLIP {res['model']} on CloudyNet (5-fold)")
        for fold_res in res["per_fold"]:
            print(f"  Fold {fold_res['fold']}: acc={fold_res['acc']:.2f}%  "
                  f"norm_acc={fold_res['norm_acc']:.2f}%")
        print(f"Normalised accuracy: "
              f"{res['norm_acc_mean']:.2f} ± {res['norm_acc_std']:.2f}%")
        print(f"  (paper CloudyNet-H = 87.1 ± 0.3%)")

    # Save a small JSON sidecar so scripts/collect_results.py can pick this up.
    if args.output is None:
        runs_dir = Path(__file__).resolve().parents[1] / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        slug = args.model.lower().replace("-", "_").replace("/", "_")
        args.output = runs_dir / f"zero_shot_{args.task}_{slug}.json"
    out = {"task": args.task, **res}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
