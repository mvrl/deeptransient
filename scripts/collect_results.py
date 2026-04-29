#!/usr/bin/env python3
"""Collect best-checkpoint metrics from runs/ and emit a Markdown table.

Usage:
    python scripts/collect_results.py [--write-readme]

Writes a Markdown table to stdout.  With --write-readme the table is
spliced into README.md between the markers
``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
README = ROOT / "README.md"

# Display order and labels.  Each row may correspond to multiple runs (one per fold).
TRANSIENT_ROWS: list[tuple[str, str, str]] = [
    # (display label, run-dir glob, paper number)
    ("Laffont et al. (2014) — paper",                "—",                                       "4.2"),
    ("TransientNet-I (AlexNet, ImageNet) — paper",   "—",                                       "4.05"),
    ("TransientNet-P (AlexNet, Places365) — paper",  "—",                                       "3.87"),
    ("TransientNet-H (AlexNet, Hybrid) — paper",     "—",                                       "3.83"),
    ("",                                              "",                                        ""),
    ("AlexNet-I  (ImageNet, SGD)",                    "transientnet_alexnet_imagenet",          ""),
    ("AlexNet-P  (Places365, SGD)",                   "transientnet_alexnet_places365",         ""),
    ("ResNet-18  (ImageNet, Adam)",                   "transientnet_resnet18_imagenet_adam",    ""),
    ("ResNet-50  (ImageNet, Adam)",                   "transientnet_resnet50_imagenet_adam",    ""),
    ("ResNet-50  (Places365, Adam)",                  "transientnet_resnet50_places365_adam",   ""),
    ("EfficientNet-B0 (ImageNet, Adam)",              "transientnet_efficientnet_b0_imagenet_adam", ""),
    ("ViT-B/16    (ImageNet-1K, AdamW)",              "transientnet_vit_b16_imagenet_adam",     ""),
    ("CLIP ViT-B/32  (frozen, linear probe, Adam)",   "transientnet_clip_b32_frozen_adam",      ""),
    ("CLIP ViT-B/32  (full fine-tune from frozen)",   "transientnet_clip_b32_finetune",         ""),
    ("CLIP ViT-L/14  (frozen, linear probe, Adam)",   "transientnet_clip_l14_frozen_adam",      ""),
]

CLOUDY_ROWS: list[tuple[str, str, str]] = [
    ("Lu et al. (2014) — paper",                     "—",                                       "53.1 ± 2.2"),
    ("CloudyNet-I (AlexNet, ImageNet) — paper",      "—",                                       "85.7 ± 0.5"),
    ("CloudyNet-P (AlexNet, Places365) — paper",     "—",                                       "86.1 ± 0.6"),
    ("CloudyNet-H (AlexNet, Hybrid) — paper",        "—",                                       "87.1 ± 0.3"),
]


def best_metric(run_name: str) -> str:
    if not run_name or run_name == "—":
        return "—"
    ckpt_path = RUNS_DIR / run_name / "checkpoint_best.pth"
    if not ckpt_path.exists():
        return "*pending*"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metric = ckpt.get("metric")
    if metric is None:
        return "?"
    return f"{metric:.2f}"


def cloudy_5fold(run_prefix: str) -> str:
    """Aggregate normalised accuracy ± std across folds 0..4."""
    accs = []
    for fold in range(5):
        ckpt_path = RUNS_DIR / f"{run_prefix}_fold{fold}" / "checkpoint_best.pth"
        if not ckpt_path.exists():
            return "*pending*"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Stored metric is normalised accuracy in [0, 1].
        accs.append(ckpt["metric"] * 100.0)
    import statistics
    return f"{statistics.mean(accs):.1f} ± {statistics.stdev(accs):.1f}"


def render_transient_table() -> str:
    lines = [
        "| Method | This repo (mean MSE × 100, ↓) | Paper |",
        "|--------|------------------------------:|------:|",
    ]
    for label, run_name, paper in TRANSIENT_ROWS:
        if not label:
            lines.append("|        |                                |       |")
            continue
        if run_name == "—":
            ours = "—"
        else:
            ours = best_metric(run_name)
        paper_disp = paper or "—"
        lines.append(f"| {label} | {ours} | {paper_disp} |")
    return "\n".join(lines)


def render_cloudy_table() -> str:
    lines = [
        "| Method | This repo (norm. acc., ↑) | Paper |",
        "|--------|--------------------------:|------:|",
    ]
    for label, _run_name, paper in CLOUDY_ROWS:
        lines.append(f"| {label} | — | {paper} |")
    # Our reproductions, if any.
    for backbone in ("alexnet", "resnet50", "clip_vit_b32"):
        for init in ("imagenet", "places365", "clip"):
            prefix = f"cloudynet_{backbone}_{init}"
            if not (RUNS_DIR / f"{prefix}_fold0").exists():
                continue
            lines.append(f"| CloudyNet {backbone} ({init}, 5-fold) | {cloudy_5fold(prefix)} | — |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-readme", action="store_true",
                        help="Splice tables into README.md.")
    args = parser.parse_args()

    transient_md = render_transient_table()
    cloudy_md = render_cloudy_table()

    block = (
        "### Transient attribute prediction\n\n"
        f"{transient_md}\n\n"
        "### Two-class weather classification (5-fold normalised accuracy)\n\n"
        f"{cloudy_md}\n"
    )

    print(block)

    if args.write_readme:
        text = README.read_text()
        start = "<!-- RESULTS:START -->"
        end = "<!-- RESULTS:END -->"
        if start not in text or end not in text:
            print(f"\n[warn] {start} / {end} markers not found in {README}; not written.")
            return
        before, _, rest = text.partition(start)
        _, _, after = rest.partition(end)
        new = f"{before}{start}\n\n{block}\n{end}{after}"
        README.write_text(new)
        print(f"\n[ok] README updated at {README}")


if __name__ == "__main__":
    main()
