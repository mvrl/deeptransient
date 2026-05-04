#!/usr/bin/env python3
"""Push a trained DeepTransient checkpoint to HuggingFace Hub.

This script uploads the best checkpoint, a model card (README.md), and an
optional example image to the specified Hub repository.

Usage::

    python push_to_hub.py \\
        --checkpoint runs/transientnet_resnet50/checkpoint_best.pth \\
        --repo-id mvrl/deeptransient-transientnet-resnet50

Requirements::

    pip install huggingface_hub

You must be logged in::

    huggingface-cli login
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deeptransient.hub import push_to_hub  # noqa: F401


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Push a DeepTransient checkpoint to HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to the checkpoint file (checkpoint_best.pth).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        metavar="USER/REPO",
        help="HuggingFace Hub repository id (will be created if absent).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private.",
    )
    parser.add_argument(
        "--example-image",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional example image to include in the repository.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    push_to_hub(
        checkpoint_path=args.checkpoint,
        repo_id=args.repo_id,
        private=args.private,
        example_image=args.example_image,
    )


if __name__ == "__main__":
    main()

